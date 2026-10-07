"""M5 / M6：会话 JSONL、大输出落盘、微压缩 / 摘要压缩、AGENTS.md、Skills、子智能体、MCP、@ 文件引用。"""

import json
import os
import sys
from pathlib import Path

import pytest

from coda.agent.events import (
    Compacted,
    Decision,
    DenyApprover,
    ListSink,
    PermissionLog,
    SubagentUpdate,
)
from coda.agent.loop import Agent
from coda.config import ContextConfig, McpServerConfig, Permissions
from coda.context.compact import SUMMARY_MARK, ContextManager
from coda.context.memory import add_memory, load_memory
from coda.context.skills import discover_skills
from coda.llm import ContextTooLongError
from coda.state.session import Session, list_sessions, load_session, repair_messages
from tests.fakes import FakeLLM, Step, call

DATA = Path(__file__).parent / "data"


class AllowAll:
    def __init__(self):
        self.requests = []

    def ask(self, request, cancel):
        self.requests.append(request)
        return Decision(True)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def make(repo, steps, *, mode="default", approver=None, llm=None, **kw):
    sink = ListSink()
    llm = llm or FakeLLM(steps)
    kw.setdefault("system_prompt", "sys")
    agent = Agent(llm, repo, sink, approver or DenyApprover(), mode=mode, **kw)
    return agent, llm, sink


def tool_msgs(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


# ---------------------------------------------------------------- 会话 JSONL


def test_session_jsonl_roundtrip_keeps_reasoning(repo):
    session = Session.create(repo, "fake")
    steps = [
        Step(thinking="先读文件", calls=[call("read_file", {"path": "app.py"})]),
        Step(thinking="读完了", text="f 返回 1"),
    ]
    agent, _, _ = make(repo, steps, session=session)
    agent.run_turn("看看 app.py")
    assert session.path.is_file()
    loaded = load_session(session.path)
    assert loaded.messages == agent.messages
    assert loaded.messages[2]["reasoning_content"] == "先读文件"
    assert loaded.user_inputs == ["看看 app.py"] and loaded.turns == 1
    assert loaded.usage.input_tokens == agent.llm.tracker.total.input_tokens
    kinds = [json.loads(line)["type"] for line in session.path.read_text().splitlines()]
    assert kinds[0] == "meta" and "tool" in kinds and kinds[-1] == "turn_end"
    infos = list_sessions(repo)
    assert len(infos) == 1 and infos[0].first_prompt == "看看 app.py" and infos[0].turns == 1


def test_session_created_lazily(repo):
    session = Session.create(repo)
    make(repo, [], session=session)
    assert not session.path.exists()


def test_repair_missing_tool_results():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "grep", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
    ]
    fixed, repaired = repair_messages(msgs)
    assert repaired and fixed[-1]["tool_call_id"] == "b"
    assert fixed[-1]["content"].startswith("Error[interrupted]")


def test_resume_continues_conversation(repo):
    session = Session.create(repo)
    agent, _, _ = make(repo, [Step(text="记住了")], session=session)
    agent.run_turn("记住数字 7")

    agent2, llm2, _ = make(repo, [Step(text="7")])
    agent2.load(load_session(session.path), Session.open(session.path))
    agent2.run_turn("刚才的数字？")
    sent = llm2.requests[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "记住数字 7"
    # 继续写同一个文件，再次加载能看到两轮
    again = load_session(session.path)
    assert again.turns == 2 and again.messages[-1]["content"] == "7"


def test_permission_decisions_recorded(repo):
    session = Session.create(repo)
    steps = [Step(calls=[call("bash", {"command": "touch x"})]), Step(text="好")]
    agent, _, sink = make(repo, steps, session=session)
    agent.run_turn("建文件")
    assert sink.of(PermissionLog)[0].decision == "denied"
    recs = [json.loads(x) for x in session.path.read_text().splitlines()]
    perm = [r for r in recs if r["type"] == "permission"]
    assert perm and perm[0]["tool"] == "bash" and perm[0]["decision"] == "denied"


# ---------------------------------------------------------------- 大输出落盘


def test_large_output_offloaded_and_readable(repo):
    cmd = (
        f"{sys.executable} -c \"print('\\n'.join('line %d ' % i + 'x' * 60 for i in range(500)))\""
    )
    steps = [Step(calls=[call("bash", {"command": cmd})]), Step(text="看到了")]
    agent, _, _ = make(repo, steps, mode="yolo")
    agent.run_turn("跑一下")
    text = tool_msgs(agent.messages)[0]
    assert "完整内容已保存到" in text and len(text) < 8000
    assert "line 0 " in text and "line 499 " in text and "line 250 " not in text
    saved = Path(text.split("完整内容已保存到 ", 1)[1].split("。", 1)[0])
    assert saved.is_file() and "line 250 " in saved.read_text()

    # 落盘目录在工作区外，但 read_file 不需要询问（DenyApprover 下也能读）
    agent.set_mode("default")
    agent.llm.steps = [
        # 落盘文件前 3 行是 exit_code 和 <stdout>，所以第 250 行输出在文件第 253 行
        Step(calls=[call("read_file", {"path": str(saved), "offset": 253, "limit": 2})]),
        Step(text="ok"),
    ]
    agent.run_turn("读中间")
    assert "line 250 " in tool_msgs(agent.messages)[-1]


def test_offload_disabled_truncates(repo):
    cmd = f"{sys.executable} -c \"print('y' * 50000)\""
    steps = [Step(calls=[call("bash", {"command": cmd})]), Step(text="ok")]
    agent, _, _ = make(repo, steps, mode="yolo", context=ContextConfig(offload=False))
    agent.run_turn("x")
    text = tool_msgs(agent.messages)[0]
    assert "完整内容已保存到" not in text and "中间省略" in text and len(text) < 31000


# ---------------------------------------------------------------- 微压缩 / 摘要压缩


def _big_files(repo, n):
    for i in range(n):
        (repo / f"f{i}.py").write_text("".join(f"# {i} {'z' * 36}\n" for _ in range(100)))


def test_micro_compaction_clears_old_tool_results(repo):
    _big_files(repo, 6)  # 每个约 4k 字符 ≈ 1.3k token
    reads = [call("read_file", {"path": f"f{i}.py"}) for i in range(6)]
    steps = [
        Step(calls=reads, input_tokens=100),
        Step(text="读完了", input_tokens=14_000),  # 预算 20k：70%，超过 60% 微压缩阈值
        Step(text="继续"),
    ]
    llm = FakeLLM(steps, budget=20_000)
    agent, _, sink = make(repo, [], llm=llm)
    agent.run_turn("读四个文件")
    assert not sink.of(Compacted)
    agent.run_turn("然后呢")
    (c,) = sink.of(Compacted)
    assert c.kind == "micro" and c.after < c.before
    sent_tools = [m["content"] for m in llm.requests[-1] if m["role"] == "tool"]
    assert all(t.startswith("[已清理：read_file f") for t in sent_tools[:3])
    assert all(not t.startswith("[已清理") for t in sent_tools[3:])  # 保留最近 3 次
    assert not llm.summary_requests


def test_summary_compaction_keeps_user_words(repo):
    steps = [
        Step(text="好的，不改公共接口", input_tokens=9000),  # 90%：超过摘要阈值
        Step(text="继续做"),
    ]
    llm = FakeLLM(steps, summaries=["## 任务目标\n重构 retriever"], budget=10_000)
    session = Session.create(repo)
    agent, _, sink = make(repo, [], llm=llm, session=session)
    agent.ctx.todos = [{"content": "改 retriever", "status": "in_progress"}]
    agent.run_turn("重构 retriever，不要改公共接口")
    agent.run_turn("第二步")
    (c,) = sink.of(Compacted)
    assert c.kind == "summary"
    sent = llm.requests[-1]
    assert [m["role"] for m in sent] == ["system", "user", "user"]
    summary = sent[1]["content"]
    assert SUMMARY_MARK in summary and "重构 retriever，不要改公共接口" in summary
    assert "## 任务目标\n重构 retriever" in summary and "改 retriever" in summary
    assert sent[2]["content"] == "第二步"
    # 给摘要模型的记录里有用户原话；会话文件恢复后从压缩后的状态开始
    assert "不要改公共接口" in llm.summary_requests[0][1]["content"]
    assert load_session(session.path).messages[:3] == agent.messages[:3]


def test_summary_cut_never_orphans_tool_messages():
    cfg = ContextConfig()
    mgr = ContextManager(cfg, lambda: 10_000, lambda msgs: "摘要")

    def tc(i):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "grep", "arguments": "{}"}}
            ],
        }

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "一个很长的任务"}]
    for i in range(6):
        msgs += [tc(i), {"role": "tool", "tool_call_id": f"c{i}", "content": "r" * 3000}]
    msgs.append({"role": "assistant", "content": "做完了"})
    result = mgr.summary(msgs, 9000, ["一个很长的任务"], "", "", keep_tokens=2500)
    assert result is not None
    tail = result.messages[2:]
    assert tail[0]["role"] == "assistant"  # 当前这一轮太长：退到两步之间切
    ids = {c["id"] for m in tail for c in m.get("tool_calls") or []}
    assert all(m["tool_call_id"] in ids for m in tail if m["role"] == "tool")
    assert "一个很长的任务" in result.messages[1]["content"]


def test_context_overflow_compacts_and_retries(repo):
    steps = [
        Step(text="a"),
        Step(error=ContextTooLongError("maximum context length exceeded")),
        Step(text="压缩后好了"),
    ]
    llm = FakeLLM(steps, summaries=["之前聊了 a"])
    agent, _, sink = make(repo, [], llm=llm)
    agent.run_turn("一")
    end = agent.run_turn("二")
    assert end.status == "done" and end.final_text == "压缩后好了"
    assert sink.of(Compacted)[0].reason == "overflow"
    assert SUMMARY_MARK in llm.requests[-1][1]["content"]


def test_manual_compact_with_focus(repo):
    llm = FakeLLM([Step(text="a"), Step(text="b")], summaries=["s"])
    agent, _, _ = make(repo, [], llm=llm)
    agent.run_turn("一")
    agent.run_turn("二")
    result = agent.compact("关注接口")
    assert result is not None and result.kind == "summary"
    assert "关注接口" in llm.summary_requests[0][0]["content"]
    assert len(agent.messages) < 5


# ---------------------------------------------------------------- AGENTS.md 与 Skills


def test_agents_md_in_system_prompt(repo, isolated_home):
    (isolated_home / "AGENTS.md").write_text("回答用中文")
    (repo / "AGENTS.md").write_text("测试命令：uv run pytest")
    agent, _, _ = make(repo, [], system_prompt=None)
    prompt = agent.messages[0]["content"]
    assert "回答用中文" in prompt and "测试命令：uv run pytest" in prompt
    assert prompt.index("回答用中文") < prompt.index("测试命令")  # 个人在前，项目在后
    assert "pytest-debug" in prompt  # Skills 目录


def test_memory_add(repo):
    path = add_memory(repo, '"不要改公共接口"')
    add_memory(repo, "函数要有类型注解")
    text = path.read_text()
    assert "## 约定" in text and "- 不要改公共接口\n- 函数要有类型注解" in text
    assert load_memory(repo)[-1].text.endswith("函数要有类型注解")


def test_skills_discovery_and_load(repo):
    d = repo / ".coda" / "skills" / "pytest-debug"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: pytest-debug\ndescription: 项目版\n---\n项目自己的步骤")
    skills = discover_skills(repo)
    assert skills["pytest-debug"].source == "项目" and skills["git-commit"].source == "内置"
    steps = [Step(calls=[call("load_skill", {"name": "pytest-debug"})]), Step(text="ok")]
    agent, _, _ = make(repo, steps)
    agent.run_turn("测试挂了")
    assert "项目自己的步骤" in tool_msgs(agent.messages)[0]


# ---------------------------------------------------------------- 子智能体


def test_subagent_isolated_and_readonly(repo):
    main = [
        Step(calls=[call("task", {"description": "查 f", "prompt": "f 在哪里定义？"})]),
        Step(text="f 在 app.py"),
    ]
    sub = [
        Step(calls=[call("bash", {"command": "ls"}), call("grep", {"pattern": "def f"})]),
        Step(text="结论：f 定义在 app.py:1"),
    ]
    llm = FakeLLM(main, sub_steps=sub)
    agent, _, sink = make(repo, [], llm=llm)
    agent.run_turn("f 在哪")
    (result,) = tool_msgs(agent.messages)
    assert "结论：f 定义在 app.py:1" in result and "子智能体「查 f」" in result
    # 子智能体没有 bash，中间的搜索结果不进主对话
    sub_tools = [m["content"] for m in llm.sub_requests[-1] if m["role"] == "tool"]
    assert sub_tools[0].startswith("Error[invalid_args]") and "'bash'" in sub_tools[0]
    assert "app.py:1:def f" in sub_tools[1]
    assert all(
        "def f():" not in m.get("content", "") for m in llm.requests[-1] if m["role"] == "tool"
    )
    ups = sink.of(SubagentUpdate)
    assert ups and ups[-1].step == 2 and any(u.tool == "grep" for u in ups)
    # 子智能体的用量计入总量，但不影响主对话的上下文估算（main=False）
    assert llm.tracker.last_input_tokens == 200 and llm.tracker.calls == 4


def test_parallel_subagents(repo):
    tasks = [call("task", {"description": f"查{i}", "prompt": f"问题{i}"}) for i in range(3)]
    main = [Step(calls=tasks), Step(text="汇总")]
    sub = [Step(calls=[call("glob", {"pattern": "*.py"})]) for _ in range(3)]
    sub += [Step(text="结论") for _ in range(3)]
    llm = FakeLLM(main, sub_steps=sub)
    agent, _, _ = make(repo, [], llm=llm)
    end = agent.run_turn("并行查")
    assert end.status == "done"
    results = tool_msgs(agent.messages)
    assert len(results) == 3 and all("结论" in r for r in results)


def test_subagent_allowed_in_plan(repo):
    # task 是只读工具，plan 模式也能用
    main = [Step(calls=[call("task", {"description": "x", "prompt": "y"})]), Step(text="ok")]
    llm = FakeLLM(main, sub_steps=[Step(text="结论")])
    agent, _, _ = make(repo, [], llm=llm, mode="plan")
    agent.run_turn("调查")
    assert "结论" in tool_msgs(agent.messages)[0]


# ---------------------------------------------------------------- MCP


@pytest.fixture(scope="module")
def mcp_manager(tmp_path_factory):
    from coda.mcp_client import McpManager

    old = os.environ.get("CODA_HOME")
    os.environ["CODA_HOME"] = str(tmp_path_factory.mktemp("mcp_home"))
    servers = {
        "echo": McpServerConfig(command=sys.executable, args=[str(DATA / "echo_mcp_server.py")]),
        "broken": McpServerConfig(command="coda-no-such-command"),
    }
    mgr = McpManager(servers, Path.cwd())
    mgr.start()
    mgr.wait_ready()
    yield mgr
    mgr.stop()
    if old is None:
        os.environ.pop("CODA_HOME", None)
    else:
        os.environ["CODA_HOME"] = old


def test_mcp_connect_and_status(mcp_manager):
    assert mcp_manager.servers["echo"].status == "connected"
    assert mcp_manager.servers["broken"].status == "failed"
    status = mcp_manager.status_text()
    assert "mcp__echo__echo" in status and "找不到命令" in status
    schema = next(t for t in mcp_manager.tools() if t.name == "mcp__echo__echo").schema()
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_mcp_tool_asks_by_default(repo, mcp_manager):
    steps = [Step(calls=[call("mcp__echo__echo", {"text": "你好"})]), Step(text="ok")]
    approver = AllowAll()
    agent, _, _ = make(repo, steps, approver=approver)
    mcp_manager.register(agent.tools)
    agent.run_turn("调 MCP")
    assert approver.requests[0].always == ["mcp__echo__echo"]
    assert tool_msgs(agent.messages)[0] == "echo: 你好"


def test_mcp_allow_rule_plan_and_errors(repo, mcp_manager):
    steps = [
        Step(calls=[call("mcp__echo__echo", {"text": "a"}), call("mcp__echo__fail", {})]),
        Step(text="ok"),
    ]
    agent, _, _ = make(repo, steps, permissions=Permissions(allow=["mcp__echo__*"]))
    mcp_manager.register(agent.tools)
    agent.run_turn("x")
    ok, err = tool_msgs(agent.messages)
    assert ok == "echo: a" and err.startswith("Error[tool_error]")

    plan, _, _ = make(
        repo, [Step(calls=[call("mcp__echo__echo", {"text": "b"})]), Step(text="ok")], mode="plan"
    )
    mcp_manager.register(plan.tools)
    plan.run_turn("y")
    assert tool_msgs(plan.messages)[0].startswith("Error[denied]")


# ---------------------------------------------------------------- @ 文件引用


def test_at_mention_attaches_file_and_counts_as_read(repo):
    edit = {"path": "app.py", "old": "return 1", "new": "return 2"}
    steps = [Step(calls=[call("edit_file", edit)]), Step(text="改好了")]
    agent, llm, _ = make(repo, steps, mode="accept-edits")
    agent.run_turn("把 @app.py 的返回值改成 2")
    sent = llm.requests[0][-1]["content"]
    assert sent.startswith("把 @app.py 的返回值改成 2") and "2→    return 1" in sent
    assert "return 2" in (repo / "app.py").read_text()
    assert agent.user_inputs == ["把 @app.py 的返回值改成 2"]  # 原话不含附件


def test_headless_json_has_session(repo, capsys):
    from coda.config import load_settings
    from coda.headless import run_headless

    llm = FakeLLM([Step(text="你好")])
    code = run_headless("hi", load_settings(repo), repo, output="json", llm=llm)
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and Path(data["session"]).is_file() and data["compactions"] == []
