"""主循环、执行器与权限：用 FakeLLM 回放脚本，断言消息历史和事件序列。"""

import threading

import pytest

from coda.agent.events import (
    Decision,
    DenyApprover,
    ListSink,
    TextDelta,
    ToolEnd,
    TurnEnd,
)
from coda.agent.loop import Agent
from coda.safety.shell import is_readonly_command
from tests.fakes import FakeLLM, Step, call


class ScriptedApprover:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.requests = []

    def ask(self, request, cancel):
        self.requests.append(request)
        return self.decisions.pop(0)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def make(repo, steps, approver=None, mode="default", **kw):
    sink = ListSink()
    llm = FakeLLM(steps)
    agent = Agent(llm, repo, sink, approver or DenyApprover(), mode=mode, system_prompt="sys", **kw)
    return agent, llm, sink


def test_plain_answer(repo):
    agent, llm, sink = make(repo, [Step(text="你好")])
    end = agent.run_turn("hi")
    assert end.status == "done" and end.final_text == "你好"
    assert "".join(e.text for e in sink.of(TextDelta)) == "你好"
    assert agent.messages[-1] == {"role": "assistant", "content": "你好"}


def test_tool_roundtrip_and_reasoning_kept(repo):
    steps = [
        Step(thinking="先读文件", calls=[call("read_file", {"path": "app.py"})]),
        Step(text="f 返回 1"),
    ]
    agent, llm, sink = make(repo, steps)
    agent.run_turn("f 返回什么")
    second = llm.requests[1]
    assert second[2]["reasoning_content"] == "先读文件"
    assert second[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second[3]["role"] == "tool" and "1→def f():" in second[3]["content"]
    assert [e.name for e in sink.of(ToolEnd)] == ["read_file"]


def test_invalid_json_args_fed_back(repo):
    steps = [Step(calls=[call("read_file", '{"path": "app.py"')]), Step(text="好")]
    agent, llm, _ = make(repo, steps)
    assert agent.run_turn("x").status == "done"
    assert llm.requests[1][-1]["content"].startswith("Error[invalid_args]: 参数不是合法的 JSON")


def test_validation_and_unknown_tool(repo):
    steps = [Step(calls=[call("read_file", {"offset": 0}), call("nope")]), Step(text="好")]
    agent, llm, _ = make(repo, steps)
    agent.run_turn("x")
    tool_msgs = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert "path: Field required" in tool_msgs[0]["content"]
    assert "没有名为 'nope' 的工具" in tool_msgs[1]["content"]


def test_edit_asks_in_default_and_denial_reason_fed_back(repo):
    steps = [
        Step(calls=[call("read_file", {"path": "app.py"})]),
        Step(calls=[call("edit_file", {"path": "app.py", "old": "return 1", "new": "return 2"})]),
        Step(text="好的，不改了"),
    ]
    approver = ScriptedApprover(Decision(False, reason="别改返回值"))
    agent, llm, _ = make(repo, steps, approver)
    agent.run_turn("改一下")
    assert approver.requests[0].tool == "edit_file"
    assert "+    return 2" in approver.requests[0].preview
    last = llm.requests[2][-1]["content"]
    assert last.startswith("Error[denied]") and "别改返回值" in last
    assert "return 1" in (repo / "app.py").read_text()


def test_always_allow_remembered(repo):
    edit = {"path": "app.py", "old": "return 1", "new": "return 2"}
    edit2 = {"path": "app.py", "old": "return 2", "new": "return 3"}
    steps = [
        Step(calls=[call("read_file", {"path": "app.py"})]),
        Step(calls=[call("edit_file", edit)]),
        Step(calls=[call("edit_file", edit2)]),
        Step(text="done"),
    ]
    approver = ScriptedApprover(Decision(True, always=True))
    agent, _, _ = make(repo, steps, approver)
    agent.run_turn("x")
    assert len(approver.requests) == 1 and "return 3" in (repo / "app.py").read_text()


def test_headless_denies_bash_but_allows_readonly(repo):
    steps = [
        Step(calls=[call("bash", {"command": "ls"}), call("bash", {"command": "touch x"})]),
        Step(text="ok"),
    ]
    agent, llm, _ = make(repo, steps)
    agent.run_turn("x")
    tools = [m["content"] for m in llm.requests[1] if m["role"] == "tool"]
    assert tools[0].startswith("exit_code: 0") and "app.py" in tools[0]
    assert tools[1].startswith("Error[denied]") and "无头模式" in tools[1]
    assert not (repo / "x").exists()


def test_plan_mode_denies_edits(repo):
    steps = [Step(calls=[call("write_file", {"path": "n.py", "content": "x"})]), Step(text="ok")]
    agent, llm, _ = make(repo, steps, mode="plan")
    agent.run_turn("x")
    assert "plan 模式" in llm.requests[1][-1]["content"]


def test_write_outside_workspace_denied(repo, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "x.py"
    steps = [
        Step(calls=[call("write_file", {"path": str(outside), "content": "x"})]),
        Step(text="ok"),
    ]
    agent, llm, _ = make(repo, steps, mode="yolo")
    agent.run_turn("x")
    assert "工作区外" in llm.requests[1][-1]["content"] and not outside.exists()


def test_parallel_readonly_keeps_order(repo):
    (repo / "b.py").write_text("B\n")
    calls = [
        call("read_file", {"path": "app.py"}),
        call("read_file", {"path": "b.py"}),
        call("grep", {"pattern": "B"}),
    ]
    agent, llm, _ = make(repo, [Step(calls=calls), Step(text="ok")])
    agent.run_turn("x")
    msgs = [m for m in llm.requests[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in msgs] == ["call_1", "call_2", "call_3"]
    assert "def f" in msgs[0]["content"] and "B" in msgs[1]["content"]


def test_repeat_failure_hint(repo):
    bad = call("read_file", {"path": "missing.py"})
    steps = [Step(calls=[bad]), Step(calls=[bad]), Step(calls=[bad]), Step(text="放弃")]
    agent, llm, _ = make(repo, steps)
    agent.run_turn("x")
    assert "连续失败 3 次" in llm.requests[3][-1]["content"]
    assert "连续失败" not in llm.requests[2][-1]["content"]


def test_max_steps(repo):
    steps = [Step(calls=[call("glob", {"pattern": "*.py"})]) for _ in range(3)]
    agent, _, _ = make(repo, steps, max_steps=3)
    assert agent.run_turn("x").status == "max_steps"


def test_interrupt_during_stream_then_continue(repo):
    steps = [Step(text="很长的回答" * 50, delay=0.005), Step(text="继续")]
    agent, llm, _ = make(repo, steps)
    threading.Timer(0.05, agent.interrupt).start()
    end = agent.run_turn("x")
    assert end.status == "interrupted"
    assert agent.messages[-1]["content"].endswith("[用户中断了这次回复]")
    assert agent.run_turn("接着说").status == "done"
    # 历史合法：user / assistant / user
    assert [m["role"] for m in llm.requests[1]] == ["system", "user", "assistant", "user"]


def test_interrupt_during_tools_fills_every_call(repo):
    calls = [call("bash", {"command": "sleep 20"}), call("read_file", {"path": "app.py"})]
    agent, llm, sink = make(repo, [Step(calls=calls), Step(text="ok")], mode="yolo")
    threading.Timer(0.3, agent.interrupt).start()
    end = agent.run_turn("x")
    assert end.status == "interrupted"
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert all(m["content"].startswith("Error[interrupted]") for m in tool_msgs)
    # 下一轮请求里每个 tool_call 都有对应的 tool 消息
    agent.run_turn("继续")
    ids = {tc["id"] for tc in llm.requests[1][2]["tool_calls"]}
    assert ids == {m["tool_call_id"] for m in llm.requests[1] if m["role"] == "tool"}


def test_llm_error_becomes_turn_end(repo):
    from coda.llm import LLMError

    class Boom(FakeLLM):
        def stream(self, *a, **k):
            raise LLMError("401")

    sink = ListSink()
    agent = Agent(Boom([]), repo, sink, DenyApprover(), system_prompt="s")
    end = agent.run_turn("x")
    assert end.status == "error" and end.error == "401"
    assert isinstance(sink.events[-1], TurnEnd)


@pytest.mark.parametrize(
    "cmd,ok",
    [
        ("ls -la", True),
        ("git status", True),
        ("git diff HEAD~1", True),
        ("git push", False),
        ("ls; rm -rf x", False),
        ("cat a > b", False),
        ("echo $(whoami)", False),
        ("ls && touch x", False),
        ("rm a", False),
    ],
)
def test_readonly_commands(cmd, ok):
    assert is_readonly_command(cmd) is ok
