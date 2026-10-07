"""Hooks、检查点 / undo、todo_write、完成闸门：FakeLLM 回放 + 真实的小 pytest 仓库。"""

import sys
import textwrap

import pytest

from coda.agent.events import (
    Decision,
    DenyApprover,
    FilesChanged,
    ListSink,
    TodoUpdate,
    VerifyEnd,
    VerifyStart,
)
from coda.agent.loop import Agent
from coda.config import Hooks, HookSpec, Permissions, VerifyConfig
from coda.verify.gate import detect_verify_command
from coda.verify.junit import parse_junit
from tests.fakes import FakeLLM, Step, call

PY = sys.executable


class AllowAll:
    def __init__(self):
        self.requests = []

    def ask(self, request, cancel):
        self.requests.append(request)
        return Decision(True)


def make(repo, steps, mode="accept-edits", approver=None, **kw):
    sink = ListSink()
    llm = FakeLLM(steps)
    kw.setdefault("verify", VerifyConfig(enabled=False))
    agent = Agent(llm, repo, sink, approver or DenyApprover(), mode=mode, system_prompt="sys", **kw)
    return agent, llm, sink


def tool_msgs(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def read_and_edit(old, new, path="app.py"):
    return [
        Step(calls=[call("read_file", {"path": path})]),
        Step(calls=[call("edit_file", {"path": path, "old": old, "new": new})]),
    ]


# ---------------------------------------------------------------- 检查点 / undo


def test_undo_restores_last_turn_and_new_file(repo):
    steps = [
        *read_and_edit("return 1", "return 2"),
        Step(calls=[call("write_file", {"path": "new.py", "content": "y = 1\n"})]),
        Step(text="改好了"),
        Step(calls=[call("edit_file", {"path": "app.py", "old": "return 2", "new": "return 3"})]),
        Step(text="又改了"),
    ]
    agent, llm, sink = make(repo, steps)
    agent.run_turn("第一轮")
    agent.run_turn("第二轮")
    assert "return 3" in (repo / "app.py").read_text()
    assert set(agent.checkpoints.session_stats()) == {"app.py", "new.py"}

    restored = agent.undo()
    assert [p.name for p in restored] == ["app.py"]
    assert "return 2" in (repo / "app.py").read_text()
    agent.undo()
    assert "return 1" in (repo / "app.py").read_text() and not (repo / "new.py").exists()
    assert agent.undo() == []
    assert agent.checkpoints.session_diffs() == {}
    # 撤销后要求重新读取，并且告诉了模型
    assert agent.ctx.filestate.check(repo / "app.py") == "not_read"
    assert "/undo" in agent.messages[-1]["content"]
    assert sink.of(FilesChanged)[-1].stats == {}


def test_turn_files_ignores_unchanged(repo):
    steps = [
        *read_and_edit("return 1", "return 2"),
        *read_and_edit("return 2", "return 1")[1:],
        Step(text="ok"),
    ]
    agent, _, _ = make(repo, steps)
    agent.run_turn("x")
    assert agent.checkpoints.turn_files() == []


# ---------------------------------------------------------------- Hooks


def test_post_hook_output_appended(repo):
    hooks = Hooks(
        PostToolUse=[
            HookSpec(
                matcher="edit_file|write_file", command='echo "lint: $CODA_FILE"; cat > /dev/null'
            )
        ]
    )
    agent, llm, sink = make(
        repo, [*read_and_edit("return 1", "return 2"), Step(text="ok")], hooks=hooks
    )
    agent.run_turn("x")
    result = tool_msgs(llm.requests[2])[-1]
    assert "[PostToolUse Hook" in result and f"lint: {repo / 'app.py'}" in result


def test_post_hook_reformat_does_not_cause_stale(repo):
    # Hook 改了文件（模拟 ruff format），下一次编辑不应报 stale
    hooks = Hooks(
        PostToolUse=[HookSpec(matcher="edit_file", command='echo "# fmt" >> "$CODA_FILE"')]
    )
    steps = [
        *read_and_edit("return 1", "return 2"),
        Step(calls=[call("edit_file", {"path": "app.py", "old": "return 2", "new": "return 3"})]),
        Step(text="ok"),
    ]
    agent, llm, _ = make(repo, steps, hooks=hooks)
    agent.run_turn("x")
    assert not tool_msgs(llm.requests[3])[-1].startswith("Error")


def test_pre_hook_veto(repo):
    script = repo / "veto.py"
    script.write_text(
        "import json,sys\nd=json.load(sys.stdin)\n"
        "if 'secret' in d['args'].get('command',''):\n"
        "    print('不许碰 secret', file=sys.stderr); sys.exit(2)\n"
    )
    hooks = Hooks(PreToolUse=[HookSpec(matcher="bash", command=f"{PY} {script}")])
    steps = [
        Step(
            calls=[call("bash", {"command": "echo secret"}), call("bash", {"command": "echo ok"})]
        ),
        Step(text="done"),
    ]
    agent, llm, _ = make(repo, steps, mode="yolo", hooks=hooks)
    agent.run_turn("x")
    first, second = tool_msgs(llm.requests[1])
    assert first.startswith("Error[denied]: PreToolUse Hook 否决：不许碰 secret")
    assert "ok" in second


def test_hook_timeout_does_not_break(repo):
    hooks = Hooks(PostToolUse=[HookSpec(matcher="read_file", command="sleep 5", timeout=0.3)])
    agent, llm, _ = make(
        repo, [Step(calls=[call("read_file", {"path": "app.py"})]), Step(text="ok")], hooks=hooks
    )
    agent.run_turn("x")
    assert "超时" in tool_msgs(llm.requests[1])[0]


# ---------------------------------------------------------------- 权限：会话规则与高危


def test_always_allow_adds_command_prefix_rule(repo):
    class Always:
        def __init__(self):
            self.n = 0

        def ask(self, request, cancel):
            self.n += 1
            return Decision(True, always=True)

    approver = Always()
    steps = [
        Step(calls=[call("bash", {"command": "touch a"})]),
        Step(calls=[call("bash", {"command": "touch b"})]),
        Step(calls=[call("bash", {"command": "mkdir c"})]),
        Step(text="ok"),
    ]
    agent, _, _ = make(repo, steps, mode="default", approver=approver)
    agent.run_turn("x")
    assert approver.n == 2  # touch 只问一次，mkdir 再问一次
    assert [str(r) for r in agent.policy.session_rules] == ["bash(touch*)", "bash(mkdir*)"]


def test_danger_request_never_remembered(repo):
    class AlwaysYes:
        def __init__(self):
            self.reqs = []

        def ask(self, request, cancel):
            self.reqs.append(request)
            return Decision(True, always=True)

    approver = AlwaysYes()
    steps = [Step(calls=[call("bash", {"command": "rm -r nothing_here"})])] * 2 + [Step(text="ok")]
    agent, _, _ = make(repo, steps, mode="yolo", approver=approver)
    agent.run_turn("x")
    assert len(approver.reqs) == 2 and approver.reqs[0].danger and approver.reqs[0].always == []


def test_blocked_command_never_asks(repo):
    approver = AllowAll()
    agent, llm, _ = make(
        repo,
        [Step(calls=[call("bash", {"command": "sudo ls"})]), Step(text="ok")],
        mode="yolo",
        approver=approver,
    )
    agent.run_turn("x")
    assert approver.requests == []
    assert "危险命令已被拦截" in tool_msgs(llm.requests[1])[0]


def test_settings_permissions_applied(repo):
    perms = Permissions(deny=["bash(git push*)"])
    agent, llm, _ = make(
        repo,
        [Step(calls=[call("bash", {"command": "git push"})]), Step(text="ok")],
        mode="yolo",
        permissions=perms,
    )
    agent.run_turn("x")
    assert "git push*" in tool_msgs(llm.requests[1])[0]


# ---------------------------------------------------------------- todo_write 与 plan 模式


def test_todo_write_updates_and_validates(repo):
    good = {
        "todos": [
            {"content": "定位", "status": "completed"},
            {"content": "修改", "status": "in_progress"},
        ]
    }
    bad = {
        "todos": [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]
    }
    steps = [
        Step(calls=[call("todo_write", good)]),
        Step(calls=[call("todo_write", bad)]),
        Step(text="ok"),
    ]
    agent, llm, sink = make(repo, steps, mode="default")
    agent.run_turn("x")
    assert sink.of(TodoUpdate)[0].todos[1] == {"content": "修改", "status": "in_progress"}
    assert len(sink.of(TodoUpdate)) == 1
    assert "◐ 修改" in tool_msgs(llm.requests[1])[0]
    assert tool_msgs(llm.requests[2])[-1].startswith("Error[invalid_args]")


def test_todo_stale_reminder(repo):
    todos = {"todos": [{"content": "a", "status": "in_progress"}]}
    steps = (
        [Step(calls=[call("todo_write", todos)])]
        + [Step(calls=[call("glob", {"pattern": "*.py"})]) for _ in range(8)]
        + [Step(text="ok")]
    )
    agent, llm, _ = make(repo, steps)
    agent.run_turn("x")
    assert "任务清单已经 8 步没有更新" in tool_msgs(llm.requests[9])[-1]
    assert "没有更新" not in tool_msgs(llm.requests[8])[-1]


def test_plan_mode_reminder_on_user_message(repo):
    agent, llm, _ = make(repo, [Step(text="计划")], mode="plan")
    agent.run_turn("调查一下")
    assert "plan 模式" in llm.requests[0][-1]["content"]
    assert llm.requests[0][-1]["content"].startswith("调查一下")


# ---------------------------------------------------------------- 完成闸门


@pytest.fixture
def pyrepo(tmp_path):
    (tmp_path / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        textwrap.dedent(
            """\
            from calc import add, mul

            def test_add():
                assert add(1, 2) == 3

            def test_mul():
                assert mul(2, 3) == 6

            def test_preexisting_broken():
                assert 1 == 2
            """
        )
    )
    (tmp_path / "conftest.py").write_text("")
    return tmp_path


def gate_cfg(**kw):
    return VerifyConfig(command=f"{PY} -m pytest -q -p no:cacheprovider", **kw)


def test_gate_ignores_baseline_failure_and_passes(pyrepo):
    steps = [*read_and_edit("return a * b", "return b * a", "calc.py"), Step(text="改好了")]
    agent, llm, sink = make(pyrepo, steps, verify=gate_cfg())
    end = agent.run_turn("x")
    assert end.status == "done" and end.verify == "passed"
    base, final = sink.of(VerifyEnd)
    assert base.kind == "baseline" and base.baseline_failures == 1
    assert final.ok and "1 failed" in final.summary and "2 passed" in final.summary
    assert len(llm.requests) == 3  # 没有回填


def test_gate_feeds_back_new_failure_then_fixed(pyrepo):
    steps = [
        *read_and_edit("return a + b", "return a - b", "calc.py"),
        Step(text="改好了"),  # 闸门发现 test_add 新增失败，回填
        Step(
            calls=[
                call("edit_file", {"path": "calc.py", "old": "return a - b", "new": "return a + b"})
            ]
        ),
        Step(text="修好了"),
    ]
    agent, llm, sink = make(pyrepo, steps, verify=gate_cfg())
    end = agent.run_turn("x")
    assert end.status == "done" and end.verify == "passed"
    feedback = llm.requests[3][-1]
    assert feedback["role"] == "user" and "完成闸门" in feedback["content"]
    assert "tests/test_calc.py::test_add" in feedback["content"]
    assert "test_preexisting_broken" not in feedback["content"]
    finals = [e for e in sink.of(VerifyEnd) if e.kind == "final"]
    assert finals[0].feedback and finals[0].new_failures == ["tests/test_calc.py::test_add"]
    assert finals[1].ok
    assert len(sink.of(VerifyStart)) == 3  # 基线 + 两次 final


def test_gate_gives_up_after_max_rounds(pyrepo):
    steps = [
        *read_and_edit("return a + b", "return a - b", "calc.py"),
        Step(text="好了"),
        Step(text="真的好了"),
        Step(text="确实好了"),
    ]
    agent, llm, sink = make(pyrepo, steps, verify=gate_cfg(max_rounds=2))
    end = agent.run_turn("x")
    assert end.status == "done" and end.verify == "failed"
    finals = [e for e in sink.of(VerifyEnd) if e.kind == "final"]
    assert [e.feedback for e in finals] == [True, True, False] and finals[-1].gave_up


def test_gate_skipped_without_edits_and_when_model_ran_tests(pyrepo):
    agent, llm, sink = make(pyrepo, [Step(text="只是解释")], verify=gate_cfg())
    assert agent.run_turn("x").verify is None and not sink.of(VerifyStart)

    cmd = gate_cfg().command
    steps = [
        *read_and_edit("return a * b", "return b * a", "calc.py"),
        Step(calls=[call("bash", {"command": cmd + " -k 'add or mul'"})]),
        Step(text="ok"),
    ]
    agent, _, sink = make(pyrepo, steps, mode="yolo", verify=gate_cfg())
    agent.run_turn("x")
    # 模型跑的命令不完全等于验证命令，闸门仍会再跑一次
    assert [e.kind for e in sink.of(VerifyEnd)] == ["baseline", "final"]


def test_gate_flags_test_file_edits(pyrepo):
    steps = [
        *read_and_edit("assert 1 == 2", "assert 1 == 1", "tests/test_calc.py"),
        Step(text="ok"),
    ]
    agent, _, sink = make(pyrepo, steps, verify=gate_cfg())
    agent.run_turn("x")
    final = sink.of(VerifyEnd)[-1]
    assert final.ok and final.tests_edited == ["tests/test_calc.py"]


def test_gate_non_pytest_command_uses_exit_code(repo):
    steps = [*read_and_edit("return 1", "return 2"), Step(text="ok"), Step(text="还是不行")]
    cfg = VerifyConfig(command="grep -q 'return 1' app.py", max_rounds=1)
    agent, llm, sink = make(repo, steps, verify=cfg)
    end = agent.run_turn("x")
    assert end.verify == "failed"
    assert "验证命令 exit 1" in llm.requests[3][-1]["content"]


def test_gate_interrupted_baseline(pyrepo):
    import threading

    cfg = VerifyConfig(command="sleep 10")
    steps = [*read_and_edit("return a * b", "return b * a", "calc.py"), Step(text="ok")]
    agent, _, sink = make(pyrepo, steps, verify=cfg)
    threading.Timer(0.5, agent.interrupt).start()
    end = agent.run_turn("x")
    assert end.status == "interrupted"


def test_detect_verify_command(tmp_path):
    assert detect_verify_command(tmp_path) is None
    (tmp_path / "tests").mkdir()
    assert detect_verify_command(tmp_path) == "python -m pytest -q"
    (tmp_path / "uv.lock").write_text("")
    assert detect_verify_command(tmp_path) == "uv run pytest -q"


def test_parse_junit(tmp_path):
    p = tmp_path / "j.xml"
    p.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_x.TestA" name="test_a">'
        '<failure message="boom">Traceback\nAssertionError: boom</failure></testcase>'
        '<testcase classname="tests.test_x" name="test_b"/>'
        '<testcase classname="tests.test_y" name="test_c"><error message="e"/></testcase>'
        "</testsuite></testsuites>"
    )
    assert parse_junit(p) == {
        "tests/test_x.py::TestA::test_a": "Traceback\nAssertionError: boom",
        "tests/test_y.py::test_c": "e",
    }
    assert parse_junit(tmp_path / "missing.xml") is None
