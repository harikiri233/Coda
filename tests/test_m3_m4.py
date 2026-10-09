"""检查点 / undo、权限规则、todo_write 与 plan 模式：FakeLLM 回放。"""

import pytest

from coda.agent.events import (
    Decision,
    DenyApprover,
    FilesChanged,
    ListSink,
    TodoUpdate,
)
from coda.agent.loop import Agent
from coda.config import Permissions
from tests.fakes import FakeLLM, Step, call


class AllowAll:
    def __init__(self):
        self.requests = []

    def ask(self, request, cancel):
        self.requests.append(request)
        return Decision(True)


def make(repo, steps, mode="accept-edits", approver=None, **kw):
    sink = ListSink()
    llm = FakeLLM(steps)
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
