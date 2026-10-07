"""TUI 无头测试：App.run_test + Pilot 驱动界面，FakeLLM 提供确定的模型回复。"""

import asyncio

import pytest

from coda.config import load_settings
from coda.tui.app import CodaApp
from coda.tui.screens.permission import PermissionScreen
from coda.tui.widgets.chat import AssistantMessage, ThinkingBlock, ToolLine, UserMessage
from tests.fakes import FakeLLM, Step, call


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


def make_app(repo, steps, **kw):
    llm = FakeLLM(steps)
    return CodaApp(load_settings(repo), repo, llm=llm, **kw), llm


async def wait_idle(app, pilot, timeout=5.0):
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if not app.busy:
            return
    raise AssertionError("本轮没有结束")


async def send(pilot, text):
    app = pilot.app
    app.prompt.text = text
    await pilot.press("enter")


async def test_stream_answer_and_thinking(repo):
    app, llm = make_app(repo, [Step(thinking="想一想", text="# 标题\n\n你好 **世界**")])
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "hi")
        await wait_idle(app, pilot)
        assert [m.content for m in app.query(UserMessage)][0].plain == "hi"
        think = app.query_one(ThinkingBlock)
        assert think.text == "想一想" and think.collapsed and think.title.startswith("思考 ")
        assert app.query_one(AssistantMessage).text == "# 标题\n\n你好 **世界**"
        assert app.prompt.input_history == ["hi"]


async def test_tool_lines_and_sidebar_usage(repo):
    steps = [
        Step(calls=[call("read_file", {"path": "app.py"}), call("grep", {"pattern": "zzz"})]),
        Step(text="ok"),
    ]
    app, _ = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "读一下")
        await wait_idle(app, pilot)
        lines = [str(t._line.content) for t in app.query(ToolLine)]
        assert "read_file app.py" in lines[0] and "2 行" in lines[0]
        assert "0 处匹配" in lines[1]
        assert "缓存命中" in str(app.query_one("#side-usage").content)


async def test_permission_allow_shows_diff_and_updates_sidebar(repo):
    edit = {"path": "app.py", "old": "return 1", "new": "return 2"}
    steps = [
        Step(calls=[call("read_file", {"path": "app.py"})]),
        Step(calls=[call("edit_file", edit)]),
        Step(text="改好了"),
    ]
    app, _ = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "改")
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(app.screen, PermissionScreen):
                break
        assert isinstance(app.screen, PermissionScreen)
        assert "+    return 2" in app.screen.request.preview
        await pilot.press("y")
        await wait_idle(app, pilot)
        assert "return 2" in (repo / "app.py").read_text()
        assert app.query(".diff")
        assert "app.py" in str(app.query_one("#side-changes").content)


async def test_permission_deny_with_reason(repo):
    steps = [Step(calls=[call("bash", {"command": "touch x"})]), Step(text="好")]
    app, llm = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "建文件")
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(app.screen, PermissionScreen):
                break
        await pilot.press("n")
        await pilot.pause(0.1)
        for ch in "不要建":
            await pilot.press(ch)
        await pilot.press("enter")
        await wait_idle(app, pilot)
        last = llm.requests[1][-1]["content"]
        assert last.startswith("Error[denied]") and "不要建" in last
        assert not (repo / "x").exists()


async def test_escape_interrupts_stream_then_continue(repo):
    steps = [Step(text="很长的回答。" * 200, delay=0.01), Step(text="继续好了")]
    app, llm = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "讲讲")
        await pilot.pause(0.3)
        assert app.busy
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert "已中断" in " ".join(str(n.content) for n in app.query(".notice"))
        await send(pilot, "接着说")
        await wait_idle(app, pilot)
        assert [m["role"] for m in llm.requests[1]] == ["system", "user", "assistant", "user"]


async def test_escape_while_permission_open(repo):
    steps = [Step(calls=[call("bash", {"command": "touch x"})]), Step(text="ok")]
    app, _ = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "x")
        for _ in range(60):
            await pilot.pause(0.05)
            if isinstance(app.screen, PermissionScreen):
                break
        await pilot.press("escape")
        await wait_idle(app, pilot)
        assert not isinstance(app.screen, PermissionScreen)
        tool_msgs = [m for m in app.agent.messages if m["role"] == "tool"]
        assert tool_msgs[0]["content"].startswith("Error[interrupted]")


async def test_queue_while_running(repo):
    steps = [Step(text="第一轮" * 200, delay=0.01), Step(text="第二轮")]
    app, llm = make_app(repo, steps)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "一")
        await pilot.pause(0.1)
        await send(pilot, "二")
        assert app.queue == ["二"]
        await wait_idle(app, pilot)
        await asyncio.sleep(0)
        await wait_idle(app, pilot)
        assert len(llm.requests) == 2 and llm.requests[1][-1]["content"] == "二"


async def test_mode_cycle_and_commands(repo):
    app, _ = make_app(repo, [])
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("shift+tab")
        assert app.agent.mode == "accept-edits"
        await send(pilot, "/mode plan")
        await pilot.pause(0.05)
        assert app.agent.mode == "plan" and "plan" in str(app.query_one("#topbar").content)
        await send(pilot, "/help")
        await pilot.pause(0.05)
        assert "/clear" in str(list(app.query(".notice"))[-1].content)


async def test_history_navigation(repo):
    app, _ = make_app(repo, [])
    async with app.run_test(size=(140, 40)) as pilot:
        app.prompt.input_history = ["first", "second"]
        await pilot.press("up")
        assert app.prompt.text == "second"
        await pilot.press("up")
        assert app.prompt.text == "first"
        await pilot.press("down", "down")
        assert app.prompt.text == ""


async def test_sidebar_toggle_and_narrow(repo):
    app, _ = make_app(repo, [])
    async with app.run_test(size=(100, 40)) as pilot:
        assert not app.sidebar.display
        await pilot.press("ctrl+b")
        assert app.sidebar.display


# ---------------------------------------------------------------- M3 / M4


async def wait_screen(app, pilot, kind):
    for _ in range(60):
        await pilot.pause(0.05)
        if isinstance(app.screen, kind):
            return
    raise AssertionError(f"没有出现 {kind.__name__}")


async def test_undo_and_diff_screen(repo):
    from coda.tui.screens.diff_screen import DiffScreen

    edit = {"path": "app.py", "old": "return 1", "new": "return 2"}
    steps = [
        Step(calls=[call("read_file", {"path": "app.py"})]),
        Step(calls=[call("edit_file", edit)]),
        Step(text="ok"),
    ]
    app, _ = make_app(repo, steps, mode="accept-edits")
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "改")
        await wait_idle(app, pilot)
        assert "app.py" in str(app.query_one("#side-changes").content)
        await send(pilot, "/diff")
        await wait_screen(app, pilot, DiffScreen)
        await pilot.press("escape")
        await pilot.pause(0.05)
        await send(pilot, "/undo")
        await pilot.pause(0.1)
        assert "return 1" in (repo / "app.py").read_text()
        assert "暂无" in str(app.query_one("#side-changes").content)


async def test_todo_sidebar(repo):
    todos = {
        "todos": [
            {"content": "定位问题", "status": "completed"},
            {"content": "修改代码", "status": "in_progress"},
        ]
    }
    app, _ = make_app(repo, [Step(calls=[call("todo_write", todos)]), Step(text="ok")])
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "x")
        await wait_idle(app, pilot)
        assert "1/2" in str(app.query_one("#side-todo-title").content)
        assert "修改代码" in str(app.query_one("#side-todo").content)


async def test_danger_permission_has_no_always(repo):
    steps = [Step(calls=[call("bash", {"command": "git push --force"})]), Step(text="ok")]
    app, llm = make_app(repo, steps, mode="yolo")
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "推送")
        await wait_screen(app, pilot, PermissionScreen)
        assert app.screen.request.danger and app.screen.request.always == []
        await pilot.press("a")  # 没有"总是允许"，按 a 无效
        await pilot.pause(0.1)
        assert isinstance(app.screen, PermissionScreen)
        await pilot.press("n")
        await pilot.press("enter")
        await wait_idle(app, pilot)
        assert llm.requests[1][-1]["content"].startswith("Error[denied]")


async def test_gate_box_shown(tmp_path):
    import json
    import sys

    (tmp_path / "m.py").write_text("X = 1\n")
    (tmp_path / "test_m.py").write_text("from m import X\n\ndef test_x():\n    assert X == 1\n")
    (tmp_path / ".coda").mkdir()
    (tmp_path / ".coda" / "settings.json").write_text(
        json.dumps({"verify": {"command": f"{sys.executable} -m pytest -q -p no:cacheprovider"}})
    )
    edit = {"path": "m.py", "old": "X = 1", "new": "X = 2"}
    steps = [
        Step(calls=[call("read_file", {"path": "m.py"})]),
        Step(calls=[call("edit_file", edit)]),
        Step(text="改好了"),
        Step(calls=[call("edit_file", {"path": "m.py", "old": "X = 2", "new": "X = 1"})]),
        Step(text="修好了"),
    ]
    app, _ = make_app(tmp_path, steps, mode="accept-edits")
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "改")
        await wait_idle(app, pilot, timeout=30)
        boxes = [str(b.content) for b in app.query(".gate")]
        assert len(boxes) == 2
        assert "新增 1 个失败" in boxes[0] and "验证通过" in boxes[1]


# ---------------------------------------------------------------- M5 / M6


async def test_resume_session_renders_history(repo):
    from coda.state.session import load_session

    app, _ = make_app(
        repo, [Step(calls=[call("grep", {"pattern": "def f"})]), Step(text="在 app.py")]
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "f 在哪")
        await wait_idle(app, pilot)
        path = app.agent.session.path
    assert load_session(path).turns == 1

    app2, llm2 = make_app(repo, [Step(text="继续")], resume=path)
    async with app2.run_test(size=(140, 40)) as pilot:
        await pilot.pause(0.2)
        users = [str(m.content) for m in app2.query(UserMessage)]
        assert users == ["f 在哪"]
        assert any("grep" in str(s.content) for s in app2.query(".history"))
        assert app2.last_answer == "在 app.py"
        await send(pilot, "然后")
        await wait_idle(app2, pilot)
        assert [m["role"] for m in llm2.requests[0]][-3:] == ["tool", "assistant", "user"]
        assert app2.agent.session.path == path


async def test_resume_picker(repo):
    from coda.tui.screens.picker import PickerScreen

    app, _ = make_app(repo, [Step(text="一")])
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "第一个会话")
        await wait_idle(app, pilot)
        await send(pilot, "/clear")
        await pilot.pause(0.1)
        await send(pilot, "/resume")
        await wait_screen(app, pilot, PickerScreen)
        assert "第一个会话" in str(app.screen.items[0][1])
        await pilot.press("enter")
        await pilot.pause(0.2)
        assert [str(m.content) for m in app.query(UserMessage)] == ["第一个会话"]


async def test_subagent_block(repo):
    from coda.tui.widgets.chat import SubagentBlock

    main = [Step(calls=[call("task", {"description": "梳理", "prompt": "查 f"})]), Step(text="好")]
    sub = [Step(calls=[call("grep", {"pattern": "def f"})]), Step(text="f 在 app.py:1")]
    llm = FakeLLM(main, sub_steps=sub)
    app = CodaApp(load_settings(repo), repo, llm=llm)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "调查")
        await wait_idle(app, pilot)
        block = app.query_one(SubagentBlock)
        assert block.done and block.collapsed and block.tool_count == 1
        assert "梳理" in block.title and "2 步" in block.title
        assert any("grep" in str(s.content) for s in block.query(".tool-line"))


async def test_slash_completion(repo):
    app, _ = make_app(repo, [])
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("/", "c", "o")
        await pilot.pause(0.05)
        assert app.completer.active and app.completer.kind == "command"
        ids = [app.completer.get_option_at_index(i).id for i in range(app.completer.option_count)]
        assert ids == ["/compact", "/cost", "/copy"]
        await pilot.press("down", "enter")
        await pilot.pause(0.1)
        assert "缓存命中" in str(list(app.query(".notice"))[-1].content)
        assert app.prompt.text == ""


async def test_at_file_completion(repo):
    (repo / "pkg").mkdir()
    (repo / "pkg" / "retriever.py").write_text("x = 1\n")
    app, _ = make_app(repo, [])
    async with app.run_test(size=(140, 40)) as pilot:
        for ch in "看 @retr":
            await pilot.press(ch if ch != " " else "space")
        await pilot.pause(0.05)
        assert app.completer.kind == "file" and app.completer.selected() == "pkg/retriever.py"
        await pilot.press("tab")
        assert app.prompt.text == "看 @pkg/retriever.py "
        assert not app.completer.active


async def test_model_picker_switches(repo, monkeypatch):
    from coda.tui.screens.picker import PickerScreen

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    app, _ = make_app(repo, [])
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "/model")
        await wait_screen(app, pilot, PickerScreen)
        keys = [k for k, _ in app.screen.items]
        await pilot.press(*["down"] * keys.index("qwen3-coder"), "enter")
        await pilot.pause(0.1)
        assert app.model_name == "qwen3-coder" and app.agent.llm is app.llm
        assert app.agent.subagents.llm is app.llm
        assert "qwen3-coder" in str(app.query_one("#topbar").content)


async def test_manual_compact_command(repo):
    from coda.tui.widgets.chat import CompactNotice

    llm = FakeLLM([Step(text="a"), Step(text="b")], summaries=["之前的摘要"])
    app = CodaApp(load_settings(repo), repo, llm=llm)
    async with app.run_test(size=(140, 40)) as pilot:
        await send(pilot, "一")
        await wait_idle(app, pilot)
        await send(pilot, "二")
        await wait_idle(app, pilot)
        await send(pilot, "/compact 关注接口")
        await wait_idle(app, pilot)
        await pilot.pause(0.1)
        assert app.query(CompactNotice)
        assert "手动" in str(app.query_one(CompactNotice).content)
