"""CodaApp：全屏 TUI。

并发模型：
  主线程   Textual 事件循环——渲染、键盘、弹层
  worker   @work(thread=True) 里跑同步的 Agent.run_turn；事件经 TuiSink 批量 call_from_thread
           回到主线程，由 apply_events 映射成组件；权限询问经 TuiApprover 推出模态弹层并阻塞等待。
  MCP      后台线程里的 asyncio 循环（mcp_client.py），连上后回到主线程注册工具。

Esc：运行中中断本轮（流式阶段关闭 HTTP 流，工具阶段杀子进程组）；空闲时清空输入框。
运行中继续输入的消息进入队列，本轮结束后自动发送。
会话逐条写入 JSONL，`coda -c` / `coda --resume` / `/resume` 恢复。
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Collapsible, Markdown, Static

from coda.agent import events as ev
from coda.agent.loop import Agent
from coda.agent.prompt import _git_branch
from coda.config import Mode, Settings
from coda.context.compact import SUMMARY_MARK, _brief_args
from coda.llm import LLMClient, LLMError
from coda.mcp_client import McpManager, ServerState
from coda.state.session import Session, load_session
from coda.tui.bridge import TuiApprover, TuiSink
from coda.tui.commands import COMMANDS, handle_command, session_items
from coda.tui.widgets.chat import (
    AssistantMessage,
    CompactNotice,
    Notice,
    SubagentBlock,
    ThinkingBlock,
    ToolLine,
    UserMessage,
    VerifyBox,
)
from coda.tui.widgets.completer import Completer
from coda.tui.widgets.prompt_input import PromptInput
from coda.tui.widgets.sidebar import Sidebar

MODE_STYLE = {"default": "yellow", "accept-edits": "green", "plan": "blue", "yolo": "red"}
NARROW = 110
IDLE_HINT = (
    "Enter 发送 · Ctrl+J 换行 · / 命令 · @ 引用文件 · ↑↓ 历史 · Esc 中断 · "
    "Shift+Tab 切模式 · Ctrl+B 侧栏 · Ctrl+O 展开/折叠 · Ctrl+Q 退出"
)
HISTORY_TOOL_LINES = 3  # 恢复会话时每一步最多显示几条工具调用


def _display_user_text(content: str) -> str:
    """去掉发送时附加的系统提醒（@ 文件内容、plan 模式提醒）。"""
    return content.split("\n\n<system-reminder>", 1)[0]


class CodaApp(App):
    CSS_PATH = "coda.tcss"
    TITLE = "Coda"
    BINDINGS = [
        Binding("escape", "interrupt", "中断", show=False),
        Binding("shift+tab", "cycle_mode", "切换模式", show=False, priority=True),
        Binding("ctrl+b", "toggle_sidebar", "侧栏", show=False),
        Binding("ctrl+o", "toggle_folds", "展开/折叠", show=False),
    ]

    def __init__(
        self,
        settings: Settings,
        workdir: Path,
        *,
        model: str | None = None,
        mode: Mode | None = None,
        first_prompt: str | None = None,
        max_steps: int | None = None,
        llm: LLMClient | None = None,
        resume: Path | None = None,
        pick_resume: bool = False,
        mcp: McpManager | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.workdir = workdir.resolve()
        self.model_name = model or settings.model
        self.llm = llm or LLMClient(settings.profile(self.model_name))
        self.agent = Agent(
            self.llm,
            self.workdir,
            TuiSink(self),
            TuiApprover(self),
            mode=mode or settings.mode,
            max_steps=max_steps or settings.max_steps,
            permissions=settings.permissions,
            hooks=settings.hooks,
            verify=settings.verify,
            context=settings.context,
            session=Session.create(self.workdir, self.model_name),
        )
        self.mcp = mcp or McpManager(settings.mcpServers, self.workdir)
        self.first_prompt = first_prompt
        self.resume_path = resume
        self.pick_resume = pick_resume
        self.branch = _git_branch(self.workdir)
        self.queue: list[str] = []
        self.busy = False  # 界面视角的"本轮未结束"：收到 TurnEnd 才清除
        self.sidebar_visible = True
        self._user_hid_sidebar = False
        self._thinking: ThinkingBlock | None = None
        self._answer: AssistantMessage | None = None
        self._tools: dict[str, ToolLine | SubagentBlock] = {}
        self._verify: VerifyBox | None = None
        self._turn_start = 0.0
        self._step = 0
        self._activity = "运行中"
        self.last_answer = ""

    # ---------------------------------------------------------------- 布局

    def compose(self) -> ComposeResult:
        cfg = self.settings.context
        yield Static(id="topbar")
        with Horizontal(id="body"):
            yield VerticalScroll(id="chat")
            yield Sidebar(self.llm.profile.context_budget, cfg.micro_ratio, cfg.summary_ratio)
        with Vertical(id="bottom"):
            yield Completer(self.workdir, COMMANDS)
            yield Static(Text(IDLE_HINT), id="hint")
            yield PromptInput(id="input")

    def on_mount(self) -> None:
        self.chat = self.query_one("#chat", VerticalScroll)
        self.chat.anchor()
        # 定时器和弹层关闭后的回调里直接用引用：弹层打开或界面退出时按 id 查询会找不到节点
        self.hint = self.query_one("#hint", Static)
        self.topbar = self.query_one("#topbar", Static)
        self.sidebar = self.query_one(Sidebar)
        self.prompt = self.query_one(PromptInput)
        self.completer = self.query_one(Completer)
        self.completer.display = False
        self.prompt.completer = self.completer
        self.prompt.focus()
        self.refresh_topbar()
        self._auto_sidebar(self.size.width)
        self.set_interval(0.5, self._tick)
        self.chat.mount(Notice(self._banner(), "info"))
        if self.mcp.servers:
            self.mcp.start(on_ready=lambda st: self.call_from_thread(self._mcp_ready, st))
        if self.resume_path is not None:
            self.call_later(self.resume_session, self.resume_path)
        elif self.pick_resume:
            self.call_later(self.pick_session)
        if self.first_prompt:
            self.submit(self.first_prompt)

    def on_unmount(self) -> None:
        self.mcp.stop()

    def _banner(self) -> str:
        profile = self.llm.profile
        think = "开" if self.llm.thinking_enabled() else "关"
        gate = self.agent.gate
        if gate.command:
            how = "自动探测" if gate.detected else "配置"
            verify = f"完成闸门：{gate.command}（{how}）" + ("" if gate.enabled else "，已关闭")
        else:
            verify = "完成闸门：未找到测试命令，/init 或 /verify <命令> 设置"
        extra = []
        if self.agent.memory_files:
            extra.append("AGENTS.md " + "、".join(f.scope for f in self.agent.memory_files))
        if self.agent.skills:
            extra.append(f"Skills {len(self.agent.skills)} 个")
        if self.mcp.servers:
            extra.append(f"MCP 连接中：{'、'.join(self.mcp.servers)}")
        loaded = f"\n已加载：{' · '.join(extra)}" if extra else ""
        return (
            f"Coda · {self.model_name}（{profile.model}，思考{think}） · 工作区 {self.workdir}\n"
            f"{verify}{loaded}\n"
            "bash 工具没有沙箱，权限检查只用来防误操作。"
        )

    def refresh_topbar(self) -> None:
        mode = self.agent.mode
        think = "思考开" if self.llm.thinking_enabled() else "思考关"
        branch = f"  [dim]{escape(self.branch)}[/]" if self.branch else ""
        home = str(Path.home())
        where = str(self.workdir).replace(home, "~", 1)
        self.topbar.update(
            Text.from_markup(
                f" [b]Coda[/]  {escape(where)}{branch}   [b]{escape(self.model_name)}[/] · {think}"
                f"   模式 [b {MODE_STYLE[mode]}]{mode}[/]"
            )
        )

    def _auto_sidebar(self, width: int) -> None:
        if not self._user_hid_sidebar:
            self.sidebar.display = width >= NARROW

    def on_resize(self, event) -> None:
        if hasattr(self, "sidebar"):
            self._auto_sidebar(event.size.width)

    def _tick(self) -> None:
        hint = getattr(self, "hint", None)
        if hint is None or not hint.is_attached:
            return
        if self.busy:
            elapsed = time.monotonic() - self._turn_start
            queued = f" · 已排队 {len(self.queue)} 条" if self.queue else ""
            step = f" · 第 {self._step} 步" if self._step else ""
            hint.update(
                Text.from_markup(
                    f"[b yellow]● {self._activity}[/]{step} · 已用时 {elapsed:.0f}s{queued} · [b]Esc[/] 中断"
                )
            )
        else:
            hint.update(Text(IDLE_HINT, style="dim"))

    # ---------------------------------------------------------------- 输入

    async def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        text = message.text
        if text.startswith("/"):
            await handle_command(self, text)
            return
        self.submit(text)

    def _start_busy(self, activity: str = "运行中") -> None:
        self._turn_start = time.monotonic()
        self._step = 0
        self._activity = activity
        self.busy = True  # 立刻标记，避免 worker 启动前的输入绕过队列
        self._tick()

    def submit(self, text: str) -> None:
        if self.busy:
            self.queue.append(text)
            self.chat.mount(UserMessage(text, queued=True))
            self._tick()
            return
        self.chat.mount(UserMessage(text))
        self.chat.anchor()
        self._start_busy()
        self.run_agent(text)

    @work(thread=True, group="agent")
    def run_agent(self, text: str) -> None:
        self.agent.run_turn(text)

    async def notice(self, text: str, level: str = "info") -> None:
        await self.chat.mount(Notice(text, level))

    # ---------------------------------------------------------------- 压缩 / 模型 / 会话

    def start_compact(self, focus: str = "") -> None:
        self._start_busy("正在压缩上下文")
        self._compact_worker(focus)

    @work(thread=True, group="agent")
    def _compact_worker(self, focus: str) -> None:
        try:
            result = self.agent.compact(focus, reason="manual")
            error = None
        except LLMError as e:
            result, error = None, str(e)
        self.call_from_thread(self._compact_done, result is not None, error)

    async def _compact_done(self, ok: bool, error: str | None) -> None:
        self.busy = False
        self._tick()
        if error:
            await self.notice(f"压缩失败：{error}", "error")
        elif not ok:
            await self.notice("没有可以压缩的历史（最近一轮本身就占满了保留区）。", "warning")
        if self.queue:
            nxt = self.queue.pop(0)
            self._start_busy()
            self.run_agent(nxt)

    async def switch_model(self, key: str) -> None:
        try:
            profile = self.settings.profile(key)
            llm = LLMClient(profile, tracker=self.llm.tracker)
        except (KeyError, LLMError) as e:
            await self.notice(f"切换失败：{e}", "error")
            return
        self.llm = llm
        self.model_name = key
        self.agent.set_llm(llm, key)
        self.sidebar.update_context(self.agent.context_tokens(), profile.context_budget)
        self.refresh_topbar()
        think = "开" if llm.thinking_enabled() else "关"
        await self.notice(
            f"已切换到 {key}（{profile.model}，思考{think}），对话历史保留。"
            + ("" if llm.is_deepseek else "\n历史里的思考内容不会发给这个模型。")
        )

    def pick_session(self) -> None:
        from coda.tui.screens.picker import PickerScreen

        items = session_items(self)
        if not items:
            self.call_later(self.notice, "本项目还没有历史会话。")
            return

        async def chosen(path: str | None) -> None:
            if path and (self.agent.session is None or Path(path) != self.agent.session.path):
                await self.resume_session(Path(path))

        self.push_screen(PickerScreen("选择要继续的会话（本项目）", items), chosen)

    async def resume_session(self, path: Path) -> None:
        if self.busy:
            await self.notice("运行中不能切换会话，先按 Esc 中断。", "warning")
            return
        try:
            loaded = load_session(path)
        except OSError as e:
            await self.notice(f"读取会话失败：{e}", "error")
            return
        self.agent.load(loaded, Session.open(path))
        await self.chat.remove_children()
        await self.chat.mount(
            Notice(
                f"已恢复会话 {path.stem}（{loaded.turns} 轮，{len(loaded.messages)} 条消息）。"
                "文件检查点不跨会话保留，/undo 只能撤销之后的修改；修改文件前会重新读取。",
                "info",
            )
        )
        await self._render_history(loaded.messages)
        tracker = self.llm.tracker
        self.sidebar.update_usage(
            tracker.total, self.agent.context_tokens(), self.llm.profile.context_budget
        )
        self.sidebar.update_todos(loaded.todos)
        self.chat.scroll_end(animate=False)
        self.chat.anchor()

    async def _render_history(self, messages: list[dict]) -> None:
        results = {
            m.get("tool_call_id"): m.get("content") or ""
            for m in messages
            if m.get("role") == "tool"
        }
        widgets = []
        for m in messages[1:]:
            role, content = m.get("role"), m.get("content") or ""
            if role == "user":
                if SUMMARY_MARK in content:
                    widgets.append(Notice("⇣ 更早的历史已压缩成摘要（用户原话原样保留）", "info"))
                elif not content.startswith("<system-reminder>"):
                    widgets.append(UserMessage(_display_user_text(content)))
            elif role == "assistant":
                calls = m.get("tool_calls") or []
                for c in calls[:HISTORY_TOOL_LINES]:
                    fn = c["function"]
                    res = results.get(c["id"], "")
                    color = "red" if res.startswith("Error[") else "green"
                    line = f"[{color}]●[/] [b]{escape(fn['name'])}[/] {escape(_brief_args(fn['arguments'], 100))}"
                    widgets.append(Static(Text.from_markup(line), classes="tool-line history"))
                if len(calls) > HISTORY_TOOL_LINES:
                    widgets.append(
                        Static(
                            f"  …另有 {len(calls) - HISTORY_TOOL_LINES} 个工具调用",
                            classes="tool-line history",
                        )
                    )
                if content:
                    widgets.append(Markdown(content, classes="assistant"))
                    self.last_answer = content
        if widgets:
            await self.chat.mount_all(widgets)

    def _mcp_ready(self, st: ServerState) -> None:
        if st.status == "connected":
            names = self.mcp.register(self.agent.tools, st.name)
            self.chat.mount(
                Notice(
                    f"MCP {st.name} 已连接（{st.elapsed:.1f}s），{len(names)} 个工具。/mcp 查看",
                    "info",
                )
            )
        else:
            self.chat.mount(Notice(f"MCP {st.name} 连接失败：{st.error}。/mcp 查看", "warning"))

    # ---------------------------------------------------------------- 事件 → 组件

    async def _end_thinking(self) -> None:
        if self._thinking is not None:
            self._thinking.finish()
            self._thinking = None

    async def _end_answer(self) -> None:
        if self._answer is not None:
            await self._answer.finish()
            if self._answer.text.strip():
                self.last_answer = self._answer.text
            self._answer = None

    async def apply_events(self, batch: list[ev.Event]) -> None:
        for e in batch:
            await self._apply(e)

    async def _apply(self, e: ev.Event) -> None:
        if isinstance(e, ev.StepStart):
            self._step = e.step
        elif isinstance(e, ev.ThinkingDelta):
            if self._thinking is None:
                self._thinking = ThinkingBlock(self.settings.show_thinking)
                await self.chat.mount(self._thinking)
            self._thinking.append(e.text)
        elif isinstance(e, ev.TextDelta):
            await self._end_thinking()
            if self._answer is None:
                self._answer = AssistantMessage()
                await self.chat.mount(self._answer)
            await self._answer.write(e.text)
        elif isinstance(e, ev.AssistantEnd):
            await self._end_thinking()
            await self._end_answer()
        elif isinstance(e, ev.ToolStart):
            if e.name == "task":
                widget: ToolLine | SubagentBlock = SubagentBlock(e.call_id, e.desc)
            else:
                widget = ToolLine(e.call_id, e.name, e.desc)
            self._tools[e.call_id] = widget
            await self.chat.mount(widget)
        elif isinstance(e, ev.SubagentUpdate):
            block = self._tools.get(e.call_id)
            if isinstance(block, SubagentBlock):
                block.progress(e)
        elif isinstance(e, ev.ToolEnd):
            line = self._tools.pop(e.call_id, None)
            if line is not None:
                line.finish(e)
        elif isinstance(e, ev.TodoUpdate):
            self.sidebar.update_todos(e.todos)
        elif isinstance(e, ev.FilesChanged):
            self.sidebar.update_changes(e.stats)
        elif isinstance(e, ev.VerifyStart):
            self._verify = VerifyBox(e.kind, e.command)
            await self.chat.mount(self._verify)
        elif isinstance(e, ev.VerifyEnd):
            box = self._verify or VerifyBox(e.kind, e.command)
            if self._verify is None:
                await self.chat.mount(box)
            box.finish(e)
            self._verify = None
        elif isinstance(e, ev.UsageUpdate):
            self.sidebar.update_usage(e.total, e.context_tokens, e.context_budget)
        elif isinstance(e, ev.Compacted):
            await self.chat.mount(CompactNotice(e))
            self.sidebar.update_context(e.after)
        elif isinstance(e, ev.Notice):
            await self.notice(e.text, e.level)
        elif isinstance(e, ev.TurnEnd):
            await self._end_thinking()
            await self._end_answer()
            self._tools.clear()
            self.busy = False
            if e.status == "interrupted":
                await self.notice("⏹ 已中断。可以直接补充说明后继续。", "warning")
            elif e.status == "error":
                await self.notice(f"✗ {e.error}", "error")
            if self.agent.checkpoints.turn_files() and e.status != "error":
                n = len(self.agent.checkpoints.turn_files())
                await self.notice(f"本轮修改了 {n} 个文件 · /diff 查看 · /undo 撤销", "info")
            self._tick()
            if self.queue:
                nxt = self.queue.pop(0)
                # 排队时已经显示过这条消息，这里只启动下一轮
                self._start_busy()
                self.run_agent(nxt)

    # ---------------------------------------------------------------- 按键动作

    def action_interrupt(self) -> None:
        if self.busy:
            self.queue.clear()
            self.agent.interrupt()
        else:
            self.prompt.text = ""

    def action_cycle_mode(self) -> None:
        self.agent.policy.cycle_mode()
        self.refresh_topbar()

    def set_mode(self, mode: Mode) -> None:
        self.agent.set_mode(mode)
        self.refresh_topbar()

    def action_toggle_sidebar(self) -> None:
        self.sidebar.display = not self.sidebar.display
        self._user_hid_sidebar = not self.sidebar.display

    def action_toggle_folds(self) -> None:
        folds = list(self.chat.query(Collapsible))
        if not folds:
            return
        expand = any(f.collapsed for f in folds)
        for f in folds:
            f.collapsed = not expand
