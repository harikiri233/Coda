"""对话流里的组件：每种事件对应一种组件，app 只负责"事件 → 组件"的映射。"""

from __future__ import annotations

import time

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from coda.agent.events import Compacted, SubagentUpdate, ToolEnd, VerifyEnd
from coda.headless import result_mark, short, verify_line

OUTPUT_PREVIEW_LINES = 6


class UserMessage(Static):
    def __init__(self, text: str, queued: bool = False) -> None:
        super().__init__(Text(text), classes="user")
        if queued:
            self.add_class("queued")


class Notice(Static):
    def __init__(self, text: str, level: str = "info") -> None:
        super().__init__(Text(text), classes=f"notice {level}")


class AssistantMessage(Markdown):
    """流式 Markdown 回答。写入走 MarkdownStream：渲染跟不上时自动合并积压的片段。"""

    def __init__(self) -> None:
        super().__init__(classes="assistant")
        self._stream: MarkdownStream | None = None
        self.text = ""

    async def write(self, fragment: str) -> None:
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        self.text += fragment
        await self._stream.write(fragment)

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class ThinkingBlock(Collapsible):
    """思考过程：流式显示在灰色折叠块里，结束后自动折叠成"思考 3.2s"。"""

    def __init__(self, show_content: bool = True) -> None:
        self._body = Static("", classes="thinking-body")
        super().__init__(
            self._body, title="思考中…", collapsed=not show_content, classes="thinking"
        )
        self._started = time.monotonic()
        self.text = ""
        self.done = False

    def append(self, text: str) -> None:
        self.text += text
        self._body.update(Text(self.text))

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        self.title = f"思考 {time.monotonic() - self._started:.1f}s"
        self.collapsed = True


class ToolLine(Vertical):
    """一次工具调用：一行摘要 + 可选的 diff / 折叠的输出。"""

    def __init__(self, call_id: str, name: str, desc: str) -> None:
        super().__init__(classes="tool")
        self.call_id = call_id
        self.tool_name = name
        self.desc = desc
        self._line = Static(self._render_line("dim", "…"), classes="tool-line")

    def compose(self) -> ComposeResult:
        yield self._line

    def _render_line(self, color: str, summary: str) -> Text:
        dot = "○" if summary == "…" else "●"
        line = f"[{color}]{dot}[/] [b]{escape(self.tool_name)}[/] {escape(short(self.desc, 110))}"
        if summary:
            line += f"  [dim]→ {escape(summary)}[/]"
        return Text.from_markup(line)

    def finish(self, ev: ToolEnd) -> None:
        color, summary = result_mark(ev)
        if ev.display.get("offloaded"):
            summary += " · 完整输出已存盘"
        self._line.update(self._render_line(color, summary))
        diff = ev.display.get("diff")
        if ev.ok and diff:
            self.mount(
                Static(
                    Syntax(diff.rstrip("\n"), "diff", background_color="default"), classes="diff"
                )
            )
            self._show_hooks(ev)
            return
        if ev.name == "todo_write":
            return
        self._show_hooks(ev)
        show_output = ev.name == "bash" or ev.name.startswith("mcp__")
        output = ev.display.get("output") if show_output else None
        if not ev.ok:
            output = ev.text
        if output:
            lines = output.splitlines()
            head = "\n".join(lines[:OUTPUT_PREVIEW_LINES])
            if len(lines) <= OUTPUT_PREVIEW_LINES:
                self.mount(Static(Text(head), classes="tool-output"))
            else:
                self.mount(
                    Collapsible(
                        Static(Text(output), classes="tool-output"),
                        title=f"{head.splitlines()[0][:80]} …（共 {len(lines)} 行，点开查看）",
                        collapsed=True,
                        classes="tool-fold",
                    )
                )

    def _show_hooks(self, ev: ToolEnd) -> None:
        for h in ev.display.get("hooks", []):
            ok = h["code"] == 0 and not h["timed_out"]
            status = "超时" if h["timed_out"] else ("通过" if ok else f"exit {h['code']}")
            color = "green" if ok else "yellow"
            line = f"  [dim]↳ PostToolUse[/] {escape(short(h['command'], 60))} [{color}]{status}[/]"
            out = h["output"].strip()
            if out and not ok:
                line += "\n" + "\n".join(f"    [dim]{escape(x)}[/]" for x in out.splitlines()[:8])
            self.mount(Static(Text.from_markup(line), classes="tool-output"))


class VerifyBox(Static):
    """完成闸门：运行中显示命令，结束后显示通过 / 新增失败。基线只显示一行灰字。"""

    def __init__(self, kind: str, command: str) -> None:
        what = "记录修改前的测试基线" if kind == "baseline" else "完成闸门：运行验证"
        super().__init__(
            Text.from_markup(f"[dim]⧗ {what}  {escape(command)} …[/]"),
            classes="gate" if kind == "final" else "gate-baseline",
        )

    def finish(self, ev: VerifyEnd) -> None:
        self.update(Text.from_markup(verify_line(ev)))
        if ev.kind == "final":
            self.add_class("ok" if ev.ok else ("retry" if ev.feedback else "fail"))


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class SubagentBlock(Collapsible):
    """子智能体：标题实时显示步数和 token，展开能看到它调用了哪些工具和最终结论。"""

    def __init__(self, call_id: str, desc: str) -> None:
        self._body = Vertical(classes="sub-body")
        self.call_id = call_id
        self.desc = desc
        self.step = 0
        self.tokens = 0
        self.tool_count = 0
        self.done = False
        super().__init__(self._body, title=self._make_title(), collapsed=True, classes="subagent")

    def _make_title(self, mark: str = "[yellow]◌[/]", tail: str = "") -> str:
        tail = tail or f"第 {self.step} 步 · {self.tool_count} 次工具 · {_k(self.tokens)} token"
        return f"{mark} 子智能体 · {escape(self.desc)}  [dim]{escape(tail)}[/]"

    def progress(self, ev: SubagentUpdate) -> None:
        if self.done:
            return
        self.step, self.tokens = ev.step, ev.tokens
        if ev.tool:
            self.tool_count += 1
            color = "green" if ev.ok else "red"
            line = f"[{color}]●[/] [b]{escape(ev.tool)}[/] {escape(short(ev.desc, 90))}"
            self._body.mount(Static(Text.from_markup(line), classes="tool-line"))
        self.title = self._make_title()

    def finish(self, ev: ToolEnd) -> None:
        self.done = True
        color, summary = result_mark(ev)
        self.title = self._make_title(f"[{color}]●[/]", summary)
        answer = ev.display.get("answer") if ev.ok else ev.text
        if answer:
            self._body.mount(Markdown(answer, classes="sub-answer"))


class CompactNotice(Static):
    def __init__(self, ev: Compacted) -> None:
        what = "微压缩" if ev.kind == "micro" else "摘要压缩"
        why = {"manual": "手动 /compact", "overflow": "超出窗口兜底", "auto": "自动"}.get(
            ev.reason, ev.reason
        )
        text = f"[b cyan]⇣ 上下文{what}[/]（{why}）  {_k(ev.before)} → {_k(ev.after)} token · {escape(ev.detail)}"
        if ev.kind == "summary":
            text += "\n[dim]用户原话已原样保留；任务清单和改过的文件已附在摘要里。[/]"
        super().__init__(Text.from_markup(text), classes="compact")
