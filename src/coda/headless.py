"""无头模式 coda -p：不打开界面，单次执行一个任务。

- --output text：在终端实时打印思考（灰色）、工具调用一行摘要和流式回答。
- --output json：只在结束时向 stdout 输出一个 JSON（最终回答、状态、步数、工具调用、用量），供脚本和评测使用；
  过程信息打印到 stderr。
- 需要确认的操作一律拒绝（DenyApprover），原因回填给模型。
- 和 TUI 一样写会话 JSONL（评测从这里统计），JSON 输出里带 session 路径；-c 可以接着上次的会话执行。
- 配置了 MCP Server 时先连接（最多等握手超时），连上的工具照常可用（默认仍需确认，所以要用 allow 规则放行）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.markup import escape

from coda.agent.events import (
    AssistantEnd,
    Compacted,
    DenyApprover,
    Event,
    ListSink,
    Notice,
    StepStart,
    TextDelta,
    ThinkingDelta,
    ToolEnd,
    ToolStart,
    TurnEnd,
    VerifyEnd,
    VerifyStart,
)
from coda.agent.loop import Agent
from coda.config import Mode, Settings
from coda.llm import LLMClient
from coda.mcp_client import McpManager
from coda.state.session import Session, load_session

OutputFormat = Literal["text", "json"]


def short(text: str, n: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def result_mark(ev: ToolEnd) -> tuple[str, str]:
    """工具结果的圆点颜色和摘要：绿=成功，红=失败，黄=被拒绝 / 中断。"""
    if ev.ok:
        color = "green"
    elif ev.error_type in ("denied", "interrupted"):
        color = "yellow"
    else:
        color = "red"
    summary = ev.display.get("summary") or ("" if ev.ok else f"Error[{ev.error_type}]")
    return color, summary


def verify_line(ev: VerifyEnd) -> str:
    """完成闸门结果的一行描述（rich markup），TUI 和无头模式共用。"""
    if ev.kind == "baseline":
        if ev.baseline_failures > 0:
            base = f"修改前已有 {ev.baseline_failures} 个失败，不计入本次"
        elif ev.baseline_failures < 0:
            base = "修改前验证命令就失败"
        else:
            base = "修改前全部通过"
        return f"[dim]基线 · {escape(ev.summary)} · {base} · {ev.elapsed:.1f}s[/]"
    tail = f"[dim]{escape(ev.command)} · {ev.elapsed:.1f}s[/]"
    if ev.ok:
        line = f"[b green]✓ 验证通过[/]  {escape(ev.summary)}  {tail}"
        if ev.warning:
            line += f"\n[yellow]{escape(ev.warning)}[/]"
    else:
        names = "、".join(ev.new_failures[:5]) + ("…" if len(ev.new_failures) > 5 else "")
        if ev.feedback:
            head = f"[b yellow]✗ 新增 {len(ev.new_failures)} 个失败，已回给 Agent 修复（{ev.round} 轮）[/]"
        elif ev.gave_up:
            head = f"[b red]⚠ 验证未通过[/]（已回填 {ev.round or '多'} 轮仍失败）"
        else:
            head = "[b red]⚠ 验证未通过[/]"
        line = f"{head}  {escape(ev.summary)}  {tail}\n  [red]{escape(names)}[/]"
    if ev.tests_edited:
        line += f"\n[yellow]注意：本轮修改了测试文件 {escape('、'.join(ev.tests_edited))}，请检查是否合理。[/]"
    return line


class TextPrinter:
    """把事件实时打印到终端。"""

    def __init__(self, console: Console, show_thinking: bool = True) -> None:
        self.console = console
        self.show_thinking = show_thinking
        self._in = None  # 当前正在流式输出的类型：thinking / text

    def _end_stream(self) -> None:
        if self._in:
            self.console.file.write("\n")
            self.console.file.flush()
            self._in = None

    def _stream(self, kind: str, text: str, style: str | None = None) -> None:
        if self._in != kind:
            self._end_stream()
            self._in = kind
            if kind == "thinking":
                self.console.print("[dim]思考：[/]", end="")
        if style:
            self.console.print(text, style=style, end="", markup=False, highlight=False)
        else:
            self.console.file.write(text)
            self.console.file.flush()

    def emit(self, ev: Event) -> None:
        if isinstance(ev, ThinkingDelta):
            if self.show_thinking:
                self._stream("thinking", ev.text, "dim")
        elif isinstance(ev, TextDelta):
            self._stream("text", ev.text)
        elif isinstance(ev, AssistantEnd):
            self._end_stream()
        elif isinstance(ev, ToolStart):
            pass
        elif isinstance(ev, ToolEnd):
            self._end_stream()
            color, summary = result_mark(ev)
            line = f"[{color}]●[/] [b]{ev.name}[/] {escape(short(ev.desc))}"
            if summary:
                line += f"  [dim]→ {escape(summary)}[/]"
            self.console.print(line, highlight=False)
            if not ev.ok:
                self.console.print(f"  [dim]{escape(short(ev.text, 200))}[/]", highlight=False)
        elif isinstance(ev, Notice):
            self._end_stream()
            color = {"info": "cyan", "warning": "yellow", "error": "red"}[ev.level]
            self.console.print(f"[{color}]{escape(ev.text)}[/]")
        elif isinstance(ev, VerifyStart):
            self._end_stream()
            what = "记录基线" if ev.kind == "baseline" else "完成闸门"
            self.console.print(f"[dim]⧗ {what}：{escape(ev.command)}[/]")
        elif isinstance(ev, VerifyEnd):
            self.console.print(verify_line(ev), highlight=False)
        elif isinstance(ev, Compacted):
            self._end_stream()
            what = "微压缩" if ev.kind == "micro" else "摘要压缩"
            self.console.print(
                f"[cyan]⇣ 上下文{what}：{ev.before:,} → {ev.after:,} token · {escape(ev.detail)}[/]"
            )
        elif isinstance(ev, StepStart):
            pass
        elif isinstance(ev, TurnEnd):
            self._end_stream()


class TeeSink:
    def __init__(self, *sinks: Any) -> None:
        self.sinks = sinks

    def emit(self, ev: Event) -> None:
        for s in self.sinks:
            s.emit(ev)


def run_headless(
    prompt: str,
    settings: Settings,
    workdir: Path,
    *,
    model: str | None = None,
    mode: Mode | None = None,
    output: OutputFormat = "text",
    max_steps: int | None = None,
    llm: LLMClient | None = None,
    resume: Path | None = None,
) -> int:
    """返回进程退出码：0 正常结束，1 出错，2 达到最大步数或被中断。"""
    profile_name = model or settings.model
    llm = llm or LLMClient(settings.profile(profile_name))
    collected = ListSink()
    console = Console(file=sys.stdout if output == "text" else sys.stderr, highlight=False)
    printer = TextPrinter(console, settings.show_thinking and output == "text")
    sink = TeeSink(collected, printer)
    session = Session.open(resume) if resume else Session.create(workdir, profile_name)
    agent = Agent(
        llm,
        workdir,
        sink,
        DenyApprover(),
        mode=mode or settings.mode,
        max_steps=max_steps or settings.max_steps,
        permissions=settings.permissions,
        hooks=settings.hooks,
        verify=settings.verify,
        interactive=False,
        context=settings.context,
        session=session,
    )
    if resume:
        agent.load(load_session(resume), session)
    mcp = McpManager(settings.mcpServers, workdir)
    if mcp.servers:
        mcp.start()
        mcp.wait_ready()
        mcp.register(agent.tools)
        for st in mcp.servers.values():
            if st.status != "connected":
                console.print(f"[yellow]MCP {escape(st.name)} 连接失败：{escape(st.error)}[/]")
    start = time.monotonic()
    try:
        end = agent.run_turn(prompt)
    except KeyboardInterrupt:
        agent.interrupt()
        end = TurnEnd("interrupted", 0)
    finally:
        mcp.stop()
    elapsed = time.monotonic() - start
    usage = llm.tracker.total

    if output == "json":
        tools = [
            {
                "name": e.name,
                "desc": e.desc,
                "ok": e.ok,
                "error_type": e.error_type,
                "elapsed": round(e.elapsed, 3),
            }
            for e in collected.of(ToolEnd)
        ]
        data = {
            "status": end.status,
            "result": end.final_text,
            "error": end.error,
            "steps": end.steps,
            "model": profile_name,
            "mode": agent.mode,
            "elapsed": round(elapsed, 2),
            "tool_calls": tools,
            "verify": end.verify,
            "session": str(session.path),
            "compactions": [
                {"kind": c.kind, "before": c.before, "after": c.after}
                for c in collected.of(Compacted)
            ],
            "usage": {**usage.to_dict(), "cache_hit_rate": round(usage.cache_hit_rate, 4)},
        }
        sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    else:
        if end.error:
            console.print(f"[red]错误：{escape(end.error)}[/]")
        status = {
            "done": "完成",
            "interrupted": "已中断",
            "max_steps": "达到最大步数",
            "error": "出错",
        }[end.status]
        console.print(
            f"[dim]── {status} · {end.steps} 步 · {elapsed:.1f}s · 输入 {usage.input_tokens:,} "
            f"(缓存 {usage.cache_hit_rate:.0%}) · 输出 {usage.output_tokens:,} · ${usage.cost_usd:.4f}[/]"
        )
    return {"done": 0, "error": 1}.get(end.status, 2)
