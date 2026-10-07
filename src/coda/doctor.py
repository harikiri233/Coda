"""coda doctor：检查配置、Key，并对每个模型档案做一次文本调用和一次工具调用。"""

from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from coda.config import coda_home, get_api_key, load_settings
from coda.llm import LLMClient, LLMError

_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path"}},
            "required": ["path"],
        },
    },
}


def _probe(client: LLMClient) -> tuple[str, str]:
    """返回 (文本调用结果, 工具调用结果)。"""
    start = time.perf_counter()
    reply = client.stream(
        [{"role": "user", "content": "Reply with exactly: ok"}], thinking=False, max_tokens=16
    )
    text = f"✓ {reply.content.strip()[:20]!r} {time.perf_counter() - start:.1f}s"

    start = time.perf_counter()
    reply = client.stream(
        [{"role": "user", "content": "Use the read_file tool to read README.md."}],
        tools=[_PROBE_TOOL],
    )
    elapsed = time.perf_counter() - start
    if not reply.tool_calls:
        return text, f"✗ 没有发起工具调用（finish_reason={reply.finish_reason}）"
    call = reply.tool_calls[0]
    args = call.parse_arguments()
    think = f"，思考 {len(reply.reasoning)} 字" if reply.reasoning else ""
    return text, f"✓ {call.name}({args}) {elapsed:.1f}s{think}"


def run_doctor(workdir: Path, console: Console | None = None) -> bool:
    console = console or Console()
    console.print(f"[b]Coda doctor[/]  工作区 {workdir}  全局配置 {coda_home()}")
    try:
        settings = load_settings(workdir)
    except ValueError as e:
        console.print(f"[red]配置错误：{e}[/]")
        return False

    table = Table(show_lines=False)
    for col in ("档案", "模型", "Key", "文本调用", "工具调用"):
        table.add_column(col)
    all_ok = True
    for name, profile in settings.models.items():
        mark = "★ " if name == settings.model else ""
        if not get_api_key(profile.api_key_env):
            table.add_row(
                mark + name, profile.model, f"[yellow]缺少 {profile.api_key_env}[/]", "-", "-"
            )
            if name == settings.model:
                all_ok = False
            continue
        try:
            text, tool = _probe(LLMClient(profile))
        except (LLMError, ValueError) as e:
            text, tool = f"[red]✗ {e}[/]", "-"
            all_ok = False
        table.add_row(mark + name, profile.model, "✓", text, tool)
    console.print(table)
    console.print("[dim]★ 为默认模型；只有默认模型缺 Key 或调用失败时自检判为失败。[/]")
    return all_ok
