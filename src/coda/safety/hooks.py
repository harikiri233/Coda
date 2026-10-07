"""Hooks：在工具执行前后运行用户配置的命令。

- 配置在 settings.json 的 hooks.PreToolUse / hooks.PostToolUse，matcher 是工具名正则。
- Hook 进程从 stdin 收到 JSON（事件名、工具名、参数，PostToolUse 还有结果），
  环境变量 CODA_TOOL 为工具名，CODA_FILE 为文件类工具操作的绝对路径。
- PreToolUse：退出码 2 表示否决，stderr 作为拒绝理由回给模型。
- PostToolUse：stdout（和非 0 退出时的 stderr）追加到工具结果后面，典型用法是编辑后跑 ruff check。
- 单个 Hook 超时默认 10 秒；超时或启动失败只记录，不影响主流程。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from coda.config import Hooks, HookSpec
from coda.tools.bash import run_shell

HookEvent = Literal["PreToolUse", "PostToolUse"]
MAX_HOOK_OUTPUT = 4000


@dataclass
class HookOutcome:
    command: str
    code: int | None
    output: str
    timed_out: bool = False

    @property
    def vetoed(self) -> bool:
        return self.code == 2


class HookRunner:
    def __init__(self, hooks: Hooks, workdir: Path) -> None:
        self.hooks = hooks
        self.workdir = workdir

    def _matching(self, event: HookEvent, tool: str) -> list[HookSpec]:
        specs = self.hooks.PreToolUse if event == "PreToolUse" else self.hooks.PostToolUse
        out = []
        for spec in specs:
            try:
                if re.fullmatch(spec.matcher, tool):
                    out.append(spec)
            except re.error:
                continue
        return out

    def has(self, event: HookEvent, tool: str) -> bool:
        return bool(self._matching(event, tool))

    def run(
        self,
        event: HookEvent,
        tool: str,
        args: dict[str, Any],
        *,
        path: Path | None = None,
        result: str | None = None,
        cancel: threading.Event | None = None,
    ) -> list[HookOutcome]:
        payload: dict[str, Any] = {
            "event": event,
            "tool": tool,
            "args": args,
            "cwd": str(self.workdir),
        }
        if result is not None:
            payload["result"] = result
        env = {"CODA_TOOL": tool, "CODA_FILE": str(path) if path else ""}
        outcomes = []
        for spec in self._matching(event, tool):
            r = run_shell(
                spec.command,
                self.workdir,
                timeout=spec.timeout,
                cancel=cancel,
                stdin=json.dumps(payload, ensure_ascii=False).encode(),
                env=env,
            )
            if event == "PreToolUse":
                text = r.stderr.strip() or r.stdout.strip()
            else:
                text = r.stdout.strip()
                if r.code not in (0, None) and r.stderr.strip():
                    text = f"{text}\n{r.stderr.strip()}".strip()
            if len(text) > MAX_HOOK_OUTPUT:
                text = text[:MAX_HOOK_OUTPUT] + "\n…（Hook 输出已截断）"
            outcome = HookOutcome(spec.command, r.code, text, r.state == "timeout")
            outcomes.append(outcome)
            if event == "PreToolUse" and outcome.vetoed:
                break
        return outcomes
