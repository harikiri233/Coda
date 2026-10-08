"""工具执行器：一次回复里的多个 tool_call 先逐个过权限，再执行。

- 参数解析 / 校验失败、未知工具：回一条 Error[invalid_args]，让模型重发，不中断循环。
- 权限询问必须串行（一次只弹一个确认框）。PreToolUse Hook 在权限通过后、执行前运行，可否决。
- 执行：连续的只读调用并行，写操作和 bash 按原顺序串行。
  文件修改前先存检查点（/undo 用），并通知完成闸门（第一次修改前跑基线）。
- PostToolUse Hook 的输出追加到工具结果后面（如编辑后 ruff check 的报错）。
- 中断：给每个还没完成的调用补一条 Error[interrupted]。OpenAI 协议要求每个 tool_call
  都有对应的 tool 消息，不补齐下一次请求会报 400。
- 重复失败：同一工具、同样参数连续失败 3 次，在结果后追加"换一种方法"的提醒。
- 大输出：结果超过阈值就落盘，模型只看到头尾和文件路径（context/offload.py）。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from coda.agent.events import (
    Approver,
    Decision,
    EventSink,
    FilesChanged,
    PermissionLog,
    PermissionRequest,
    TodoUpdate,
    ToolEnd,
    ToolStart,
)
from coda.context.offload import Offloader, truncate_for_context
from coda.llm import ToolCall
from coda.safety.hooks import HookRunner
from coda.safety.policy import PermissionPolicy
from coda.safety.shell import is_readonly_command
from coda.state.checkpoints import Checkpoints
from coda.tools.base import (
    ErrorType,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    format_validation_error,
)

if TYPE_CHECKING:
    from coda.verify.gate import CompletionGate

MAX_PARALLEL = 4
REPEAT_LIMIT = 3


@dataclass
class _Prepared:
    call: ToolCall
    tool: Tool | None = None
    params: Any = None
    desc: str = ""
    args: dict[str, Any] | None = None
    result: ToolResult | None = None  # 预检阶段就确定的结果（参数错误、被拒绝）
    fs_before: Any = None  # 非只读 bash 命令执行前的工作区指纹（完成闸门检测 bash 改动用）


def tool_message(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


class ToolExecutor:
    def __init__(
        self,
        tools: ToolRegistry,
        ctx: ToolContext,
        policy: PermissionPolicy,
        sink: EventSink,
        approver: Approver,
        *,
        hooks: HookRunner | None = None,
        checkpoints: Checkpoints | None = None,
        gate: CompletionGate | None = None,
        offloader: Offloader | None = None,
    ) -> None:
        self.tools = tools
        self.ctx = ctx
        self.policy = policy
        self.sink = sink
        self.approver = approver
        self.hooks = hooks
        self.checkpoints = checkpoints
        self.gate = gate
        self.offloader = offloader
        self._last_failure: tuple[str, str] | None = None
        self._failure_count = 0

    # ---- 预检：解析参数、校验、过权限 ----

    def _prepare(self, call: ToolCall) -> _Prepared:
        prep = _Prepared(call)
        tool = self.tools.get(call.name)
        if tool is None:
            known = "、".join(self.tools.names())
            prep.desc = call.name
            prep.result = ToolResult.error(
                ErrorType.INVALID_ARGS, f"没有名为 {call.name!r} 的工具。", f"可用工具：{known}"
            )
            return prep
        prep.tool = tool
        try:
            prep.args = call.parse_arguments()
        except (json.JSONDecodeError, ValueError) as e:
            prep.desc = call.arguments[:80]
            prep.result = ToolResult.error(
                ErrorType.INVALID_ARGS,
                f"参数不是合法的 JSON 对象：{e}",
                "重新发起调用，参数必须是完整的 JSON 对象；内容很长时拆成多次较小的修改。",
            )
            return prep
        try:
            prep.params = tool.validate(prep.args)
        except ValidationError as e:
            prep.desc = json.dumps(prep.args, ensure_ascii=False)[:80]
            prep.result = ToolResult.error(
                ErrorType.INVALID_ARGS, f"参数校验失败：{format_validation_error(e)}"
            )
            return prep
        prep.desc = tool.describe(prep.params, self.ctx)
        return prep

    def _authorize(self, prep: _Prepared) -> None:
        assert prep.tool is not None
        verdict = self.policy.check(prep.tool, prep.params, self.ctx)
        if verdict.verdict == "deny":
            self._log(prep, "blocked" if verdict.danger else "denied", verdict.reason)
            prep.result = ToolResult.error(
                ErrorType.DENIED,
                f"操作被拒绝：{verdict.reason}",
                "不要换一种写法绕过；需要这个操作时在回答里说明，让用户自己执行。"
                if verdict.danger
                else None,
                summary="已拦截" if verdict.danger else "已拒绝",
            )
            return
        if verdict.verdict == "ask":
            preview = prep.tool.preview(prep.params, self.ctx) if prep.tool.kind == "edit" else None
            decision: Decision = self.approver.ask(
                PermissionRequest(
                    prep.tool.name,
                    prep.tool.kind,
                    prep.desc,
                    verdict.reason,
                    preview,
                    danger=verdict.danger,
                    always=[] if verdict.danger else verdict.always,
                ),
                self.ctx.cancel,
            )
            if self.ctx.cancel.is_set():
                self._log(prep, "interrupted", verdict.reason)
                prep.result = ToolResult.error(
                    ErrorType.INTERRUPTED, "用户中断了本轮。", summary="已中断"
                )
                return
            if decision.allow:
                self._log(prep, "always" if decision.always else "allowed", verdict.reason)
            else:
                self._log(prep, "denied", decision.reason or "用户拒绝")
            if not decision.allow:
                msg = "用户拒绝了这个操作。"
                if decision.reason:
                    msg += f"理由：{decision.reason}"
                prep.result = ToolResult.error(
                    ErrorType.DENIED,
                    msg,
                    "不要原样重试；按用户的意见调整做法，不清楚时直接问用户。",
                    summary="已拒绝",
                )
                return
            if decision.always and not verdict.danger:
                self.policy.remember(verdict.always)
        self._pre_hook(prep)

    def _log(self, prep: _Prepared, decision: str, reason: str) -> None:
        assert prep.tool is not None
        self.sink.emit(PermissionLog(prep.tool.name, prep.desc, decision, reason))  # type: ignore[arg-type]

    def _pre_hook(self, prep: _Prepared) -> None:
        assert prep.tool is not None
        if self.hooks is None or not self.hooks.has("PreToolUse", prep.tool.name):
            return
        path = prep.tool.target_path(prep.params, self.ctx)
        for out in self.hooks.run(
            "PreToolUse", prep.tool.name, prep.args or {}, path=path, cancel=self.ctx.cancel
        ):
            if out.vetoed:
                reason = out.output or f"Hook `{out.command}` 否决了这个操作"
                prep.result = ToolResult.error(
                    ErrorType.DENIED, f"PreToolUse Hook 否决：{reason}", summary="Hook 否决"
                )
                return

    # ---- 执行 ----

    def _before_run(self, prep: _Prepared) -> None:
        assert prep.tool is not None
        if prep.tool.kind == "bash" and self.gate is not None:
            readonly = is_readonly_command(prep.params.command, self.ctx.workdir)
            prep.fs_before = self.gate.before_bash(readonly)
            return
        if prep.tool.kind != "edit":
            return
        path = prep.tool.target_path(prep.params, self.ctx)
        if path is None:
            return
        if self.gate is not None:
            self.gate.before_edit(self.ctx.rel(path))
        if self.checkpoints is not None:
            self.checkpoints.snapshot(path)

    def _run_one(self, prep: _Prepared) -> tuple[ToolResult, float]:
        if prep.result is not None:
            return prep.result, 0.0
        if self.ctx.cancel.is_set():
            return ToolResult.error(
                ErrorType.INTERRUPTED, "用户中断了本轮，这个调用没有执行。", summary="已中断"
            ), 0.0
        assert prep.tool is not None
        start = time.monotonic()
        try:
            self._before_run(prep)
            result = prep.tool.run_call(prep.params, self.ctx, prep.call.id)
        except Exception as e:  # 工具内部的意外错误也回给模型，不让整轮崩掉
            result = ToolResult.error(
                ErrorType.INVALID_ARGS, f"工具执行出错：{type(e).__name__}: {e}"
            )
        if result.ok and result.error_type is None:
            self._post_hook(prep, result)
        return result, time.monotonic() - start

    def _post_hook(self, prep: _Prepared, result: ToolResult) -> None:
        assert prep.tool is not None
        if self.hooks is None or not self.hooks.has("PostToolUse", prep.tool.name):
            return
        path = prep.tool.target_path(prep.params, self.ctx)
        outs = self.hooks.run(
            "PostToolUse",
            prep.tool.name,
            prep.args or {},
            path=path,
            result=result.content,
            cancel=self.ctx.cancel,
        )
        notes = []
        for out in outs:
            if out.timed_out:
                notes.append(f"[PostToolUse Hook `{out.command}` 超时]")
            elif out.output:
                notes.append(
                    f"[PostToolUse Hook `{out.command}` 输出（exit {out.code}）]\n{out.output}"
                )
        if notes:
            result.content += "\n\n" + "\n\n".join(notes)
        result.display["hooks"] = [
            {"command": o.command, "code": o.code, "output": o.output, "timed_out": o.timed_out}
            for o in outs
        ]
        # Hook（如 ruff format）可能改了文件，重新记录，避免下一次编辑误报"读后被修改"
        if prep.tool.kind == "edit" and path is not None and path.is_file():
            self.ctx.filestate.record(path)

    def _track_repeat(self, prep: _Prepared, result: ToolResult) -> ToolResult:
        if result.error_type in (None, ErrorType.INTERRUPTED, ErrorType.DENIED):
            self._last_failure, self._failure_count = None, 0
            return result
        key = (prep.call.name, prep.call.arguments)
        if key == self._last_failure:
            self._failure_count += 1
        else:
            self._last_failure, self._failure_count = key, 1
        if self._failure_count >= REPEAT_LIMIT:
            note = f"同样的调用已经连续失败 {self._failure_count} 次。不要再原样重试，换一种方法（换工具、换参数，或先读取 / 搜索确认情况）。"
            result.hint = f"{result.hint}\n{note}" if result.hint else note
        return result

    def _finish(self, prep: _Prepared, result: ToolResult, elapsed: float) -> dict[str, Any]:
        result = self._track_repeat(prep, result)
        text = result.to_text()
        ok = result.ok and result.error_type is None
        if self.offloader is not None:
            text, saved = self.offloader.maybe(text, prep.call.id, prep.call.name)
            if saved is not None:
                result.display["offloaded"] = str(saved)
                result.display.setdefault("output", result.display.get("output") or result.content)
        else:
            text = truncate_for_context(text)
        self.sink.emit(
            ToolEnd(
                prep.call.id,
                prep.call.name,
                prep.desc,
                ok,
                str(result.error_type) if result.error_type else None,
                text,
                result.display,
                elapsed,
            )
        )
        if ok and prep.tool is not None:
            if prep.tool.name == "todo_write":
                self.sink.emit(TodoUpdate(list(self.ctx.todos)))
            elif prep.tool.kind == "edit":
                path = prep.tool.target_path(prep.params, self.ctx)
                if self.gate is not None and path is not None:
                    self.gate.after_edit(self.ctx.rel(path))
                if self.checkpoints is not None:
                    self.sink.emit(FilesChanged(self.checkpoints.session_stats()))
            elif prep.tool.kind == "bash" and self.gate is not None:
                self.gate.after_bash(prep.fs_before)
                self.gate.note_bash(prep.desc, result.display.get("exit_code"))
        return tool_message(prep.call.id, text)

    def run(self, calls: list[ToolCall]) -> list[dict[str, Any]]:
        preps = [self._prepare(c) for c in calls]
        for prep in preps:
            self.sink.emit(ToolStart(prep.call.id, prep.call.name, prep.desc, prep.args or {}))
        for prep in preps:
            if prep.result is None and not self.ctx.cancel.is_set():
                self._authorize(prep)

        # 按顺序切分：连续的只读调用成一组并行执行，其余一个一组
        groups: list[list[_Prepared]] = []
        for prep in preps:
            parallel = prep.result is None and prep.tool is not None and prep.tool.read_only
            last = groups[-1][0] if groups else None
            if (
                parallel
                and last is not None
                and last.tool is not None
                and last.tool.read_only
                and last.result is None
            ):
                groups[-1].append(prep)
            else:
                groups.append([prep])

        messages: list[dict[str, Any]] = []
        for group in groups:
            if len(group) == 1:
                result, elapsed = self._run_one(group[0])
                messages.append(self._finish(group[0], result, elapsed))
                continue
            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(group))) as pool:
                outcomes = list(pool.map(self._run_one, group))
            for prep, (result, elapsed) in zip(group, outcomes, strict=True):
                messages.append(self._finish(prep, result, elapsed))
        return messages
