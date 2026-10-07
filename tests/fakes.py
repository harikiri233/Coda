"""FakeLLM：按脚本依次返回文本或工具调用，确定性地测试主循环和界面，不调用真实接口。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from coda.config import ModelProfile
from coda.llm import Reply, ToolCall, Usage, UsageTracker


@dataclass
class Step:
    text: str = ""
    thinking: str = ""
    calls: list[tuple[str, dict[str, Any] | str]] = field(default_factory=list)
    finish: str | None = None
    delay: float = 0.0  # 每个分片之间的间隔，用来测试中断
    input_tokens: int | None = None  # 这次请求报告的 prompt_tokens（测试压缩阈值用）
    error: Exception | None = None  # 这次请求直接抛出的异常（如 ContextTooLongError）


def call(name: str, args: dict[str, Any] | str | None = None) -> tuple[str, dict[str, Any] | str]:
    return name, args or {}


SUB_MARK = "你是 Coda 派生的只读调查子智能体"


class FakeLLM:
    """steps：主循环的脚本；sub_steps：子智能体的脚本（按系统提示词区分）；summaries：摘要压缩的返回。"""

    is_deepseek = False

    def __init__(
        self,
        steps: list[Step],
        *,
        sub_steps: list[Step] | None = None,
        summaries: list[str] | None = None,
        budget: int = 128_000,
    ) -> None:
        self.steps = list(steps)
        self.sub_steps = list(sub_steps or [])
        self.summaries = list(summaries or [])
        self.requests: list[list[dict[str, Any]]] = []
        self.sub_requests: list[list[dict[str, Any]]] = []
        self.summary_requests: list[list[dict[str, Any]]] = []
        self.tracker = UsageTracker()
        self.profile = ModelProfile(
            base_url="http://fake", api_key_env="X", model="fake", context_budget=budget
        )
        self._n = 0
        self._lock = threading.Lock()

    def thinking_enabled(self, override=None) -> bool:
        return False

    def complete(self, messages, *, max_tokens=None, main=False) -> Reply:
        self.summary_requests.append([dict(m) for m in messages])
        if not self.summaries:
            raise AssertionError("FakeLLM 的摘要脚本已经用完")
        return Reply(content=self.summaries.pop(0))

    def stream(
        self,
        messages,
        *,
        tools=None,
        on_text=None,
        on_thinking=None,
        cancel: threading.Event | None = None,
        **_: Any,
    ) -> Reply:
        sub = SUB_MARK in (messages[0].get("content") or "")
        with self._lock:
            (self.sub_requests if sub else self.requests).append([dict(m) for m in messages])
            queue = self.sub_steps if sub else self.steps
            if not queue:
                raise AssertionError(f"FakeLLM 的{'子智能体' if sub else ''}脚本已经用完")
            step = queue.pop(0)
            if step.error is not None:
                raise step.error
            n_req = len(self.requests)
        reply = Reply()
        for kind, text, cb in (("r", step.thinking, on_thinking), ("t", step.text, on_text)):
            for ch in _chunks(text):
                if cancel is not None and cancel.is_set():
                    reply.cancelled = True
                    break
                if step.delay:
                    time.sleep(step.delay)
                if kind == "r":
                    reply.reasoning += ch
                else:
                    reply.content += ch
                if cb:
                    cb(ch)
            if reply.cancelled:
                break
        if cancel is not None and cancel.is_set():
            reply.cancelled = True
        if not reply.cancelled:
            for name, args in step.calls:
                with self._lock:
                    self._n += 1
                    n = self._n
                raw = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                reply.tool_calls.append(ToolCall(f"call_{n}", name, raw))
        reply.finish_reason = step.finish or ("tool_calls" if reply.tool_calls else "stop")
        tokens = step.input_tokens if step.input_tokens is not None else 100 * n_req
        reply.usage = Usage(input_tokens=tokens, cached_tokens=50, output_tokens=10)
        self.tracker.add(reply.usage, main=not sub)
        return reply


def _chunks(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
