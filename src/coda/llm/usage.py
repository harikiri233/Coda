"""用量与费用统计。

DeepSeek 的缓存命中 token 在 prompt_cache_hit_tokens，OpenAI 兼容服务一般在
prompt_tokens_details.cached_tokens（逻辑沿用 PaperLens 的 models/llm.py）。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from coda.config import Price


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cost_usd += other.cost_usd
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


def usage_from_api(raw: Any, price: Price | None) -> Usage:
    """把 API 返回的 usage 对象转成 Usage，并按价格算费用。"""
    if raw is None:
        return Usage()
    input_tokens = _int(getattr(raw, "prompt_tokens", 0))
    output_tokens = _int(getattr(raw, "completion_tokens", 0))
    cached = getattr(raw, "prompt_cache_hit_tokens", None)
    if not isinstance(cached, int):
        details = getattr(raw, "prompt_tokens_details", None)
        cached = _int(getattr(details, "cached_tokens", 0))
    out_details = getattr(raw, "completion_tokens_details", None)
    reasoning = _int(getattr(out_details, "reasoning_tokens", 0))
    cost = 0.0
    if price is not None:
        miss = max(input_tokens - cached, 0)
        cost = (miss * price.input + cached * price.cache_hit + output_tokens * price.output) / 1e6
    return Usage(input_tokens, cached, output_tokens, reasoning, cost)


class UsageTracker:
    """会话级累计用量。子智能体在其他线程里也会记账，所以加锁。"""

    def __init__(self) -> None:
        self.total = Usage()
        self.calls = 0
        self.last_input_tokens = 0  # 最近一次主循环请求的输入 token，用于估算上下文占用
        self._lock = threading.Lock()

    def add(self, usage: Usage, *, main: bool = True) -> None:
        with self._lock:
            self.total += usage
            self.calls += 1
            if main and usage.input_tokens:
                self.last_input_tokens = usage.input_tokens
