"""LLM 客户端：OpenAI 兼容协议的流式调用，处理各家的协议差异。

- 流式 tool_calls：参数按 index 分片到达，逐片累积，结束后再解析。硅基流动在后续分片里
  会带空字符串的 name，所以 name 只在非空时记录（实测）。
- DeepSeek 思考模式：请求带 tools 时，历史里每条 assistant 消息都必须带 reasoning_content，
  缺失返回 400；空字符串可以通过（实测）。所以发送前给缺失的补空串。
- 其他服务商：发送前去掉 reasoning_content，避免报未知字段或按输入计费。
- 中断：cancel 事件被设置后关闭 HTTP 流，返回已收到的部分，并丢弃未完成的 tool_calls。
- 限流：SDK 自带的重试最多只等 8 秒，硅基流动的 TPM 限流按分钟计（实测），
  所以 429 再按 10 / 20 / 40 / 60 秒退避重试，仍失败才报错。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import OpenAI

from coda.config import ModelProfile, get_api_key
from coda.llm.usage import Usage, UsageTracker, usage_from_api

Message = dict[str, Any]
TextCallback = Callable[[str], None]


RATE_LIMIT_WAITS = (10, 20, 40, 60)


class LLMError(RuntimeError):
    """模型调用失败（鉴权、参数、服务端错误等），信息可直接展示给用户。"""


class ContextTooLongError(LLMError):
    """请求超过模型上下文窗口。主循环收到后先强制压缩再重试。"""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串

    def parse_arguments(self) -> dict[str, Any]:
        """解析参数；空字符串视为 {}。解析失败抛 ValueError，由执行器回填给模型。"""
        if not self.arguments.strip():
            return {}
        data = json.loads(self.arguments)
        if not isinstance(data, dict):
            raise ValueError("参数必须是 JSON 对象")
        return data

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Reply:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    cancelled: bool = False

    def to_message(self) -> Message:
        """转成写入历史的 assistant 消息。reasoning_content 总是保留，发送时再按服务商处理。"""
        msg: Message = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return msg


def _is_context_error(err: openai.APIStatusError) -> bool:
    text = str(err).lower()
    return any(
        key in text
        for key in (
            "context length",
            "context_length",
            "maximum context",
            "too many tokens",
            "上下文",
        )
    )


class LLMClient:
    def __init__(
        self,
        profile: ModelProfile,
        *,
        tracker: UsageTracker | None = None,
        client: Any = None,
        timeout: float = 120.0,
    ) -> None:
        """client 用于测试注入替身；正常使用时按档案创建 OpenAI 客户端。"""
        self.profile = profile
        self.tracker = tracker or UsageTracker()
        if client is None:
            key = get_api_key(profile.api_key_env)
            if not key:
                raise LLMError(
                    f"未找到 API Key：请设置环境变量 {profile.api_key_env}，或写入 ~/.coda/.env"
                )
            # SDK 自带重试：连接错误、408、409、429、5xx 按指数退避
            client = OpenAI(api_key=key, base_url=profile.base_url, timeout=timeout, max_retries=3)
        self._client = client
        self._sleep = time.sleep  # 测试时替换

    @property
    def is_deepseek(self) -> bool:
        return self.profile.provider == "deepseek"

    def thinking_enabled(self, override: bool | None = None) -> bool:
        return self.is_deepseek and (self.profile.thinking if override is None else override)

    # ---- 请求组装 ----

    def prepare_messages(
        self, messages: list[Message], *, thinking: bool, with_tools: bool
    ) -> list[Message]:
        """按服务商处理 reasoning_content；不修改传入的历史。"""
        out: list[Message] = []
        for m in messages:
            if m.get("role") != "assistant":
                out.append(m)
                continue
            m = dict(m)
            if self.is_deepseek and thinking and with_tools:
                m.setdefault("reasoning_content", "")
            else:
                m.pop("reasoning_content", None)
            out.append(m)
        return out

    def _kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        thinking: bool,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "messages": self.prepare_messages(messages, thinking=thinking, with_tools=bool(tools)),
        }
        if tools:
            kwargs["tools"] = tools
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        tokens = max_tokens or self.profile.max_tokens
        if tokens:
            kwargs["max_tokens"] = tokens
        if self.is_deepseek:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
            if thinking and self.profile.reasoning_effort:
                kwargs["reasoning_effort"] = self.profile.reasoning_effort
        return kwargs

    def _create(self, **kwargs: Any) -> Any:
        for attempt in range(len(RATE_LIMIT_WAITS) + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as e:
                if attempt == len(RATE_LIMIT_WAITS):
                    raise LLMError(f"模型接口持续限流（HTTP 429）：{e.message}") from e
                self._sleep(RATE_LIMIT_WAITS[attempt])
            except openai.APIStatusError as e:
                if _is_context_error(e):
                    raise ContextTooLongError(str(e)) from e
                raise LLMError(f"模型接口返回错误（HTTP {e.status_code}）：{e.message}") from e
            except openai.APIConnectionError as e:
                raise LLMError(f"无法连接模型接口 {self.profile.base_url}：{e}") from e
        raise AssertionError("unreachable")

    # ---- 调用 ----

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_text: TextCallback | None = None,
        on_thinking: TextCallback | None = None,
        cancel: threading.Event | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        main: bool = True,
    ) -> Reply:
        """流式调用。文本和思考增量通过回调实时交出；返回完整的 Reply。"""
        think = self.thinking_enabled(thinking)
        response = self._create(
            **self._kwargs(
                messages, tools=tools, thinking=think, max_tokens=max_tokens, stream=True
            )
        )
        reply = Reply()
        text: list[str] = []
        reasoning: list[str] = []
        partial: dict[int, dict[str, str]] = {}
        raw_usage = None
        try:
            for chunk in response:
                if cancel is not None and cancel.is_set():
                    reply.cancelled = True
                    break
                if getattr(chunk, "usage", None):
                    raw_usage = chunk.usage  # 硅基流动每个分片都带累计 usage，取最后一个
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                r = getattr(delta, "reasoning_content", None)
                if r:
                    reasoning.append(r)
                    if on_thinking:
                        on_thinking(r)
                if delta.content:
                    text.append(delta.content)
                    if on_text:
                        on_text(delta.content)
                for tc in delta.tool_calls or []:
                    slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = tc.function
                    if fn is not None:
                        if fn.name:
                            slot["name"] = fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
                if choice.finish_reason:
                    reply.finish_reason = choice.finish_reason
        except openai.APIError as e:
            raise LLMError(f"流式响应中断：{e}") from e
        finally:
            if reply.cancelled:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

        reply.content = "".join(text)
        reply.reasoning = "".join(reasoning)
        if not reply.cancelled:
            reply.tool_calls = [
                ToolCall(id=s["id"] or f"call_{i}", name=s["name"], arguments=s["arguments"])
                for i, s in sorted(partial.items())
            ]
        reply.usage = usage_from_api(raw_usage, self.profile.price_per_m)
        self.tracker.add(reply.usage, main=main)
        return reply

    def complete(
        self, messages: list[Message], *, max_tokens: int | None = None, main: bool = False
    ) -> Reply:
        """非流式、无工具、关闭思考的调用，用于摘要压缩等内部任务。"""
        response = self._create(
            **self._kwargs(
                messages, tools=None, thinking=False, max_tokens=max_tokens, stream=False
            )
        )
        choice = response.choices[0]
        usage = usage_from_api(getattr(response, "usage", None), self.profile.price_per_m)
        self.tracker.add(usage, main=main)
        return Reply(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=usage,
        )
