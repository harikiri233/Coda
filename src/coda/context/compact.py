"""上下文压缩：从便宜到贵。

1. 大输出落盘（offload.py，总是执行）。
2. 微压缩（达到预算 60%）：较早的工具结果换成占位符，保留最近 3 次，不调用模型。
   会破坏 prompt cache，所以能腾出的空间够多（≥ 预算 10%）才批量做一次，不每步都做。
3. 摘要压缩（达到预算 85%，或 /compact [关注点]）：调用模型（关闭思考、不带工具）按固定模板
   总结较早的历史；用户历次原话原样保留，"不要改公共接口"这类约束不会被概括掉；
   附上当前任务清单和改过的文件。切分点优先落在两轮对话之间，当前这一轮本身就太长时
   退到两步之间——都不会留下孤立的 tool 消息。
4. 被动兜底：接口返回上下文超长时，主循环强制执行第 3 步再重试。

token 计数：上一次请求的 usage.prompt_tokens 加上之后新增消息的字符数 / 3，不引入分词器。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from coda.config import ContextConfig

Message = dict[str, Any]

CHARS_PER_TOKEN = 3
MIN_MICRO_GAIN = 0.10  # 微压缩至少腾出预算的 10% 才做
KEEP_TAIL_RATIO = 0.30  # 摘要压缩后保留的近期消息不超过预算的 30%
MANUAL_KEEP_RATIO = 0.10
CLEARED_PREFIX = "[已清理："
SUMMARY_MARK = "<!-- coda:summary -->"
MAX_INPUT_CHARS = 2000  # 单条用户原话过长时截断
MAX_INPUTS_CHARS = 20_000
TRANSCRIPT_TOOL_CHARS = 1200
MAX_TRANSCRIPT_CHARS = 240_000  # 给摘要模型的记录上限（约 80k token），超出时保留开头和结尾

SUMMARY_SYSTEM = """\
你负责为一个 Coding Agent 压缩对话历史。下面是它和用户较早的一段对话记录（含工具调用和结果）。
写一份结构化摘要，让 Agent 只看摘要就能无缝继续工作。用中文，按以下标题输出，没有内容的部分写"无"：

## 任务目标
用户要完成什么（概括所有轮次的需求）。
## 已完成的修改
按文件列出改了什么（路径 + 一句话）。
## 关键发现
代码结构、相关函数和位置（路径:行号）、测试命令、踩过的坑、已经排除的方向。
## 当前进度与下一步
做到了哪一步，接下来具体要做什么。
## 未解决的错误
还没修好的报错或失败测试，附关键错误信息。

要求：只写记录里有依据的内容，不编造；保留具体的路径、函数名、命令和错误信息；不超过 1200 字。
用户的原话会另外原样附上，不需要在摘要里复述。"""


def message_chars(m: Message) -> int:
    n = len(m.get("content") or "")
    n += len(m.get("reasoning_content") or "")
    for c in m.get("tool_calls") or []:
        n += len(c["function"]["name"]) + len(c["function"]["arguments"]) + 20
    return n + 10


def estimate_tokens_of(messages: list[Message]) -> int:
    return sum(message_chars(m) for m in messages) // CHARS_PER_TOKEN


def is_real_user(m: Message) -> bool:
    """用户真正输入的消息（不是系统提醒、闸门回填或摘要）。"""
    if m.get("role") != "user":
        return False
    content = m.get("content") or ""
    return not content.startswith("<system-reminder>") and SUMMARY_MARK not in content


@dataclass
class CompactResult:
    messages: list[Message]
    kind: str  # micro / summary
    before: int
    after: int
    detail: str


class ContextManager:
    def __init__(
        self,
        cfg: ContextConfig,
        budget: Callable[[], int],
        summarize: Callable[[list[Message]], str],
    ) -> None:
        """summarize：接收给摘要模型的消息，返回摘要文本（主循环里是 llm.complete）。"""
        self.cfg = cfg
        self.budget = budget
        self.summarize = summarize
        self._base_tokens: int | None = None  # 最近一次请求的 prompt_tokens
        self._base_count = 0  # 那次请求时的消息条数

    # ---- 计数 ----

    def observe(self, prompt_tokens: int, message_count: int) -> None:
        if prompt_tokens:
            self._base_tokens, self._base_count = prompt_tokens, message_count

    def reset(self) -> None:
        self._base_tokens, self._base_count = None, 0

    def estimate(self, messages: list[Message]) -> int:
        if self._base_tokens is None or self._base_count > len(messages):
            return estimate_tokens_of(messages)
        return self._base_tokens + estimate_tokens_of(messages[self._base_count :])

    # ---- 入口 ----

    def maybe_compact(
        self,
        messages: list[Message],
        *,
        user_inputs: list[str],
        extras: str = "",
        force: bool = False,
        focus: str = "",
        reason: str = "auto",
    ) -> CompactResult | None:
        """返回压缩后的结果；不需要压缩时返回 None。force=True 直接做摘要压缩。"""
        budget = self.budget()
        before = self.estimate(messages)
        if not force:
            if before < self.cfg.micro_ratio * budget:
                return None
            result = None
            if self.cfg.micro:
                result = self.micro(messages, before)
            if result is not None:
                if result.after < self.cfg.summary_ratio * budget:
                    return result
                messages = result.messages
                before = result.after
            elif before < self.cfg.summary_ratio * budget:
                return None
        keep = MANUAL_KEEP_RATIO if reason == "manual" else KEEP_TAIL_RATIO
        return self.summary(messages, before, user_inputs, extras, focus, keep * budget)

    # ---- 微压缩 ----

    @staticmethod
    def _call_index(messages: list[Message]) -> dict[str, tuple[str, str]]:
        index = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                index[c["id"]] = (c["function"]["name"], c["function"]["arguments"])
        return index

    def micro(self, messages: list[Message], before: int) -> CompactResult | None:
        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = tool_idx[: max(len(tool_idx) - self.cfg.keep_tool_results, 0)]
        calls = self._call_index(messages)
        out = list(messages)
        cleared, saved = 0, 0
        for i in old:
            m = messages[i]
            content = m.get("content") or ""
            if content.startswith(CLEARED_PREFIX) or len(content) < 300:
                continue
            name, args = calls.get(m.get("tool_call_id", ""), ("工具", ""))
            note = f"{CLEARED_PREFIX}{name} {_brief_args(args)} 的输出，共 {len(content):,} 字符"
            saved_path = re.search(r"完整内容已保存到 (\S+?)。", content)
            if saved_path:
                note += f"，完整内容在 {saved_path.group(1)}"
            note += "。需要时重新读取或执行]"
            out[i] = {**m, "content": note}
            cleared += 1
            saved += (len(content) - len(note)) // CHARS_PER_TOKEN
        if not cleared or saved < MIN_MICRO_GAIN * self.budget():
            return None
        after = max(before - saved, 0)
        return CompactResult(out, "micro", before, after, f"清理了 {cleared} 个较早的工具结果")

    # ---- 摘要压缩 ----

    @staticmethod
    def _cut_point(messages: list[Message], keep_tokens: float) -> int | None:
        """返回保留部分的起点下标；保留部分不超过 keep_tokens。"""
        n = len(messages)
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + message_chars(messages[i]) // CHARS_PER_TOKEN
        # 下标 1 之前是系统提示词；至少要摘要掉 1 条消息
        turn_starts = [i for i in range(2, n) if is_real_user(messages[i])]
        steps = [i for i in range(2, n) if messages[i].get("role") == "assistant"]
        for candidates in (turn_starts, steps):
            ok = [i for i in candidates if suffix[i] <= keep_tokens]
            if ok:
                return min(ok)
        return n  # 最后一条也太长：全部摘要掉

    def summary(
        self,
        messages: list[Message],
        before: int,
        user_inputs: list[str],
        extras: str,
        focus: str,
        keep_tokens: float,
    ) -> CompactResult | None:
        cut = self._cut_point(messages, keep_tokens)
        if cut is None or cut <= 1:
            return None
        head, tail = messages[1:cut], messages[cut:]
        prompt = SUMMARY_SYSTEM
        if focus:
            prompt += f"\n\n用户特别要求摘要关注：{focus}"
        text = self.summarize(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": render_transcript(head)},
            ]
        )
        tail_inputs = [m.get("content") or "" for m in tail if is_real_user(m)]
        earlier = [u for u in user_inputs if not any(t.startswith(u) for t in tail_inputs)]
        summary_msg = {
            "role": "user",
            "content": build_summary_message(text, earlier, extras),
        }
        # 摘要单独作为一条 user 消息，和后面用户的原话分开（两条 user 相邻，DeepSeek 和硅基流动实测可用），
        # 这样再次压缩时还能认出哪些是用户原话
        out = [messages[0], summary_msg, *tail]
        after = estimate_tokens_of(out)
        return CompactResult(out, "summary", before, after, f"摘要了 {len(head)} 条较早的消息")


def _brief_args(arguments: str, n: int = 80) -> str:
    try:
        data = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        return arguments[:n]
    if isinstance(data, dict):
        for key in ("command", "path", "pattern", "description"):
            if key in data:
                return str(data[key])[:n]
    return json.dumps(data, ensure_ascii=False)[:n]


def render_transcript(messages: list[Message]) -> str:
    """把较早的历史渲染成给摘要模型看的文字记录。工具结果截断，思考内容不放。"""
    lines = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            if SUMMARY_MARK in content:
                lines.append(f"[更早的摘要]\n{content}")
            else:
                lines.append(f"[用户]\n{content}")
        elif role == "assistant":
            if content:
                lines.append(f"[Agent]\n{content}")
            for c in m.get("tool_calls") or []:
                fn = c["function"]
                lines.append(f"[Agent 调用 {fn['name']}] {fn['arguments'][:600]}")
        elif role == "tool":
            if len(content) > TRANSCRIPT_TOOL_CHARS:
                half = TRANSCRIPT_TOOL_CHARS // 2
                content = f"{content[:half]}\n…（省略）…\n{content[-half:]}"
            lines.append(f"[工具结果]\n{content}")
    text = "\n\n".join(lines)
    if len(text) > MAX_TRANSCRIPT_CHARS:
        head = MAX_TRANSCRIPT_CHARS // 6
        text = (
            f"{text[:head]}\n\n…（记录过长，中间省略）…\n\n{text[-(MAX_TRANSCRIPT_CHARS - head) :]}"
        )
    return text


def build_summary_message(summary: str, user_inputs: list[str], extras: str) -> str:
    parts = [
        "<system-reminder>",
        SUMMARY_MARK,
        "为节省上下文，本会话较早的部分已压缩成下面的摘要。根据摘要和之后的消息继续工作；"
        "需要具体代码时重新读取文件，不要凭记忆修改。",
    ]
    if user_inputs:
        kept, total = [], 0
        for u in reversed(user_inputs):
            if len(u) > MAX_INPUT_CHARS:
                u = u[:MAX_INPUT_CHARS] + "…（截断）"
            total += len(u)
            if total > MAX_INPUTS_CHARS:
                break
            kept.append(u)
        kept.reverse()
        quoted = "\n".join(f"{i}. {u}" for i, u in enumerate(kept, 1))
        parts.append(f"\n## 用户历次原话（原样保留，其中的要求和约束仍然有效）\n{quoted}")
    parts.append(f"\n## 摘要\n{summary.strip()}")
    if extras:
        parts.append(extras)
    parts.append("</system-reminder>")
    return "\n".join(parts)
