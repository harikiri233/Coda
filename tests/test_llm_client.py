"""LLM 客户端测试：用假的流式分片模拟两家服务商的实测格式，不调用真实接口。"""

import threading
from types import SimpleNamespace as NS

from coda.config import ModelProfile, Price
from coda.llm import LLMClient


def _profile(provider="deepseek", thinking=True):
    return ModelProfile(
        provider=provider,
        base_url="http://x",
        api_key_env="K",
        model="m",
        thinking=thinking,
        price_per_m=Price(input=0.3, output=1.2, cache_hit=0.006),
    )


def _chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    choices = (
        []
        if usage is not None and content is None and tool_calls is None and finish is None
        else [NS(delta=delta, finish_reason=finish)]
    )
    return NS(choices=choices, usage=usage)


def _tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


class FakeOpenAI:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream


USAGE = NS(
    prompt_tokens=1000,
    completion_tokens=100,
    prompt_cache_hit_tokens=800,
    completion_tokens_details=NS(reasoning_tokens=20),
)


def test_deepseek_style_fragments_assemble_two_calls():
    # DeepSeek：首片带 id 和 name，参数逐字符到达；两个并行调用按 index 区分
    chunks = [
        _chunk(reasoning="想一想"),
        _chunk(tool_calls=[_tc(0, "call_a", "read_file", "")]),
        _chunk(tool_calls=[_tc(0, args='{"pa')]),
        _chunk(tool_calls=[_tc(0, args='th": "a.py"}')]),
        _chunk(tool_calls=[_tc(1, "call_b", "read_file", '{"path": "b.py"}')]),
        _chunk(finish="tool_calls"),
        _chunk(usage=USAGE),
    ]
    fake = FakeOpenAI(FakeStream(chunks))
    client = LLMClient(_profile(), client=fake)
    thoughts = []
    reply = client.stream(
        [{"role": "user", "content": "hi"}], tools=[{}], on_thinking=thoughts.append
    )
    assert [tc.name for tc in reply.tool_calls] == ["read_file", "read_file"]
    assert reply.tool_calls[0].parse_arguments() == {"path": "a.py"}
    assert reply.tool_calls[1].id == "call_b"
    assert reply.reasoning == "想一想" and thoughts == ["想一想"]
    assert reply.finish_reason == "tool_calls"
    # 费用：200 未命中 × 0.3 + 800 命中 × 0.006 + 100 输出 × 1.2
    assert abs(reply.usage.cost_usd - (200 * 0.3 + 800 * 0.006 + 100 * 1.2) / 1e6) < 1e-12
    assert client.tracker.last_input_tokens == 1000
    sent = fake.calls[0]
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
    assert sent["stream_options"] == {"include_usage": True}


def test_siliconflow_empty_name_in_later_fragments():
    # 硅基流动：后续分片的 name 是空字符串，不能覆盖已记录的名字（实测）
    chunks = [
        _chunk(tool_calls=[_tc(0, "x1", "read_file", "")]),
        _chunk(tool_calls=[_tc(0, None, "", '{"path": "README.md"')]),
        _chunk(tool_calls=[_tc(0, None, "", "}")]),
        _chunk(finish="tool_calls"),
    ]
    client = LLMClient(_profile("openai", thinking=False), client=FakeOpenAI(FakeStream(chunks)))
    reply = client.stream([{"role": "user", "content": "hi"}], tools=[{}])
    assert reply.tool_calls[0].name == "read_file"
    assert reply.tool_calls[0].parse_arguments() == {"path": "README.md"}


def test_text_stream_and_to_message():
    chunks = [_chunk(content="你"), _chunk(content="好"), _chunk(finish="stop")]
    client = LLMClient(_profile(thinking=False), client=FakeOpenAI(FakeStream(chunks)))
    got = []
    reply = client.stream([{"role": "user", "content": "hi"}], on_text=got.append)
    assert got == ["你", "好"]
    assert reply.to_message() == {"role": "assistant", "content": "你好"}


def test_cancel_closes_stream_and_drops_partial_tool_calls():
    cancel = threading.Event()
    chunks = [_chunk(content="部分"), _chunk(tool_calls=[_tc(0, "c", "bash", '{"comm')])]

    class CancelAfterFirst(FakeStream):
        def __iter__(self):
            yield self.chunks[0]
            cancel.set()
            yield self.chunks[1]

    stream = CancelAfterFirst(chunks)
    client = LLMClient(_profile(), client=FakeOpenAI(stream))
    reply = client.stream([{"role": "user", "content": "hi"}], tools=[{}], cancel=cancel)
    assert reply.cancelled and stream.closed
    assert reply.content == "部分"
    assert reply.tool_calls == []


def test_prepare_messages_reasoning_by_provider():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    ds = LLMClient(_profile(), client=object())
    out = ds.prepare_messages(history, thinking=True, with_tools=True)
    # DeepSeek 思考 + 工具：缺失的补空串（缺失会 400，空串可通过，实测）
    assert out[1]["reasoning_content"] == "" and out[2]["reasoning_content"] == "r"
    assert "reasoning_content" not in history[1]  # 不修改原历史
    # 关闭思考或其他服务商：去掉字段
    assert (
        "reasoning_content" not in ds.prepare_messages(history, thinking=False, with_tools=True)[2]
    )
    other = LLMClient(_profile("openai"), client=object())
    assert (
        "reasoning_content"
        not in other.prepare_messages(history, thinking=True, with_tools=True)[2]
    )


def test_openai_provider_never_sends_thinking_param():
    fake = FakeOpenAI(FakeStream([_chunk(content="x", finish="stop")]))
    LLMClient(_profile("openai", thinking=True), client=fake).stream(
        [{"role": "user", "content": "q"}]
    )
    assert "extra_body" not in fake.calls[0]


def test_empty_arguments_parse_as_empty_dict():
    from coda.llm import ToolCall

    assert ToolCall("1", "todo", "").parse_arguments() == {}
