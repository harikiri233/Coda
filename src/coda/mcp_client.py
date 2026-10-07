"""MCP Client：把 settings.json 里 mcpServers 配置的 stdio Server 接成普通工具。

并发：主循环和工具是同步代码，MCP SDK 是异步的。这里起一个后台线程跑独立的 asyncio 事件循环，
每个 Server 一个长驻任务持有连接（anyio 的上下文必须在同一个任务里进入和退出）；
工具调用用 run_coroutine_threadsafe 从 worker 线程桥接过去，等待期间检查取消标志。

- 握手超时 15 秒，连不上就提示并跳过，不影响其他功能；单次调用超时 60 秒。
- 工具名改写成 mcp__{server}__{tool}，只保留 OpenAI 函数名允许的字符（[A-Za-z0-9_-]，最长 64）。
- JSON Schema 原样转发，参数也原样转发（不做本地校验，由 Server 校验）。
- 默认需要询问，可以用 allow 规则放行（见 safety/policy.py）。
- Server 的 stderr 写到 ~/.coda/logs/mcp-<name>.log；SDK 自己的日志写到 ~/.coda/logs/mcp-client.log。
  实测 mcp-server-fetch 首次运行会往 stdout 打 npm 的安装信息，SDK 解析失败时把 traceback 打到
  stderr，会弄花全屏界面，所以要把 mcp 这个 logger 接到文件上、不再向上传播。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coda.config import McpServerConfig, coda_home
from coda.tools.base import ErrorType, Tool, ToolContext, ToolRegistry, ToolResult

CONNECT_TIMEOUT = 15.0
CALL_TIMEOUT = 60.0
MAX_RESULT_CHARS = 100_000


def _route_sdk_logs() -> None:
    log = logging.getLogger("mcp")
    if any(getattr(h, "_coda", False) for h in log.handlers):
        return
    folder = coda_home() / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(folder / "mcp-client.log", encoding="utf-8")
    handler._coda = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    for name in ("mcp", "client"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.propagate = False


def tool_name(server: str, tool: str) -> str:
    raw = f"mcp__{server}__{tool}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:64]


@dataclass
class ServerState:
    name: str
    cfg: McpServerConfig
    status: str = "connecting"  # connecting / connected / failed / stopped
    error: str = ""
    tools: list[Any] = field(default_factory=list)  # mcp.types.Tool
    server_info: str = ""
    session: Any = None
    elapsed: float = 0.0


class McpTool(Tool):
    kind = "mcp"
    Params = None  # type: ignore[assignment]

    def __init__(self, manager: McpManager, server: str, spec: Any) -> None:
        self.manager = manager
        self.server = server
        self.remote_name = spec.name
        self.name = tool_name(server, spec.name)  # type: ignore[misc]
        desc = (spec.description or "").strip()
        self.description = f"[MCP {server}] {desc}"[:1024]  # type: ignore[misc]
        schema = dict(spec.input_schema or {})
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        self.input_schema = schema

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def validate(self, args: dict[str, Any]) -> Any:
        return args

    def describe(self, params: dict[str, Any], ctx: ToolContext) -> str:
        if not params:
            return ""
        first = next(iter(params.values()))
        text = first if isinstance(first, str) else json.dumps(params, ensure_ascii=False)
        return text[:120]

    def run(self, params: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.call(self.server, self.remote_name, params, ctx.cancel)


def _render_content(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource":
            res = block.resource
            parts.append(getattr(res, "text", None) or f"[资源 {getattr(res, 'uri', '')}]")
        else:
            parts.append(f"[{kind or '未知'} 内容，Coda 不支持显示]")
    structured = getattr(result, "structured_content", None)
    if not parts and structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, indent=2))
    return "\n".join(parts).strip()


class McpManager:
    def __init__(self, servers: dict[str, McpServerConfig], workdir: Path) -> None:
        self.workdir = workdir
        self.servers = {
            name: ServerState(name, cfg) for name, cfg in servers.items() if cfg.enabled
        }
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None
        self._ready: dict[str, threading.Event] = {n: threading.Event() for n in self.servers}

    # ---- 生命周期 ----

    def start(self, on_ready: Callable[[ServerState], None] | None = None) -> None:
        """在后台线程里连接所有 Server，不阻塞调用方。每个 Server 有结果（成功或失败）时回调 on_ready。"""
        if not self.servers or self._thread is not None:
            return
        _route_sdk_logs()
        started = threading.Event()

        def main() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._stop = asyncio.Event()
            started.set()
            tasks = [loop.create_task(self._serve(s, on_ready)) for s in self.servers.values()]
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()

        self._thread = threading.Thread(target=main, name="coda-mcp", daemon=True)
        self._thread.start()
        started.wait(5)

    def wait_ready(self, timeout: float = CONNECT_TIMEOUT + 2) -> None:
        deadline = time.monotonic() + timeout
        for ev in self._ready.values():
            ev.wait(max(deadline - time.monotonic(), 0))

    def stop(self) -> None:
        if self._loop is None or self._stop is None:
            return
        self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def _serve(self, st: ServerState, on_ready: Callable[[ServerState], None] | None) -> None:
        from mcp import ClientSession, StdioServerParameters, stdio_client

        assert self._stop is not None
        log_dir = coda_home() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        params = StdioServerParameters(
            command=st.cfg.command, args=st.cfg.args, env=st.cfg.env, cwd=self.workdir
        )
        reported = False

        def report() -> None:
            nonlocal reported
            if reported:
                return
            reported = True
            st.elapsed = time.monotonic() - start
            self._ready[st.name].set()
            if on_ready is not None:
                with contextlib.suppress(Exception):  # 回调出错不影响连接
                    on_ready(st)

        try:
            with (log_dir / f"mcp-{st.name}.log").open("a", encoding="utf-8") as errlog:  # noqa: SIM117
                async with (
                    stdio_client(params, errlog=errlog) as (read, write),
                    ClientSession(read, write) as session,
                ):
                    init = await asyncio.wait_for(session.initialize(), CONNECT_TIMEOUT)
                    listed = await asyncio.wait_for(session.list_tools(), CONNECT_TIMEOUT)
                    info = init.server_info
                    st.server_info = f"{info.name} {info.version}".strip() if info else ""
                    st.tools = list(listed.tools)
                    st.session = session
                    st.status = "connected"
                    report()
                    await self._stop.wait()
        except BaseException as e:  # noqa: BLE001  连接失败只提示，不影响 Coda
            if st.status != "connected":
                st.status = "failed"
                st.error = _describe_error(e)
            else:
                st.status = "stopped"
                st.error = _describe_error(e)
            report()
            if isinstance(e, KeyboardInterrupt | SystemExit):
                raise
        finally:
            st.session = None
            if st.status == "connected":
                st.status = "stopped"
            report()

    # ---- 工具 ----

    def tools(self) -> list[McpTool]:
        out = []
        for st in self.servers.values():
            if st.status == "connected":
                out += [McpTool(self, st.name, spec) for spec in st.tools]
        return out

    def register(self, registry: ToolRegistry, server: str | None = None) -> list[str]:
        names = []
        for tool in self.tools():
            if server is None or tool.server == server:
                registry.register(tool)
                names.append(tool.name)
        return names

    def call(
        self, server: str, tool: str, args: dict[str, Any], cancel: threading.Event
    ) -> ToolResult:
        st = self.servers.get(server)
        if st is None or st.session is None or self._loop is None:
            return ToolResult.error(
                ErrorType.TOOL_ERROR,
                f"MCP Server {server} 没有连接。",
                "换一种不依赖这个工具的做法。",
            )
        fut: Future = asyncio.run_coroutine_threadsafe(
            st.session.call_tool(tool, args, read_timeout_seconds=CALL_TIMEOUT), self._loop
        )
        start = time.monotonic()
        while True:
            try:
                result = fut.result(timeout=0.1)
                break
            except FutureTimeout:
                if cancel.is_set():
                    fut.cancel()
                    return ToolResult.error(
                        ErrorType.INTERRUPTED, "用户中断了 MCP 调用。", summary="已中断"
                    )
                if time.monotonic() - start > CALL_TIMEOUT + 5:
                    fut.cancel()
                    return ToolResult.error(
                        ErrorType.TIMEOUT,
                        f"MCP 调用超过 {CALL_TIMEOUT:.0f}s 没有返回。",
                        summary="超时",
                    )
            except Exception as e:
                return ToolResult.error(ErrorType.TOOL_ERROR, f"MCP 调用失败：{_describe_error(e)}")
        text = _render_content(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n…（结果过长，已截断）"
        elapsed = time.monotonic() - start
        if getattr(result, "is_error", False):
            return ToolResult.error(
                ErrorType.TOOL_ERROR,
                text or "MCP 工具返回了错误。",
                summary=f"出错 · {elapsed:.1f}s",
            )
        return ToolResult(
            True,
            text or "（工具没有返回内容）",
            display={"summary": f"{len(text):,} 字符 · {elapsed:.1f}s", "output": text},
        )

    def status_text(self) -> str:
        if not self.servers:
            return (
                "没有配置 MCP Server。在 settings.json 的 mcpServers 里添加，例如：\n"
                + json.dumps(
                    {"mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}},
                    ensure_ascii=False,
                )
            )
        lines = []
        for st in self.servers.values():
            cmd = " ".join([st.cfg.command, *st.cfg.args])
            if st.status == "connected":
                head = f"● {st.name}  已连接 {st.server_info}（{st.elapsed:.1f}s） · {cmd}"
                tools = [
                    f"    {tool_name(st.name, t.name)}  {(t.description or '').strip().splitlines()[0][:60] if t.description else ''}"
                    for t in st.tools
                ]
                lines += [head, *tools]
            elif st.status == "connecting":
                lines.append(f"○ {st.name}  连接中… · {cmd}")
            else:
                label = "连接失败" if st.status == "failed" else "已断开"
                lines.append(f"✗ {st.name}  {label}：{st.error} · {cmd}")
        lines.append(f"Server 的 stderr 日志在 {coda_home() / 'logs'}/mcp-<名称>.log")
        return "\n".join(lines)


def _describe_error(e: BaseException) -> str:
    if isinstance(e, BaseExceptionGroup):
        inner = e.exceptions[0] if e.exceptions else e
        return _describe_error(inner)
    if isinstance(e, TimeoutError | asyncio.TimeoutError):
        return f"握手超过 {CONNECT_TIMEOUT:.0f}s 没有完成"
    if isinstance(e, FileNotFoundError):
        return f"找不到命令（{e.filename or e}）"
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__
