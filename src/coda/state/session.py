"""会话持久化：每个会话一个 JSONL，同时就是 trace。

位置：~/.coda/projects/<工作区路径转写>/<session_id>.jsonl，落盘的大输出放在同名目录下。
每行一条记录，type 字段区分：
  meta        会话信息（工作区、模型、创建时间），第一次写入时自动补上
  message     追加一条消息（含 reasoning_content，DeepSeek 恢复会话后第一次请求要用）
  snapshot    消息列表被整体替换（压缩、修复）时写入完整快照；恢复时从最后一个快照开始重放
  turn_start / turn_end / tool / permission / verify / usage / todo_update / compact / model / notice
              运行过程，评测脚本从这里统计步数、工具错误率、token 和成本

消息只追加：Agent 每步之后调用 sync()，把新增的消息写盘；整体替换时调用 snapshot()。
文件在第一次写入时才创建，打开界面后什么都没做就退出不会留下空会话。API Key 不写入。
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coda.config import coda_home
from coda.llm.usage import Usage

Message = dict[str, Any]
INTERRUPTED_RESULT = "Error[interrupted]: 会话在这个工具执行完成前结束了，结果未知。"


def project_dir(workdir: Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(workdir.resolve()))
    return coda_home() / "projects" / slug


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


class Session:
    def __init__(self, path: Path, meta: dict[str, Any], *, exists: bool = False) -> None:
        self.path = path
        self.id = path.stem
        self.meta = meta
        self._started = exists  # 文件里已经有 meta
        self._written = 0  # 已写盘的消息条数
        self._lock = threading.Lock()

    @classmethod
    def create(cls, workdir: Path, model: str = "") -> Session:
        sid = _new_id()
        meta = {
            "session_id": sid,
            "workdir": str(workdir.resolve()),
            "model": model,
            "created": time.time(),
        }
        return cls(project_dir(workdir) / f"{sid}.jsonl", meta)

    @classmethod
    def open(cls, path: Path) -> Session:
        """继续写一个已有的会话文件。"""
        return cls(path, {"session_id": path.stem}, exists=True)

    @property
    def dir(self) -> Path:
        """会话目录：落盘的大输出等。对 read_file 开放读取。"""
        return self.path.with_suffix("")

    @property
    def outputs_dir(self) -> Path:
        return self.dir / "outputs"

    # ---- 写入 ----

    def _write(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                if not self._started:
                    f.write(_dumps({"type": "meta", "ts": time.time(), **self.meta}))
                    self._started = True
                for r in records:
                    f.write(_dumps(r))

    def record(self, record_type: str, /, **data: Any) -> None:
        # record_type 只能按位置传：data 里常有 kind 之类的字段（压缩类型、验证类型）
        self._write([{"type": record_type, "ts": time.time(), **data}])

    def sync(self, messages: list[Message]) -> None:
        """把 messages 里还没写盘的部分追加进去。"""
        new = messages[self._written :]
        if not new:
            return
        now = time.time()
        self._write([{"type": "message", "ts": now, "message": m} for m in new])
        self._written = len(messages)

    def snapshot(self, messages: list[Message], kind: str, **extra: Any) -> None:
        """消息列表被整体替换后写一份完整快照。"""
        self.record("snapshot", kind=kind, messages=messages, **extra)
        self._written = len(messages)

    def mark_synced(self, messages: list[Message]) -> None:
        self._written = len(messages)


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, default=str) + "\n"


# ---------------------------------------------------------------- 读取


@dataclass
class LoadedSession:
    path: Path
    meta: dict[str, Any]
    messages: list[Message]
    user_inputs: list[str] = field(default_factory=list)  # 用户历次原话（摘要压缩时原样保留）
    todos: list[dict[str, str]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    turns: int = 0
    repaired: bool = False  # 补过缺失的工具结果


def _records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 进程被杀时最后一行可能不完整
            if isinstance(rec, dict):
                yield rec


def repair_messages(messages: list[Message]) -> tuple[list[Message], bool]:
    """给没有结果的 tool_call 补一条 Error[interrupted]，否则下一次请求会报 400。"""
    out: list[Message] = []
    repaired = False
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        calls = m.get("tool_calls") if m.get("role") == "assistant" else None
        if not calls:
            continue
        seen = set()
        while i < len(messages) and messages[i].get("role") == "tool":
            seen.add(messages[i].get("tool_call_id"))
            out.append(messages[i])
            i += 1
        for c in calls:
            if c["id"] not in seen:
                out.append({"role": "tool", "tool_call_id": c["id"], "content": INTERRUPTED_RESULT})
                repaired = True
    return out, repaired


def load_session(path: Path) -> LoadedSession:
    meta: dict[str, Any] = {}
    messages: list[Message] = []
    s = LoadedSession(path, meta, messages)
    for rec in _records(path):
        kind = rec.get("type")
        if kind == "meta":
            meta.update({k: v for k, v in rec.items() if k not in ("type", "ts")})
        elif kind == "message":
            messages.append(rec["message"])
        elif kind == "snapshot":
            messages[:] = rec.get("messages", [])
        elif kind == "turn_start":
            s.user_inputs.append(rec.get("user_input", ""))
            s.turns += 1
        elif kind == "todo_update":
            s.todos = rec.get("todos", [])
        elif kind == "usage":
            total = rec.get("total") or {}
            s.usage = Usage(**{k: total.get(k, 0) for k in Usage.__dataclass_fields__})
            s.calls = rec.get("calls", s.calls)
        elif kind == "model":
            meta["model"] = rec.get("name", meta.get("model"))
    s.messages, s.repaired = repair_messages(messages)
    return s


@dataclass
class SessionInfo:
    path: Path
    id: str
    updated: float
    first_prompt: str
    turns: int
    cost: float
    model: str


def list_sessions(workdir: Path) -> list[SessionInfo]:
    """本项目的历史会话，最近更新的在前；没有任何一轮对话的会话不列出。"""
    folder = project_dir(workdir)
    if not folder.is_dir():
        return []
    out = []
    for path in folder.glob("*.jsonl"):
        first, turns, cost, model = "", 0, 0.0, ""
        for rec in _records(path):
            kind = rec.get("type")
            if kind == "turn_start":
                turns += 1
                first = first or rec.get("user_input", "")
            elif kind == "usage":
                cost = (rec.get("total") or {}).get("cost_usd", cost)
            elif kind in ("meta", "model"):
                model = rec.get("model") or rec.get("name") or model
        if turns:
            out.append(
                SessionInfo(path, path.stem, path.stat().st_mtime, first, turns, cost, model)
            )
    out.sort(key=lambda s: s.updated, reverse=True)
    return out


def latest_session(workdir: Path) -> Path | None:
    sessions = list_sessions(workdir)
    return sessions[0].path if sessions else None
