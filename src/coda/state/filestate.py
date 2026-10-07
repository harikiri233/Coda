"""先读后写：记录每个文件读取时的 mtime 和哈希。

- 没读过的已有文件不能编辑或覆盖（not_read）。
- 读过之后文件又被改了（用户手动改、Bash 改、格式化工具改），要求重新读（stale）。
  mtime 变了但内容哈希相同时视为没变，避免 touch 之类的误报。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "not_read", "stale"]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _Record:
    mtime_ns: int
    sha: str


class FileState:
    def __init__(self) -> None:
        self._records: dict[Path, _Record] = {}
        self._lock = threading.Lock()  # 只读工具会并行执行

    def record(self, path: Path, data: bytes | None = None) -> None:
        """读取或写入成功后调用。"""
        if data is None:
            data = path.read_bytes()
        rec = _Record(path.stat().st_mtime_ns, _digest(data))
        with self._lock:
            self._records[path] = rec

    def check(self, path: Path) -> Status:
        with self._lock:
            rec = self._records.get(path)
        if rec is None:
            return "not_read"
        try:
            if path.stat().st_mtime_ns == rec.mtime_ns:
                return "ok"
            return "ok" if _digest(path.read_bytes()) == rec.sha else "stale"
        except FileNotFoundError:
            return "stale"

    def forget(self, path: Path) -> None:
        with self._lock:
            self._records.pop(path, None)
