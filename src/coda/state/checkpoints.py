"""文件检查点：/undo 撤销上一轮的修改，/diff 查看本会话累计改动。

每轮对话里，每个文件第一次被 edit_file / write_file 修改前保存一份原始内容
（新建的文件记为"原本不存在"）。/undo 恢复最近一轮的修改，可以连续撤销。
bash 命令造成的修改不在检查点里，无法撤销。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from coda.tools.files import diff_stats, make_diff


@dataclass
class _Turn:
    index: int
    files: dict[Path, bytes | None] = field(default_factory=dict)


class Checkpoints:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self._turns: list[_Turn] = []
        self._turn_index = 0
        self._session_original: dict[Path, bytes | None] = {}  # 会话里第一次修改前的内容
        self._lock = threading.Lock()

    def begin_turn(self) -> None:
        self._turn_index += 1

    @staticmethod
    def _read(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def snapshot(self, path: Path) -> None:
        """修改文件前调用；同一轮里只记第一次。"""
        with self._lock:
            if not self._turns or self._turns[-1].index != self._turn_index:
                self._turns.append(_Turn(self._turn_index))
            turn = self._turns[-1]
            if path in turn.files:
                return
            data = self._read(path)
            turn.files[path] = data
            self._session_original.setdefault(path, data)

    def turn_files(self) -> list[Path]:
        """本轮内容确实变了的文件。"""
        with self._lock:
            if not self._turns or self._turns[-1].index != self._turn_index:
                return []
            items = list(self._turns[-1].files.items())
        return [p for p, data in items if self._read(p) != data]

    @property
    def can_undo(self) -> bool:
        return bool(self._turns)

    def undo(self) -> list[Path]:
        """恢复最近一轮的修改，返回恢复的文件。"""
        with self._lock:
            if not self._turns:
                return []
            turn = self._turns.pop()
        restored = []
        for path, data in turn.files.items():
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.coda-undo")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            restored.append(path)
        return restored

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workdir))
        except ValueError:
            return str(path)

    def session_diffs(self) -> dict[str, str]:
        """本会话累计改动：相对路径 → unified diff（内容没变的文件不列出）。"""
        out = {}
        with self._lock:
            items = list(self._session_original.items())
        for path, original in items:
            current = self._read(path)
            if current == original:
                continue
            old = original.decode("utf-8", errors="replace") if original is not None else ""
            new = current.decode("utf-8", errors="replace") if current is not None else ""
            rel = self._rel(path)
            diff = make_diff(old, new, rel)
            if original is None:
                diff = diff.replace(f"--- a/{rel}", "--- /dev/null", 1)
            if current is None:
                diff = diff.replace(f"+++ b/{rel}", "+++ /dev/null", 1)
            out[rel] = diff
        return out

    def session_stats(self) -> dict[str, tuple[int, int]]:
        return {rel: diff_stats(d) for rel, d in self.session_diffs().items()}
