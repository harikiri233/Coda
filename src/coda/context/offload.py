"""大输出落盘：工具结果超过阈值就写到会话目录，模型只看到头尾和文件路径，需要时用 read_file 分段读。

测试日志、长命令输出最受益。read_file 自己有 offset/limit 分页，不落盘；task 的结论已经限长，也不落盘。
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

HEAD_LINES = 40
TAIL_LINES = 40
EDGE_CHARS = 3000  # 头尾各自的字符上限（防止单行很长）
SKIP_TOOLS = {"read_file", "task"}


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80]


class Offloader:
    def __init__(self, folder: Path, threshold: int = 8000) -> None:
        self.folder = folder
        self.threshold = threshold
        self._lock = threading.Lock()
        self._n = 0

    def maybe(self, text: str, call_id: str, tool: str) -> tuple[str, Path | None]:
        """返回 (发给模型的文本, 落盘路径)；没超过阈值时原样返回。"""
        if tool in SKIP_TOOLS or len(text) <= self.threshold:
            return text, None
        with self._lock:
            self._n += 1
            n = self._n
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{n:03d}-{_safe(tool)}-{_safe(call_id)}.txt"
        path.write_text(text, encoding="utf-8")
        lines = text.splitlines()
        head = "\n".join(lines[:HEAD_LINES])[:EDGE_CHARS]
        tail = "\n".join(lines[-TAIL_LINES:])[-EDGE_CHARS:]
        omitted = max(len(lines) - HEAD_LINES - TAIL_LINES, 0)
        if omitted == 0:
            # 行数不多但单行很长：按字符取头尾
            head, tail = text[:EDGE_CHARS], text[-EDGE_CHARS:]
        note = (
            f"[输出共 {len(text):,} 字符、{len(lines)} 行，超过 {self.threshold:,} 字符，完整内容已保存到 {path}。"
            f"下面只显示开头和结尾；需要中间部分时用 read_file(path=该文件, offset=…, limit=…) 分段读，"
            f"或用 grep(pattern=…, path=该文件) 搜索。]"
        )
        body = f"{note}\n\n{head}\n\n…（中间省略 {omitted} 行）…\n\n{tail}"
        return body, path


def truncate_for_context(text: str, limit: int = 30_000) -> str:
    """关闭落盘时的兜底：只保留头尾，避免一次输出就撑满上下文。"""
    from coda.tools.base import truncate_middle

    return truncate_middle(text, limit)
