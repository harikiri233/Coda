"""输入框上方的补全下拉：输入 `/` 列出斜杠命令，输入 `@` 模糊搜索文件（遵守 .gitignore）。

焦点始终留在输入框：补全列表打开时，输入框把 ↑↓ / Tab / Enter / Esc 交给这里处理。
"""

from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from coda.tools.walk import walk_files

MAX_ITEMS = 8
MAX_FILES = 20_000
FILE_CACHE_SECONDS = 10.0


def fuzzy_score(query: str, candidate: str) -> int | None:
    """越小越好；None 表示不匹配。子串匹配优先（文件名里出现更好），其次按顺序出现的子序列。"""
    q, c = query.lower(), candidate.lower()
    if not q:
        return len(c)
    name = c.rsplit("/", 1)[-1]
    if q in name:
        return name.index(q) + len(c) // 10
    if q in c:
        return 100 + c.index(q)
    pos, gaps, last = 0, 0, -1
    for ch in q:
        found = c.find(ch, pos)
        if found < 0:
            return None
        if last >= 0:
            gaps += found - last - 1
        last, pos = found, found + 1
    return 1000 + gaps + len(c) // 10


class Completer(OptionList, can_focus=False):
    def __init__(self, workdir: Path, commands: list[tuple[str, str]]) -> None:
        super().__init__(id="completer")
        self.workdir = workdir
        self.commands = commands  # [(命令, 说明)]
        self.kind: str | None = None  # command / file
        self.token_start = 0  # 被补全的片段在输入里的起点（文件补全）
        self._files: list[str] = []
        self._files_at = 0.0

    @property
    def active(self) -> bool:
        return bool(self.display) and self.option_count > 0

    def _all_files(self) -> list[str]:
        now = time.monotonic()
        if now - self._files_at > FILE_CACHE_SECONDS:
            files = []
            for i, p in enumerate(walk_files(self.workdir)):
                if i >= MAX_FILES:
                    break
                files.append(p.relative_to(self.workdir).as_posix())
            self._files, self._files_at = files, now
        return self._files

    def hide(self) -> None:
        self.display = False
        self.kind = None
        self.clear_options()

    def _show(self, kind: str, options: list[Option]) -> None:
        self.clear_options()
        if not options:
            self.hide()
            return
        self.kind = kind
        self.add_options(options)
        self.highlighted = 0
        self.display = True

    def update_for(self, text: str, cursor: int) -> None:
        """根据输入框内容和光标位置（字符下标）决定显示什么。"""
        before = text[:cursor]
        if text.startswith("/") and " " not in text and "\n" not in text:
            q = text[1:].lower()
            matches = [(c, d) for c, d in self.commands if c[1:].startswith(q)]
            if len(matches) == 1 and matches[0][0] == text:
                self.hide()
                return
            self._show(
                "command",
                [
                    Option(Text.assemble((c, "bold"), "  ", (d, "dim")), id=c)
                    for c, d in matches[: MAX_ITEMS * 2]
                ],
            )
            return
        at = before.rfind("@")
        if at >= 0 and (at == 0 or before[at - 1].isspace()):
            query = before[at + 1 :]
            if not any(ch.isspace() for ch in query):
                scored = []
                for f in self._all_files():
                    s = fuzzy_score(query, f)
                    if s is not None:
                        scored.append((s, f))
                scored.sort()
                self.token_start = at
                self._show("file", [Option(f, id=f) for _, f in scored[:MAX_ITEMS]])
                return
        self.hide()

    def move(self, delta: int) -> None:
        if not self.option_count:
            return
        cur = self.highlighted or 0
        self.highlighted = (cur + delta) % self.option_count

    def selected(self) -> str | None:
        if self.highlighted is None or not self.option_count:
            return None
        return self.get_option_at_index(self.highlighted).id
