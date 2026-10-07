"""遍历工作区文件：遵守各级 .gitignore，跳过常见的大目录。glob 和 grep 共用。"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import pathspec

# 即使没有 .gitignore 也跳过的目录
ALWAYS_SKIP = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}


def _load_spec(directory: Path) -> pathspec.PathSpec | None:
    gi = directory / ".gitignore"
    if not gi.is_file():
        return None
    try:
        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return pathspec.GitIgnoreSpec.from_lines(lines)


def _find_git_root(start: Path) -> Path | None:
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return None


def walk_files(root: Path, *, respect_gitignore: bool = True) -> Iterator[Path]:
    """深度优先产出 root 下的文件（root 是文件时只产出它自己）。"""
    if root.is_file():
        yield root
        return
    # 从 git 根目录（或 root 本身）到 root 之间各级的 .gitignore 也要生效
    specs: list[tuple[Path, pathspec.PathSpec]] = []
    if respect_gitignore:
        top = _find_git_root(root) or root
        chain = [root, *[p for p in root.parents if p == top or top in p.parents]]
        for d in reversed(chain[1:]):
            spec = _load_spec(d)
            if spec:
                specs.append((d, spec))

    def ignored(path: Path, is_dir: bool, active: list[tuple[Path, pathspec.PathSpec]]) -> bool:
        for base, spec in active:
            rel = path.relative_to(base).as_posix()
            if is_dir:
                rel += "/"
            if spec.match_file(rel):
                return True
        return False

    stack: list[tuple[Path, list[tuple[Path, pathspec.PathSpec]]]] = [(root, specs)]
    while stack:
        directory, active = stack.pop()
        if respect_gitignore:
            spec = _load_spec(directory)
            if spec:
                active = [*active, (directory, spec)]
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            continue
        subdirs = []
        for entry in entries:
            path = Path(entry.path)
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if entry.name in ALWAYS_SKIP or ignored(path, True, active):
                    continue
                subdirs.append(path)
            elif entry.is_file() and not ignored(path, False, active):
                yield path
        for d in reversed(subdirs):
            stack.append((d, active))


@lru_cache(maxsize=256)
def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """把 glob 转成正则：** 跨目录，* 和 ? 不跨 /，支持 {a,b} 和 [abc]。"""
    out = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                i += 2
                if i < n and pattern[i] == "/":
                    out.append("(?:.*/)?")
                    i += 1
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "{":
            end = pattern.find("}", i)
            if end == -1:
                out.append(re.escape(c))
            else:
                alts = pattern[i + 1 : end].split(",")
                out.append("(?:" + "|".join(glob_to_regex(a).pattern[:-2] for a in alts) + ")")
                i = end + 1
                continue
        elif c == "[":
            end = pattern.find("]", i + 1)
            if end == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = end + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def glob_match(pattern: str, rel_posix: str) -> bool:
    """不含 / 的模式（如 *.py）只匹配文件名，和 rg --glob 的习惯一致。"""
    if "/" not in pattern:
        return bool(glob_to_regex(pattern).match(rel_posix.rsplit("/", 1)[-1]))
    return bool(glob_to_regex(pattern).match(rel_posix))


def is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return False
