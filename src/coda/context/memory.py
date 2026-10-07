"""项目记忆：AGENTS.md。

启动时加载 ~/.coda/AGENTS.md（个人）和从 git 根目录到工作区各级的 AGENTS.md（项目），放进系统提示词。
选 AGENTS.md 是因为它是 Codex 等工具共用的约定文件名。
会话中途 /memory add 追加的约定写进工作区的 AGENTS.md，同时以系统提醒告诉模型——
系统提示词在会话内不变，保持 prompt cache 前缀稳定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coda.config import coda_home

FILENAME = "AGENTS.md"
MAX_CHARS = 20_000  # 单个文件放进提示词的上限


@dataclass
class MemoryFile:
    path: Path
    scope: str  # 个人 / 项目
    text: str


def _git_root(start: Path) -> Path | None:
    for p in (start, *start.parents):
        if (p / ".git").exists():
            return p
    return None


def memory_paths(workdir: Path) -> list[tuple[Path, str]]:
    workdir = workdir.resolve()
    paths = [(coda_home() / FILENAME, "个人")]
    top = _git_root(workdir) or workdir
    chain = [workdir, *[p for p in workdir.parents if p == top or top in p.parents]]
    for d in reversed(chain):
        paths.append((d / FILENAME, "项目"))
    return paths


def load_memory(workdir: Path) -> list[MemoryFile]:
    out = []
    for path, scope in memory_paths(workdir):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n…（{path} 过长，后面的内容已截断）"
        out.append(MemoryFile(path, scope, text))
    return out


def render_memory(files: list[MemoryFile]) -> str:
    if not files:
        return ""
    parts = [
        "# 项目记忆（AGENTS.md）",
        "以下是用户和项目的约定，优先级高于你的默认习惯；与用户当前的明确要求冲突时以用户为准。",
    ]
    for f in files:
        parts.append(f'\n<agents-md scope="{f.scope}" path="{f.path}">\n{f.text}\n</agents-md>')
    return "\n".join(parts)


def add_memory(workdir: Path, note: str) -> Path:
    """在工作区的 AGENTS.md 末尾追加一条约定。"""
    path = workdir / FILENAME
    note = note.strip().strip('"').strip("'").strip()
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    if old and not old.endswith("\n"):
        old += "\n"
    if "## 约定" not in old:
        old += ("\n" if old else "") + "## 约定\n\n"
    path.write_text(f"{old}- {note}\n", encoding="utf-8")
    return path
