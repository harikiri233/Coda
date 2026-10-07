"""@文件引用：用户消息里的 @路径 在发送时附上文件内容，模型不用再调用 read_file。

完整附上的文件记为"已读"，模型可以直接编辑；超长被截断的只附开头，仍要求先读。
"""

from __future__ import annotations

import re
from pathlib import Path

from coda.tools.walk import is_binary

MENTION = re.compile(r"(?<![\w@])@([\w./\-]+[\w/])")
MAX_FILE_CHARS = 40_000
MAX_TOTAL_CHARS = 120_000


def find_mentions(text: str, workdir: Path) -> list[Path]:
    out: list[Path] = []
    for m in MENTION.finditer(text):
        p = (workdir / m.group(1)).resolve()
        if p.is_file() and (p == workdir or workdir in p.parents) and p not in out:
            out.append(p)
    return out


def expand_mentions(text: str, workdir: Path) -> tuple[str, list[Path]]:
    """返回 (附上文件内容后的消息, 完整附上的文件)。没有引用时原样返回。"""
    paths = find_mentions(text, workdir)
    if not paths:
        return text, []
    blocks, full, total = [], [], 0
    for p in paths:
        rel = p.relative_to(workdir).as_posix()
        if is_binary(p):
            blocks.append(f'<file path="{rel}">（二进制文件，未附上）</file>')
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        room = min(MAX_FILE_CHARS, MAX_TOTAL_CHARS - total)
        if room <= 0:
            blocks.append(
                f'<file path="{rel}">（附件总长度已达上限，需要时用 read_file 读取）</file>'
            )
            continue
        if len(content) > room:
            content = content[:room] + "\n…（文件过长已截断，编辑前先用 read_file 读取）"
        else:
            full.append(p)
        total += len(content)
        numbered = "\n".join(f"{i}→{line}" for i, line in enumerate(content.splitlines(), 1))
        blocks.append(f'<file path="{rel}">\n{numbered}\n</file>')
    attached = "\n\n".join(blocks)
    return (
        f"{text}\n\n<system-reminder>\n用户用 @ 引用了以下文件，内容如下（行号→内容）：\n{attached}\n</system-reminder>",
        full,
    )
