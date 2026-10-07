"""文件工具：read_file / edit_file / write_file。"""

from __future__ import annotations

import difflib
import os
from pathlib import Path

from pydantic import BaseModel, Field

from coda.tools.base import ErrorType, Tool, ToolContext, ToolResult
from coda.tools.walk import is_binary

MAX_LINES = 2000
MAX_LINE_CHARS = 2000


def _read_text(path: Path) -> tuple[str, bytes]:
    data = path.read_bytes()
    return data.decode("utf-8", errors="replace"), data


def make_diff(old: str, new: str, rel: str, context: int = 3) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        n=context,
    )
    out = []
    for line in lines:
        out.append(line if line.endswith("\n") else line + "\n")
    return "".join(out)


def diff_stats(diff: str) -> tuple[int, int]:
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _atomic_write(path: Path, text: str) -> bytes:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.coda-tmp")
    tmp.write_bytes(data)
    if path.exists():
        os.chmod(tmp, path.stat().st_mode)
    os.replace(tmp, path)
    return data


def _check_fresh(ctx: ToolContext, path: Path, rel: str) -> ToolResult | None:
    status = ctx.filestate.check(path)
    if status == "not_read":
        return ToolResult.error(
            ErrorType.NOT_READ,
            f"修改前需要先读取 {rel}。",
            "先调用 read_file 读取该文件，再修改。",
        )
    if status == "stale":
        return ToolResult.error(
            ErrorType.STALE,
            f"{rel} 在上次读取后被修改过（可能是用户、格式化工具或 bash 命令）。",
            "重新调用 read_file 读取最新内容后再修改。",
        )
    return None


# ---------------------------------------------------------------- read_file


class ReadParams(BaseModel):
    path: str = Field(description="文件路径，相对工作区或绝对路径")
    offset: int = Field(1, ge=1, description="从第几行开始读（1 起），默认 1")
    limit: int = Field(MAX_LINES, ge=1, le=MAX_LINES, description=f"最多读多少行，默认 {MAX_LINES}")


class ReadFileTool(Tool):
    name = "read_file"
    kind = "read"
    Params = ReadParams
    description = (
        "读取文本文件，输出带行号（格式：行号→内容）。默认从头读最多 2000 行；大文件用 offset/limit 分段读。"
        "修改文件之前必须先用它读过。查看文件内容用这个工具，不要用 bash 调 cat/head/tail。"
        "要找某个符号在哪，先用 grep，再按行号读附近的内容。"
    )

    def describe(self, p: ReadParams, ctx: ToolContext) -> str:
        return ctx.rel(ctx.resolve(p.path))

    def target_path(self, p: ReadParams, ctx: ToolContext) -> Path:
        return ctx.resolve(p.path)

    def run(self, p: ReadParams, ctx: ToolContext) -> ToolResult:
        path = ctx.resolve(p.path)
        rel = ctx.rel(path)
        if not path.exists():
            return ToolResult.error(
                ErrorType.NOT_FOUND, f"文件不存在：{rel}", "用 glob 查找正确的路径。"
            )
        if path.is_dir():
            return ToolResult.error(
                ErrorType.INVALID_ARGS,
                f"{rel} 是目录。",
                "用 glob 列出目录下的文件，例如 glob(pattern='*', path=该目录)。",
            )
        if is_binary(path):
            return ToolResult.error(ErrorType.INVALID_ARGS, f"{rel} 是二进制文件，无法按文本读取。")
        text, data = _read_text(path)
        ctx.filestate.record(path, data)
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult(True, f"（{rel} 是空文件）", display={"summary": "空文件"})
        start = p.offset - 1
        if start >= total:
            return ToolResult.error(
                ErrorType.INVALID_ARGS,
                f"offset={p.offset} 超出文件长度（共 {total} 行）。",
            )
        chunk = lines[start : start + p.limit]
        width = len(str(start + len(chunk)))
        out = []
        for i, line in enumerate(chunk, start + 1):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"…（本行截断，共 {len(line)} 字符）"
            out.append(f"{i:>{width}}→{line}")
        end = start + len(chunk)
        if end < total:
            out.append(
                f"\n（共 {total} 行，已显示 {p.offset}–{end} 行；用 offset={end + 1} 继续读）"
            )
        summary = f"{total} 行" if start == 0 and end == total else f"{p.offset}–{end} / {total} 行"
        return ToolResult(True, "\n".join(out), display={"summary": summary})


# ---------------------------------------------------------------- edit_file


class EditParams(BaseModel):
    path: str = Field(description="要修改的文件路径")
    old: str = Field(description="要替换的原文，必须和文件内容逐字符一致（含缩进），且在文件中唯一")
    new: str = Field(description="替换后的新内容")
    replace_all: bool = Field(False, description="为 true 时替换所有出现处（用于重命名等）")


def _closest_hint(text: str, old: str) -> str:
    """old 没找到时，给出空白差异或最相近的行，帮助模型修正。"""
    norm = " ".join(old.split())
    if norm and norm in " ".join(text.split()):
        return "原文按空白归一化后能找到，说明缩进、空格或换行不一致。按 read_file 的输出逐字符复制（不要带行号前缀）。"
    first = next((ln for ln in old.splitlines() if ln.strip()), "").strip()
    if not first:
        return "old 不能只包含空白。"
    lines = text.splitlines()
    stripped = [ln.strip() for ln in lines]
    best = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6)
    if best:
        lineno = stripped.index(best[0]) + 1
        return f"最相近的是第 {lineno} 行：{lines[lineno - 1].strip()[:200]!r}。重新 read_file 确认原文。"
    return "文件中没有相近内容，可能改错了文件；先用 grep 定位。"


class EditFileTool(Tool):
    name = "edit_file"
    kind = "edit"
    Params = EditParams
    description = (
        "通过精确字符串替换修改已有文件：把 old 替换成 new。必须先用 read_file 读过该文件。"
        "old 要逐字符复制原文（不含行号前缀），并包含足够的上下文使其在文件中唯一；"
        "要替换所有出现处时设 replace_all=true。新建文件用 write_file。不要用 bash 调 sed 改文件。"
    )

    def describe(self, p: EditParams, ctx: ToolContext) -> str:
        return ctx.rel(ctx.resolve(p.path))

    def target_path(self, p: EditParams, ctx: ToolContext) -> Path:
        return ctx.resolve(p.path)

    def _apply(self, p: EditParams, ctx: ToolContext) -> tuple[str, str] | ToolResult:
        path = ctx.resolve(p.path)
        rel = ctx.rel(path)
        if not path.is_file():
            return ToolResult.error(
                ErrorType.NOT_FOUND,
                f"文件不存在：{rel}",
                "新建文件用 write_file；路径不确定时用 glob 查找。",
            )
        if err := _check_fresh(ctx, path, rel):
            return err
        if p.old == p.new:
            return ToolResult.error(ErrorType.INVALID_ARGS, "old 和 new 相同，没有需要修改的内容。")
        if not p.old:
            return ToolResult.error(
                ErrorType.INVALID_ARGS, "old 不能为空。", "要整体重写文件用 write_file。"
            )
        text, _ = _read_text(path)
        count = text.count(p.old)
        if count == 0:
            return ToolResult.error(
                ErrorType.NO_MATCH, f"在 {rel} 中没有找到 old 原文。", _closest_hint(text, p.old)
            )
        if count > 1 and not p.replace_all:
            return ToolResult.error(
                ErrorType.AMBIGUOUS,
                f"old 在 {rel} 中出现了 {count} 次。",
                "在 old 里加入更多上下文使其唯一；或确实要全部替换时设 replace_all=true。",
            )
        new_text = text.replace(p.old, p.new) if p.replace_all else text.replace(p.old, p.new, 1)
        return text, new_text

    def preview(self, p: EditParams, ctx: ToolContext) -> str | None:
        res = self._apply(p, ctx)
        if isinstance(res, ToolResult):
            return None
        return make_diff(res[0], res[1], ctx.rel(ctx.resolve(p.path)))

    def run(self, p: EditParams, ctx: ToolContext) -> ToolResult:
        res = self._apply(p, ctx)
        if isinstance(res, ToolResult):
            return res
        old_text, new_text = res
        path = ctx.resolve(p.path)
        rel = ctx.rel(path)
        data = _atomic_write(path, new_text)
        ctx.filestate.record(path, data)
        diff = make_diff(old_text, new_text, rel)
        added, removed = diff_stats(diff)
        n = old_text.count(p.old) if p.replace_all else 1
        return ToolResult(
            True,
            f"已修改 {rel}（替换 {n} 处，+{added} -{removed}）。\n{diff}",
            display={
                "summary": f"+{added} -{removed}",
                "diff": diff,
                "added": added,
                "removed": removed,
            },
        )


# ---------------------------------------------------------------- write_file


class WriteParams(BaseModel):
    path: str = Field(description="文件路径；父目录不存在时自动创建")
    content: str = Field(description="完整的文件内容")


class WriteFileTool(Tool):
    name = "write_file"
    kind = "edit"
    Params = WriteParams
    description = (
        "写入完整文件内容：新建文件，或整体重写已有文件（覆盖已有文件前必须先 read_file）。"
        "只改局部时用 edit_file，它更安全、输出更小。不要用 bash 的 echo/cat 重定向写文件。"
    )

    def describe(self, p: WriteParams, ctx: ToolContext) -> str:
        return ctx.rel(ctx.resolve(p.path))

    def target_path(self, p: WriteParams, ctx: ToolContext) -> Path:
        return ctx.resolve(p.path)

    def _check(self, p: WriteParams, ctx: ToolContext) -> ToolResult | None:
        path = ctx.resolve(p.path)
        if path.is_dir():
            return ToolResult.error(ErrorType.INVALID_ARGS, f"{ctx.rel(path)} 是目录。")
        if path.exists():
            return _check_fresh(ctx, path, ctx.rel(path))
        return None

    def preview(self, p: WriteParams, ctx: ToolContext) -> str | None:
        if self._check(p, ctx):
            return None
        path = ctx.resolve(p.path)
        old = _read_text(path)[0] if path.exists() else ""
        return make_diff(old, p.content, ctx.rel(path))

    def run(self, p: WriteParams, ctx: ToolContext) -> ToolResult:
        if err := self._check(p, ctx):
            return err
        path = ctx.resolve(p.path)
        rel = ctx.rel(path)
        existed = path.exists()
        old = _read_text(path)[0] if existed else ""
        data = _atomic_write(path, p.content)
        ctx.filestate.record(path, data)
        diff = make_diff(old, p.content, rel)
        added, removed = diff_stats(diff)
        lines = p.content.count("\n") + (0 if p.content.endswith("\n") or not p.content else 1)
        verb = "已覆盖" if existed else "已新建"
        summary = f"+{added} -{removed}" if existed else f"新建 {lines} 行"
        return ToolResult(
            True,
            f"{verb} {rel}（{lines} 行）。",
            display={"summary": summary, "diff": diff, "added": added, "removed": removed},
        )
