"""搜索工具：glob / grep。纯 Python 实现，遵守 .gitignore。

结果为空时明确告诉模型"搜索了 N 个文件，0 处匹配"，和"执行失败"区分开（痛点 P1）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from coda.tools.base import ErrorType, Tool, ToolContext, ToolResult
from coda.tools.walk import glob_match, is_binary, walk_files

MAX_GLOB = 200
MAX_LINE = 300


def _search_root(path: str | None, ctx: ToolContext) -> Path | ToolResult:
    root = ctx.resolve(path) if path else ctx.workdir
    if not root.exists():
        return ToolResult.error(
            ErrorType.NOT_FOUND, f"路径不存在：{ctx.rel(root)}", "省略 path 从工作区根目录搜索。"
        )
    return root


def _rel_from(root: Path, p: Path) -> str:
    base = root if root.is_dir() else root.parent
    return p.relative_to(base).as_posix()


# ---------------------------------------------------------------- glob


class GlobParams(BaseModel):
    pattern: str = Field(description="glob 模式，如 **/*.py、src/**/test_*.py、*.{toml,cfg}")
    path: str | None = Field(None, description="搜索的目录，默认工作区根目录")


class GlobTool(Tool):
    name = "glob"
    kind = "read"
    Params = GlobParams
    description = (
        "按文件名模式查找文件，遵守 .gitignore，按修改时间倒序返回（最多 200 条）。"
        "不含 / 的模式（如 *.py）匹配任意目录下的文件名。按文件名找文件用这个工具，不要用 bash 调 find/ls。"
    )

    def describe(self, p: GlobParams, ctx: ToolContext) -> str:
        return f"{p.pattern}" + (f" {p.path}" if p.path else "")

    def target_path(self, p: GlobParams, ctx: ToolContext) -> Path:
        return ctx.resolve(p.path) if p.path else ctx.workdir

    def run(self, p: GlobParams, ctx: ToolContext) -> ToolResult:
        root = _search_root(p.path, ctx)
        if isinstance(root, ToolResult):
            return root
        matches: list[tuple[float, Path]] = []
        scanned = 0
        for f in walk_files(root):
            scanned += 1
            if glob_match(p.pattern, _rel_from(root, f)):
                try:
                    matches.append((f.stat().st_mtime, f))
                except OSError:
                    continue
        if not matches:
            return ToolResult(
                True,
                f"0 个文件匹配 {p.pattern!r}（共检查 {scanned} 个文件，已排除 .gitignore 忽略的文件）。",
                display={"summary": "0 个文件"},
            )
        matches.sort(key=lambda x: -x[0])
        lines = [ctx.rel(f) for _, f in matches[:MAX_GLOB]]
        if len(matches) > MAX_GLOB:
            lines.append(f"（共 {len(matches)} 个，只显示最近修改的 {MAX_GLOB} 个；请缩小范围）")
        return ToolResult(True, "\n".join(lines), display={"summary": f"{len(matches)} 个文件"})


# ---------------------------------------------------------------- grep


class GrepParams(BaseModel):
    pattern: str = Field(description="正则表达式（Python re 语法）")
    path: str | None = Field(None, description="搜索的文件或目录，默认工作区根目录")
    glob: str | None = Field(None, description="只搜匹配该 glob 的文件，如 *.py")
    mode: Literal["files", "content", "count"] = Field(
        "content", description="files 只列文件名；content 显示匹配行；count 每个文件的匹配数"
    )
    context: int = Field(0, ge=0, le=10, description="content 模式下每处匹配前后显示的行数")
    ignore_case: bool = Field(False, description="忽略大小写")
    limit: int = Field(100, ge=1, le=500, description="最多返回多少条结果")


class GrepTool(Tool):
    name = "grep"
    kind = "read"
    Params = GrepParams
    description = (
        "在文件内容中搜索正则表达式，遵守 .gitignore，跳过二进制文件。"
        "mode=content 输出 文件:行号:内容；mode=files 只列文件；mode=count 统计。"
        "搜索代码一律用这个工具，不要用 bash 调 grep/rg。没有匹配时会明确说明搜索了多少个文件。"
    )

    def describe(self, p: GrepParams, ctx: ToolContext) -> str:
        extra = " ".join(x for x in (p.path, f"--glob {p.glob}" if p.glob else None) if x)
        return f"{p.pattern}" + (f"  {extra}" if extra else "")

    def target_path(self, p: GrepParams, ctx: ToolContext) -> Path:
        return ctx.resolve(p.path) if p.path else ctx.workdir

    def run(self, p: GrepParams, ctx: ToolContext) -> ToolResult:
        try:
            rx = re.compile(p.pattern, re.IGNORECASE if p.ignore_case else 0)
        except re.error as e:
            return ToolResult.error(
                ErrorType.INVALID_REGEX,
                f"正则表达式不合法：{e}",
                "检查括号和转义；要按字面搜索特殊字符（如 ( . [ ）时用反斜杠转义。",
            )
        root = _search_root(p.path, ctx)
        if isinstance(root, ToolResult):
            return root

        out: list[str] = []
        files_matched = 0
        total_hits = 0
        scanned = 0
        truncated = False
        for f in walk_files(root):
            if p.glob and not glob_match(p.glob, _rel_from(root, f)):
                continue
            if ctx.cancel.is_set():
                break
            if is_binary(f):
                continue
            scanned += 1
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            hits = [i for i, line in enumerate(lines) if rx.search(line)]
            if not hits:
                continue
            files_matched += 1
            total_hits += len(hits)
            rel = ctx.rel(f)
            if truncated:
                continue
            if p.mode == "files":
                out.append(rel)
            elif p.mode == "count":
                out.append(f"{rel}:{len(hits)}")
            else:
                shown: set[int] = set()
                for i in hits:
                    lo, hi = max(0, i - p.context), min(len(lines), i + p.context + 1)
                    if p.context and out and lo not in shown and shown:
                        out.append("--")
                    for j in range(lo, hi):
                        if j in shown:
                            continue
                        shown.add(j)
                        sep = ":" if j == i or j in hits else "-"
                        text = lines[j]
                        if len(text) > MAX_LINE:
                            text = text[:MAX_LINE] + "…"
                        out.append(f"{rel}{sep}{j + 1}{sep}{text}")
            if len(out) >= p.limit:
                truncated = True

        if total_hits == 0:
            where = ctx.rel(root)
            scope = f"，glob={p.glob}" if p.glob else ""
            return ToolResult(
                True,
                f"0 处匹配（在 {where} 下共搜索 {scanned} 个文本文件{scope}）。这表示确实没有匹配，不是执行失败。",
                display={"summary": "0 处匹配"},
            )
        if len(out) > p.limit:
            out = out[: p.limit]
        summary = f"{total_hits} 处匹配（{files_matched} 个文件）"
        if truncated:
            out.append(
                f"（结果已截断：共 {summary}，只显示前 {p.limit} 条；请缩小范围或用 mode=files）"
            )
        return ToolResult(True, "\n".join(out), display={"summary": summary})
