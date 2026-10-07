"""权限引擎。

判定顺序：硬拦截 → deny 规则 → ask 规则 → 高危 → allow 规则 / 本会话规则 → 模式默认。

| 模式          | 读 / 搜索 | 编辑 / 写入      | bash                         |
|---------------|-----------|------------------|------------------------------|
| default       | 自动      | 询问（显示 diff）| 只读命令自动，其余询问       |
| accept-edits  | 自动      | 工作区内自动     | 同上                         |
| plan          | 自动      | 拒绝             | 只读命令自动，其余拒绝       |
| yolo          | 自动      | 自动             | 自动（硬拦截和高危除外）     |

MCP 工具（mcp__server__tool）：默认询问，allow 规则可放行（如 `mcp__fetch__fetch` 或 `mcp__fetch__*`）；
plan 模式下拒绝（无法确认外部工具有没有副作用），yolo 下自动。

规则写法 `工具名(模式)`：bash 的模式用通配符匹配命令文本，文件工具匹配工作区相对路径
（工作区外用绝对路径）；只写工具名表示匹配该工具的所有调用。

bash 命令先拆成子命令（safety/shell.py），每个子命令都满足 allow 规则或属于只读命令，
整条命令才自动放行。`pytest && rm -rf src` 不会因为匹配了 `bash(pytest*)` 就被放行；
含命令替换、写文件的重定向、嵌套 shell 的命令不走自动放行。
高危命令（git push、git reset --hard、rm -r 等）交互模式下即使 yolo 也询问，无头模式直接拒绝。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from coda.config import Mode, Permissions
from coda.safety.shell import SENSITIVE, analyze_command, suggest_prefix
from coda.tools.base import Tool, ToolContext
from coda.tools.walk import glob_match

Verdict = Literal["allow", "ask", "deny"]

MODES: tuple[Mode, ...] = ("default", "accept-edits", "plan", "yolo")

_RULE = re.compile(r"^\s*([\w.*-]+)\s*(?:\((.*)\))?\s*$", re.S)


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str | None  # None：匹配该工具的所有调用

    @classmethod
    def parse(cls, text: str) -> Rule:
        m = _RULE.match(text)
        if not m:
            raise ValueError(f"无法解析权限规则 {text!r}，格式应为 工具名(模式)")
        pattern = m.group(2)
        return cls(m.group(1), pattern.strip() if pattern is not None else None)

    def __str__(self) -> str:
        return self.tool if self.pattern is None else f"{self.tool}({self.pattern})"

    def match_command(self, command: str) -> bool:
        if self.pattern is None:
            return True
        return fnmatch.fnmatchcase(command, self.pattern)

    def match_path(self, rel: str) -> bool:
        if self.pattern is None:
            return True
        pat = self.pattern
        if pat.startswith("./"):
            pat = pat[2:]
        return glob_match(pat, rel) or fnmatch.fnmatchcase(rel, pat)


def parse_rules(texts: list[str]) -> list[Rule]:
    return [Rule.parse(t) for t in texts]


@dataclass
class PolicyResult:
    verdict: Verdict
    reason: str = ""
    danger: bool = False  # 硬拦截或高危：弹层里标红，且不提供"总是允许"
    always: list[str] = field(default_factory=list)  # 选"总是允许"时加入的会话规则


@dataclass
class PermissionPolicy:
    mode: Mode = "default"
    permissions: Permissions = field(default_factory=Permissions)
    interactive: bool = True  # 无头模式为 False：高危命令直接拒绝
    session_rules: list[Rule] = field(default_factory=list)  # 本会话"总是允许"加入的规则
    # 不需要询问就能读取的额外目录（会话目录：落盘的大输出）
    extra_read_roots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.deny = parse_rules(self.permissions.deny)
        self.ask = parse_rules(self.permissions.ask)
        self.allow = parse_rules(self.permissions.allow)

    def cycle_mode(self) -> Mode:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def remember(self, rules: list[str]) -> None:
        for text in rules:
            rule = Rule.parse(text)
            if rule not in self.session_rules:
                self.session_rules.append(rule)

    # ---- 判定 ----

    def check(self, tool: Tool, params: object, ctx: ToolContext) -> PolicyResult:
        if tool.kind == "bash":
            return self._check_bash(tool, tool.describe(params, ctx), ctx)
        if tool.kind == "mcp":
            return self._check_mcp(tool)
        path = tool.target_path(params, ctx)
        rel = ctx.rel(path) if path is not None else ""
        if tool.kind == "edit":
            return self._check_edit(tool, path, rel, ctx)
        return self._check_read(tool, path, rel, ctx)

    def _rules_for(self, rules: list[Rule], tool: Tool) -> list[Rule]:
        return [r for r in rules if r.tool == tool.name]

    def _path_hit(self, rules: list[Rule], tool: Tool, rel: str) -> Rule | None:
        return next((r for r in self._rules_for(rules, tool) if r.match_path(rel)), None)

    def _check_read(
        self, tool: Tool, path: Path | None, rel: str, ctx: ToolContext
    ) -> PolicyResult:
        if hit := self._path_hit(self.deny, tool, rel):
            return PolicyResult("deny", f"被规则 {hit} 禁止")
        if hit := self._path_hit(self.ask, tool, rel):
            return PolicyResult("ask", f"规则 {hit} 要求确认", always=[f"{tool.name}({rel})"])
        if self._path_hit(self.allow + self.session_rules, tool, rel):
            return PolicyResult("allow")
        if path is not None:
            roots = [ctx.workdir, *self.extra_read_roots]
            if not any(path == r or r in path.parents for r in roots):
                folder = path if path.is_dir() else path.parent
                return PolicyResult(
                    "ask", "读取工作区外的路径", always=[f"{tool.name}({folder}/**)"]
                )
            if SENSITIVE.search(rel):
                return PolicyResult("ask", "读取可能包含密钥的文件", always=[f"{tool.name}({rel})"])
        return PolicyResult("allow")

    def _check_edit(
        self, tool: Tool, path: Path | None, rel: str, ctx: ToolContext
    ) -> PolicyResult:
        if path is not None and not ctx.inside(path):
            return PolicyResult("deny", f"不允许写入工作区外的路径 {path}", danger=True)
        if hit := self._path_hit(self.deny, tool, rel):
            return PolicyResult("deny", f"被规则 {hit} 禁止")
        if self.mode == "plan":
            return PolicyResult(
                "deny", "当前是 plan 模式，只调查不修改。给出计划，等用户切换模式后再执行。"
            )
        always = ["edit_file", "write_file"]
        if hit := self._path_hit(self.ask, tool, rel):
            return PolicyResult("ask", f"规则 {hit} 要求确认", always=always)
        if self.mode in ("accept-edits", "yolo"):
            return PolicyResult("allow")
        if self._path_hit(self.allow + self.session_rules, tool, rel):
            return PolicyResult("allow")
        return PolicyResult("ask", "修改文件", always=always)

    def _check_mcp(self, tool: Tool) -> PolicyResult:
        def hit(rules: list[Rule]) -> Rule | None:
            return next((r for r in rules if fnmatch.fnmatchcase(tool.name, r.tool)), None)

        if rule := hit(self.deny):
            return PolicyResult("deny", f"被规则 {rule} 禁止")
        if hit(self.ask):
            return PolicyResult("ask", "规则要求确认", always=[tool.name])
        if hit(self.allow + self.session_rules) or self.mode == "yolo":
            return PolicyResult("allow")
        if self.mode == "plan":
            return PolicyResult("deny", "plan 模式下不调用外部 MCP 工具（无法确认是否有副作用）。")
        return PolicyResult("ask", "调用外部 MCP 工具", always=[tool.name])

    def _check_bash(self, tool: Tool, command: str, ctx: ToolContext) -> PolicyResult:
        chk = analyze_command(command, ctx.workdir)
        if chk.blocked:
            return PolicyResult(
                "deny", f"危险命令已被拦截（{chk.blocked}），任何模式下都不会执行。", danger=True
            )
        texts = [command, *chk.parts]
        deny = self._rules_for(self.deny, tool)
        if hit := next((r for r in deny for t in texts if r.match_command(t)), None):
            return PolicyResult("deny", f"被规则 {hit} 禁止")
        if chk.high_risk:
            if not self.interactive:
                return PolicyResult(
                    "deny", f"高危操作（{chk.high_risk}），无头模式下不执行。", danger=True
                )
            return PolicyResult("ask", f"高危操作：{chk.high_risk}", danger=True)
        ask = self._rules_for(self.ask, tool)
        if hit := next((r for r in ask for t in texts if r.match_command(t)), None):
            return PolicyResult("ask", f"规则 {hit} 要求确认")

        allow = self._rules_for(self.allow + self.session_rules, tool)
        if any(r.pattern is None for r in allow):
            return PolicyResult("allow")
        covered = [
            ro or any(r.match_command(p) for r in allow)
            for p, ro in zip(chk.parts, chk.readonly, strict=True)
        ]
        auto_ok = bool(chk.parts) and not chk.opaque and not chk.touches_outside
        if auto_ok and all(covered):
            return PolicyResult("allow")
        if self.mode == "yolo":
            return PolicyResult("allow")
        if self.mode == "plan":
            return PolicyResult("deny", "当前是 plan 模式，只允许只读命令。")

        reason = "执行 shell 命令"
        if chk.opaque:
            reason = "命令含命令替换、写文件的重定向或嵌套 shell，需要确认"
        elif chk.touches_outside:
            reason = "命令读取工作区外或可能包含密钥的路径"
        always: list[str] = []
        if not chk.opaque:
            always = sorted(
                {
                    f"bash({suggest_prefix(p)})"
                    for p, ok in zip(chk.parts, covered, strict=True)
                    if not ok
                }
            )
        return PolicyResult("ask", reason, always=always)
