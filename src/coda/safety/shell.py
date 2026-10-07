"""shell 命令分析：拆分复合命令，识别只读命令和危险命令。

- 拆分：按 ; && || | |& & 和换行拆成子命令，引号内的分隔符不拆；同时标记命令替换
  $(...) / 反引号 / <(...)，以及写文件的输出重定向（> /dev/null 和 2>&1 不算）。
- 每个子命令去掉不改变语义的前缀（VAR=1、env、nohup、time、timeout 10、xargs 等）再判断，
  `bash -c "..."` 递归分析里面的命令。
- 跟踪子命令里的 cd，`cd / && rm -rf *` 这样的组合也能识别出目标是根目录。

这里的判定只用来防误操作，不是沙箱：变量展开、别名、脚本文件里的内容都看不到。
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
INTERPRETERS = SHELLS | {"python", "python3", "perl", "ruby", "node"}

READONLY_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat", "which", "type", "echo",
    "printf", "tree", "du", "df", "date", "whoami", "id", "hostname", "uname", "uptime",
    "env", "printenv", "basename", "dirname", "realpath", "readlink", "grep", "egrep",
    "fgrep", "rg", "sort", "uniq", "cut", "tr", "diff", "cmp", "true", "false", "test",
    "[", "nproc", "free", "ps",
}  # fmt: skip
READONLY_GIT = {
    "status",
    "diff",
    "log",
    "show",
    "rev-parse",
    "ls-files",
    "blame",
    "shortlog",
    "describe",
}
FIND_WRITES = {
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-delete",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-fls",
}
VERSION_CMDS = {"python", "python3", "uv", "pip", "node", "ruff", "pytest", "git", "npm", "cargo"}

SENSITIVE = re.compile(
    r"(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa.*|id_ed25519.*|\.netrc|credentials.*)$"
)
FORK_BOMB = re.compile(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SAFE_DEVICES = {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"}


@dataclass
class Split:
    parts: list[str] = field(default_factory=list)
    ops: list[str | None] = field(default_factory=list)  # 每个子命令后面的分隔符
    substitution: bool = False
    redirect_targets: list[str] = field(default_factory=list)  # 写文件的重定向目标
    parse_error: bool = False


def split_command(command: str) -> Split:
    out = Split()
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)

    def flush(op: str | None) -> None:
        text = "".join(buf).strip()
        buf.clear()
        if text:
            out.parts.append(text)
            out.ops.append(op)
        elif out.ops and op is not None:
            out.ops[-1] = op

    while i < n:
        c = command[i]
        if quote == "'":
            buf.append(c)
            if c == "'":
                quote = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            if command[i + 1] != "\n":  # 行尾续行符直接去掉
                buf.append(command[i : i + 2])
            i += 2
            continue
        if quote == '"':
            if c == '"':
                quote = None
            elif c == "`" or command.startswith("$(", i):
                out.substitution = True
            buf.append(c)
            i += 1
            continue
        # 引号外
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "`" or command.startswith(("$(", "<(", ">("), i):
            out.substitution = True
            buf.append(c)
            i += 1
            continue
        if c == "#" and (not buf or buf[-1].isspace()):
            while i < n and command[i] != "\n":
                i += 1
            continue
        if c in ";\n":
            flush(c)
            i += 1
            continue
        if c == "&":
            if command.startswith("&&", i):
                flush("&&")
                i += 2
            elif command.startswith("&>", i) or (buf and buf[-1] in "<>"):
                buf.append(c)
                i += 1
            else:
                flush("&")
                i += 1
            continue
        if c == "|":
            if command.startswith("||", i):
                flush("||")
                i += 2
            else:
                flush("|")
                i += 2 if command.startswith("|&", i) else 1
            continue
        if c == ">":
            j = i + 1
            if j < n and command[j] in ">|":
                j += 1
            if j < n and command[j] == "&":  # >&2、2>&1：复制文件描述符
                k = j + 1
                while k < n and (command[k].isdigit() or command[k] == "-"):
                    k += 1
                if k > j + 1:
                    buf.append(command[i:k])
                    i = k
                    continue
                j += 1
            k = j
            while k < n and command[k] in " \t":
                k += 1
            m = k
            while m < n and command[m] not in " \t;&|<>\n":
                m += 1
            target = command[k:m].strip("'\"")
            if target not in _SAFE_DEVICES:
                out.redirect_targets.append(target)
            buf.append(command[i:m])
            i = m
            continue
        buf.append(c)
        i += 1
    if quote:
        out.parse_error = True
    flush(None)
    return out


def core_argv(argv: list[str]) -> list[str]:
    """去掉不改变语义的前缀，返回真正执行的命令。"""
    args = list(argv)
    while args:
        head = args[0]
        if _ASSIGN.match(head):
            args = args[1:]
        elif head == "env":
            args = args[1:]
            while args and (args[0].startswith("-") or _ASSIGN.match(args[0])):
                args = args[1:]
        elif head in ("nohup", "time", "nice", "command", "builtin", "exec", "stdbuf"):
            args = args[1:]
            while args and args[0].startswith("-"):
                args = args[2:] if args[0] == "-n" else args[1:]
        elif head == "timeout":
            args = args[1:]
            while args and args[0].startswith("-"):
                args = args[1:]
            args = args[1:]  # 时长
        elif head == "xargs":
            args = args[1:]
            while args and args[0].startswith("-"):
                takes_value = args[0] in ("-I", "-n", "-P", "-L", "-d", "-s", "-E", "-a")
                args = args[2:] if takes_value else args[1:]
        else:
            break
    return args


def _git_sub(argv: list[str]) -> tuple[str, list[str]]:
    args = argv[1:]
    while args and args[0].startswith("-"):
        args = args[2:] if args[0] in ("-C", "-c") else args[1:]
    return (args[0], args[1:]) if args else ("", [])


def is_readonly_argv(argv: list[str]) -> bool:
    if not argv:
        return False
    cmd = argv[0]
    if len(argv) == 2 and cmd in VERSION_CMDS and argv[1] in ("--version", "-V"):
        return True
    if cmd == "git":
        sub, rest = _git_sub(argv)
        if sub in READONLY_GIT:
            return not any(a.startswith("--output") for a in rest)
        if sub == "branch":
            return all(
                a in ("-a", "-r", "-v", "-vv", "--all", "--list", "--show-current") for a in rest
            )
        if sub == "remote":
            return all(a in ("-v", "--verbose") for a in rest)
        if sub == "stash":
            return bool(rest) and rest[0] in ("list", "show")
        return False
    if cmd == "find":
        return not any(a in FIND_WRITES for a in argv)
    return cmd in READONLY_COMMANDS


def _expand(target: str, cwd: Path | None, home: Path) -> Path | None:
    """把命令里的路径参数解析成绝对路径；含无法确定的变量时返回 None。"""
    t = target.replace("${HOME}", str(home)).replace("$HOME", str(home))
    if "$" in t:
        return None
    if t.startswith("~"):
        t = os.path.expanduser(t)
    # 通配符：取第一个通配符之前的目录（/* → /，src/*.py → src）
    m = re.search(r"[*?\[]", t)
    if m:
        t = t[: m.start()]
        t = t if t.endswith("/") or not t else os.path.dirname(t) or "."
        t = t or "."
    p = Path(t)
    if not p.is_absolute():
        if cwd is None:
            return None
        p = cwd / p
    return Path(os.path.realpath(p))


def _inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


@dataclass
class CommandCheck:
    parts: list[str]  # 规范化后的子命令文本，用于规则匹配
    readonly: list[bool]
    blocked: str | None = None  # 硬拦截原因
    high_risk: str | None = None  # 高危原因
    opaque: bool = False  # 含命令替换 / 写文件重定向 / 解析失败 / 嵌套 shell，不能靠规则自动放行
    touches_outside: bool = False  # 只读命令读取了工作区外或敏感路径


class _Analyzer:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.home = Path(os.path.realpath(Path.home()))
        self.cwd: Path | None = workdir
        self.check = CommandCheck([], [])

    def block(self, reason: str) -> None:
        self.check.blocked = self.check.blocked or reason

    def risk(self, reason: str) -> None:
        self.check.high_risk = self.check.high_risk or reason

    def run(self, command: str, depth: int = 0) -> None:
        if FORK_BOMB.search(command):
            self.block("fork 炸弹")
        sp = split_command(command)
        if sp.substitution or sp.parse_error or sp.redirect_targets:
            self.check.opaque = True
        for t in sp.redirect_targets:
            if t.startswith("/dev/"):
                self.block(f"写入设备文件 {t}")
            else:
                p = _expand(t, self.cwd, self.home)
                if p is None or not _inside(p, self.workdir):
                    self.risk(f"重定向写入工作区外的 {t}")
        for idx, part in enumerate(sp.parts):
            try:
                argv = shlex.split(part)
            except ValueError:
                self.check.opaque = True
                self.check.parts.append(" ".join(part.split()))
                self.check.readonly.append(False)
                continue
            core = core_argv(argv)
            if not core:
                continue
            if core[0] in SHELLS and "-c" in core and depth < 3:
                pos = core.index("-c")
                self.check.opaque = True
                if pos + 1 < len(core):
                    self.run(core[pos + 1], depth + 1)
                continue
            if core[0] == "eval":
                self.check.opaque = True
                self.run(" ".join(core[1:]), depth + 1)
                continue
            # curl ... | sh
            nxt_op = sp.ops[idx] if idx < len(sp.ops) else None
            if core[0] in ("curl", "wget") and nxt_op == "|" and idx + 1 < len(sp.parts):
                try:
                    nxt = core_argv(shlex.split(sp.parts[idx + 1]))
                except ValueError:
                    nxt = []
                if nxt and (nxt[0] in INTERPRETERS or nxt[0] == "sudo"):
                    self.block("把下载的内容直接交给解释器执行")
            self.check_danger(core)
            readonly = is_readonly_argv(core)
            if readonly and self.reads_outside(core):
                self.check.touches_outside = True
            self.check.parts.append(" ".join(core))
            self.check.readonly.append(readonly)
            if core[0] == "cd":
                self.cwd = _expand(core[1], self.cwd, self.home) if len(core) > 1 else self.home

    def reads_outside(self, core: list[str]) -> bool:
        for a in core[1:]:
            if a.startswith("-"):
                continue
            if SENSITIVE.search(a):
                return True
            if a.startswith(("/", "~", "..")):
                p = _expand(a, self.cwd, self.home)
                if p is None or not _inside(p, self.workdir):
                    return True
        return False

    def check_danger(self, core: list[str]) -> None:
        cmd = core[0]
        args = core[1:]
        if cmd in ("sudo", "su", "doas", "pkexec"):
            self.block(f"提权命令 {cmd}")
        elif cmd.startswith("mkfs") or cmd in ("fdisk", "parted", "wipefs", "sfdisk"):
            self.block(f"磁盘分区 / 格式化命令 {cmd}")
        elif cmd == "dd" and any(a.startswith("of=/dev/") for a in args):
            self.block("dd 写入设备")
        elif cmd in ("shutdown", "reboot", "halt", "poweroff"):
            self.block(f"关机 / 重启命令 {cmd}")
        elif cmd == "rm":
            self.check_rm(args)
        elif cmd == "git":
            sub, rest = _git_sub(core)
            if sub == "push":
                force = any(
                    a in ("-f", "--force", "--force-with-lease", "--mirror", "--delete")
                    or a.startswith("+")
                    for a in rest
                )
                self.risk("git push --force" if force else "git push 会修改远程仓库")
            elif sub == "reset" and "--hard" in rest:
                self.risk("git reset --hard 会丢弃未提交的修改")
            elif sub == "clean":
                self.risk("git clean 会删除未跟踪的文件")
            elif sub in ("checkout", "restore") and (
                "." in rest or "--" in rest and rest[-1] == "."
            ):
                self.risk(f"git {sub} . 会丢弃工作区修改")
        elif cmd in ("chmod", "chown", "chgrp") and any(a in ("-R", "--recursive") for a in args):
            self.risk(f"{cmd} -R 递归修改权限")
        elif cmd == "find" and "-delete" in args:
            self.risk("find -delete 会批量删除文件")

    def check_rm(self, args: list[str]) -> None:
        recursive = False
        targets: list[str] = []
        end_opts = False
        for a in args:
            if not end_opts and a == "--":
                end_opts = True
            elif not end_opts and a.startswith("--"):
                recursive = recursive or a == "--recursive"
            elif not end_opts and a.startswith("-") and len(a) > 1:
                recursive = recursive or "r" in a or "R" in a
            else:
                targets.append(a)
        for t in targets:
            p = _expand(t, self.cwd, self.home)
            if p is None:
                if recursive:
                    self.risk(f"无法确定 rm -r 的目标 {t}")
                continue
            if p == Path("/") or p == self.home or self.home.is_relative_to(p):
                self.block(f"删除 {p}")
            elif not _inside(p, self.workdir):
                if recursive:
                    self.block(f"rm -r 作用于工作区外的 {p}")
                else:
                    self.risk(f"删除工作区外的文件 {p}")
            elif p == self.workdir:
                self.risk("删除整个工作区的内容")
        if recursive:
            self.risk("rm -r 递归删除")


def analyze_command(command: str, workdir: Path) -> CommandCheck:
    a = _Analyzer(Path(os.path.realpath(workdir)))
    a.run(command)
    return a.check


def is_readonly_command(command: str, workdir: Path | None = None) -> bool:
    """整条命令都是只读的，且没有命令替换和写文件的重定向。"""
    chk = analyze_command(command, workdir or Path.cwd())
    return bool(chk.parts) and not chk.opaque and all(chk.readonly) and not chk.touches_outside


def suggest_prefix(part: str) -> str:
    """ "总是允许"时为子命令生成的前缀规则：git commit -m x → git commit*，uv run pytest -q → uv run pytest*。"""
    words = part.split()
    if not words:
        return part
    n = 1
    if words[0] in ("git", "npm", "pnpm", "yarn", "cargo", "docker", "pip", "make", "go", "poetry"):
        n = 2
    elif words[0] in ("uv", "python", "python3", "npx") and len(words) > 1:
        n = 3 if words[1] in ("run", "-m", "exec", "tool") else 2
    return " ".join(words[:n]) + "*"
