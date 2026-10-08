"""完成闸门：模型准备结束一轮时，用测试结果这个客观信号判断是否真的完成。

  第一次修改文件前   → 运行一次验证命令，记录基线失败集合（verify.baseline=true 时）
  模型准备结束时     → 本轮改过文件、且改完后没有验证通过过，就再运行一次
  新增失败 = 当前 − 基线
    为空 → 结束，显示"✓ 验证通过"
    不空 → 失败用例名 + 截断后的错误信息作为 <system-reminder> 回填，继续循环
  回填 max_rounds 次后仍有新增失败 → 停下，如实显示"⚠ 验证未通过"

bash 命令（sed -i、python 脚本、git checkout 等）也会改文件：非只读命令执行前后各取一次工作区指纹
（路径 → mtime、大小），有变化就和 edit_file 一样标记为改过；第一条非只读命令执行前同样先跑基线。

有基线才能区分"Agent 引入的失败"和"仓库原本就有的失败"，否则闸门会一直卡住，
或者模型去"修"不相关的测试。验证命令含 pytest 时自动追加 --junitxml 解析失败用例；
其他命令只看退出码。
"""

from __future__ import annotations

import re
import shlex
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from coda.agent.events import EventSink, VerifyEnd, VerifyStart
from coda.config import VerifyConfig
from coda.safety.shell import split_command
from coda.tools.bash import run_shell
from coda.tools.walk import walk_files
from coda.verify.junit import parse_junit

MAX_FEEDBACK_FAILURES = 5
MAX_FEEDBACK_CHARS = 6000
_SUMMARY = re.compile(r"\b\d+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b")
NO_TESTS = 5  # pytest：没有收集到测试


def detect_verify_command(workdir: Path) -> str | None:
    """探测 Python 项目的测试命令：有 pytest 配置或 tests 目录时用 pytest。"""
    has_tests = (workdir / "tests").is_dir() or (workdir / "test").is_dir()
    pyproject = workdir / "pyproject.toml"
    configured = (workdir / "pytest.ini").is_file() or (
        pyproject.is_file()
        and "[tool.pytest" in pyproject.read_text(encoding="utf-8", errors="replace")
    )
    if not (has_tests or configured):
        return None
    if (workdir / "uv.lock").is_file() and shutil.which("uv"):
        return "uv run pytest -q"
    return "python -m pytest -q"


Fingerprint = dict[str, tuple[int, int]]


def fingerprint(root: Path) -> Fingerprint:
    """工作区文件指纹：相对路径 → (mtime_ns, 大小)。遵守 .gitignore，跳过 .venv 等目录。"""
    out: Fingerprint = {}
    for p in walk_files(root):
        try:
            st = p.stat()
        except OSError:
            continue
        out[p.relative_to(root).as_posix()] = (st.st_mtime_ns, st.st_size)
    return out


def changed_paths(before: Fingerprint, after: Fingerprint) -> list[str]:
    """新增、删除或内容（mtime / 大小）变化的文件。"""
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))


def _summary_line(output: str) -> str:
    for line in reversed(output.splitlines()):
        if _SUMMARY.search(line):
            return line.strip("= ").strip()
    return ""


def _tail(text: str, lines: int = 60) -> str:
    return "\n".join(text.rstrip().splitlines()[-lines:])


@dataclass
class VerifyResult:
    command: str
    state: str  # done / timeout / interrupted
    code: int | None
    failures: dict[str, str] | None  # None：没有可解析的用例信息，只看退出码
    summary: str
    output: str
    elapsed: float

    @property
    def passed(self) -> bool:
        return self.state == "done" and self.code in (0, NO_TESTS)

    def label(self) -> str:
        if self.state == "timeout":
            return "超时"
        if self.state == "interrupted":
            return "已中断"
        return self.summary or f"exit {self.code}"


@dataclass
class _TurnState:
    baseline: VerifyResult | None = None
    baseline_done: bool = False
    dirty: bool = False  # 改过文件且之后还没验证通过
    rounds: int = 0  # 已回填的次数
    tests_edited: set[str] = field(default_factory=set)
    edited: bool = False


class CompletionGate:
    def __init__(
        self, cfg: VerifyConfig, workdir: Path, sink: EventSink, cancel: threading.Event
    ) -> None:
        self.cfg = cfg
        self.workdir = workdir
        self.sink = sink
        self.cancel = cancel
        self.enabled = cfg.enabled
        self.command = cfg.command or detect_verify_command(workdir)
        self.detected = cfg.command is None and self.command is not None
        self.turn = _TurnState()
        self.last_status: str | None = None  # passed / failed / skipped / None（本轮没触发）

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.command)

    def begin_turn(self) -> None:
        self.turn = _TurnState()
        self.last_status = None

    # ---- 运行验证命令 ----

    def _command_with_junit(self, junit: Path) -> str:
        assert self.command
        sp = split_command(self.command)
        if "pytest" in self.command and len(sp.parts) == 1 and not sp.redirect_targets:
            return f"{self.command} --junitxml={shlex.quote(str(junit))}"
        return self.command

    def run(self) -> VerifyResult:
        assert self.command
        with tempfile.TemporaryDirectory(prefix="coda-verify-") as tmp:
            junit = Path(tmp) / "junit.xml"
            cmd = self._command_with_junit(junit)
            r = run_shell(cmd, self.workdir, timeout=self.cfg.timeout, cancel=self.cancel)
            failures = parse_junit(junit) if cmd != self.command else None
        output = (r.stdout + ("\n" + r.stderr if r.stderr.strip() else "")).strip()
        return VerifyResult(
            self.command, r.state, r.code, failures, _summary_line(output), output, r.elapsed
        )

    # ---- 与主循环 / 执行器的接口 ----

    def after_edit(self, rel_path: str) -> None:
        """编辑成功后调用：标记本轮改过文件，结束前需要验证。"""
        t = self.turn
        t.edited = True
        t.dirty = True
        name = rel_path.rsplit("/", 1)[-1]
        if (
            rel_path.startswith(("tests/", "test/"))
            or "/tests/" in rel_path
            or name.startswith("test_")
        ):
            t.tests_edited.add(rel_path)

    def before_edit(self, rel_path: str) -> None:
        """执行器在每次 edit_file / write_file 执行前调用。第一次修改前跑基线。"""
        t = self.turn
        if not self.active or t.baseline_done or not self.cfg.baseline:
            return
        t.baseline_done = True
        self.sink.emit(VerifyStart("baseline", self.command or ""))
        result = self.run()
        if result.state == "interrupted":
            self.sink.emit(
                VerifyEnd("baseline", result.command, False, "已中断", elapsed=result.elapsed)
            )
            return
        t.baseline = result
        count = (
            len(result.failures) if result.failures is not None else (0 if result.passed else -1)
        )
        self.sink.emit(
            VerifyEnd(
                "baseline",
                result.command,
                result.passed,
                result.label(),
                baseline_failures=count,
                elapsed=result.elapsed,
            )
        )

    def before_bash(self, readonly: bool) -> Fingerprint | None:
        """非只读 bash 命令执行前调用：第一次时跑基线，返回指纹供执行后比对。"""
        if not self.active or readonly:
            return None
        self.before_edit("")
        return fingerprint(self.workdir)

    def after_bash(self, before: Fingerprint | None) -> list[str]:
        """返回这条命令改动的文件，并像编辑工具一样标记。"""
        if before is None:
            return []
        changed = changed_paths(before, fingerprint(self.workdir))
        for rel in changed:
            self.after_edit(rel)
        return changed

    def note_bash(self, command: str, exit_code: int | None) -> None:
        """模型自己用 bash 完整跑过验证命令且通过，就不必再跑一次。"""
        same = bool(self.command) and command.split() == (self.command or "").split()
        if same and exit_code == 0 and self.turn.edited:
            self.turn.dirty = False

    def _new_failures(self, cur: VerifyResult) -> tuple[dict[str, str], str | None]:
        """返回 (新增失败, 警告)。"""
        base = self.turn.baseline
        if cur.state == "timeout":
            return {"（验证命令超时）": _tail(cur.output, 30)}, None
        base_set = base.failures if base is not None else {}
        if cur.failures is not None and base_set is not None:
            new = {k: v for k, v in cur.failures.items() if k not in base_set}
            if not new and not cur.passed and (base is None or base.passed):
                new = {f"（验证命令 exit {cur.code}）": _tail(cur.output)}
            return new, None
        if cur.passed:
            return {}, None
        if base is not None and not base.passed:
            return {}, f"验证命令在修改前就失败（exit {base.code}），无法判断是否引入了新问题。"
        return {f"（验证命令 exit {cur.code}）": _tail(cur.output)}, None

    def _feedback(self, cur: VerifyResult, new: dict[str, str]) -> str:
        lines = [
            "<system-reminder>",
            f"完成闸门：你修改代码后运行 `{cur.command}`，出现了 {len(new)} 个修改前没有的失败"
            f"（{cur.label()}）。在结束之前修复它们；如果确认失败与本次任务无关、或者无法修复，"
            "在回答里如实说明原因，不要声称已经完成。",
        ]
        budget = MAX_FEEDBACK_CHARS
        for name, msg in list(new.items())[:MAX_FEEDBACK_FAILURES]:
            block = f"\n## {name}\n{msg}"
            if len(block) > budget:
                block = block[:budget] + "\n…"
            lines.append(block)
            budget -= len(block)
            if budget <= 0:
                break
        if len(new) > MAX_FEEDBACK_FAILURES:
            lines.append(f"\n……另有 {len(new) - MAX_FEEDBACK_FAILURES} 个失败未列出。")
        lines.append(
            f"\n（第 {self.turn.rounds}/{self.cfg.max_rounds} 次回填）\n</system-reminder>"
        )
        return "\n".join(lines)

    def check(self) -> str | None:
        """模型准备结束时调用。返回要回填给模型的提醒；None 表示可以结束。"""
        t = self.turn
        if not self.active or not t.dirty:
            if t.edited and not self.active:
                self.last_status = "skipped"
            elif t.edited:
                self.last_status = "passed"  # 模型自己跑过验证命令且通过
            return None
        self.sink.emit(VerifyStart("final", self.command or ""))
        cur = self.run()
        if cur.state == "interrupted":
            self.sink.emit(VerifyEnd("final", cur.command, False, "已中断", elapsed=cur.elapsed))
            return None
        new, warning = self._new_failures(cur)
        tests = sorted(t.tests_edited)
        base_count = len(t.baseline.failures or {}) if t.baseline is not None else 0
        if not new:
            t.dirty = False
            self.last_status = "passed"
            self.sink.emit(
                VerifyEnd(
                    "final",
                    cur.command,
                    True,
                    cur.label(),
                    baseline_failures=base_count,
                    elapsed=cur.elapsed,
                    tests_edited=tests,
                    warning=warning,
                )
            )
            return None
        if t.rounds >= self.cfg.max_rounds:
            t.dirty = False
            self.last_status = "failed"
            self.sink.emit(
                VerifyEnd(
                    "final",
                    cur.command,
                    False,
                    cur.label(),
                    new_failures=list(new),
                    baseline_failures=base_count,
                    elapsed=cur.elapsed,
                    gave_up=True,
                    tests_edited=tests,
                )
            )
            return None
        t.rounds += 1
        self.sink.emit(
            VerifyEnd(
                "final",
                cur.command,
                False,
                cur.label(),
                new_failures=list(new),
                baseline_failures=base_count,
                elapsed=cur.elapsed,
                feedback=True,
                round=t.rounds,
                tests_edited=tests,
            )
        )
        return self._feedback(cur, new)
