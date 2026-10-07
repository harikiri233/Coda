"""bash 工具：在工作区根目录执行命令。

- 自动 set -o pipefail，管道中间一步失败不会被吞掉（痛点 P1）。
- 分开返回退出码、stdout、stderr。
- 新建进程组运行；超时或 Esc 中断时杀掉整个进程组，连同子进程一起结束。
- stdin 接 /dev/null，交互式命令不会卡住。

run_shell 也给完成闸门和 Hooks 复用。
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from coda.tools.base import ErrorType, Tool, ToolContext, ToolResult, truncate_middle

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
MAX_OUTPUT = 400_000  # 只防内存失控；超过 8k 的结果由执行器落盘，模型只看到头尾


@dataclass
class ShellResult:
    state: Literal["done", "timeout", "interrupted"]
    code: int | None
    stdout: str
    stderr: str
    elapsed: float


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    except ProcessLookupError:
        pass


def run_shell(
    command: str,
    cwd: Path,
    *,
    timeout: float,
    cancel: threading.Event | None = None,
    stdin: bytes | None = None,
    env: dict[str, str] | None = None,
) -> ShellResult:
    script = f"set -o pipefail\n{command}"
    # 不写 .pyc：编辑很快、文件大小不变时，按 mtime（秒）+ 大小校验的字节码缓存可能没失效，跑到旧代码
    full_env = {
        **os.environ,
        "CODA": "1",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "PYTHONDONTWRITEBYTECODE": "1",
        **(env or {}),
    }
    # 输出写临时文件而不是管道，避免输出过多时管道写满导致子进程阻塞
    with (
        tempfile.TemporaryFile() as out,
        tempfile.TemporaryFile() as err,
        tempfile.TemporaryFile() as inp,
    ):
        if stdin:
            inp.write(stdin)
            inp.seek(0)
        start = time.monotonic()
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=cwd,
            stdin=inp if stdin else subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=full_env,
        )
        state: Literal["done", "timeout", "interrupted"] = "done"
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                state = "interrupted"
                break
            if time.monotonic() - start > timeout:
                state = "timeout"
                break
            time.sleep(0.05)
        if state != "done":
            _kill_group(proc)
        elapsed = time.monotonic() - start
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8", errors="replace")
        stderr = err.read().decode("utf-8", errors="replace")
    return ShellResult(state, proc.returncode, stdout, stderr, elapsed)


class BashParams(BaseModel):
    command: str = Field(description="要执行的 bash 命令")
    timeout: int = Field(
        DEFAULT_TIMEOUT,
        ge=1,
        le=MAX_TIMEOUT,
        description=f"超时秒数，默认 {DEFAULT_TIMEOUT}，最大 {MAX_TIMEOUT}",
    )
    description: str = Field("", description="用一句话说明这条命令做什么（显示给用户）")


class BashTool(Tool):
    name = "bash"
    kind = "bash"
    Params = BashParams
    description = (
        "在工作区根目录用 bash 执行命令（已开启 pipefail），返回退出码、stdout 和 stderr。"
        "用于运行测试、构建、git、包管理等。不要用它读文件（用 read_file）、搜索（用 grep/glob）"
        "或改文件（用 edit_file/write_file）。命令之间不保留 cd 和环境变量，需要时写在同一条命令里。"
        "不支持交互式命令（stdin 为空）；长时间运行的命令设置合适的 timeout。"
        "危险命令（sudo、rm -rf 工作区外、curl | sh 等）会被直接拦截。"
    )

    def describe(self, p: BashParams, ctx: ToolContext) -> str:
        return p.command

    def run(self, p: BashParams, ctx: ToolContext) -> ToolResult:
        r = run_shell(p.command, ctx.workdir, timeout=p.timeout, cancel=ctx.cancel)
        body = []
        if r.stdout.strip():
            body.append(f"<stdout>\n{truncate_middle(r.stdout.rstrip(), MAX_OUTPUT)}\n</stdout>")
        if r.stderr.strip():
            body.append(
                f"<stderr>\n{truncate_middle(r.stderr.rstrip(), MAX_OUTPUT // 2)}\n</stderr>"
            )
        output = "\n".join(body) or "（无输出）"
        display = {
            "output": (r.stdout + r.stderr).rstrip(),
            "exit_code": r.code,
            "elapsed": r.elapsed,
        }

        if r.state == "timeout":
            return ToolResult.error(
                ErrorType.TIMEOUT,
                f"命令超过 {p.timeout}s 未结束，已终止。已有输出：\n{output}",
                "确认命令不会等待输入；需要更久时调大 timeout，或缩小范围（如只跑相关测试）。",
                summary=f"超时 {p.timeout}s",
                **display,
            )
        if r.state == "interrupted":
            return ToolResult.error(
                ErrorType.INTERRUPTED,
                f"用户中断了命令。已有输出：\n{output}",
                summary="已中断",
                **display,
            )
        summary = f"exit {r.code} · {r.elapsed:.1f}s"
        text = f"exit_code: {r.code}\n{output}"
        # 退出码非 0 不算工具错误：测试失败等是正常的信号，交给模型判断
        return ToolResult(True, text, display={"summary": summary, **display})
