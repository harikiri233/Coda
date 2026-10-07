"""命令行入口。

coda                          打开 TUI
coda "任务"                    打开 TUI 并直接提交第一条任务
coda -c / coda --resume       继续最近一次会话 / 从列表选择会话
coda -p "任务" [--mode ...]    无头模式：不打开界面，单次执行
coda doctor                   自检

位置参数（任务）和子命令（doctor）在 click 里会冲突，所以先看第一个参数再分派到两个 Typer 应用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from coda import __version__
from coda.config import MODES_HELP, load_settings

app = typer.Typer(add_completion=False, help="Coda：终端 Coding Agent")
doctor_app = typer.Typer(add_completion=False)


@app.command()
def run(
    prompt: str | None = typer.Argument(None, help="第一条任务（可选）"),
    print_mode: str | None = typer.Option(
        None, "-p", "--print", help="无头模式：执行这条任务后退出，不打开界面"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="模型档案名"),
    mode: str | None = typer.Option(None, "--mode", help=MODES_HELP),
    output: str = typer.Option("text", "--output", help="无头模式输出格式：text 或 json"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="单轮最大步数"),
    cont: bool = typer.Option(False, "-c", "--continue", help="继续本项目最近一次会话"),
    resume: bool = typer.Option(False, "--resume", help="从本项目的历史会话列表中选择一个继续"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="不连接 MCP Server"),
    version: bool = typer.Option(False, "--version", help="显示版本"),
) -> None:
    """打开交互式界面；带 -p 时进入无头模式。

    注意：bash 工具没有沙箱，权限检查只用来防误操作，需要强隔离时请在容器里运行。
    """
    if version:
        typer.echo(f"coda {__version__}")
        raise typer.Exit()
    workdir = Path.cwd()
    try:
        settings = load_settings(workdir)
        if model:
            settings.profile(model)
    except (ValueError, KeyError) as e:
        typer.secho(f"配置错误：{e}", fg="red", err=True)
        raise typer.Exit(1) from None
    if mode is not None and mode not in ("default", "accept-edits", "plan", "yolo"):
        typer.secho(f"未知的权限模式 {mode!r}。{MODES_HELP}", fg="red", err=True)
        raise typer.Exit(1)
    if output not in ("text", "json"):
        typer.secho("--output 只能是 text 或 json", fg="red", err=True)
        raise typer.Exit(1)

    from coda.llm import LLMError
    from coda.state.session import latest_session

    resume_path = None
    if cont:
        resume_path = latest_session(workdir)
        if resume_path is None and print_mode is None:
            typer.secho("本项目还没有历史会话，开始新会话。", fg="yellow", err=True)
    if no_mcp:
        settings.mcpServers = {}

    if print_mode is not None:
        from coda.headless import run_headless

        try:
            code = run_headless(
                print_mode,
                settings,
                workdir,
                model=model,
                mode=mode,  # type: ignore[arg-type]
                output=output,  # type: ignore[arg-type]
                max_steps=max_steps,
                resume=resume_path,
            )
        except LLMError as e:
            typer.secho(str(e), fg="red", err=True)
            raise typer.Exit(1) from None
        raise typer.Exit(code)

    from coda.tui.app import CodaApp

    try:
        tui = CodaApp(
            settings,
            workdir,
            model=model,
            mode=mode,  # type: ignore[arg-type]
            first_prompt=prompt,
            max_steps=max_steps,
            resume=resume_path,
            pick_resume=resume,
        )
    except LLMError as e:
        typer.secho(str(e), fg="red", err=True)
        raise typer.Exit(1) from None
    tui.run()


@doctor_app.command()
def doctor() -> None:
    """检查配置与 API Key，并对每个模型做一次文本调用和工具调用。"""
    from coda.doctor import run_doctor

    ok = run_doctor(Path.cwd())
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    if sys.argv[1:2] == ["doctor"]:
        sys.argv.pop(1)
        doctor_app(prog_name="coda doctor")
    else:
        app(prog_name="coda")
