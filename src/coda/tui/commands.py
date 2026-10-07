"""斜杠命令。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.text import Text

from coda.config import update_project_settings
from coda.context.memory import add_memory
from coda.safety.policy import MODES
from coda.state.session import Session, list_sessions
from coda.verify.gate import detect_verify_command

if TYPE_CHECKING:
    from coda.tui.app import CodaApp

COMMANDS: list[tuple[str, str]] = [
    ("/help", "显示帮助"),
    ("/clear", "清空对话，开始新会话"),
    ("/compact", "摘要压缩上下文，可加关注点：/compact 保留接口讨论"),
    ("/cost", "本会话的 token、缓存命中率和花费"),
    ("/model", "切换模型：/model 打开选择器，或 /model <档案名>"),
    ("/thinking", "开关思考模式（仅 DeepSeek）：/thinking on | off"),
    ("/mode", "查看或切换权限模式：default / accept-edits / plan / yolo"),
    ("/resume", "从历史会话列表中选择一个继续"),
    ("/undo", "撤销上一轮的文件修改（可连续撤销）"),
    ("/diff", "查看本会话所有文件的累计改动"),
    ("/verify", "完成闸门：/verify on | off | <测试命令>"),
    ("/rules", "查看权限规则和本会话总是允许的规则"),
    ("/init", "调查仓库，生成 AGENTS.md 并探测测试命令"),
    ("/memory", '查看项目记忆，或追加一条约定：/memory add "不要改公共接口"'),
    ("/skills", "列出可用的 Skills"),
    ("/mcp", "查看 MCP Server 的连接状态和工具"),
    ("/copy", "复制最后一条回答到剪贴板"),
    ("/exit", "退出"),
]

KEYS = (
    "按键：Enter 发送 · Ctrl+J 换行 · ↑↓ 翻历史 · Esc 中断 · Shift+Tab 切模式 · Ctrl+B 侧栏 · "
    "Ctrl+O 展开/折叠全部\n输入 / 补全命令，输入 @ 引用文件（发送时附上文件内容）。"
)

INIT_PROMPT = """\
请调查这个仓库，在根目录生成（已存在则更新）AGENTS.md，供以后的编码会话阅读。要求：
- 先用 glob / grep / read_file 了解项目：README、pyproject.toml 等配置、目录结构、测试目录和 CI 配置。
- 内容简洁（建议 30–60 行），只写对修改代码有用的信息：项目用途一句话；构建、运行、测试、lint 的确切命令；
  目录结构和各模块职责；代码风格与约定（从现有代码和配置里归纳，不要编造）；容易踩的坑。
- 不要列出每个文件，不要写通用的编程建议。
- 已有 AGENTS.md 时保留其中人工写的约定，只补充和修正。
{verify}"""


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


async def handle_command(app: CodaApp, text: str) -> None:
    name, _, arg = text.partition(" ")
    arg = arg.strip()
    agent = app.agent
    if name == "/help":
        width = max(len(c) for c, _ in COMMANDS)
        lines = [f"{c.ljust(width)}  {d}" for c, d in COMMANDS]
        await app.notice("\n".join(lines) + "\n" + KEYS)
    elif name in ("/exit", "/quit"):
        if app.busy:
            agent.interrupt()
        app.exit()
    elif name == "/clear":
        if app.busy:
            await app.notice("运行中不能清空，先按 Esc 中断。", "warning")
            return
        agent.clear(new_session=Session.create(app.workdir, app.model_name))
        await app.chat.remove_children()
        app.sidebar.update_context(0)
        await app.notice("已清空对话，开始新会话。文件检查点保留，/undo 仍可撤销之前的修改。")
    elif name == "/compact":
        if app.busy:
            await app.notice("运行中不能压缩，先按 Esc 中断或等本轮结束。", "warning")
            return
        if len(agent.messages) <= 2:
            await app.notice("对话还很短，不需要压缩。")
            return
        app.start_compact(arg)
    elif name == "/cost":
        await app.notice(cost_text(app))
    elif name == "/model":
        await _model(app, arg)
    elif name == "/thinking":
        await _thinking(app, arg)
    elif name == "/mode":
        if not arg:
            await app.notice(f"当前模式：{agent.mode}（可选：{' / '.join(MODES)}）")
        elif arg in MODES:
            app.set_mode(arg)  # type: ignore[arg-type]
            await app.notice(f"已切换到 {arg} 模式。")
        else:
            await app.notice(f"未知模式 {arg!r}，可选：{' / '.join(MODES)}", "error")
    elif name == "/resume":
        if app.busy:
            await app.notice("运行中不能切换会话，先按 Esc 中断。", "warning")
            return
        app.pick_session()
    elif name == "/undo":
        if app.busy:
            await app.notice("运行中不能撤销，先按 Esc 中断。", "warning")
            return
        if not agent.checkpoints.can_undo:
            await app.notice("没有可以撤销的修改。", "warning")
            return
        restored = agent.undo()
        files = "\n".join(f"  {agent.ctx.rel(p)}" for p in restored)
        await app.notice(
            f"已撤销上一轮对 {len(restored)} 个文件的修改：\n{files}\n"
            "（bash 命令造成的修改不在撤销范围内）"
        )
    elif name == "/diff":
        from coda.tui.screens.diff_screen import DiffScreen

        diffs = agent.checkpoints.session_diffs()
        if not diffs:
            await app.notice("本会话还没有修改文件（bash 命令造成的修改不在统计范围内）。")
            return
        app.push_screen(DiffScreen(diffs))
    elif name == "/verify":
        await _verify(app, arg)
    elif name == "/rules":
        await _rules(app)
    elif name == "/init":
        await _init(app)
    elif name == "/memory":
        await _memory(app, arg)
    elif name == "/skills":
        if not agent.skills:
            await app.notice("没有可用的 Skill。在 .coda/skills/<名称>/SKILL.md 里添加。")
            return
        lines = [f"{s.name}（{s.source}）  {s.description}" for s in agent.skills.values()]
        lines.append("模型会按需调用 load_skill 加载正文。目录：.coda/skills/、~/.coda/skills/")
        await app.notice("\n".join(lines))
    elif name == "/mcp":
        await app.notice(app.mcp.status_text())
    elif name == "/copy":
        if app.last_answer:
            app.copy_to_clipboard(app.last_answer)
            await app.notice("已复制最后一条回答。")
        else:
            await app.notice("还没有可复制的回答。", "warning")
    else:
        await app.notice(f"未知命令 {name}，输入 /help 查看可用命令。", "warning")


def cost_text(app: CodaApp) -> str:
    tracker = app.llm.tracker
    u = tracker.total
    miss = u.input_tokens - u.cached_tokens
    lines = [
        f"模型调用 {tracker.calls} 次（含子智能体和摘要压缩） · 当前模型 {app.model_name}",
        f"输入 {u.input_tokens:,} token：缓存命中 {u.cached_tokens:,} · 未命中 {miss:,} · "
        f"命中率 {u.cache_hit_rate:.1%}",
        f"输出 {u.output_tokens:,} token（其中思考 {u.reasoning_tokens:,}）",
        f"花费 ${u.cost_usd:.4f}",
        f"上下文 {_k(app.agent.context_tokens())} / {_k(app.llm.profile.context_budget)}",
    ]
    if app.llm.profile.price_per_m is None:
        lines.append("（这个模型档案没有配置价格 price_per_m，花费按 0 计算）")
    return "\n".join(lines)


async def _model(app: CodaApp, arg: str) -> None:
    if app.busy:
        await app.notice("运行中不能切换模型，先按 Esc 中断或等本轮结束。", "warning")
        return
    if arg:
        await app.switch_model(arg)
        return
    from coda.tui.screens.picker import PickerScreen

    items = []
    for key, prof in app.settings.models.items():
        think = " · 思考" if prof.provider == "deepseek" and prof.thinking else ""
        label = Text.assemble(
            (key, "bold"),
            f"  {prof.model}",
            (f"  {prof.base_url} · {_k(prof.context_budget)} 预算{think}", "dim"),
        )
        items.append((key, label))

    async def chosen(key: str | None) -> None:
        if key and key != app.model_name:
            await app.switch_model(key)

    app.push_screen(PickerScreen("选择模型", items, current=app.model_name), chosen)


async def _thinking(app: CodaApp, arg: str) -> None:
    llm = app.llm
    if not llm.is_deepseek:
        await app.notice("思考模式开关只对 DeepSeek 生效。", "warning")
        return
    if arg not in ("on", "off"):
        state = "开" if llm.thinking_enabled() else "关"
        await app.notice(f"思考模式：{state}。用法：/thinking on | off")
        return
    llm.profile = llm.profile.model_copy(update={"thinking": arg == "on"})
    app.refresh_topbar()
    await app.notice(f"思考模式已{'开启' if arg == 'on' else '关闭'}（仅本会话）。")


async def _memory(app: CodaApp, arg: str) -> None:
    agent = app.agent
    sub, _, rest = arg.partition(" ")
    if sub == "add" and rest.strip():
        path = add_memory(app.workdir, rest)
        note = rest.strip().strip('"').strip("'")
        agent.note(f"用户新增了一条项目约定（已写入 {path.name}），之后的工作都要遵守：{note}")
        await app.notice(f"已写入 {path}，并告知 Agent。")
        return
    if sub:
        await app.notice('用法：/memory 查看 · /memory add "约定内容"', "warning")
        return
    if not agent.memory_files:
        await app.notice('没有加载 AGENTS.md。可以用 /init 生成，或 /memory add "约定" 追加一条。')
        return
    lines = [f"{f.scope}  {f.path}（{len(f.text.splitlines())} 行）" for f in agent.memory_files]
    lines.append("会话开始时加载进系统提示词；/memory add 追加的约定会立刻告知 Agent。")
    await app.notice("\n".join(lines))


def session_items(app: CodaApp) -> list[tuple[str, Text]]:
    items = []
    for s in list_sessions(app.workdir):
        when = time.strftime("%m-%d %H:%M", time.localtime(s.updated))
        first = " ".join(s.first_prompt.split())[:60]
        current = "  · 当前" if app.agent.session and s.path == app.agent.session.path else ""
        label = Text.assemble(
            (when, "bold"),
            f"  {first}",
            (f"  {s.turns} 轮 · ${s.cost:.4f} · {s.model}{current}", "dim"),
        )
        items.append((str(s.path), label))
    return items


async def _verify(app: CodaApp, arg: str) -> None:
    gate = app.agent.gate
    if arg in ("on", "off"):
        gate.enabled = arg == "on"
        if gate.enabled and not gate.command:
            await app.notice("已开启，但还没有验证命令：用 /verify <命令> 设置。", "warning")
        else:
            await app.notice(f"完成闸门已{'开启' if gate.enabled else '关闭'}（仅本会话）。")
    elif arg:
        gate.command = arg
        gate.detected = False
        gate.enabled = True
        path = update_project_settings(app.workdir, {"verify": {"command": arg}})
        await app.notice(f"验证命令设为 `{arg}`，已写入 {path}。")
    else:
        state = "开启" if gate.enabled else "关闭"
        cmd = gate.command or "（未设置）"
        src = "（自动探测）" if gate.detected else ""
        await app.notice(
            f"完成闸门：{state} · 命令 {cmd}{src} · 基线 {'开' if gate.cfg.baseline else '关'}"
            f" · 最多回填 {gate.cfg.max_rounds} 轮\n用法：/verify on | off | <命令>"
        )


async def _rules(app: CodaApp) -> None:
    p = app.agent.policy
    lines = [f"模式：{p.mode}"]
    for label, rules in (("deny", p.deny), ("ask", p.ask), ("allow", p.allow)):
        lines.append(f"{label}：{'、'.join(map(str, rules)) or '（无）'}")
    lines.append(f"本会话总是允许：{'、'.join(map(str, p.session_rules)) or '（无）'}")
    lines.append("规则写在 ~/.coda/settings.json 或 .coda/settings.json 的 permissions 里。")
    await app.notice("\n".join(lines))


async def _init(app: CodaApp) -> None:
    if app.busy:
        await app.notice("运行中不能执行 /init，先按 Esc 中断或等本轮结束。", "warning")
        return
    gate = app.agent.gate
    detected = detect_verify_command(app.workdir)
    verify = ""
    if not app.settings.verify.command and detected:
        path = update_project_settings(app.workdir, {"verify": {"command": detected}})
        gate.command = detected
        gate.detected = False
        await app.notice(f"探测到测试命令 `{detected}`，已写入 {path}（完成闸门会用它）。")
        verify = (
            f"- 测试命令暂定为 `{detected}`，确认它能正常运行；不对的话在回答里告诉我正确的命令。"
        )
    elif not detected and not app.settings.verify.command:
        verify = "- 没有探测到测试命令。如果项目有测试，在回答最后单独一行写出确切的测试命令。"
    app.submit(INIT_PROMPT.format(verify=verify).rstrip())
