"""Agent 主循环（ReAct）：模型回复 → 执行工具 → 结果回填 → 再请求，直到模型不再调用工具。

同步代码，在 TUI 的 worker 线程或无头模式的主线程里运行；与外界只通过 EventSink / Approver 交互。

每一步开始前检查上下文：达到预算 60% 做微压缩、85% 做摘要压缩（context/compact.py）；
接口返回上下文超长时强制摘要压缩再重试一次。模型准备结束时经过完成闸门：改过代码就跑测试，
新增失败回填给模型继续修。消息和运行过程逐条写入会话 JSONL（state/session.py），可以随时恢复。
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from coda.agent import reminders
from coda.agent.events import (
    Approver,
    AssistantEnd,
    Compacted,
    Event,
    EventSink,
    FilesChanged,
    Notice,
    PermissionLog,
    StepStart,
    TextDelta,
    ThinkingDelta,
    TodoUpdate,
    ToolEnd,
    TurnEnd,
    TurnStart,
    UsageUpdate,
    VerifyEnd,
)
from coda.agent.executor import ToolExecutor
from coda.agent.prompt import build_system_prompt
from coda.agent.subagent import SubagentRunner, TaskTool
from coda.config import ContextConfig, Hooks, Mode, Permissions, VerifyConfig
from coda.context.compact import CompactResult, ContextManager
from coda.context.memory import MemoryFile, load_memory, render_memory
from coda.context.mentions import expand_mentions
from coda.context.offload import Offloader
from coda.context.skills import Skill, discover_skills, render_catalog
from coda.llm import ContextTooLongError, LLMClient, LLMError, Reply
from coda.safety.hooks import HookRunner
from coda.safety.policy import PermissionPolicy
from coda.state.checkpoints import Checkpoints
from coda.state.filestate import FileState
from coda.state.session import LoadedSession, Session
from coda.tools import ToolRegistry, builtin_tools
from coda.tools.base import ToolContext
from coda.tools.skill import LoadSkillTool
from coda.tools.todo import format_todos
from coda.verify.gate import CompletionGate

Message = dict[str, Any]

INTERRUPTED_NOTE = "[用户中断了这次回复]"
SUMMARY_MAX_TOKENS = 4000


class RecordingSink:
    """把运行过程写进会话 JSONL，再转给真正的 sink（界面 / 无头输出）。"""

    def __init__(self, inner: EventSink, agent: Agent) -> None:
        self.inner = inner
        self.agent = agent

    def emit(self, ev: Event) -> None:
        session = self.agent.session
        if session is not None:
            with contextlib.suppress(OSError):  # 写盘失败不影响运行
                self._record(session, ev)
        self.inner.emit(ev)

    def _record(self, s: Session, ev: Event) -> None:
        if isinstance(ev, TurnStart):
            s.record("turn_start", user_input=ev.user_input)
        elif isinstance(ev, TurnEnd):
            s.record(
                "turn_end",
                status=ev.status,
                steps=ev.steps,
                error=ev.error,
                verify=ev.verify,
                final_text=ev.final_text,
            )
        elif isinstance(ev, ToolEnd):
            s.record(
                "tool",
                call_id=ev.call_id,
                name=ev.name,
                desc=ev.desc[:300],
                ok=ev.ok,
                error_type=ev.error_type,
                elapsed=round(ev.elapsed, 3),
                offloaded=ev.display.get("offloaded"),
            )
        elif isinstance(ev, PermissionLog):
            s.record(
                "permission",
                tool=ev.tool,
                desc=ev.desc[:300],
                decision=ev.decision,
                reason=ev.reason,
            )
        elif isinstance(ev, VerifyEnd):
            s.record(
                "verify",
                kind=ev.kind,
                ok=ev.ok,
                summary=ev.summary,
                new_failures=ev.new_failures,
                baseline_failures=ev.baseline_failures,
                round=ev.round,
                gave_up=ev.gave_up,
                elapsed=round(ev.elapsed, 2),
            )
        elif isinstance(ev, UsageUpdate):
            s.record(
                "usage",
                last=ev.last.to_dict(),
                total=ev.total.to_dict(),
                calls=self.agent.llm.tracker.calls,
                context_tokens=ev.context_tokens,
            )
        elif isinstance(ev, TodoUpdate):
            s.record("todo_update", todos=ev.todos)
        elif isinstance(ev, Notice) and ev.level != "info":
            s.record("notice", level=ev.level, text=ev.text)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        workdir: Path,
        sink: EventSink,
        approver: Approver,
        *,
        mode: Mode = "default",
        max_steps: int = 60,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        permissions: Permissions | None = None,
        hooks: Hooks | None = None,
        verify: VerifyConfig | None = None,
        interactive: bool = True,
        context: ContextConfig | None = None,
        session: Session | None = None,
        subagents: bool = True,
        skills: dict[str, Skill] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        self.llm = llm
        self.workdir = workdir.resolve()
        self.session = session
        self.sink: EventSink = RecordingSink(sink, self)
        self.max_steps = max_steps
        self.cancel = threading.Event()
        self.ctx = ToolContext(self.workdir, FileState(), self.cancel)
        self.policy = PermissionPolicy(mode, permissions or Permissions(), interactive)
        self.checkpoints = Checkpoints(self.workdir)
        self.gate = CompletionGate(verify or VerifyConfig(), self.workdir, self.sink, self.cancel)
        self.context_cfg = context or ContextConfig()
        self.offloader = Offloader(self._outputs_dir(), self.context_cfg.offload_chars)
        self.policy.extra_read_roots = [self.offloader.folder]
        self.context = ContextManager(
            self.context_cfg, lambda: self.llm.profile.context_budget, self._summarize
        )

        self.skills = discover_skills(self.workdir) if skills is None else skills
        self.memory_files: list[MemoryFile] = load_memory(self.workdir)
        self.subagents = SubagentRunner(
            llm,
            self.workdir,
            self.sink,
            self.cancel,
            offloader=self.offloader,
            extra_read_roots=self.policy.extra_read_roots,
        )
        if tools is None:
            tools = builtin_tools()
            if subagents:
                tools.register(TaskTool(self.subagents))
            if self.skills:
                tools.register(LoadSkillTool(self.skills))
        if enabled_tools is not None:
            # 评测 E1（只给 bash）用；MCP 工具在之后注册，不受影响
            tools = ToolRegistry([tools.get(n) for n in tools.names() if n in enabled_tools])
        self.tools = tools
        self.executor = ToolExecutor(
            self.tools,
            self.ctx,
            self.policy,
            self.sink,
            approver,
            hooks=HookRunner(hooks or Hooks(), self.workdir),
            checkpoints=self.checkpoints,
            gate=self.gate,
            offloader=self.offloader if self.context_cfg.offload else None,
        )
        if system_prompt is None:
            system_prompt = build_system_prompt(
                self.workdir,
                memory=render_memory(self.memory_files),
                skills=render_catalog(self.skills),
            )
            if enabled_tools is not None:
                names = "、".join(self.tools.names()) or "（无）"
                system_prompt += (
                    f"\n# 本次可用的工具\n只有：{names}。上文提到的其他工具在本次会话中不可用，"
                    "读文件、搜索和修改都用可用的工具完成（如 bash 里的 cat -n、grep -rn、sed -i 或 python 脚本）。\n"
                )
        self.messages: list[Message] = [{"role": "system", "content": system_prompt}]
        self.user_inputs: list[str] = []  # 用户历次原话，摘要压缩时原样保留
        self.running = False

    def _outputs_dir(self) -> Path:
        if self.session is not None:
            return self.session.outputs_dir
        # 没有会话（测试、评测里不落会话）时放临时目录，第一次落盘时才创建
        return Path(tempfile.gettempdir()) / f"coda-outputs-{os.getpid()}-{id(self):x}"

    # ---- 外部控制 ----

    def interrupt(self) -> None:
        """Esc：设置取消标志。流式阶段在分片之间检查，工具阶段杀掉子进程组。"""
        self.cancel.set()

    @property
    def mode(self) -> Mode:
        return self.policy.mode

    def set_mode(self, mode: Mode) -> None:
        self.policy.mode = mode

    def set_llm(self, llm: LLMClient, name: str = "") -> None:
        """切换模型（/model）。历史消息不变，reasoning_content 由客户端按服务商处理。"""
        self.llm = llm
        self.subagents.llm = llm
        self.context.reset()
        if self.session is not None:
            self.session.record("model", name=name or llm.profile.model, model=llm.profile.model)

    def _sync(self) -> None:
        if self.session is not None:
            with contextlib.suppress(OSError):
                self.session.sync(self.messages)

    def _replace_messages(self, messages: list[Message], kind: str) -> None:
        self.messages = messages
        if self.session is not None:
            self.session.snapshot(messages, kind)

    def clear(self, new_session: Session | None = None) -> None:
        """开始新的对话。传入 new_session 时之后写入新的会话文件。"""
        self.messages = self.messages[:1]
        self.user_inputs = []
        self.ctx.filestate = FileState()
        self.ctx.todos = []
        self.context.reset()
        if new_session is not None:
            self.session = new_session
            self._move_outputs()
        self.sink.emit(TodoUpdate([]))

    def _move_outputs(self) -> None:
        self.offloader.folder = self._outputs_dir()
        self.policy.extra_read_roots[:] = [self.offloader.folder]

    def load(self, loaded: LoadedSession, session: Session | None = None) -> None:
        """恢复会话：重放消息（含 reasoning_content）、用户原话、任务清单和累计用量。

        系统提示词沿用会话里保存的那份，前缀不变，恢复后第一次请求就能命中缓存。
        文件读取记录不恢复：恢复后修改文件前要重新读取。
        """
        if loaded.messages and loaded.messages[0].get("role") == "system":
            self.messages = list(loaded.messages)
        else:
            self.messages = [self.messages[0], *loaded.messages]
        self.user_inputs = list(loaded.user_inputs)
        self.ctx.filestate = FileState()
        self.ctx.todos = list(loaded.todos)
        self.context.reset()
        tracker = self.llm.tracker
        tracker.total = loaded.usage
        tracker.calls = loaded.calls
        self.session = session
        if session is not None:
            self._move_outputs()
            if loaded.repaired:
                session.snapshot(self.messages, "repair")
            else:
                session.mark_synced(self.messages)
        self.sink.emit(TodoUpdate(list(self.ctx.todos)))

    def undo(self) -> list[Path]:
        """撤销最近一轮的文件修改。恢复的文件要求重新读取后才能编辑。"""
        restored = self.checkpoints.undo()
        for p in restored:
            self.ctx.filestate.forget(p)
        if restored:
            self.messages.append(
                {
                    "role": "user",
                    "content": reminders.wrap(
                        "用户用 /undo 撤销了上一轮对以下文件的修改，它们已恢复到修改前的内容：\n"
                        + "\n".join(f"- {self.ctx.rel(p)}" for p in restored)
                        + "\n之后如需修改，先重新读取。"
                    ),
                }
            )
            self._sync()
            self.sink.emit(FilesChanged(self.checkpoints.session_stats()))
        return restored

    def note(self, text: str) -> None:
        """会话中途的信息（如 /memory add）以系统提醒追加，不改系统提示词。"""
        self.messages.append({"role": "user", "content": reminders.wrap(text)})
        self._sync()

    # ---- 上下文 ----

    def context_tokens(self) -> int:
        return self.context.estimate(self.messages)

    def _summarize(self, messages: list[Message]) -> str:
        return self.llm.complete(messages, max_tokens=SUMMARY_MAX_TOKENS).content

    def _extras(self) -> str:
        parts = []
        if self.ctx.todos:
            parts.append(f"\n## 当前任务清单\n{format_todos(self.ctx.todos)}")
        changed = self.checkpoints.session_stats()
        if changed:
            files = "\n".join(f"- {p}（+{a} -{r}）" for p, (a, r) in changed.items())
            parts.append(f"\n## 本会话修改过的文件\n{files}")
        return "\n".join(parts)

    def compact(self, focus: str = "", *, reason: str = "manual") -> CompactResult | None:
        """摘要压缩（/compact、上下文超长兜底）。在 worker 线程里调用（要请求模型）。"""
        if len(self.messages) <= 2:
            return None
        result = self.context.maybe_compact(
            self.messages,
            user_inputs=self.user_inputs,
            extras=self._extras(),
            force=True,
            focus=focus,
            reason=reason,
        )
        self._apply_compaction(result, reason)
        return result

    def _apply_compaction(self, result: CompactResult | None, reason: str) -> None:
        if result is None:
            return
        self._replace_messages(result.messages, f"compact-{result.kind}")
        self.context.reset()
        if self.session is not None:
            self.session.record(
                "compact",
                kind=result.kind,
                reason=reason,
                before=result.before,
                after=result.after,
                detail=result.detail,
            )
        self.sink.emit(
            Compacted(result.kind, result.before, result.after, result.detail, reason)  # type: ignore[arg-type]
        )

    def _maybe_compact(self) -> None:
        try:
            result = self.context.maybe_compact(
                self.messages, user_inputs=self.user_inputs, extras=self._extras()
            )
        except LLMError as e:
            self.sink.emit(Notice("warning", f"摘要压缩失败，本次跳过：{e}"))
            return
        self._apply_compaction(result, "auto")

    # ---- 主循环 ----

    def _call_model(self) -> Reply:
        count = len(self.messages)
        reply = self.llm.stream(
            self.messages,
            tools=self.tools.schemas(),
            on_text=lambda t: self.sink.emit(TextDelta(t)),
            on_thinking=lambda t: self.sink.emit(ThinkingDelta(t)),
            cancel=self.cancel,
        )
        self.context.observe(reply.usage.input_tokens, count)
        return reply

    def _emit_usage(self, reply: Reply) -> None:
        tracker = self.llm.tracker
        self.sink.emit(
            UsageUpdate(
                reply.usage,
                tracker.total,
                tracker.last_input_tokens,
                self.llm.profile.context_budget,
            )
        )

    def run_turn(self, user_input: str) -> TurnEnd:
        self.cancel.clear()
        self.running = True
        try:
            end = self._run(user_input)
        except LLMError as e:
            end = TurnEnd("error", 0, error=str(e))
        except Exception as e:  # 兜底：界面不能因为主循环异常而崩溃
            end = TurnEnd("error", 0, error=f"{type(e).__name__}: {e}")
        finally:
            self.running = False
            self._sync()
        end.verify = self.gate.last_status
        self.sink.emit(end)
        return end

    def _user_message(self, user_input: str) -> Message:
        content, attached = expand_mentions(user_input, self.workdir)
        for p in attached:
            self.ctx.filestate.record(p)  # 完整附上的文件视为已读
        if self.mode == "plan":
            content = f"{content}\n\n{reminders.PLAN_MODE}"
        return {"role": "user", "content": content}

    def _run(self, user_input: str) -> TurnEnd:
        self.sink.emit(TurnStart(user_input))
        self.checkpoints.begin_turn()
        self.gate.begin_turn()
        self.user_inputs.append(user_input)
        self.messages.append(self._user_message(user_input))
        self._sync()
        final_text = ""
        since_todo = 0
        for step in range(1, self.max_steps + 1):
            self.sink.emit(StepStart(step))
            self._maybe_compact()
            try:
                reply = self._call_model()
            except ContextTooLongError:
                # 被动兜底：强制摘要压缩后重试一次
                self.sink.emit(Notice("warning", "上下文超出模型窗口，先压缩再重试。"))
                if self.compact(reason="overflow") is None:
                    raise
                reply = self._call_model()

            self.sink.emit(
                AssistantEnd(
                    reply.content,
                    reply.reasoning,
                    len(reply.tool_calls),
                    reply.finish_reason,
                    reply.cancelled,
                )
            )
            self._emit_usage(reply)

            if reply.cancelled:
                # 未完成的 tool_calls 已在客户端丢弃；保留已输出的文字，便于用户接着说
                if reply.content or reply.reasoning:
                    msg = reply.to_message()
                    msg["content"] = (reply.content + "\n\n" + INTERRUPTED_NOTE).strip()
                    self.messages.append(msg)
                return TurnEnd("interrupted", step, reply.content)

            self.messages.append(reply.to_message())
            self._sync()
            if reply.content:
                final_text = reply.content

            if not reply.tool_calls:
                if reply.finish_reason == "length":
                    self.sink.emit(Notice("warning", "回复达到输出长度上限被截断。"))
                feedback = self.gate.check()
                if self.cancel.is_set():
                    return TurnEnd("interrupted", step, final_text)
                if feedback is None:
                    return TurnEnd("done", step, final_text)
                self.messages.append({"role": "user", "content": feedback})
                continue

            if reply.finish_reason == "length":
                self.sink.emit(Notice("warning", "回复被长度上限截断，工具参数可能不完整。"))
            results = self.executor.run(reply.tool_calls)
            if any(tc.name == "todo_write" for tc in reply.tool_calls):
                since_todo = 0
            else:
                since_todo += 1
            note = reminders.todo_stale(self.ctx.todos, since_todo)
            if note and results:
                results[-1]["content"] += "\n\n" + note
                since_todo = 0
            self.messages.extend(results)
            self._sync()
            if self.cancel.is_set():
                return TurnEnd("interrupted", step, final_text)

        self.sink.emit(Notice("warning", f"达到最大步数 {self.max_steps}，本轮停止。"))
        return TurnEnd("max_steps", self.max_steps, final_text)
