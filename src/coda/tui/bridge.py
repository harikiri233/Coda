"""worker 线程 → 界面的桥接。

- TuiSink：主循环在 worker 线程里 emit 事件。文本 / 思考增量先在 worker 里累积，
  每 50ms 通过 call_from_thread 批量交给主线程一次（模型每秒几十个分片，逐片刷新 Markdown 会卡）；
  其他事件到来时先把积压的增量冲刷出去，保证顺序不乱。
- TuiApprover：权限询问时在主线程推出模态弹层，worker 用 threading.Event 阻塞等结果；
  等待期间检查取消标志，Esc 中断时关掉弹层并返回拒绝。
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from coda.agent.events import Decision, Event, PermissionRequest, TextDelta, ThinkingDelta

if TYPE_CHECKING:
    from coda.tui.app import CodaApp

FLUSH_INTERVAL = 0.05


class TuiSink:
    def __init__(self, app: CodaApp) -> None:
        self.app = app
        self._pending: list[Event] = []
        self._last_flush = 0.0
        self._lock = threading.Lock()

    def _coalesce(self, ev: Event) -> None:
        last = self._pending[-1] if self._pending else None
        if type(last) is type(ev) and isinstance(ev, TextDelta | ThinkingDelta):
            last.text += ev.text  # type: ignore[union-attr]
        else:
            self._pending.append(type(ev)(ev.text))  # type: ignore[call-arg, union-attr]

    def emit(self, ev: Event) -> None:
        with self._lock:
            if isinstance(ev, TextDelta | ThinkingDelta):
                self._coalesce(ev)
                if time.monotonic() - self._last_flush < FLUSH_INTERVAL:
                    return
            else:
                self._pending.append(ev)
            batch, self._pending = self._pending, []
            self._last_flush = time.monotonic()
        if not batch:
            return
        if threading.get_ident() == self.app._thread_id:
            # /undo 等斜杠命令在主线程里调用主循环的方法，直接排进事件循环
            self.app.call_later(self.app.apply_events, batch)
        else:
            self.app.call_from_thread(self.app.apply_events, batch)


class TuiApprover:
    def __init__(self, app: CodaApp) -> None:
        self.app = app

    def ask(self, request: PermissionRequest, cancel: threading.Event) -> Decision:
        from coda.tui.screens.permission import PermissionScreen

        done = threading.Event()
        box: list[Decision] = []

        def on_result(decision: Decision | None) -> None:
            box.append(decision or Decision(False, reason="用户中断了本轮"))
            done.set()

        screen = PermissionScreen(request)
        self.app.call_from_thread(self.app.push_screen, screen, on_result)
        while not done.wait(0.1):
            if cancel.is_set():
                self.app.call_from_thread(screen.dismiss, None)
                done.wait(1)
                break
        return box[0] if box else Decision(False, reason="用户中断了本轮")
