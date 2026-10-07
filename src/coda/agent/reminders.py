"""系统提醒：动态信息以 <system-reminder> 追加在消息末尾，不改系统提示词，保持前缀稳定（prompt cache）。"""

from __future__ import annotations

from coda.tools.todo import format_todos

TODO_STALE_STEPS = 8


def wrap(text: str) -> str:
    return f"<system-reminder>\n{text}\n</system-reminder>"


PLAN_MODE = wrap(
    "当前是 plan 模式：只调查，不修改。可以读文件、搜索、运行只读命令；不要尝试编辑文件或执行有副作用的命令"
    "（会被拒绝）。调查完成后给出具体的执行计划：要改哪些文件、怎么改、如何验证，并列出需要用户决定的问题。"
    "用户确认后会切换模式让你执行。"
)


def todo_stale(todos: list[dict[str, str]], steps: int) -> str | None:
    """清单里还有未完成的项，但已经很多步没更新了。"""
    if not todos or all(t["status"] == "completed" for t in todos):
        return None
    if steps < TODO_STALE_STEPS:
        return None
    return wrap(
        f"任务清单已经 {steps} 步没有更新。如果进度有变化，用 todo_write 更新（完成的标 completed，"
        f"正在做的标 in_progress）；不需要的话忽略这条提醒，不要向用户提及。当前清单：\n{format_todos(todos)}"
    )
