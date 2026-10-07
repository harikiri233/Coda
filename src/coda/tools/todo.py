"""todo_write：任务清单。整表替换，同时只能有一项进行中；界面侧栏实时显示。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from coda.tools.base import ErrorType, Tool, ToolContext, ToolResult

STATUS_MARK = {"pending": "☐", "in_progress": "◐", "completed": "✔"}


class TodoItem(BaseModel):
    content: str = Field(description="这一步要做什么，一句话")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="pending 未开始 / in_progress 进行中 / completed 已完成"
    )


class TodoParams(BaseModel):
    todos: list[TodoItem] = Field(description="完整的任务清单（整表替换，不是追加）")


def format_todos(todos: list[dict[str, str]]) -> str:
    return "\n".join(f"{STATUS_MARK[t['status']]} {t['content']}" for t in todos) or "（清单为空）"


class TodoWriteTool(Tool):
    name = "todo_write"
    kind = "read"  # 不改文件，自动放行
    Params = TodoParams
    description = (
        "维护本次任务的步骤清单，用户能在界面上实时看到进度。"
        "任务需要 3 步以上、或用户一次提了多个要求时，先列清单再动手；简单任务不要用。"
        "每次传完整清单（整表替换）。开始做某一步前把它标为 in_progress，同一时间只能有一项 in_progress；"
        "做完立刻标为 completed，不要攒到最后一起改。发现新的步骤就加进去，不再需要的步骤删掉。"
    )

    def describe(self, p: TodoParams, ctx: ToolContext) -> str:
        done = sum(t.status == "completed" for t in p.todos)
        return f"{done}/{len(p.todos)} 已完成"

    def run(self, p: TodoParams, ctx: ToolContext) -> ToolResult:
        doing = [t for t in p.todos if t.status == "in_progress"]
        if len(doing) > 1:
            return ToolResult.error(
                ErrorType.INVALID_ARGS,
                f"同时有 {len(doing)} 项 in_progress。",
                "同一时间只保留一项 in_progress，其余标为 pending 或 completed。",
            )
        ctx.todos = [t.model_dump() for t in p.todos]
        done = sum(t.status == "completed" for t in p.todos)
        return ToolResult(
            True,
            f"任务清单已更新（{done}/{len(p.todos)} 已完成）：\n{format_todos(ctx.todos)}",
            display={"summary": f"{done}/{len(p.todos)} 已完成", "todos": ctx.todos},
        )
