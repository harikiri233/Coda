from coda.tools.base import ErrorType, Tool, ToolContext, ToolRegistry, ToolResult
from coda.tools.bash import BashTool
from coda.tools.files import EditFileTool, ReadFileTool, WriteFileTool
from coda.tools.search import GlobTool, GrepTool
from coda.tools.todo import TodoWriteTool


def builtin_tools() -> ToolRegistry:
    """内置工具。task / load_skill 在 M6 加入。"""
    return ToolRegistry(
        [
            ReadFileTool(),
            GlobTool(),
            GrepTool(),
            EditFileTool(),
            WriteFileTool(),
            BashTool(),
            TodoWriteTool(),
        ]
    )


__all__ = [
    "ErrorType",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "builtin_tools",
]
