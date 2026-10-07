"""测试用的最小 MCP Server（stdio）：echo 和 fail 两个工具。"""

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo-test")


@server.tool(description="原样返回输入的文本")
def echo(text: str) -> str:
    return f"echo: {text}"


@server.tool(description="总是失败")
def fail(reason: str = "boom") -> str:
    raise ValueError(reason)


if __name__ == "__main__":
    server.run("stdio")
