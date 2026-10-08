# Coda

Coda 是面向国产大模型的终端 Coding Agent，界面是 Textual 全屏 TUI。不依赖任何 Agent 框架，只用 `openai` SDK 和 `mcp` SDK。

```bash
uv sync
uv run coda doctor                         # 自检：Key、文本调用、工具调用
uv run coda                                # 打开全屏 TUI
uv run coda "解释 RRF 融合的实现"            # 打开 TUI 并提交第一条任务
uv run coda -c                             # 继续本项目最近一次会话
uv run coda --resume                       # 从历史会话列表中选择
uv run coda -p "任务" [--mode accept-edits] [--output json] [--max-steps 40] [-c]   # 无头模式
```

API Key 放在环境变量或 `~/.coda/.env` 里（`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`）。Coda 不读工作区里的 `.env`。

## 界面

- 按键：Enter 发送 · Ctrl+J 换行 · ↑↓ 翻历史 · Esc 中断 · Shift+Tab 切换权限模式 · Ctrl+B 侧栏 · Ctrl+O 展开/折叠 · Ctrl+Q 退出。
- 输入 `/` 弹出命令补全；输入 `@` 模糊搜索文件（遵守 .gitignore）。发送时会附上被引用文件的内容，Agent 可以直接编辑这个文件。
- 命令：`/help /clear /compact /cost /model /thinking /mode /resume /undo /diff /verify /rules /init /memory /skills /mcp /copy /exit`。
- 侧栏依次显示任务清单、本会话改动、上下文占用（含 60% / 85% 压缩阈值）、token、缓存命中率和花费。

## 权限

有四种模式：`default`（编辑和非只读命令要确认）、`accept-edits`（编辑自动执行）、`plan`（只读）、`yolo`（全部自动执行）。

判定顺序：硬拦截 → deny 规则 → ask 规则 → 高危 → allow 规则 → 模式默认。复合命令按 `; && || |` 拆成子命令，每个子命令都满足规则或属于只读命令时，整条命令才会自动放行。

- 硬拦截（任何模式都拒绝）：`sudo`；`rm -rf` 作用于 `/`、`~` 或工作区外；`mkfs`；`dd of=/dev/*`；fork 炸弹；`curl ... | sh` 等。
- 高危（交互模式下即使是 yolo 也会询问，无头模式直接拒绝）：`git push`、`git reset --hard`、`git clean`、`rm -r`、`chmod -R` 等。
- MCP 工具默认询问，可以用 allow 规则放行，如 `mcp__fetch__*`。plan 模式下拒绝 MCP 工具。

bash 工具没有沙箱，权限检查只用来防误操作。需要强隔离时请在容器里运行。

## 上下文与会话

- 大输出落盘：工具结果超过 8k 字符时，完整内容写到会话目录，模型只看到开头 40 行、结尾 40 行和文件路径，需要时用 `read_file` 分段读。
- 微压缩：上下文达到预算的 60% 时，把较早的工具结果换成占位符，保留最近 3 次，不调用模型。这一步会破坏 prompt cache，所以只在能腾出足够空间时批量做一次。
- 摘要压缩：上下文达到 85%，或执行 `/compact [关注点]` 时，调用模型按固定模板总结较早的历史。用户历次原话原样保留，当前任务清单和改过的文件附在摘要里。切分点不会留下孤立的 tool 消息。接口返回上下文超长时，会强制压缩后重试一次。
- 系统提示词在会话内不变，动态信息以 `<system-reminder>` 追加，这样前缀保持稳定，能命中 DeepSeek 的缓存。`/cost` 可以查看命中率。
- 每个会话对应一个 `~/.coda/projects/<工作区>/<id>.jsonl`，逐行记录消息（含 `reasoning_content`）、工具执行、权限决定、压缩、用量和验证结果。这个文件同时是 trace。恢复会话时会给中断的工具调用补上结果。
- 项目记忆：启动时加载 `~/.coda/AGENTS.md` 和工作区各级的 `AGENTS.md`。`/memory add "约定"` 追加一条约定，并立刻告知 Agent。
- Skills：放在 `.coda/skills/<名称>/SKILL.md`（以及 `~/.coda/skills/`）。系统提示词里只放目录，正文由模型调用 `load_skill` 按需加载。内置 `pytest-debug` 和 `git-commit` 两个 Skill。
- 子智能体：`task` 工具派生只读调查 Agent，只能用 read_file / glob / grep，最多 30 步，只返回结论。多个 task 并行执行，最多 3 个同时运行。界面上每个子智能体显示为一个折叠块，标题实时更新步数和 token。

## 配置示例

全局 `~/.coda/settings.json` 和项目 `.coda/settings.json` 会合并：标量以项目为准，规则列表拼接。

```json
{
  "model": "deepseek-flash",
  "mode": "default",
  "permissions": {
    "allow": ["bash(uv run pytest*)", "bash(uv run ruff*)", "mcp__fetch__*"],
    "ask": ["read_file(.env*)"],
    "deny": ["bash(git push*)"]
  },
  "hooks": {
    "PostToolUse": [{"matcher": "edit_file|write_file", "command": "uv run ruff check \"$CODA_FILE\""}]
  },
  "verify": {"command": "uv run pytest -q", "baseline": true, "max_rounds": 3},
  "context": {"offload_chars": 8000, "micro_ratio": 0.6, "summary_ratio": 0.85},
  "mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"]}}
}
```

- 模型档案：内置 `deepseek-flash`（默认，开启思考）、`qwen3-coder`、`glm-4.5-air`（后两个走硅基流动）。可以在 `models` 里添加或覆盖，每个档案有自己的 `context_budget`（默认 128k）和价格 `price_per_m`。`/model` 可以在会话中途切换。切到非 DeepSeek 的模型时，历史里的思考内容不会发送。
- Hooks：Hook 进程从 stdin 收到 JSON（工具名、参数、结果），环境变量 `CODA_FILE` 是文件路径。PreToolUse 以退出码 2 退出表示否决；PostToolUse 的 stdout 会追加到工具结果后面。
- 完成闸门：改过代码后（包括 bash 命令改动的文件，按执行前后的工作区指纹检测），模型准备结束时自动运行 `verify.command`，结果和修改前的基线比较，只把新增的失败回填给模型（pytest 通过 junitxml 精确到用例）。没有配置时，如果存在 `tests/` 目录，自动探测为 `uv run pytest -q`。`/verify off` 关闭闸门。
- `tools`：只启用列出的内置工具，如 `["bash"]`（评测 E1 用）；不写表示全部启用。
- MCP：只支持 stdio。启动后在后台连接 MCP Server，握手超时 15 秒，连不上只给提示，不影响其他功能。Server 的 stderr 写到 `~/.coda/logs/`。


## 开发

```bash
uv run pytest -q        # 214 个测试：FakeLLM 回放主循环，Textual run_test 驱动界面，本地 MCP Server
uv run ruff check src tests && uv run ruff format src tests
```
