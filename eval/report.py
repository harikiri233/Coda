"""评测报告：python -m eval.report → eval/report.md（读取 eval/runs/results.jsonl）。

指标（每个配置，多次运行取平均）：
- 解决率：恢复隐藏测试后完整测试套件全部通过。
- 虚假完成率：Agent 正常结束（status=done）但没有解决，占全部运行的比例。
- 闸门挽回：闸门回填过失败、最终解决的运行数。
- 平均步数、输入 / 输出 token、缓存命中率、成本、耗时；工具错误率（按类型）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from eval.runner import CONFIGS, RESULTS
from eval.workspace import EVAL_DIR


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def load() -> list[dict]:
    if not RESULTS.exists():
        return []
    return [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line]


def session_stats(row: dict) -> dict:
    """从会话 JSONL 读：大输出落盘次数、上下文峰值（主循环请求的输入 token）。"""
    out = {"offloaded": 0, "peak_context": 0}
    path = row.get("session")
    if not path or not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["type"] == "tool" and rec.get("offloaded"):
            out["offloaded"] += 1
        elif rec["type"] == "usage":
            out["peak_context"] = max(out["peak_context"], rec.get("context_tokens", 0))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    extra = [session_stats(r) for r in rows]
    tool_calls = sum(r["tool_calls"] for r in rows)
    errors = Counter(e for r in rows for e in r["tool_errors"])
    inp = sum(r["input_tokens"] for r in rows)
    return {
        "n": n,
        "resolved": sum(r["resolved"] for r in rows) / n,
        "false_done": sum(r["status"] == "done" and not r["resolved"] for r in rows) / n,
        "gate_saved": sum(r["gate_feedback_rounds"] > 0 and r["resolved"] for r in rows),
        "gate_triggered": sum(r["gate_feedback_rounds"] > 0 for r in rows),
        "steps": mean(r["steps"] for r in rows),
        "input": mean(r["input_tokens"] for r in rows),
        "output": mean(r["output_tokens"] for r in rows),
        "cache": sum(r["cached_tokens"] for r in rows) / inp if inp else 0.0,
        "cost": mean(r["cost_usd"] for r in rows),
        "elapsed": mean(r["elapsed"] for r in rows),
        "tool_error_rate": sum(errors.values()) / tool_calls if tool_calls else 0.0,
        "errors": dict(errors.most_common()),
        "status": dict(Counter(r["status"] for r in rows)),
        "micro": sum(c["kind"] == "micro" for r in rows for c in r["compactions"]),
        "summary": sum(c["kind"] == "summary" for r in rows for c in r["compactions"]),
        "offloaded": sum(e["offloaded"] for e in extra),
        "peak_context": mean(e["peak_context"] for e in extra),
        "peak_context_max": max(e["peak_context"] for e in extra),
        "self_tested": sum(
            any(name == "bash" for name in r["tool_names"]) and r["verify"] != "skipped"
            for r in rows
        ),
        "reps": max(r["rep"] for r in rows),
    }


def render(rows: list[dict]) -> str:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    configs = [c for c in CONFIGS if c in by_cfg]
    stats = {c: summarize(by_cfg[c]) for c in configs}
    reps = max((r["rep"] for r in rows), default=0)
    n_tasks = len({r["task"] for r in rows})

    lines = [
        "# Coda 评测结果",
        "",
        f"{n_tasks} 个任务（PaperLens 6 + toolz 7 + more-itertools 7；12 个 bug 修复 + 8 个小功能），"
        f"每个配置最多运行 {reps} 轮（见“运行数”列）。解决 = 恢复隐藏测试后完整测试套件全部通过。"
        f"无头模式、yolo 权限、最多 {40} 步。",
        "",
        "## 总表",
        "",
        "| 配置 | 说明 | 运行数 | 解决率 | 虚假完成率 | 平均步数 | 输入 token | 输出 token | 缓存命中 | 单次成本 | 平均耗时 | 工具错误率 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        s = stats[c]
        lines.append(
            f"| {c} | {CONFIGS[c]['desc']} | {s['n']} | {_pct(s['resolved'])} | {_pct(s['false_done'])} "
            f"| {s['steps']:.1f} | {s['input']:,.0f} | {s['output']:,.0f} | {_pct(s['cache'])} "
            f"| ${s['cost']:.4f} | {s['elapsed']:.0f}s | {_pct(s['tool_error_rate'])} |"
        )

    lines += [
        "",
        "## 完成闸门",
        "",
        "| 配置 | 闸门回填过的运行 | 其中最终解决（挽回） | 结束时验证结果 | 结束状态分布 |",
        "|---|---|---|---|---|",
    ]
    for c in configs:
        s = stats[c]
        status = "，".join(f"{k} {v}" for k, v in s["status"].items())
        verify = "，".join(
            f"{k or '未触发'} {v}" for k, v in Counter(r["verify"] for r in by_cfg[c]).items()
        )
        lines.append(f"| {c} | {s['gate_triggered']} | {s['gate_saved']} | {verify} | {status} |")

    ctx = [c for c in ("E0", "E3", "C0", "C1") if c in stats]
    if any(c.startswith("C") for c in ctx):
        lines += [
            "",
            "## 上下文工程（压缩触发情况）",
            "",
            "| 配置 | 说明 | 运行数 | 解决率 | 微压缩 | 摘要压缩 | 大输出落盘 | 上下文峰值（平均 / 最大） | 输入 token | 缓存命中 | 单次成本 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in ctx:
            s = stats[c]
            lines.append(
                f"| {c} | {CONFIGS[c]['desc']} | {s['n']} | {_pct(s['resolved'])} | {s['micro']} "
                f"| {s['summary']} | {s['offloaded']} | {s['peak_context']:,.0f} / {s['peak_context_max']:,} "
                f"| {s['input']:,.0f} | {_pct(s['cache'])} | ${s['cost']:.4f} |"
            )

    lines += ["", "## 按任务类型 / 仓库的解决率", ""]
    groups = [
        ("bug", "kind"),
        ("feature", "kind"),
        ("paperlens", "repo"),
        ("toolz", "repo"),
        ("more-itertools", "repo"),
    ]
    lines += [
        "| 配置 | " + " | ".join(g for g, _ in groups) + " |",
        "|---" * (len(groups) + 1) + "|",
    ]
    for c in configs:
        cells = []
        for g, field in groups:
            sub = [r for r in by_cfg[c] if r[field] == g]
            cells.append(_pct(sum(r["resolved"] for r in sub) / len(sub)) if sub else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines += ["", "## 工具错误（按类型）", ""]
    for c in configs:
        errs = "，".join(f"{k} {v}" for k, v in stats[c]["errors"].items()) or "无"
        lines.append(f"- {c}：{errs}")

    lines += ["", "## 逐任务结果（✓ 解决 / ✗ 未解决，每格为各轮结果）", ""]
    tasks = sorted({r["task"] for r in rows})
    lines += ["| 任务 | " + " | ".join(configs) + " |", "|---" * (len(configs) + 1) + "|"]
    for t in tasks:
        cells = []
        for c in configs:
            runs = sorted((r for r in by_cfg[c] if r["task"] == t), key=lambda r: r["rep"])
            cells.append("".join("✓" if r["resolved"] else "✗" for r in runs) or "-")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    total = sum(r["cost_usd"] for r in rows)
    lines += ["", f"全部 {len(rows)} 次运行合计花费 ${total:.3f}（按各模型档案的单价估算）。", ""]
    return "\n".join(lines)


NOTES = """
## 方法

- 任务：`eval/spec.py` 定义改动和提示词（issue 口吻，只描述现象）。`python -m eval.build` 应用改动后跑完整测试，
  失败的用例就是隐藏测试，并从工作区删掉；删掉后剩下的测试必须全部通过。判定时把原始测试文件放回去跑完整套件，
  所以“解决”同时要求隐藏测试通过、没有改坏别的功能。
- 运行：每次在全新的工作区里执行 `coda -p … --mode yolo --output json --max-steps 40 --no-mcp`，单次最长 1200 秒；
  工作区预先写好 `verify.command`（相当于执行过 /init）。统计来自无头模式的 JSON 输出和会话 JSONL。
- 成本按各模型档案的单价估算（DeepSeek 区分缓存命中；Qwen3-Coder 按硅基流动 $0.07 / $0.28 每百万 token）。
  DeepSeek 高峰 / 非高峰价格不同，Coda 的档案统一按高峰价计。

## 结论

1. **基线**：完整 Coda 在 40 次运行中全部解决，平均 9.6 步、73 秒、$0.006，缓存命中率 90%。
2. **消融差异很小，不能下强结论**：E1–E4 共 5 次失败，其中 4 次落在 `pl-dup-citation`（E1 两次、E2 和 E3 各一次），C0 唯一的失败也是它。
   这个任务的隐藏测试要求 `cited_numbers` 本身去重，而失败的运行都在下游的 `check_citations` 里去重：按 issue 描述的行为是对的，
   隐藏测试却不过。这是任务定义的歧义，不是 Agent 能力的差异。去掉这个任务后，E0–E3、C0、C1 全部解决，E4 错 1 次（`mi-windowed-step`）。
   这 20 个任务对 deepseek-flash 偏简单，区分不出各个机制。
3. **只给 bash（E1）**：解决率没有下降（失败的 2 次都是上面的歧义任务），输入 token 反而少 27%（52k vs 72k），
   因为模型用 `sed -n` / `grep -rn` 一次取到需要的片段。在小仓库、小改动上，专用工具的价值没有体现在解决率上；
   体现在可控性上：E1 的修改全部绕过了先读后写检查、编辑预览和 `/undo` 检查点。
4. **完成闸门（E2）**：所有配置里闸门都没有回填过失败，因为 E0 中 38/40 次运行模型自己跑了测试，通过后才结束。
   关闭闸门后虚假完成率 2.5%（1/40），E0 是 0；但这 1 次正是歧义任务，开着闸门也拦不住：闸门只能看到工作区里的测试，
   隐藏测试按设计不在其中。闸门的代价是每次运行多约 33 秒（基线 + 结束前各跑一次测试套件；平均耗时 73s vs 40s）。
   结论：对会主动跑测试的模型，闸门是兜底；这组任务没有测出它的收益。
5. **关闭思考（E4）**：成本持平，但步数 +22%（11.7 vs 9.6）、输入 token +28%、工具参数错误更多（3.7% vs 2.2%）。
   思考模式用推理 token 换掉了一部分试错步骤。
6. **上下文工程**：默认 128k 预算下压缩从未触发（上下文峰值平均 9.7k、最大 19k），所以 E3 和 E0 没有差别。
   把预算压到 16k 测一轮：完整上下文工程（C0）触发微压缩 11 次，上下文峰值从平均 9.7k 降到 7.9k，平均输入 token 56.7k，
   比 E0 第一轮（66.1k）少 14%；只保留摘要压缩（C1）触发摘要压缩 5 次，平均输入 71.9k，单次成本比 C0 高 37%（$0.0077 vs $0.0056）。
   微压缩会改写历史、使缓存失效，所以 C0 的缓存命中率最低（84% vs E0 的 90%），但省下的 token 更多。
   两种配置的解决率都没有因压缩下降（C0 唯一的失败仍是歧义任务）。
7. **换模型（E5，Qwen3-Coder-30B-A3B，1 轮）**：解决率 65%，平均 14 步、$0.016（E0 的 2.7 倍：输入 token 是 E0 的 3 倍，
   硅基流动也没有缓存折扣）；平均耗时 586 秒，其中包含限流排队。7 次失败：2 次自称完成但隐藏测试不过，
   1 次把任务误判成“不是代码任务”直接结束，1 次 40 步用完，2 次因限流排队超过 1200 秒的单次上限，1 次网关返回 HTTP 400。
   同一套 harness 下，模型之间的差距远大于各个机制之间的差距。

## 评测期间对 Coda 的改动

- 完成闸门原来只跟踪 edit_file / write_file。E1 只给 bash 时闸门根本不会触发，于是加了 bash 改动检测：
  非只读命令执行前后各取一次工作区指纹（mtime + 大小），有变化就和编辑工具一样标记、首次修改前跑基线。
- `settings.tools` 白名单（E1 用），并在系统提示词里说明本次可用的工具。
- 硅基流动的 TPM 限流按分钟计，SDK 自带的重试最多只等 8 秒；LLM 客户端对 429 再按 10 / 20 / 40 / 60 秒退避。
- 新增 3 个测试，共 214 个。
"""


def main() -> None:
    rows = load()
    path = EVAL_DIR / "report.md"
    path.write_text(render(rows) + NOTES, encoding="utf-8")
    summary = {
        c: summarize([r for r in rows if r["config"] == c])
        for c in CONFIGS
        if any(r["config"] == c for r in rows)
    }
    (EVAL_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(path)


if __name__ == "__main__":
    main()
