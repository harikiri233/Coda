"""运行评测：python -m eval.runner --config E0 [--config E2 ...] [--tasks id ...] [--reps 1] [--workers 4]

每次运行：
1. 按 task.json 生成工作区（埋好 bug / 删掉实现、去掉隐藏测试、git 提交、安装依赖）。
2. 把配置写进工作区的 .coda/settings.json，执行
   coda -p "<提示词>" --mode yolo --output json --max-steps 40 --no-mcp
   硬拦截和高危拒绝在 yolo 下依然生效（无头模式直接拒绝高危命令）。
3. 恢复隐藏测试，跑完整测试套件：全部通过才算解决（同时检查有没有改坏别的功能）。
4. 从 JSON 输出和会话 JSONL 里统计步数、token、成本、工具错误、闸门回填等。

结果追加到 eval/runs/results.jsonl（同一 config/task/rep 已有结果时跳过），报告用 python -m eval.report 生成。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.workspace import EVAL_DIR, prepare, restore_hidden, run_tests

CODA_ROOT = EVAL_DIR.parent
CODA_BIN = CODA_ROOT / ".venv" / "bin" / "coda"
RUNS = EVAL_DIR / "runs"
RESULTS = RUNS / "results.jsonl"
MAX_STEPS = 40
RUN_TIMEOUT = 1200
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_WAIT = 90

# 硅基流动 Qwen3-Coder-30B-A3B-Instruct：输入 $0.07、输出 $0.28 / 百万 token（2026-10 官网价格），无缓存折扣
QWEN_PRICE = {"input": 0.07, "output": 0.28, "cache_hit": 0.07}

CONFIGS: dict[str, dict] = {
    "E0": {"desc": "完整 Coda（deepseek-flash，开思考）", "settings": {}},
    "E1": {"desc": "只给 bash 工具", "settings": {"tools": ["bash"]}},
    "E2": {"desc": "关闭完成闸门", "settings": {"verify": {"enabled": False}}},
    "E3": {
        "desc": "关闭大输出落盘和微压缩",
        "settings": {"context": {"offload": False, "micro": False}},
    },
    "E4": {
        "desc": "关闭思考模式",
        "settings": {"models": {"deepseek-flash": {"thinking": False}}},
    },
    # 上下文压力测试：这些任务的上下文峰值约 4k–19k token，128k 预算下压缩从不触发，
    # 所以把预算压到 16k（60% ≈ 9.6k 触发微压缩，85% ≈ 13.6k 触发摘要压缩）再对比
    "C0": {
        "desc": "上下文预算 16k，完整上下文工程",
        "settings": {"models": {"deepseek-flash": {"context_budget": 16000}}},
    },
    "C1": {
        "desc": "上下文预算 16k，关闭大输出落盘和微压缩（只剩摘要压缩）",
        "settings": {
            "models": {"deepseek-flash": {"context_budget": 16000}},
            "context": {"offload": False, "micro": False},
        },
    },
    "E5": {
        "desc": "换成 Qwen3-Coder-30B-A3B（硅基流动），其余同 E0",
        "settings": {
            "model": "qwen3-coder",
            "models": {"qwen3-coder": {"price_per_m": QWEN_PRICE}},
        },
    },
}

_lock = threading.Lock()


def load_tasks() -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((EVAL_DIR / "tasks").glob("*/task.json"))
    ]


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _read_env_file(path: Path) -> dict[str, str]:
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#") and value.strip():
                values[key.strip()] = value.strip().strip("'\"")
    return values


def child_env(home: Path) -> dict[str, str]:
    """子进程环境：独立的 CODA_HOME；Key 从 ~/.coda/.env 读入；去掉 Coda 自己的虚拟环境，
    让工作区里的 python / uv 用工作区自己的环境。"""
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
    }
    venv_bin = str(CODA_ROOT / ".venv" / "bin")
    env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin)
    for key, value in _read_env_file(Path.home() / ".coda" / ".env").items():
        env.setdefault(key, value)
    env["CODA_HOME"] = str(home)
    return env


def _session_stats(path: str | None) -> dict:
    """从会话 JSONL 统计闸门回填和最终验证状态。"""
    stats = {"gate_feedback_rounds": 0, "gate_final": None}
    if not path or not Path(path).is_file():
        return stats
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") == "verify" and rec.get("kind") == "final":
            if not rec["ok"] and not rec.get("gave_up"):
                stats["gate_feedback_rounds"] += 1
            stats["gate_final"] = "passed" if rec["ok"] else "failed"
    return stats


def run_one(task: dict, config: str, rep: int) -> dict:
    run_dir = RUNS / config / f"{task['id']}-r{rep}"
    ws = run_dir / "ws"
    home = RUNS / "_home"
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare(task, ws)
    settings_path = ws / ".coda" / "settings.json"
    settings = _deep_merge(
        json.loads(settings_path.read_text(encoding="utf-8")), CONFIGS[config]["settings"]
    )
    settings_path.write_text(json.dumps(settings, ensure_ascii=False) + "\n", encoding="utf-8")

    cmd = [
        str(CODA_BIN),
        "-p",
        task["prompt"],
        "--mode",
        "yolo",
        "--output",
        "json",
        "--max-steps",
        str(MAX_STEPS),
        "--no-mcp",
    ]
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=ws, env=child_env(home), capture_output=True, text=True, timeout=RUN_TIMEOUT
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr, code = "timeout", -1
    wall = time.monotonic() - start
    (run_dir / "stderr.txt").write_text(stderr[-20000:], encoding="utf-8")
    try:
        out = json.loads(stdout)
    except json.JSONDecodeError:
        out = {
            "status": "crash",
            "error": (stderr or stdout)[-2000:],
            "tool_calls": [],
            "usage": {},
        }

    # Agent 的改动（恢复隐藏测试之前统计，不含 .coda/ 和 .venv/）
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--stat"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()
    # 判定：恢复隐藏测试，跑完整测试套件
    restore_hidden(task, ws)
    test_code, failed, tail = run_tests(ws, task["workspace_test"])
    hidden_failed = [n for n in failed if n in task["hidden"]]

    tools = out.get("tool_calls", [])
    usage = out.get("usage", {})
    record = {
        "config": config,
        "task": task["id"],
        "repo": task["repo"],
        "kind": task["kind"],
        "rep": rep,
        "exit_code": code,
        "status": out.get("status"),
        "error": out.get("error"),
        "resolved": test_code == 0,
        "hidden_passed": not hidden_failed and test_code in (0, 1),
        "failed_tests": failed[:20],
        "steps": out.get("steps", 0),
        "elapsed": round(wall, 1),
        "input_tokens": usage.get("input_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "cost_usd": usage.get("cost_usd", 0.0),
        "tool_calls": len(tools),
        "tool_errors": [t["error_type"] for t in tools if not t["ok"]],
        "tool_names": [t["name"] for t in tools],
        "compactions": out.get("compactions", []),
        "verify": out.get("verify"),
        "diff_stat": diff.splitlines()[-1] if diff else "",
        "session": out.get("session"),
        **_session_stats(out.get("session")),
    }
    (run_dir / "result.json").write_text(
        json.dumps(
            {**record, "agent_output": out, "test_tail": tail}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    # 只保留结果，删掉工作区（paperlens 的 .venv 很大）
    shutil.rmtree(ws, ignore_errors=True)
    return record


def done_keys() -> set[tuple[str, str, int]]:
    if not RESULTS.exists():
        return set()
    keys = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        keys.add((r["config"], r["task"], r["rep"]))
    return keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", choices=list(CONFIGS), required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks = [t for t in load_tasks() if not args.tasks or t["id"] in args.tasks]
    done = done_keys()
    jobs = [
        (t, c, r)
        for r in range(1, args.reps + 1)
        for c in args.config
        for t in tasks
        if (c, t["id"], r) not in done
    ]
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"共 {len(jobs)} 次运行（已有结果的跳过）", flush=True)

    def work(job: tuple[dict, str, int]) -> None:
        task, config, rep = job
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            try:
                rec = run_one(task, config, rep)
            except Exception as e:  # 单次运行出错不影响其他运行
                print(f"✗ {config} {task['id']} r{rep}：{type(e).__name__}: {e}", flush=True)
                return
            # 服务商限流（硅基流动 TPM 上限）不算 Agent 的失败：等一会儿整轮重跑
            if rec["status"] == "error" and "429" in (rec["error"] or ""):
                print(
                    f"… {config} {task['id']} r{rep} 被限流，{RATE_LIMIT_WAIT}s 后重试（{attempt}）",
                    flush=True,
                )
                time.sleep(RATE_LIMIT_WAIT)
                continue
            break
        else:
            print(f"✗ {config} {task['id']} r{rep}：多次限流，放弃", flush=True)
            return
        with _lock, RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "✓" if rec["resolved"] else "✗"
        print(
            f"{mark} {config} {task['id']:<24} r{rep} {rec['status']:<9} {rec['steps']:>2} 步 "
            f"{rec['elapsed']:>6.0f}s ${rec['cost_usd']:.4f}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))


if __name__ == "__main__":
    main()
