"""生成任务集：python -m eval.build → eval/tasks/<id>/task.json

对每个任务：
1. 原始仓库跑一遍测试，确认全部通过（基线干净）。
2. 应用改动（埋 bug / 删实现）后再跑一遍，失败的用例记为隐藏测试；没有失败说明改动没被测试覆盖，报错。
3. 删掉隐藏测试后再跑一遍，确认工作区里剩下的测试全部通过——Agent 只能靠提示词和读代码定位问题。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from eval.spec import REPOS, TASKS
from eval.workspace import EVAL_DIR, prepare, run_tests

TASKS_DIR = EVAL_DIR / "tasks"


def build_one(task: dict, tmp: Path) -> dict:
    repo = REPOS[task["repo"]]
    ws = prepare(task, tmp / task["id"], hide=False)
    code, failed, tail = run_tests(ws, repo["test"])
    if not failed:
        raise RuntimeError(f"{task['id']}：改动后没有测试失败（exit {code}）\n{tail}")
    task = {**task, "hidden": failed}
    ws = prepare(task, tmp / task["id"], hide=True)
    code, still, tail = run_tests(ws, repo["test"])
    if code != 0 or still:
        raise RuntimeError(f"{task['id']}：删掉隐藏测试后仍有失败 {still}\n{tail}")
    return {
        **task,
        "workspace_test": repo["test"],
        "judge": f"{repo['test']} " + " ".join(f"'{n}'" for n in failed),
    }


def main() -> None:
    only = set(sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="coda-build-") as t:
        tmp = Path(t)
        clean: set[str] = set()
        for task in TASKS:
            if only and task["id"] not in only:
                continue
            if task["repo"] not in clean:
                ws = prepare({**task, "kind": "bug", "mutations": []}, tmp / "clean")
                code, failed, tail = run_tests(ws, REPOS[task["repo"]]["test"])
                if code != 0:
                    raise RuntimeError(f"{task['repo']} 原始测试未通过：{failed}\n{tail}")
                clean.add(task["repo"])
            out = build_one(task, tmp)
            path = TASKS_DIR / task["id"] / "task.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{task['id']:<24} 隐藏测试 {len(out['hidden'])} 个：{', '.join(out['hidden'])}")


if __name__ == "__main__":
    main()
