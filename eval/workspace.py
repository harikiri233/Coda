"""准备任务工作区：解压仓库快照 → 埋 bug / 删实现 → 删隐藏测试 → git 提交 → 安装依赖。

build.py 和 runner.py 共用。所有操作都是确定性的，同一个 task.json 每次得到同样的工作区。
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from eval.spec import REPOS

EVAL_DIR = Path(__file__).resolve().parent
ARCHIVES = EVAL_DIR / "repos"


def extract(repo: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVES / REPOS[repo]["archive"]) as tar:
        tar.extractall(dest, filter="data")


def apply_mutations(root: Path, mutations: list[list[str]]) -> None:
    for rel, old, new in mutations:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise ValueError(f"{rel}: 要替换的原文出现了 {text.count(old)} 次")
        path.write_text(text.replace(old, new), encoding="utf-8")


def apply_stub(root: Path, rel: str, func: str) -> None:
    """把模块级函数 func 的函数体换成 raise NotImplementedError，保留签名和文档字符串。"""
    path = root / rel
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == func
    )
    first = node.body[0]
    has_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    keep_until = first.end_lineno if has_doc else first.lineno - 1  # 1-based，包含
    indent = lines[first.lineno - 1][: first.col_offset]
    stub = [f"{indent}raise NotImplementedError\n"]
    new = lines[:keep_until] + stub + lines[node.end_lineno :]
    path.write_text("".join(new), encoding="utf-8")


def remove_tests(root: Path, node_ids: list[str]) -> None:
    """从测试文件里删掉指定的测试函数 / 方法（node id 形如 path::Class::test 或 path::test）。"""
    by_file: dict[str, list[list[str]]] = {}
    for nid in node_ids:
        path, *names = nid.split("::")
        by_file.setdefault(path, []).append(names)
    for rel, targets in by_file.items():
        path = root / rel
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        spans: set[tuple[int, int, str]] = set()
        classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
        for names in targets:
            fn = _find_test(tree, classes, names)
            start = min([fn.lineno, *[d.lineno for d in fn.decorator_list]])
            spans.add((start, fn.end_lineno, " " * fn.col_offset))
        out = _cut(text, spans, keep_pass=False)
        if not _parses(out):  # 类里的测试方法全删光了：原位置留一个 pass 保持语法合法
            out = _cut(text, spans, keep_pass=True)
        path.write_text(out, encoding="utf-8")


def _find_test(tree: ast.Module, classes: dict[str, ast.ClassDef], names: list[str]):
    """按 node id 找测试函数；方法不在类体里时沿同一文件里定义的基类查找（继承来的测试）。"""

    def funcs(body):
        return {n.name: n for n in body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}

    if len(names) == 1:
        return funcs(tree.body)[names[0]]
    todo = [classes[names[0]]]
    while todo:
        cls = todo.pop(0)
        if names[-1] in funcs(cls.body):
            return funcs(cls.body)[names[-1]]
        todo += [classes[b.id] for b in cls.bases if isinstance(b, ast.Name) and b.id in classes]
    raise KeyError("::".join(names))


def _cut(text: str, spans: set[tuple[int, int, str]], *, keep_pass: bool) -> str:
    lines = text.splitlines(keepends=True)
    for start, end, indent in sorted(spans, reverse=True):
        lines[start - 1 : end] = [f"{indent}pass\n"] if keep_pass else []
    return "".join(lines)


def _parses(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def sh(cmd: str, cwd: Path, *, timeout: float = 900, env: dict[str, str] | None = None):
    return subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
    )


def git_commit(root: Path) -> None:
    sh(
        "git init -q && git add -A && "
        "git -c user.name=eval -c user.email=eval@example.com commit -q -m init",
        root,
    )


def prepare(task: dict, dest: Path, *, hide: bool = True, setup: bool = True) -> Path:
    """在 dest 下生成任务工作区，返回工作区路径。"""
    if dest.exists():
        shutil.rmtree(dest)
    extract(task["repo"], dest)
    if task["kind"] == "bug":
        apply_mutations(dest, task["mutations"])
    else:
        apply_stub(dest, *task["stub"])
    if hide and task.get("hidden"):
        remove_tests(dest, task["hidden"])
    repo = REPOS[task["repo"]]
    # 相当于用户在项目里执行过 /init：写好验证命令
    (dest / ".coda").mkdir(exist_ok=True)
    settings = {"verify": {"command": repo["test"]}}
    (dest / ".coda" / "settings.json").write_text(json.dumps(settings) + "\n", encoding="utf-8")
    (dest / ".gitignore").open("a", encoding="utf-8").write("\n.venv/\n.coda/\n")
    git_commit(dest)
    if setup and repo["setup"]:
        r = sh(repo["setup"], dest)
        if r.returncode != 0:
            raise RuntimeError(f"安装依赖失败：{r.stderr[-2000:]}")
    return dest


def junit_failures(root: Path, xml_path: Path) -> list[str]:
    """pytest junitxml → 失败用例的 node id（参数化的 [..] 去掉，去重）。"""
    failed: list[str] = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        parts = case.get("classname", "").split(".")
        name = case.get("name", "").split("[", 1)[0]
        for i in range(len(parts), 0, -1):
            rel = "/".join(parts[:i]) + ".py"
            if (root / rel).is_file():
                nid = "::".join([rel, *parts[i:], name])
                if nid not in failed:
                    failed.append(nid)
                break
    return failed


def run_tests(
    root: Path, cmd: str, args: str = "", *, timeout: float = 900
) -> tuple[int, list[str], str]:
    """运行测试，返回 (退出码, 失败用例, 输出末尾)。"""
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        r = sh(f"{cmd} {args} --junitxml={xml}", root, timeout=timeout)
        failed = junit_failures(root, xml) if xml.exists() else []
    return r.returncode, failed, (r.stdout + r.stderr)[-3000:]


def restore_hidden(task: dict, root: Path) -> None:
    """把含隐藏测试的测试文件恢复成原始版本（同时撤销 Agent 对这些文件的修改）。"""
    with tempfile.TemporaryDirectory() as tmp:
        extract(task["repo"], Path(tmp))
        for rel in {nid.split("::")[0] for nid in task["hidden"]}:
            shutil.copy2(Path(tmp) / rel, root / rel)
