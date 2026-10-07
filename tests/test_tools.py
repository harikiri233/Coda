"""6 个基础工具与统一返回协议。"""

import os
import threading
import time

import pytest

from coda.state.filestate import FileState
from coda.tools import ErrorType, ToolContext, builtin_tools
from coda.tools.walk import glob_match

TOOLS = builtin_tools()


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (tmp_path / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("build/\n*.log\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "gen.py").write_text("def add(): pass\n", encoding="utf-8")
    (tmp_path / "debug.log").write_text("add\n", encoding="utf-8")
    return ToolContext(tmp_path.resolve(), FileState())


def run(name, ctx, **args):
    tool = TOOLS.get(name)
    return tool.run(tool.validate(args), ctx)


def test_schema_has_no_titles():
    schema = TOOLS.get("grep").schema()
    params = schema["function"]["parameters"]
    assert params["required"] == ["pattern"]
    assert "title" not in params and "title" not in params["properties"]["pattern"]


def test_error_format():
    r = run(
        "read_file", ToolContext(__import__("pathlib").Path("/tmp"), FileState()), path="nope.txt"
    )
    assert r.to_text().startswith("Error[not_found]: ")
    assert "\nHint: " in r.to_text()


# ---- read_file


def test_read_with_line_numbers_and_paging(ctx):
    r = run("read_file", ctx, path="src/pkg/core.py", offset=2, limit=2)
    assert r.ok
    assert r.content.splitlines()[0] == "2→    return a + b"
    assert "offset=4" in r.content


def test_read_directory_and_binary(ctx):
    assert run("read_file", ctx, path="src").error_type == ErrorType.INVALID_ARGS
    (ctx.workdir / "x.bin").write_bytes(b"\x00\x01")
    assert "二进制" in run("read_file", ctx, path="x.bin").content


# ---- glob / grep


def test_glob_respects_gitignore(ctx):
    r = run("glob", ctx, pattern="*.py")
    files = set(r.content.splitlines())
    assert files == {"src/pkg/core.py", "src/pkg/__init__.py"}


def test_glob_no_match_is_not_failure(ctx):
    r = run("glob", ctx, pattern="*.rs")
    assert r.ok and r.error_type is None and "0 个文件" in r.content


def test_glob_match_rules():
    assert glob_match("**/*.py", "a/b/c.py") and glob_match("**/*.py", "c.py")
    assert glob_match("src/*.py", "src/a.py") and not glob_match("src/*.py", "src/x/a.py")
    assert glob_match("*.{toml,cfg}", "x/setup.cfg")


def test_grep_content_and_gitignore(ctx):
    r = run("grep", ctx, pattern=r"def add")
    assert r.content == "src/pkg/core.py:1:def add(a, b):"
    assert r.display["summary"] == "1 处匹配（1 个文件）"


def test_grep_modes_and_context(ctx):
    assert run("grep", ctx, pattern="def", mode="count").content == "src/pkg/core.py:2"
    assert run("grep", ctx, pattern="def", mode="files").content == "src/pkg/core.py"
    r = run("grep", ctx, pattern="def sub", context=1)
    assert "src/pkg/core.py-4-" in r.content and "src/pkg/core.py:5:def sub" in r.content


def test_grep_zero_matches_says_how_many_files(ctx):
    r = run("grep", ctx, pattern="nothing_here")
    assert r.ok and "0 处匹配" in r.content and "不是执行失败" in r.content


def test_grep_invalid_regex(ctx):
    r = run("grep", ctx, pattern="foo(")
    assert r.error_type == ErrorType.INVALID_REGEX


# ---- edit_file / write_file


def test_edit_requires_read_first(ctx):
    r = run("edit_file", ctx, path="src/pkg/core.py", old="a + b", new="b + a")
    assert r.error_type == ErrorType.NOT_READ


def test_edit_success_returns_diff(ctx):
    run("read_file", ctx, path="src/pkg/core.py")
    r = run("edit_file", ctx, path="src/pkg/core.py", old="return a + b", new="return b + a")
    assert r.ok and r.display["added"] == 1 and r.display["removed"] == 1
    assert "+    return b + a" in r.display["diff"]
    # 自己写入后不算"读后被改"，可以继续编辑
    r2 = run("edit_file", ctx, path="src/pkg/core.py", old="return b + a", new="return a + b")
    assert r2.ok


def test_edit_stale_after_external_change(ctx):
    path = ctx.workdir / "src/pkg/core.py"
    run("read_file", ctx, path="src/pkg/core.py")
    time.sleep(0.01)
    path.write_text(path.read_text() + "# changed\n")
    os.utime(path, None)
    r = run("edit_file", ctx, path="src/pkg/core.py", old="a + b", new="b + a")
    assert r.error_type == ErrorType.STALE


def test_edit_ambiguous_and_replace_all(ctx):
    run("read_file", ctx, path="src/pkg/core.py")
    r = run("edit_file", ctx, path="src/pkg/core.py", old="(a, b)", new="(x, y)")
    assert r.error_type == ErrorType.AMBIGUOUS and "2 次" in r.content
    r = run("edit_file", ctx, path="src/pkg/core.py", old="(a, b)", new="(x, y)", replace_all=True)
    assert r.ok and (ctx.workdir / "src/pkg/core.py").read_text().count("(x, y)") == 2


def test_edit_no_match_hints_whitespace(ctx):
    run("read_file", ctx, path="src/pkg/core.py")
    r = run("edit_file", ctx, path="src/pkg/core.py", old="def add(a, b):\n  return a + b", new="x")
    assert r.error_type == ErrorType.NO_MATCH and "空白" in r.hint


def test_edit_no_match_hints_closest_line(ctx):
    run("read_file", ctx, path="src/pkg/core.py")
    r = run("edit_file", ctx, path="src/pkg/core.py", old="def add(a, c):", new="x")
    assert "第 1 行" in r.hint


def test_write_new_file_and_overwrite_needs_read(ctx):
    r = run("write_file", ctx, path="new/mod.py", content="x = 1\n")
    assert r.ok and (ctx.workdir / "new/mod.py").read_text() == "x = 1\n"
    assert run("write_file", ctx, path="README.md", content="hi").error_type == ErrorType.NOT_READ
    run("read_file", ctx, path="README.md")
    assert run("write_file", ctx, path="README.md", content="hi\n").ok


def test_edit_preview_does_not_write(ctx):
    run("read_file", ctx, path="src/pkg/core.py")
    tool = TOOLS.get("edit_file")
    params = tool.validate({"path": "src/pkg/core.py", "old": "a - b", "new": "b - a"})
    assert "+    return b - a" in tool.preview(params, ctx)
    assert "a - b" in (ctx.workdir / "src/pkg/core.py").read_text()


# ---- bash


def test_bash_exit_code_and_streams(ctx):
    r = run("bash", ctx, command="echo out; echo err >&2; exit 3")
    assert r.ok and r.content.startswith("exit_code: 3")
    assert "<stdout>\nout\n</stdout>" in r.content and "<stderr>\nerr\n</stderr>" in r.content


def test_bash_pipefail(ctx):
    r = run("bash", ctx, command="false | cat")
    assert r.display["exit_code"] == 1


def test_bash_runs_in_workdir(ctx):
    assert str(ctx.workdir) in run("bash", ctx, command="pwd").content


def test_bash_timeout_kills_group(ctx):
    start = time.monotonic()
    r = run("bash", ctx, command="sleep 30 & sleep 30", timeout=1)
    assert r.error_type == ErrorType.TIMEOUT and time.monotonic() - start < 5


def test_bash_interrupt(ctx):
    timer = threading.Timer(0.3, ctx.cancel.set)
    timer.start()
    r = run("bash", ctx, command="sleep 30")
    assert r.error_type == ErrorType.INTERRUPTED
