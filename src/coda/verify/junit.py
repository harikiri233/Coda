"""解析 pytest 的 junitxml，得到失败用例集合（用例名 → 截断后的错误信息）。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

MAX_MESSAGE = 1500


def _case_id(case: ET.Element) -> str:
    cls = case.get("classname", "")
    name = case.get("name", "")
    if not cls:
        return name
    # tests.test_x.TestA → tests/test_x.py::TestA::name 这种 pytest 风格更便于模型定位
    parts = cls.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test"):
            module = "/".join(parts[:i]) + ".py"
            rest = parts[i:]
            return "::".join([module, *rest, name])
    return f"{cls}::{name}"


def parse_junit(path: Path) -> dict[str, str] | None:
    """返回失败和出错的用例；文件不存在或无法解析时返回 None。"""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    failures: dict[str, str] = {}
    for case in root.iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is None:
                continue
            message = (node.get("message") or "").strip()
            body = (node.text or "").strip()
            text = body or message
            if len(text) > MAX_MESSAGE:
                text = "…" + text[-MAX_MESSAGE:]  # 断言信息通常在最后
            failures[_case_id(case)] = text
            break
    return failures
