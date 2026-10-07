"""Skills：两层注入、按需加载。

每个 Skill 是一个目录，里面的 SKILL.md 以 YAML 头部写 name 和 description：

    ---
    name: pytest-debug
    description: 排查失败的 pytest 用例
    ---
    正文……

系统提示词只放目录（名称 + 一句描述），模型需要时调用 load_skill 取正文。
查找顺序（同名时前面的优先）：项目 .coda/skills/ → 个人 ~/.coda/skills/ → 内置 coda/skills/。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coda.config import coda_home

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    source: str  # 项目 / 个人 / 内置

    def body(self) -> str:
        _, text = parse_frontmatter(self.path.read_text(encoding="utf-8", errors="replace"))
        return text.strip()


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """只支持单行的 key: value，够 SKILL.md 用，不引入 YAML 依赖。"""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1 :])
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return {}, text


def skill_dirs(workdir: Path) -> list[tuple[Path, str]]:
    return [
        (workdir / ".coda" / "skills", "项目"),
        (coda_home() / "skills", "个人"),
        (BUILTIN_DIR, "内置"),
    ]


def discover_skills(workdir: Path) -> dict[str, Skill]:
    found: dict[str, Skill] = {}
    for root, source in skill_dirs(workdir):
        if not root.is_dir():
            continue
        for md in sorted(root.glob("*/SKILL.md")):
            meta, _ = parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
            name = meta.get("name") or md.parent.name
            if name in found or not meta.get("description"):
                continue
            found[name] = Skill(name, meta["description"], md, source)
    return found


def render_catalog(skills: dict[str, Skill]) -> str:
    if not skills:
        return ""
    lines = [
        "# Skills",
        "下面是可用的 Skill（操作指南）。任务和某个 Skill 的描述相符时，先调用 load_skill 读取正文再按它做；"
        "不相关就不要加载。",
    ]
    lines += [f"- {s.name}：{s.description}" for s in skills.values()]
    return "\n".join(lines)
