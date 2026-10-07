"""load_skill：按名称加载 Skill 正文（系统提示词里只有目录）。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from coda.context.skills import Skill
from coda.tools.base import ErrorType, Tool, ToolContext, ToolResult


class SkillParams(BaseModel):
    name: str = Field(description="Skill 名称，见系统提示词里的 Skills 目录")


class LoadSkillTool(Tool):
    name = "load_skill"
    kind = "read"
    Params = SkillParams
    description = (
        "加载一个 Skill 的完整操作指南。系统提示词的 Skills 目录里只有名称和描述；"
        "任务和某个 Skill 相符时先加载，再按指南操作。同一个 Skill 在一次会话里加载一次即可。"
    )

    def __init__(self, skills: dict[str, Skill]) -> None:
        self.skills = skills

    def describe(self, p: SkillParams, ctx: ToolContext) -> str:
        return p.name

    def run(self, p: SkillParams, ctx: ToolContext) -> ToolResult:
        skill = self.skills.get(p.name)
        if skill is None:
            known = "、".join(self.skills) or "（无）"
            return ToolResult.error(
                ErrorType.NOT_FOUND, f"没有名为 {p.name!r} 的 Skill。", f"可用：{known}"
            )
        body = skill.body()
        return ToolResult(
            True,
            f'<skill name="{skill.name}" source="{skill.source}">\n{body}\n</skill>',
            display={"summary": f"{skill.source} · {len(body.splitlines())} 行"},
        )
