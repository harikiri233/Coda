---
name: git-commit
description: 用户要求提交代码、整理改动或写 commit message 时使用
---
# 整理改动并提交

1. **看清改了什么**：`git status --short`、`git diff --stat`，再用 `git diff` 看具体内容（已暂存的用 `git diff --cached`）。
2. **检查不该提交的东西**：调试打印、临时文件、`.env` / 密钥、无关的格式化改动、大文件。发现了先告诉用户，不要擅自删除用户的文件。
3. **按目的拆分**：一个提交只做一件事。多个不相关的改动分开暂存（`git add 路径`），不要无脑 `git add .`。
4. **提交前验证**：跑一遍测试和 lint（命令见项目 AGENTS.md）。
5. **写 commit message**：沿用仓库已有的风格（先看 `git log --oneline -10`）。没有约定时用：
   ```
   <类型>: <一句话说明做了什么>

   <为什么改、怎么改的要点；可选>
   ```
   类型用 feat / fix / refactor / test / docs / chore。标题不超过 72 个字符，说清"做了什么"而不是"改了哪个文件"。
6. **只提交，不推送**：`git push`、`git commit --amend`、`git reset --hard`、强制推送都要用户明确要求才做。
