import json

import pytest

from coda.config import get_api_key, load_settings, update_project_settings


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_defaults_include_builtin_models(tmp_path):
    s = load_settings(tmp_path)
    assert s.model == "deepseek-flash"
    assert s.profile().provider == "deepseek"
    assert "qwen3-coder" in s.models


def test_project_overrides_scalars_and_concats_rules(tmp_path, isolated_home):
    _write(
        isolated_home / "settings.json",
        {
            "mode": "accept-edits",
            "permissions": {"allow": ["bash(uv run pytest*)"]},
        },
    )
    _write(
        tmp_path / ".coda" / "settings.json",
        {
            "mode": "plan",
            "permissions": {"allow": ["bash(make*)"], "deny": ["bash(git push*)"]},
            "models": {"deepseek-flash": {"thinking": False}},
        },
    )
    s = load_settings(tmp_path)
    assert s.mode == "plan"
    assert s.permissions.allow == ["bash(uv run pytest*)", "bash(make*)"]
    assert s.permissions.deny == ["bash(git push*)"]
    # 模型档案按键合并：只改 thinking，其余字段保留默认
    assert s.profile().thinking is False
    assert s.profile().base_url == "https://api.deepseek.com"


def test_invalid_json_reports_path(tmp_path):
    p = tmp_path / ".coda" / "settings.json"
    p.parent.mkdir()
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match="settings.json"):
        load_settings(tmp_path)


def test_unknown_profile(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        load_settings(tmp_path).profile("nope")


def test_api_key_from_env_file_and_env(isolated_home, monkeypatch):
    (isolated_home / ".env").write_text(
        '# c\nDEEPSEEK_API_KEY="from-file"\nEMPTY=\n', encoding="utf-8"
    )
    assert get_api_key("DEEPSEEK_API_KEY") == "from-file"
    assert get_api_key("EMPTY") is None
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert get_api_key("DEEPSEEK_API_KEY") == "from-env"


def test_workspace_env_is_ignored(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=project-secret\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert get_api_key("DEEPSEEK_API_KEY") is None


def test_update_project_settings(tmp_path):
    update_project_settings(tmp_path, {"verify": {"command": "pytest -q"}})
    update_project_settings(tmp_path, {"verify": {"baseline": False}})
    s = load_settings(tmp_path)
    assert s.verify.command == "pytest -q"
    assert s.verify.baseline is False
