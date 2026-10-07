import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """每个测试使用独立的 CODA_HOME，并清掉真实 Key，避免读到本机配置或误调用接口。

    CODA_HOME 放在 tmp_path 之外：很多测试把 tmp_path 当工作区，会话 JSONL 写进去会被 grep 搜到。
    """
    home = tmp_path_factory.mktemp("coda_home")
    monkeypatch.setenv("CODA_HOME", str(home))
    for key in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    return home
