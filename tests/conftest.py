import pytest

pytest_plugins = ["tests.investment_ui_fixtures"]


@pytest.fixture(autouse=True)
def isolated_research_cache(tmp_path, monkeypatch):
    """Tests never read or write the real research answer cache."""
    from backend import research_cache

    monkeypatch.setenv("RESEARCH_CACHE_DIR", str(tmp_path / "research_cache"))
    monkeypatch.setattr(research_cache, "_cache", None)
    monkeypatch.setattr(research_cache, "local_embedding", lambda: None)
