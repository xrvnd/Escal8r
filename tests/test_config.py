import pytest

from escal8r.config import ConfigError, load_settings

ENV = {
    "API_URL": "https://helpdesk.test/v1/",
    "API_KEY": "secret-key",
    "CANDIDATE_EMAIL": "me@example.com",
}


@pytest.fixture
def env(monkeypatch):
    for name in (*ENV, "CANDIDATE_NAME", "OPENROUTER_KEY", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    return monkeypatch


def test_loads_from_environment(env):
    settings = load_settings(env_file=None)
    assert settings.helpdesk_url == "https://helpdesk.test/v1"
    assert settings.candidate_name is None
    assert settings.max_run_cost_usd == 0.10


def test_missing_variables_are_named(env):
    env.delenv("API_KEY")
    with pytest.raises(ConfigError, match="API_KEY"):
        load_settings(env_file=None)


def test_repr_hides_secrets(env):
    env.setenv("OPENROUTER_KEY", "sk-or-secret")
    text = repr(load_settings(env_file=None))
    assert "secret-key" not in text
    assert "sk-or-secret" not in text
    assert "me@example.com" not in text
