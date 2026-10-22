"""Project .env loading. A blank permission value must leave cloud analysis off."""

import os

from class_quality.env import load_local_env
from class_quality.judge import cloud_enabled


def test_dotenv_values_are_loaded_and_blank_permission_stays_off(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_CLOUD_PERMITTED=\nOLLAMA_API_KEY=local-test-key\n", encoding="utf-8")

    assert load_local_env(env_file) is True
    assert cloud_enabled() is False
    assert os.environ["OLLAMA_API_KEY"] == "local-test-key"


def test_only_the_value_one_enables_cloud(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_CLOUD_PERMITTED=true\n", encoding="utf-8")
    load_local_env(env_file)
    assert cloud_enabled() is False

    monkeypatch.delenv("OLLAMA_CLOUD_PERMITTED", raising=False)
    env_file.write_text("OLLAMA_CLOUD_PERMITTED=1\n", encoding="utf-8")
    load_local_env(env_file)
    assert cloud_enabled() is True


def test_existing_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_CLOUD_PERMITTED", "0")
    monkeypatch.setenv("OLLAMA_API_KEY", "from-the-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_CLOUD_PERMITTED=1\nOLLAMA_API_KEY=from-the-file\n", encoding="utf-8")

    load_local_env(env_file)

    assert os.environ["OLLAMA_CLOUD_PERMITTED"] == "0"
    assert os.environ["OLLAMA_API_KEY"] == "from-the-shell"
    assert cloud_enabled() is False


def test_missing_file_leaves_the_environment_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    assert load_local_env(tmp_path / ".env") is False
    assert os.environ.get("OLLAMA_API_KEY") is None
