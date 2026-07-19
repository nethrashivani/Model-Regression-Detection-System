import pytest

from src.llm_provider import get_api_key, get_provider_config
from src.models import EmailClassification, PromptConfig
from src.prompt_loader import list_available_versions, load_prompt_config


def test_v1_loads_successfully():
    config = load_prompt_config("v1")
    assert isinstance(config, PromptConfig)
    assert config.version == "v1"
    assert len(config.few_shot_examples) == 4


def test_missing_version_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="No prompt file found"):
        load_prompt_config("does-not-exist")


def test_list_available_versions_includes_v1():
    assert "v1" in list_available_versions()


def test_email_classification_rejects_invalid_category():
    with pytest.raises(Exception):
        EmailClassification(category="not-a-real-category", summary="test")


def test_email_classification_rejects_empty_summary():
    with pytest.raises(Exception):
        EmailClassification(category="billing", summary="   ")


def test_email_classification_accepts_valid_input():
    result = EmailClassification(category="technical", summary="App crashes on upload.")
    assert result.category == "technical"


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    cfg = get_provider_config()
    assert cfg.base_url == "https://api.groq.com/openai/v1"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_provider_config()


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = get_provider_config()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        get_api_key(cfg)


def test_ollama_needs_no_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    cfg = get_provider_config()
    assert get_api_key(cfg) == "ollama"
