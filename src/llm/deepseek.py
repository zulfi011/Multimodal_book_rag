"""DeepSeek R1 model integration supporting local Ollama and cloud APIs."""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import get_settings


def get_deepseek_llm(
    provider: Optional[str] = None,
    temperature: float = 0.6,
) -> BaseChatModel:
    """Initialize DeepSeek R1 model based on configured provider.

    Supported providers:
    - 'ollama' (default local): Uses langchain_ollama.ChatOllama
    - 'deepseek_api': Uses official DeepSeek API via langchain_openai.ChatOpenAI
    - 'groq': Uses Groq's hosted DeepSeek R1 distillation via langchain_groq.ChatGroq
    - 'openrouter': Uses OpenRouter hosted DeepSeek R1

    Args:
        provider: Provider identifier ('ollama', 'deepseek_api', 'groq', 'openrouter').
        temperature: Sampling temperature for reasoning and generation.

    Returns:
        BaseChatModel instance configured for DeepSeek R1.
    """
    settings = get_settings()
    active_provider = (provider or settings.llm_provider).lower()

    if active_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=temperature,
        )

    elif active_provider == "deepseek_api":
        from langchain_openai import ChatOpenAI

        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required when using provider 'deepseek_api'."
            )

        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
        )

    elif active_provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is required when using provider 'groq'."
            )

        return ChatGroq(
            model_name=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
        )

    elif active_provider == "openrouter":
        from langchain_openai import ChatOpenAI

        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required when using provider 'openrouter'."
            )

        return ChatOpenAI(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=4000,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider '{active_provider}'. "
            "Supported providers are: 'ollama', 'deepseek_api', 'groq', 'openrouter'."
        )

