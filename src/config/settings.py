"""Application configuration and environment settings."""

import os
from pathlib import Path
from dataclasses import dataclass, field
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    """Project-wide application settings."""

    # Project Paths
    base_dir: Path = BASE_DIR
    data_raw_dir: Path = BASE_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
    vector_store_dir: Path = BASE_DIR / os.getenv("VECTOR_STORE_DIR", "data/vectorstore")

    # LLM Settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    # Ollama Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")

    # Cloud API Settings
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1")

    # Embedding Model Settings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # Semantic Chunking Settings
    breakpoint_threshold_type: str = os.getenv("BREAKPOINT_THRESHOLD_TYPE", "percentile")
    breakpoint_threshold_amount: float = float(os.getenv("BREAKPOINT_THRESHOLD_AMOUNT", "95.0"))

    # Retrieval Settings
    top_k_chunks: int = int(os.getenv("TOP_K_CHUNKS", "3"))


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
