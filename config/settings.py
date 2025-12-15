import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Settings:
    # DB
    db_url: str = os.getenv("DB_URL", "sqlite:///support.db")

    # LLM / OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
