from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="F1DI_", env_file=".env", extra="ignore")

    env: str = "local"
    storage_url: str = "sqlite:///./f1di.db"
    vector_backend: str = "memory"
    model_backend: str = "rules"
    confidence_min_driver: float = 0.58
    confidence_min_engineer: float = 0.25
    kafka_bootstrap_servers: str = "localhost:9092"
    telemetry_topic: str = "telemetry.windows"
    insight_topic: str = "driver.insights"
    log_level: str = "INFO"
    knowledge_path: str = "data/knowledge"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "f1_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM advice layer
    llm_backend: str = "rules"  # "rules" | "anthropic" | "openai_compatible"
    llm_advice_model: str = "claude-opus-4-8"
    anthropic_api_key: str = ""
    llm_open_source_model: str = "llama3.1"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_timeout_ms: float = 2000.0

    def runtime_errors(self) -> list[str]:
        errors: list[str] = []
        if self.env not in {"local", "test", "production"}:
            errors.append("F1DI_ENV must be one of: local, test, production")
        if self.vector_backend not in {"memory", "qdrant", "pgvector"}:
            errors.append("F1DI_VECTOR_BACKEND must be one of: memory, qdrant, pgvector")
        if self.llm_backend not in {"rules", "anthropic", "openai_compatible"}:
            errors.append("F1DI_LLM_BACKEND must be one of: rules, anthropic, openai_compatible")
        if self.env == "production":
            if not self.storage_url:
                errors.append("F1DI_STORAGE_URL is required in production")
            if self.vector_backend == "memory":
                errors.append("F1DI_VECTOR_BACKEND=memory is not allowed in production")
            if self.llm_backend == "anthropic" and not self.anthropic_api_key:
                errors.append("F1DI_ANTHROPIC_API_KEY is required when LLM_BACKEND=anthropic")
        return errors

    def validate_runtime(self) -> None:
        errors = self.runtime_errors()
        if errors:
            raise RuntimeError("; ".join(errors))


settings = Settings()
