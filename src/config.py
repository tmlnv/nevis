from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 2048


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgres://nevis:nevis@db:5432/nevis?sslmode=disable"

    api_bearer_token: str

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "nvidia/nemotron-3-embed-1b:free"

    summary_base_url: str = ""
    summary_api_key: str = ""
    summary_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    ai_timeout_seconds: float = 60.0

    # Calibrated against the seeded corpus (see README). Nemotron's cosine scores
    # are compressed: unrelated documents topped out at 0.174 and genuine matches
    # started at 0.233, so 0.20 sits in the gap.
    client_score_threshold: float = 0.10
    document_score_threshold: float = 0.20
    search_result_limit: int = 20

    def summary_url(self) -> str:
        return f"{(self.summary_base_url or self.openrouter_base_url).rstrip('/')}/chat/completions"

    def embeddings_url(self) -> str:
        return f"{self.openrouter_base_url.rstrip('/')}/embeddings"

    def summary_key(self) -> str:
        return self.summary_api_key or self.openrouter_api_key
