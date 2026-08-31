from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 2048


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgres://nevis:nevis@db:5432/nevis?sslmode=disable"

    api_bearer_token: str

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-large"

    summary_base_url: str = ""
    summary_api_key: str = ""
    summary_model: str = "google/gemini-2.5-flash-lite"

    ai_timeout_seconds: float = 60.0

    # Calibrated by sweeping a 70-document corpus and 18 labelled queries (see README).
    # 0.40 is the F1 peak for this model and the point where precision reaches 1.0
    # while both nonsense queries still return nothing.
    client_score_threshold: float = 0.10
    document_score_threshold: float = 0.40
    search_result_limit: int = 20

    def summary_url(self) -> str:
        return f"{(self.summary_base_url or self.openrouter_base_url).rstrip('/')}/chat/completions"

    def embeddings_url(self) -> str:
        return f"{self.openrouter_base_url.rstrip('/')}/embeddings"

    def summary_key(self) -> str:
        return self.summary_api_key or self.openrouter_api_key
