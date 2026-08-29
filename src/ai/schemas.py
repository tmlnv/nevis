"""Provider trust boundary. `Out` is what we send to OpenRouter, `In` is what we receive."""

from pydantic import BaseModel, ConfigDict


class EmbeddingRequestOut(BaseModel):
    model: str
    input: list[str]


class EmbeddingDataIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    embedding: list[float]


class EmbeddingResponseIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    data: list[EmbeddingDataIn]


class ChatMessageOut(BaseModel):
    role: str
    content: str


class ChatCompletionRequestOut(BaseModel):
    model: str
    messages: list[ChatMessageOut]
    max_tokens: int
    temperature: float


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None


class ChatChoiceIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: ChatMessageIn


class ChatCompletionResponseIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    choices: list[ChatChoiceIn]
