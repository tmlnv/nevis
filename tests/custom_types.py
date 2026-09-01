from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated

import httpx
import pytest

from src.ai.client import AIClient

type Fixture[T] = Annotated[T, pytest.fixture]
type JsonPrimitive = None | bool | int | float | str
type JsonValue = JsonPrimitive | Sequence[JsonValue] | Mapping[str, JsonValue]
type Factory = Callable[..., Awaitable[dict[str, JsonValue]]]
type Build = Callable[
    [Callable[[httpx.Request], httpx.Response]], tuple[AIClient, list[httpx.Request]]
]
