import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Settings

_bearer = HTTPBearer(auto_error=False, description="Shared API bearer token.")


async def require_bearer(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    settings = await request.state.dishka_container.get(Settings)
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(supplied, settings.api_bearer_token):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
