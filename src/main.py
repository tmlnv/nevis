import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dishka import make_async_container
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI

from src.di import AppProvider, RequestProvider
from src.errors import register_exception_handlers
from src.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    await app.state.dishka_container.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Nevis Search API",
        version="1.0.0",
        description="Search across clients and their documents.",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(router)
    setup_dishka(make_async_container(AppProvider(), RequestProvider()), app)
    return app


app = create_app()
