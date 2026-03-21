import os
from contextlib import asynccontextmanager

from gemini_webapi import GeminiClient

from endpoints import state
from app import app


async def _create_client() -> GeminiClient:
    secure_1psid = os.getenv("GEMINI_SECURE_1PSID")
    secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS")
    proxy = os.getenv("GEMINI_PROXY") or None

    if not secure_1psid:
        raise RuntimeError("GEMINI_SECURE_1PSID must be set.")

    client = GeminiClient(secure_1psid, secure_1psidts, proxy=proxy)
    await client.init(auto_close=False, auto_refresh=True)
    return client


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.client = await _create_client()
    try:
        yield
    finally:
        if state.client is not None:
            await state.client.close()


app.router.lifespan_context = lifespan