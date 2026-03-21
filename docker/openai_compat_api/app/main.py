import os
from contextlib import asynccontextmanager

from gemini_webapi import GeminiClient

from endpoints import state
from app import app


async def _create_client() -> GeminiClient:
    secure_1psid = os.getenv("GEMINI_SECURE_1PSID")
    secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS")
    proxy = os.getenv("GEMINI_PROXY") or None

    print(f"DEBUG: GEMINI_SECURE_1PSID present: {bool(secure_1psid)}")
    print(f"DEBUG: GEMINI_SECURE_1PSIDTS present: {bool(secure_1psidts)}")

    if not secure_1psid:
        raise RuntimeError("GEMINI_SECURE_1PSID must be set.")

    print("INFO: Creating GeminiClient...")
    client = GeminiClient(secure_1psid, secure_1psidts, proxy=proxy)

    print("INFO: Initializing GeminiClient...")
    await client.init(auto_close=False, auto_refresh=True)
    print("INFO: GeminiClient initialized successfully")

    return client


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        print("INFO: Initializing Gemini client...")
        state.client = await _create_client()
        print("INFO: Gemini client initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize Gemini client: {e}")
        state.client = None
    try:
        yield
    finally:
        if state.client is not None:
            print("INFO: Closing Gemini client...")
            await state.client.close()
            print("INFO: Gemini client closed")


app.router.lifespan_context = lifespan