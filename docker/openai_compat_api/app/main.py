import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from gemini_webapi import GeminiClient, set_log_level

from endpoints import state
from app import app


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _create_client() -> GeminiClient:
    secure_1psid = os.getenv("GEMINI_SECURE_1PSID")
    secure_1psidts = os.getenv("GEMINI_SECURE_1PSIDTS")
    proxy = os.getenv("GEMINI_PROXY") or None

    # Configure logging level based on debug setting
    debug_raw = os.getenv("OPENAI_COMPAT_DEBUG", "false")
    debug_enabled = debug_raw.strip().lower() in ("true", "1", "yes", "on")
    log_level = "DEBUG" if debug_enabled else "INFO"
    set_log_level(log_level)

    print(f"DEBUG: GEMINI_SECURE_1PSID present: {bool(secure_1psid)}")
    print(f"DEBUG: GEMINI_SECURE_1PSIDTS present: {bool(secure_1psidts)}")
    print(f"INFO: OPENAI_COMPAT_DEBUG raw value: {debug_raw!r}")
    print(f"INFO: Setting Gemini log level to: {log_level}")

    if not secure_1psid:
        raise RuntimeError("GEMINI_SECURE_1PSID must be set.")

    print("INFO: Creating GeminiClient...")
    client = GeminiClient(secure_1psid, secure_1psidts, proxy=proxy)

    print("INFO: Initializing GeminiClient...")
    await client.init(auto_close=False, auto_refresh=True)
    print("INFO: GeminiClient initialized successfully")

    use_account_language = _env_flag("OPENAI_COMPAT_USE_ACCOUNT_LANGUAGE", True)
    language_override = (os.getenv("OPENAI_COMPAT_LANGUAGE") or "").strip()

    if language_override:
        original_language = client.language
        client.language = language_override
        print(
            "INFO: Overriding Gemini language "
            f"from {original_language!r} to {client.language!r} via OPENAI_COMPAT_LANGUAGE"
        )
    elif use_account_language:
        # Keep the account-derived language from gemini-webapi init.
        # Use English as baseline if account metadata does not provide a language.
        if not client.language:
            client.language = "en"
        print(
            "INFO: Using account language from Gemini session metadata "
            f"({client.language!r})"
        )
    else:
        # Disable language pinning so the model can infer language from conversation.
        original_language = client.language
        client.language = ""
        print(
            "INFO: OPENAI_COMPAT_USE_ACCOUNT_LANGUAGE=false, disabling fixed language "
            f"(was {original_language!r})"
        )

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