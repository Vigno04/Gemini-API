import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import orjson

from gemini_webapi import GeminiClient


def _unix_ts() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    temperature: float | None = None
    stream: bool = False


class AppState:
    client: GeminiClient | None = None


state = AppState()


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


app = FastAPI(title="Gemini OpenAI-compatible API", lifespan=lifespan)


def _require_auth(authorization: str | None):
    expected = os.getenv("OPENAI_COMPAT_API_KEY")
    if not expected:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    rendered: list[str] = []
    for msg in messages:
        role = msg.role.lower()
        if role not in {"system", "user", "assistant"}:
            continue

        if isinstance(msg.content, str):
            text = msg.content
        else:
            # OpenAI-style content array support: [{"type":"text","text":"..."}, ...]
            text_parts = [
                str(part.get("text", ""))
                for part in msg.content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            text = "\n".join([part for part in text_parts if part])

        if text:
            rendered.append(f"{role}: {text}")

    if not rendered:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    rendered.append("assistant:")
    return "\n".join(rendered)


@app.get("/v1/models")
@app.get("/models")
@app.get("/v/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)

    # Keep this static and stable for OpenAI clients.
    data = [
        {"id": "gemini-3-flash", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-3-pro", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {
            "id": "gemini-3-flash-thinking",
            "object": "model",
            "created": 0,
            "owned_by": "gemini-webapi",
        },
    ]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    model = payload.model or os.getenv("OPENAI_COMPAT_DEFAULT_MODEL", "gemini-3-flash")
    prompt = _messages_to_prompt(payload.messages)

    if payload.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = _unix_ts()

        async def stream_events():
            async for chunk in client.generate_content_stream(
                prompt=prompt,
                model=model,
            ):
                delta = chunk.text_delta
                if not delta:
                    continue

                payload_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    output = await client.generate_content(prompt=prompt, model=model)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = _unix_ts()
    text = output.text

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.post("/v1/completions")
async def completions(
    payload: CompletionRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    model = payload.model or os.getenv("OPENAI_COMPAT_DEFAULT_MODEL", "gemini-3-flash")

    if payload.stream:
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = _unix_ts()

        async def stream_events():
            async for chunk in client.generate_content_stream(
                prompt=payload.prompt,
                model=model,
            ):
                delta = chunk.text_delta
                if not delta:
                    continue

                payload_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
            }
            yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    output = await client.generate_content(prompt=payload.prompt, model=model)

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = _unix_ts()

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": output.text, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
