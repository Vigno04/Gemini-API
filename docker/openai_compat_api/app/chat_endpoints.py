import os
import uuid
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Any
from fastapi import Header, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import orjson

from gemini_webapi import GeminiClient

from app import app

from models import (
    ChatCompletionRequest,
    CompletionRequest,
    ImageGenerationRequest,
)
from utils import (
    _require_auth,
    _messages_to_prompt,
    _estimate_tokens,
    _estimate_cost,
    _unix_ts,
    _debug_log,
    _debug_dump_request,
    _debug_dump_response,
)
from state import state


IMAGE_EDIT_MARKER = "{OPENAI_COMPAT_IMAGE_EDIT}"
IMAGE_GENERATION_MARKER = "{OPENAI_COMPAT_IMAGE_GENERATION}"
_COMPAT_MARKERS = (IMAGE_EDIT_MARKER, IMAGE_GENERATION_MARKER)


def _account_status_payload(client: GeminiClient | None) -> dict[str, Any] | None:
    if client is None:
        return None

    account_status = getattr(client, "account_status", None)
    if account_status is None:
        return None

    payload: dict[str, Any] = {}

    code = getattr(account_status, "value", None)
    if code is not None:
        payload["code"] = int(code)

    name = getattr(account_status, "name", None)
    if name:
        payload["name"] = str(name)

    description = getattr(account_status, "description", None)
    if description:
        payload["description"] = str(description)

    if payload:
        return payload

    return {"raw": str(account_status)}

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _effective_stream(request_stream: bool) -> bool:
    return request_stream or _env_flag("OPENAI_COMPAT_FORCE_STREAM", False)


def _selected_policy_gem_id(inline_images: bool) -> str | None:
    key = "inline_images" if inline_images else "no_inline_images"
    gem_id = state.policy_gem_ids.get(key)
    if gem_id:
        return gem_id
    return None


def _generation_timeout_seconds() -> float:
    raw = os.getenv("OPENAI_COMPAT_GENERATION_TIMEOUT_SECONDS", "120").strip()
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    if value <= 0:
        return 120.0
    return value


def _extract_b64_from_data_url(data_url: str) -> str:
    if not data_url.startswith("data:") or "," not in data_url:
        raise ValueError("Invalid data URL")
    return data_url.split(",", 1)[1]


def _is_usable_image_url(value: str | None) -> bool:
    if not value or not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return len(candidate) > len("https://")
    if candidate.startswith("data:"):
        return "," in candidate and len(candidate.split(",", 1)[1]) > 0
    return False


async def _image_to_openai_item(
    img: Any,
    response_format: str,
    revised_prompt: str | None = None,
) -> dict[str, str]:
    image_url = getattr(img, "url", None)
    image_url = image_url if _is_usable_image_url(image_url) else None

    if response_format == "url":
        if image_url:
            item = {"url": image_url}
        else:
            with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                saved_path = await img.save(path=temp_dir)
                image_bytes = Path(saved_path).read_bytes()
                b64_data = base64.b64encode(image_bytes).decode("ascii")
            item = {"url": f"data:image/png;base64,{b64_data}"}
    else:
        b64_data = None
        if image_url and image_url.startswith("data:"):
            b64_data = _extract_b64_from_data_url(image_url)
        else:
            with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                saved_path = await img.save(path=temp_dir)
                image_bytes = Path(saved_path).read_bytes()
                b64_data = base64.b64encode(image_bytes).decode("ascii")

        item = {"b64_json": b64_data}

    if revised_prompt is not None:
        item["revised_prompt"] = revised_prompt

    return item


def _validate_image_response_format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"url", "b64_json"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid response_format. Supported values: url, b64_json",
        )
    return normalized


def _extension_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return ".png"

    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/avif": ".avif",
        "image/heic": ".heic",
        "image/heif": ".heif",
    }
    return mapping.get(mime_type.strip().lower(), ".png")


def _extension_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    suffix = Path(filename).suffix.lower()
    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".avif",
        ".heic",
        ".heif",
    }:
        return ".jpg" if suffix == ".jpeg" else ".tiff" if suffix == ".tif" else suffix
    return None


def _save_temp_image_file(image_bytes: bytes, extension: str) -> Path:
    # Keep a valid image extension so Gemini recognizes this as image input.
    with tempfile.NamedTemporaryFile(
        suffix=extension,
        prefix="openai_compat_input_",
        delete=False,
    ) as tmp:
        tmp.write(image_bytes)
        return Path(tmp.name)


def _strip_image_edit_marker(value: str | None) -> str:
    if not value:
        return ""
    sanitized = value
    for marker in _COMPAT_MARKERS:
        sanitized = sanitized.replace(marker, "")
    return sanitized


class _StreamingMarkerSanitizer:
    def __init__(self) -> None:
        self._buffer = ""
        self._hold = max(len(marker) for marker in _COMPAT_MARKERS) - 1

    def push(self, text: str) -> str:
        if not text:
            return ""
        self._buffer += text
        if len(self._buffer) <= self._hold:
            return ""

        split_at = len(self._buffer) - self._hold
        emit = self._buffer[:split_at]
        self._buffer = self._buffer[split_at:]
        return _strip_image_edit_marker(emit)

    def flush(self) -> str:
        if not self._buffer:
            return ""
        remaining = self._buffer
        self._buffer = ""
        return _strip_image_edit_marker(remaining)


def _prepend_tool_context(prompt: str, tools: list[Any] | None) -> str:
    if not tools:
        return prompt

    tool_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type", "")).strip().lower()
        if tool_type != "function":
            continue
        function_obj = tool.get("function")
        if not isinstance(function_obj, dict):
            continue
        name = str(function_obj.get("name", "")).strip()
        if name:
            tool_names.append(name)

    if not tool_names:
        return prompt

    tools_block = (
        "[OPENAI_COMPAT_TOOLS]\n"
        f"Application-provided tools for this request: {', '.join(tool_names)}\n"
        "If a required capability is available via these tools, you may rely on them."
    )
    return f"system: {tools_block}\n{prompt}"


def _upstream_failure_message() -> str:
    return (
        "I could not complete this request because the upstream Gemini stream was interrupted. "
        "Please retry the request."
    )


def _sse_response(event_generator: Any, background: BackgroundTask | None = None) -> StreamingResponse:
    # Keep SSE frames unbuffered across common proxies and clients.
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers=headers,
        background=background,
    )


async def _read_image_payload(value: str | UploadFile, field_name: str) -> tuple[bytes, str]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise HTTPException(status_code=400, detail=f"{field_name} is empty")

        try:
            if raw.startswith("data:"):
                mime_part = raw.split(",", 1)[0].lower()
                mime_type = mime_part[5:].split(";", 1)[0] if mime_part.startswith("data:") else None
                image_bytes = base64.b64decode(_extract_b64_from_data_url(raw), validate=True)
                return image_bytes, _extension_from_mime(mime_type)
            if raw.startswith("http://") or raw.startswith("https://"):
                raise HTTPException(status_code=400, detail=f"URL {field_name} is not supported")
            # Plain base64 without metadata: default to PNG extension.
            return base64.b64decode(raw, validate=True), ".png"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid {field_name} format") from exc

    data = await value.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} file is empty")
    ext = _extension_from_filename(value.filename) or _extension_from_mime(value.content_type)
    return data, ext


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint that shows the status of the Gemini client."""
    client_status = "initialized" if state.client is not None else "not_initialized"
    response = {
        "status": "healthy" if state.client is not None else "unhealthy",
        "gemini_client": client_status,
        "account_status": _account_status_payload(state.client),
        "timestamp": _unix_ts()
    }
    _debug_log(f"Health check: {response}")
    return response


@app.get("/v1/models")
@app.get("/models")
@app.get("/v/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")

    discovered_models = state.client.list_models() or []
    data: list[dict[str, Any]] = []

    for model in discovered_models:
        if getattr(model, "is_available", True) is False:
            continue

        model_id = getattr(model, "model_name", None) or getattr(model, "name", None)
        if not model_id:
            continue

        model_item: dict[str, Any] = {
            "id": str(model_id),
            "object": "model",
            "created": 0,
            "owned_by": "gemini-webapi",
        }

        display_name = getattr(model, "display_name", None)
        if display_name:
            model_item["display_name"] = str(display_name)

        description = getattr(model, "description", None)
        if description:
            model_item["description"] = str(description)

        data.append(model_item)

    # Keep backward-compatible defaults if dynamic discovery returns nothing.
    if not data:
        data = [
            {"id": "gemini-3-flash", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
            {"id": "gemini-3-pro", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
            {"id": "gemini-3-flash-thinking", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        ]

    _debug_log(f"Models listed: count={len(data)}, account_status={_account_status_payload(state.client)}")
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    authorization: str | None = Header(default=None),
) -> Any:
    _require_auth(authorization)
    _debug_dump_request("POST /v1/chat/completions", payload)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    model = payload.model or os.getenv("OPENAI_COMPAT_DEFAULT_MODEL", "gemini-3-flash")
    prompt, files = _messages_to_prompt(payload.messages)
    
    if isinstance(payload.response_format, dict) and payload.response_format.get("type") == "json_object":
        prompt += "\n\n(IMPORTANT: Please respond with a valid JSON object.)"

    use_temporary_chats = os.getenv("OPENAI_COMPAT_USE_TEMPORARY_CHATS", "true").lower() == "true"
    inline_images = _env_flag("OPENAI_COMPAT_INLINE_IMAGES", True)
    return_reasoning = _env_flag("OPENAI_COMPAT_RETURN_REASONING", True)
    gem_id = _selected_policy_gem_id(inline_images)
    routed_prompt = _prepend_tool_context(prompt, payload.tools)
    files_for_generation: list[Any] | None = files if files else None

    stream_enabled = _effective_stream(payload.stream)

    _debug_log(
        f"Chat completion request: model={model}, messages={len(payload.messages)}, "
        f"stream_requested={payload.stream}, stream_enabled={stream_enabled}, gem_selected={bool(gem_id)}"
    )

    if stream_enabled:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = _unix_ts()
        generate_timeout = _generation_timeout_seconds()

        async def stream_events():
            try:
                # Emit an initial SSE comment to open the stream immediately.
                yield b": stream-start\n\n"

                full_response_text = ""
                full_response_thoughts = ""
                thought_started = False
                thought_ended = False
                streamed_images = []
                role_sent = False
                text_sanitizer = _StreamingMarkerSanitizer()
                thoughts_sanitizer = _StreamingMarkerSanitizer()

                async for chunk in client.generate_content_stream(
                    prompt=routed_prompt,
                    model=model,
                    files=files_for_generation,
                    temporary=use_temporary_chats,
                    gem=gem_id,
                    timeout=generate_timeout,
                ):
                    if hasattr(chunk, "images") and chunk.images:
                        streamed_images = chunk.images

                    delta = text_sanitizer.push(chunk.text_delta or "")
                    thoughts_delta = thoughts_sanitizer.push(chunk.thoughts_delta or "") if return_reasoning else ""

                    if delta:
                        full_response_text += delta
                    if thoughts_delta:
                        full_response_thoughts += thoughts_delta

                    chunk_pieces: list[str] = []
                    if thoughts_delta:
                        if not thought_started:
                            chunk_pieces.append("<thinking>\n")
                            thought_started = True
                        chunk_pieces.append(thoughts_delta)

                    if delta:
                        if thought_started and not thought_ended:
                            chunk_pieces.append("\n</thinking>\n\n")
                            thought_ended = True
                        chunk_pieces.append(delta)

                    content_to_send = "".join(chunk_pieces)
                    if not content_to_send:
                        continue

                    delta_payload: dict[str, Any] = {"content": content_to_send}
                    if not role_sent:
                        delta_payload["role"] = "assistant"
                        role_sent = True

                    payload_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_payload,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

                trailing_thoughts_delta = thoughts_sanitizer.flush() if return_reasoning else ""
                trailing_text_delta = text_sanitizer.flush()
                trailing_pieces: list[str] = []

                if trailing_thoughts_delta:
                    full_response_thoughts += trailing_thoughts_delta
                    if not thought_started:
                        trailing_pieces.append("<thinking>\n")
                        thought_started = True
                    trailing_pieces.append(trailing_thoughts_delta)

                if trailing_text_delta:
                    full_response_text += trailing_text_delta
                    if thought_started and not thought_ended:
                        trailing_pieces.append("\n</thinking>\n\n")
                        thought_ended = True
                    trailing_pieces.append(trailing_text_delta)

                if trailing_pieces:
                    trailing_delta_payload: dict[str, Any] = {"content": "".join(trailing_pieces)}
                    if not role_sent:
                        trailing_delta_payload["role"] = "assistant"
                        role_sent = True

                    trailing_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": trailing_delta_payload,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(trailing_chunk) + b"\n\n"

                if thought_started and not thought_ended:
                    close_thinking_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "\n</thinking>\n\n"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield b"data: " + orjson.dumps(close_thinking_chunk) + b"\n\n"

                if inline_images and streamed_images:
                    image_markdown_list = []
                    for img in streamed_images:
                        try:
                            with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                                saved_path = await img.save(path=temp_dir)
                                image_bytes = Path(saved_path).read_bytes()
                                b64_data = base64.b64encode(image_bytes).decode("ascii")
                                image_markdown_list.append(f"![Generated Image](data:image/png;base64,{b64_data})")
                        except Exception as e:
                            _debug_log(f"Failed to process image during stream fallback: {e}")
                            if hasattr(img, "url") and getattr(img, "url") and getattr(img, "url").startswith("http"):
                                image_markdown_list.append(f"![{getattr(img, 'alt', 'Generated Image')}]({img.url})")

                    if image_markdown_list:
                        image_string = "\n\n" + "\n\n".join(image_markdown_list)
                        full_response_text += image_string
                        for i in range(0, len(image_string), 8192):
                            part = image_string[i:i + 8192]
                            payload_chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": part},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

                full_content = ""
                if full_response_thoughts:
                    full_content = (
                        f"<thinking>\n{full_response_thoughts}\n</thinking>\n\n"
                        f"{full_response_text}"
                    )
                else:
                    full_content = full_response_text
                full_content = _strip_image_edit_marker(full_content)

                prompt_tokens = _estimate_tokens(prompt)
                completion_tokens = _estimate_tokens(full_content)
                total_tokens = prompt_tokens + completion_tokens
                reasoning_tokens = _estimate_tokens(full_response_thoughts) if full_response_thoughts else 0

                cost = _estimate_cost(model, prompt_tokens, completion_tokens)
                state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "completion_tokens_details": {
                            "reasoning_tokens": reasoning_tokens
                        }
                    },
                }
                _debug_dump_response(
                    "POST /v1/chat/completions (stream final)",
                    {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": full_content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": final_chunk["usage"],
                    },
                )
                yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"
                return
            except Exception as exc:
                _debug_log(f"Upstream generation failed during stream response: {type(exc).__name__}: {exc}")
                failure_text = _upstream_failure_message()

                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": failure_text},
                            "finish_reason": None,
                        }
                    ],
                }
                yield b"data: " + orjson.dumps(first_chunk) + b"\n\n"

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": _estimate_tokens(prompt),
                        "completion_tokens": _estimate_tokens(failure_text),
                        "total_tokens": _estimate_tokens(prompt) + _estimate_tokens(failure_text),
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                }
                yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
                yield b"data: [DONE]\n\n"
                return

        def cleanup_files(file_paths):
            for p in file_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        return _sse_response(
            stream_events(),
            background=BackgroundTask(cleanup_files, files) if files else None,
        )

    try:
        output = await asyncio.wait_for(
            client.generate_content(
                prompt=routed_prompt,
                model=model,
                files=files_for_generation,
                temporary=use_temporary_chats,
                gem=gem_id,
            ),
            timeout=_generation_timeout_seconds(),
        )
        _debug_log("Non-stream request resolved with direct generation mode")
    except Exception as exc:
        _debug_log(f"Upstream generation failed during non-stream response: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=502, detail=_upstream_failure_message()) from exc
    finally:
        for fpath in files:
            try:
                os.remove(fpath)
            except Exception:
                pass

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = _unix_ts()

    # Combine thoughts and text for thinking models
    thoughts = _strip_image_edit_marker(output.thoughts) if return_reasoning else ""
    text = _strip_image_edit_marker(output.text)
    full_content = ""

    if thoughts:
        full_content = f"<thinking>\n{thoughts}\n</thinking>\n\n{text}"
    else:
        full_content = text

    # Add generated images to the response if any
    if inline_images and output.images:
        image_markdown_list = []
        for img in output.images:
            try:
                with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                    saved_path = await img.save(path=temp_dir)
                    image_bytes = Path(saved_path).read_bytes()
                    b64_data = base64.b64encode(image_bytes).decode("ascii")
                    image_markdown_list.append(f"![Generated Image](data:image/png;base64,{b64_data})")
            except Exception as e:
                _debug_log(f"Failed to process image: {e}")
                if hasattr(img, "url") and getattr(img, "url") and getattr(img, "url").startswith("http"):
                    image_markdown_list.append(f"![{getattr(img, 'alt', 'Generated Image')}]({img.url})")
                    
        if image_markdown_list:
            full_content += "\n\n" + "\n\n".join(image_markdown_list)

    # Estimate token usage
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(full_content)
    total_tokens = prompt_tokens + completion_tokens
    reasoning_tokens = _estimate_tokens(thoughts) if thoughts else 0

    # Track usage
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

    _debug_log(f"Chat completion response: id={completion_id}, tokens={total_tokens}, images={len(output.images) if hasattr(output, 'images') else 0}")

    response_payload = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "completion_tokens_details": {
                "reasoning_tokens": reasoning_tokens
            }
        },
    }
    _debug_dump_response("POST /v1/chat/completions", response_payload)
    return response_payload


@app.post("/v1/completions")
async def completions(
    payload: CompletionRequest,
    authorization: str | None = Header(default=None),
) -> Any:
    _require_auth(authorization)
    _debug_dump_request("POST /v1/completions", payload)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    model = payload.model or os.getenv("OPENAI_COMPAT_DEFAULT_MODEL", "gemini-3-flash")
    use_temporary_chats = os.getenv("OPENAI_COMPAT_USE_TEMPORARY_CHATS", "true").lower() == "true"
    inline_images = _env_flag("OPENAI_COMPAT_INLINE_IMAGES", True)
    return_reasoning = _env_flag("OPENAI_COMPAT_RETURN_REASONING", True)

    stream_enabled = _effective_stream(payload.stream)

    _debug_log(
        f"Completion request: model={model}, stream_requested={payload.stream}, "
        f"stream_enabled={stream_enabled}"
    )

    if stream_enabled:
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = _unix_ts()

        async def stream_events():
            # Emit an initial SSE comment to open the stream immediately.
            yield b": stream-start\n\n"

            full_response_text = ""  # Accumula il testo completo
            full_response_thoughts = ""  # Accumula i pensieri completi
            thought_started = False
            thought_ended = False
            streamed_images = []
            text_sanitizer = _StreamingMarkerSanitizer()
            thoughts_sanitizer = _StreamingMarkerSanitizer()

            async for chunk in client.generate_content_stream(
                prompt=payload.prompt,
                model=model,
                temporary=use_temporary_chats,
            ):
                if hasattr(chunk, "images") and chunk.images:
                    streamed_images = chunk.images
                    
                delta = text_sanitizer.push(chunk.text_delta or "")
                thoughts_delta = thoughts_sanitizer.push(chunk.thoughts_delta or "") if return_reasoning else ""
                if not thoughts_delta:
                    thoughts_delta = ""

                content_to_send = ""

                if thoughts_delta:
                    if not thought_started:
                        content_to_send += "<thinking>\n"
                        thought_started = True
                    content_to_send += thoughts_delta

                if delta:
                    if thought_started and not thought_ended:
                        content_to_send += "\n</thinking>\n\n"
                        thought_ended = True
                    content_to_send += delta

                if delta:
                    full_response_text += delta  # Accumula testo
                if thoughts_delta:
                    full_response_thoughts += thoughts_delta  # Accumula pensieri

                if content_to_send:
                    payload_chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": content_to_send, "finish_reason": None}],
                    }
                    yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            trailing_thoughts_delta = thoughts_sanitizer.flush() if return_reasoning else ""
            trailing_text_delta = text_sanitizer.flush()
            trailing_content = ""

            if trailing_thoughts_delta:
                if not thought_started:
                    trailing_content += "<thinking>\n"
                    thought_started = True
                trailing_content += trailing_thoughts_delta
                full_response_thoughts += trailing_thoughts_delta

            if trailing_text_delta:
                if thought_started and not thought_ended:
                    trailing_content += "\n</thinking>\n\n"
                    thought_ended = True
                trailing_content += trailing_text_delta
                full_response_text += trailing_text_delta

            if trailing_content:
                payload_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": trailing_content, "finish_reason": None}],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            if thought_started and not thought_ended:
                payload_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": "\n</thinking>\n\n", "finish_reason": None}],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            # Combina pensieri e testo per il contenuto finale
            full_content = ""
            if full_response_thoughts:
                full_content = f"<thinking>\n{full_response_thoughts}\n</thinking>\n\n{full_response_text}"
            else:
                full_content = full_response_text
            full_content = _strip_image_edit_marker(full_content)

            if inline_images and streamed_images:
                image_markdown_list = []
                for img in streamed_images:
                    try:
                        with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                            saved_path = await img.save(path=temp_dir)
                            image_bytes = Path(saved_path).read_bytes()
                            b64_data = base64.b64encode(image_bytes).decode("ascii")
                            image_markdown_list.append(f"![Generated Image](data:image/png;base64,{b64_data})")
                    except Exception as e:
                        if hasattr(img, "url") and getattr(img, "url") and getattr(img, "url").startswith("http"):
                            image_markdown_list.append(f"![{getattr(img, 'alt', 'Generated Image')}]({img.url})")
                
                if image_markdown_list:
                    image_string = "\n\n" + "\n\n".join(image_markdown_list)
                    full_response_text += image_string
                    full_content += image_string
                    
                    # Split the large base64 string into smaller chunks for SSE delivery
                    chunk_size = 8192
                    for i in range(0, len(image_string), chunk_size):
                        part = image_string[i:i+chunk_size]
                        payload_chunk = {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "text": part, "finish_reason": None}],
                        }
                        yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            # Chunk finale con token calcolati
            prompt_tokens = _estimate_tokens(payload.prompt)
            completion_tokens = _estimate_tokens(full_content)
            total_tokens = prompt_tokens + completion_tokens

            # Track usage
            cost = _estimate_cost(model, prompt_tokens, completion_tokens)
            state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

            final_chunk = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            _debug_dump_response(
                "POST /v1/completions (stream final)",
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": full_content, "finish_reason": "stop"}],
                    "usage": final_chunk["usage"],
                },
            )
            yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _sse_response(stream_events())

    output = await client.generate_content(prompt=payload.prompt, model=model, temporary=use_temporary_chats)

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = _unix_ts()

    # Combine thoughts and text for thinking models
    thoughts = _strip_image_edit_marker(output.thoughts) if return_reasoning else ""
    text = _strip_image_edit_marker(output.text)
    full_content = ""

    if thoughts:
        full_content = f"<thinking>\n{thoughts}\n</thinking>\n\n{text}"
    else:
        full_content = text
    full_content = _strip_image_edit_marker(full_content)

    # Add generated images to the response if any
    if inline_images and output.images:
        image_markdown_list = []
        for img in output.images:
            try:
                with tempfile.TemporaryDirectory(prefix="openai_compat_img_") as temp_dir:
                    saved_path = await img.save(path=temp_dir)
                    image_bytes = Path(saved_path).read_bytes()
                    b64_data = base64.b64encode(image_bytes).decode("ascii")
                    image_markdown_list.append(f"![Generated Image](data:image/png;base64,{b64_data})")
            except Exception as e:
                if hasattr(img, "url") and getattr(img, "url") and getattr(img, "url").startswith("http"):
                    image_markdown_list.append(f"![{getattr(img, 'alt', 'Generated Image')}]({img.url})")
                    
        if image_markdown_list:
            full_content += "\n\n" + "\n\n".join(image_markdown_list)

    # Estimate token usage
    prompt_tokens = _estimate_tokens(payload.prompt)
    completion_tokens = _estimate_tokens(full_content)
    total_tokens = prompt_tokens + completion_tokens

    # Track usage
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

    response_payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": full_content, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    _debug_dump_response("POST /v1/completions", response_payload)
    return response_payload


@app.get("/health")
async def health() -> dict[str, Any]:
    return await health_check()