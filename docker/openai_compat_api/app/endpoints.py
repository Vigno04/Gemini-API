import os
import uuid
import base64
import tempfile
import time
import asyncio
from pathlib import Path
from typing import Any
from fastapi import Header, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
import orjson

from gemini_webapi import GeminiClient

from app import app

from models import (
    ChatCompletionRequest,
    CompletionRequest,
    ModerationRequest,
    EmbeddingRequest,
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
from usage_tracker import UsageTracker


class AppState:
    client: GeminiClient | None = None
    uploaded_files: dict[str, dict[str, Any]] = {}
    usage_tracker: UsageTracker = UsageTracker()
    policy_gem_ids: dict[str, str] = {}


state = AppState()


IMAGE_EDIT_MARKER = "{OPENAI_COMPAT_IMAGE_EDIT}"
IMAGE_GENERATION_MARKER = "{OPENAI_COMPAT_IMAGE_GENERATION}"


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
    cleaned = (
        value.replace(IMAGE_EDIT_MARKER, "")
        .replace(IMAGE_GENERATION_MARKER, "")
    )
    return cleaned.strip()


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
                output = await asyncio.wait_for(
                    client.generate_content(
                        prompt=routed_prompt,
                        model=model,
                        files=files_for_generation,
                        temporary=use_temporary_chats,
                        gem=gem_id,
                    ),
                    timeout=generate_timeout,
                )
                _debug_log("Stream request resolved with direct generation mode")

                full_response_text = _strip_image_edit_marker(output.text)
                full_response_thoughts = (
                    _strip_image_edit_marker(output.thoughts) if return_reasoning else ""
                )

                full_content = ""
                if full_response_thoughts:
                    full_content = (
                        f"<thinking>\n{full_response_thoughts}\n</thinking>\n\n"
                        f"{full_response_text}"
                    )
                else:
                    full_content = full_response_text

                first_chunk_content = full_content[:2048]
                rest_content = full_content[2048:]

                payload_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": first_chunk_content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

                chunk_size = 2048
                for i in range(0, len(rest_content), chunk_size):
                    part = rest_content[i:i + chunk_size]
                    if not part:
                        continue
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
                            _debug_log(f"Failed to process image during stream fallback: {e}")
                            if hasattr(img, "url") and getattr(img, "url") and getattr(img, "url").startswith("http"):
                                image_markdown_list.append(f"![{getattr(img, 'alt', 'Generated Image')}]({img.url})")

                    if image_markdown_list:
                        image_string = "\n\n" + "\n\n".join(image_markdown_list)
                        full_content += image_string
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

        return StreamingResponse(
            stream_events(), 
            media_type="text/event-stream",
            background=BackgroundTask(cleanup_files, files) if files else None
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
            full_response_text = ""  # Accumula il testo completo
            full_response_thoughts = ""  # Accumula i pensieri completi
            thought_started = False
            thought_ended = False
            streamed_images = []

            async for chunk in client.generate_content_stream(
                prompt=payload.prompt,
                model=model,
                temporary=use_temporary_chats,
            ):
                if hasattr(chunk, "images") and chunk.images:
                    streamed_images = chunk.images
                    
                delta = chunk.text_delta or ""
                thoughts_delta = chunk.thoughts_delta if return_reasoning else ""
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

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    output = await client.generate_content(prompt=payload.prompt, model=model, temporary=use_temporary_chats)

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = _unix_ts()

    # Combine thoughts and text for thinking models
    thoughts = output.thoughts if return_reasoning else ""
    text = output.text
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


@app.post("/v1/images/generations")
async def create_image(
    payload: ImageGenerationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    _debug_dump_request("POST /v1/images/generations", payload)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    response_format = _validate_image_response_format(payload.response_format)
    use_temporary_chats = _env_flag("OPENAI_COMPAT_USE_TEMPORARY_CHATS", True)

    _debug_log(
        f"Image generation request: prompt='{payload.prompt[:50]}...', model={payload.model}, "
        f"n={payload.n}, response_format={response_format}"
    )

    output = await client.generate_content(
        prompt=payload.prompt,
        model=payload.model,
        temporary=use_temporary_chats,
    )

    if not output.images:
        raise HTTPException(
            status_code=500,
            detail="No image was generated. Try rephrasing your prompt.",
        )

    requested_count = max(1, min(payload.n, len(output.images)))
    data_items = []
    for image in output.images[:requested_count]:
        data_items.append(
            await _image_to_openai_item(
                image,
                response_format=response_format,
                revised_prompt=payload.prompt,
            )
        )

    response_payload = {
        "created": _unix_ts(),
        "data": data_items,
    }
    _debug_dump_response("POST /v1/images/generations", response_payload)
    return response_payload


@app.post("/v1/images/edits")
async def edit_image(
    image: str | UploadFile = Form(...),
    prompt: str | None = Form(None),
    mask: str | UploadFile | None = Form(None),
    model: str = Form("gemini-3-flash"),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
    user: str | None = Form(None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    image_debug: dict[str, Any]
    if not isinstance(image, str):
        image_upload: Any = image
        image_bytes_for_debug = await image_upload.read()
        image_debug = {
            "type": "upload",
            "filename": image_upload.filename,
            "content_type": image_upload.content_type,
            "byte_length": len(image_bytes_for_debug),
        }
        await image_upload.seek(0)
    else:
        image_debug = {
            "type": "string",
            "value": image,
        }

    mask_debug: dict[str, Any] | None = None
    if mask is not None:
        if not isinstance(mask, str):
            mask_upload: Any = mask
            mask_bytes_for_debug = await mask_upload.read()
            mask_debug = {
                "type": "upload",
                "filename": mask_upload.filename,
                "content_type": mask_upload.content_type,
                "byte_length": len(mask_bytes_for_debug),
            }
            await mask_upload.seek(0)
        else:
            mask_debug = {
                "type": "string",
                "value": mask,
            }

    _debug_dump_request(
        "POST /v1/images/edits",
        {
            "prompt": prompt,
            "model": model,
            "n": n,
            "size": size,
            "response_format": response_format,
            "user": user,
            "image": image_debug,
            "mask": mask_debug,
        },
    )

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    response_format = _validate_image_response_format(response_format)
    use_temporary_chats = _env_flag("OPENAI_COMPAT_USE_TEMPORARY_CHATS", True)

    _debug_log(
        f"Image edit request: prompt='{prompt[:50] if prompt else None}...', model={model}, "
        f"n={n}, response_format={response_format}, has_mask={mask is not None}"
    )

    image_bytes, image_ext = await _read_image_payload(image, "image")

    if mask is not None:
        # Validate mask input for better client compatibility, even if not used by Gemini.
        await _read_image_payload(mask, "mask")

    edit_prompt = prompt or "Edit this image"
    temp_image_path = _save_temp_image_file(image_bytes, image_ext)

    try:
        output = await client.generate_content(
            prompt=edit_prompt,
            model=model,
            files=[temp_image_path],
            temporary=use_temporary_chats,
        )
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not output.images:
        raise HTTPException(
            status_code=500,
            detail="No edited image was generated. Try rephrasing your prompt.",
        )

    requested_count = max(1, min(n, len(output.images)))
    data_items = []
    for generated in output.images[:requested_count]:
        data_items.append(
            await _image_to_openai_item(
                generated,
                response_format=response_format,
                revised_prompt=prompt or "",
            )
        )

    response_payload = {
        "created": _unix_ts(),
        "data": data_items,
    }
    _debug_dump_response("POST /v1/images/edits", response_payload)
    return response_payload


@app.post("/v1/images/variations")
async def create_image_variations(
    image: str | UploadFile = Form(...),
    model: str = Form("gemini-3-flash"),
    n: int = Form(1),
    response_format: str = Form("url"),
    size: str = Form("1024x1024"),
    user: str | None = Form(None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    image_debug: dict[str, Any]
    if not isinstance(image, str):
        image_upload: Any = image
        image_bytes_for_debug = await image_upload.read()
        image_debug = {
            "type": "upload",
            "filename": image_upload.filename,
            "content_type": image_upload.content_type,
            "byte_length": len(image_bytes_for_debug),
        }
        await image_upload.seek(0)
    else:
        image_debug = {
            "type": "string",
            "value": image,
        }

    _debug_dump_request(
        "POST /v1/images/variations",
        {
            "model": model,
            "n": n,
            "size": size,
            "response_format": response_format,
            "user": user,
            "image": image_debug,
        },
    )

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    response_format = _validate_image_response_format(response_format)
    use_temporary_chats = _env_flag("OPENAI_COMPAT_USE_TEMPORARY_CHATS", True)

    _debug_log(
        f"Image variations request: model={model}, n={n}, response_format={response_format}"
    )

    image_bytes, image_ext = await _read_image_payload(image, "image")

    variation_prompt = "Create image variations while preserving the main subject and style."
    temp_image_path = _save_temp_image_file(image_bytes, image_ext)

    try:
        output = await client.generate_content(
            prompt=variation_prompt,
            model=model,
            files=[temp_image_path],
            temporary=use_temporary_chats,
        )
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not output.images:
        raise HTTPException(
            status_code=500,
            detail="No image variations were generated.",
        )

    requested_count = max(1, min(n, len(output.images)))
    data_items = []
    for generated in output.images[:requested_count]:
        data_items.append(
            await _image_to_openai_item(
                generated,
                response_format=response_format,
            )
        )

    response_payload = {
        "created": _unix_ts(),
        "data": data_items,
    }
    _debug_dump_response("POST /v1/images/variations", response_payload)
    return response_payload


@app.post("/v1/moderations")
async def moderations(
    payload: ModerationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    # Handle both single string and list of strings
    inputs = [payload.input] if isinstance(payload.input, str) else payload.input

    results = []
    for text in inputs:
        try:
            # Ask Gemini to analyze the content for moderation
            moderation_prompt = f"""Analyze this text for content that might be inappropriate, harmful, or violate safety guidelines.
Respond with only 'safe' or 'unsafe' followed by a brief reason.

Text to analyze: {text}"""

            response = await client.generate_content(
                prompt=moderation_prompt,
                model=payload.model,
                temporary=True
            )

            # Simple heuristic: if response contains "unsafe" or similar, mark as flagged
            response_text = response.text.lower()
            is_flagged = "unsafe" in response_text or "inappropriate" in response_text or "harmful" in response_text

            results.append({
                "flagged": is_flagged,
                "categories": {
                    "hate": "hate" in response_text,
                    "hate/threatening": "threat" in response_text,
                    "self-harm": "self-harm" in response_text or "suicide" in response_text,
                    "sexual": "sexual" in response_text or "explicit" in response_text,
                    "sexual/minors": "minor" in response_text or "child" in response_text,
                    "violence": "violence" in response_text or "violent" in response_text,
                    "violence/graphic": "graphic" in response_text,
                },
                "category_scores": {cat: 0.0 for cat in [
                    "hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence", "violence/graphic"
                ]}
            })

        except Exception:
            # If moderation fails, assume safe
            results.append({
                "flagged": False,
                "categories": {cat: False for cat in [
                    "hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence", "violence/graphic"
                ]},
                "category_scores": {cat: 0.0 for cat in [
                    "hate", "hate/threatening", "self-harm", "sexual", "sexual/minors", "violence", "violence/graphic"
                ]}
            })

    # Track usage for moderation requests (approximate)
    total_moderation_tokens = sum(len(text) // 4 for text in inputs)  # Rough token estimate
    moderation_cost = _estimate_cost(payload.model, total_moderation_tokens, total_moderation_tokens)
    state.usage_tracker.track_request(payload.model, total_moderation_tokens, total_moderation_tokens, moderation_cost)

    return {
        "id": f"modr-{uuid.uuid4().hex}",
        "model": payload.model,
        "results": results
    }


@app.post("/v1/embeddings")
async def embeddings(
    payload: EmbeddingRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    # Gemini doesn't have native embeddings, so we'll return an error
    # In a real implementation, you might use a different service or approximate
    raise HTTPException(
        status_code=501,
        detail="Embeddings are not supported by Gemini. Consider using a different embedding service."
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    # Gemini doesn't support audio transcription natively
    raise HTTPException(
        status_code=501,
        detail="Audio transcription is not supported by Gemini. Consider using a different transcription service."
    )


@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    # Gemini doesn't support audio translation natively
    raise HTTPException(
        status_code=501,
        detail="Audio translation is not supported by Gemini. Consider using a different translation service."
    )


@app.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    # For now, we'll store files temporarily and return file info
    # In a real implementation, you'd upload to cloud storage
    file_id = f"file-{uuid.uuid4().hex}"

    # Read file content
    content = await file.read()

    # Store temporarily (in memory for this demo)
    state.uploaded_files[file_id] = {
        "content": content,
        "filename": file.filename,
        "purpose": purpose,
        "bytes": len(content),
        "created_at": int(time.time())
    }

    return {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": purpose
    }


@app.get("/v1/files")
async def list_files(
    purpose: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    files = []
    for file_id, file_info in state.uploaded_files.items():
        if purpose is None or file_info["purpose"] == purpose:
            files.append({
                "id": file_id,
                "object": "file",
                "bytes": file_info["bytes"],
                "created_at": file_info["created_at"],
                "filename": file_info["filename"],
                "purpose": file_info["purpose"]
            })

    return {
        "object": "list",
        "data": files
    }


@app.post("/v1/assistants")
async def create_assistant(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Assistants API is not supported by Gemini. Consider using chat completions instead.")


@app.get("/v1/assistants")
async def list_assistants(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Assistants API is not supported by Gemini. Consider using chat completions instead.")


@app.post("/v1/threads")
async def create_thread(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Threads API is not supported by Gemini. Consider using chat completions instead.")


@app.post("/v1/threads/{thread_id}/runs")
async def create_thread_run(
    thread_id: str,
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Threads API is not supported by Gemini. Consider using chat completions instead.")


@app.post("/v1/fine_tuning/jobs")
async def create_fine_tuning_job(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Fine-tuning is not supported by Gemini.")


@app.get("/v1/fine_tuning/jobs")
async def list_fine_tuning_jobs(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Fine-tuning is not supported by Gemini.")


@app.post("/v1/batches")
async def create_batch(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Batch API is not supported by Gemini.")


@app.get("/v1/batches")
async def list_batches(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Batch API is not supported by Gemini.")


@app.post("/v1/uploads")
async def create_upload(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Uploads API is not supported by Gemini.")


@app.post("/v1/vector_stores")
async def create_vector_store(
    request: dict[str, Any],
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Vector stores are not supported by Gemini.")


@app.get("/v1/vector_stores")
async def list_vector_stores(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    raise HTTPException(status_code=501, detail="Vector stores are not supported by Gemini.")


@app.get("/v1/organization")
async def get_organization(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    # Return a mock organization response
    return {
        "object": "organization",
        "id": "org-gemini-api",
        "name": "Gemini API",
        "created": 1600000000,
        "is_default": True
    }


@app.get("/v1/usage")
async def get_usage(
    start_date: str | None = None,
    end_date: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    return state.usage_tracker.get_usage(start_date, end_date)


@app.get("/health")
async def health() -> dict[str, Any]:
    return await health_check()