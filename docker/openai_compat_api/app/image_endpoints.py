import base64
import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, Request, UploadFile

from app import app
from models import ImageGenerationRequest
from state import state
from utils import _debug_dump_request, _debug_dump_response, _debug_log, _require_auth, _unix_ts


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _generation_timeout_seconds() -> float:
    raw = os.getenv("OPENAI_COMPAT_GENERATION_TIMEOUT_SECONDS", "120").strip()
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    if value <= 0:
        return 120.0
    return value


def _upstream_failure_message() -> str:
    return (
        "Upstream Gemini image generation was interrupted or returned an incomplete response. "
        "Please retry the request."
    )


def _collect_output_images(output: Any) -> list[Any]:
    # Prefer generated images from any candidate, not only the currently chosen one.
    candidates = list(getattr(output, "candidates", []) or [])
    chosen = getattr(output, "chosen", None)

    generated_pool: list[Any] = []
    generic_pool: list[Any] = []

    def _push_unique(pool: list[Any], items: list[Any]) -> None:
        seen = {id(item) for item in pool}
        for item in items:
            if id(item) not in seen:
                pool.append(item)
                seen.add(id(item))

    for idx, candidate in enumerate(candidates):
        cand_generated = list(getattr(candidate, "generated_images", []) or [])
        cand_images = list(getattr(candidate, "images", []) or [])
        _debug_log(
            "Image candidate summary: "
            f"index={idx}, chosen={idx == chosen}, generated_images={len(cand_generated)}, images={len(cand_images)}"
        )
        _push_unique(generated_pool, cand_generated)
        _push_unique(generic_pool, cand_images)

    if generated_pool:
        return generated_pool
    if generic_pool:
        return generic_pool

    # Backward-compatible fallback for outputs without candidates.
    return list(getattr(output, "images", []) or [])


def _build_image_edit_prompt(user_prompt: str | None) -> str:
    base = (user_prompt or "Edit this image").strip() or "Edit this image"
    return (
        "Edit the attached image according to the request below. "
        "You must return an edited image, not a text-only response.\n"
        f"Request: {base}"
    )


def _debug_candidate_text_preview(output: Any, label: str) -> None:
    candidates = list(getattr(output, "candidates", []) or [])
    for idx, candidate in enumerate(candidates):
        text = str(getattr(candidate, "text", "") or "")
        preview = text[:120] + ("..." if len(text) > 120 else "")
        _debug_log(
            f"{label} candidate text: index={idx}, length={len(text)}, preview={preview!r}"
        )


def _extract_data_urls_from_output_text(output: Any) -> list[str]:
    candidates = list(getattr(output, "candidates", []) or [])
    if not candidates:
        return []

    data_url_pattern = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+")
    results: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        text = str(getattr(candidate, "text", "") or "")
        for match in data_url_pattern.findall(text):
            if match not in seen:
                seen.add(match)
                results.append(match)

    return results


def _openai_item_from_data_url(
    data_url: str,
    response_format: str,
    revised_prompt: str | None = None,
) -> dict[str, str]:
    if response_format == "url":
        item: dict[str, str] = {"url": data_url}
    else:
        item = {"b64_json": _extract_b64_from_data_url(data_url)}

    if revised_prompt is not None:
        item["revised_prompt"] = revised_prompt

    return item


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


def _is_upload_file_like(value: Any) -> bool:
    # Starlette and FastAPI upload objects both expose these attributes.
    return (
        value is not None
        and hasattr(value, "read")
        and hasattr(value, "seek")
        and hasattr(value, "filename")
    )


def _summarize_payload_value(value: Any) -> Any:
    if isinstance(value, str):
        preview = value[:120]
        if len(value) > 120:
            preview += "..."
        return {
            "type": "str",
            "length": len(value),
            "starts_with": value[:24],
            "preview": preview,
        }

    if _is_upload_file_like(value):
        return {
            "type": type(value).__name__,
            "filename": getattr(value, "filename", None),
            "content_type": getattr(value, "content_type", None),
        }

    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "items": [_summarize_payload_value(item) for item in value[:2]],
        }

    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": sorted(list(value.keys())),
        }

    return {
        "type": type(value).__name__ if value is not None else None,
        "value": value,
    }


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _normalize_media_field(value: Any) -> Any:
    if isinstance(value, list):
        return _normalize_media_field(value[0]) if value else None

    if isinstance(value, dict):
        if "url" in value:
            return value.get("url")
        if "b64_json" in value:
            return value.get("b64_json")
        nested = value.get("image_url")
        if isinstance(nested, dict) and "url" in nested:
            return nested.get("url")
        return None

    return value


def _parse_int_field(raw_value: Any, field_name: str, default: int) -> int:
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}") from exc


async def _parse_image_request_body(request: Request) -> dict[str, Any]:
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")

        image_value = _first_present(
            body,
            ["image", "image[]", "images", "image_url", "input_image", "file", "file[]"],
        )
        mask_value = _first_present(body, ["mask", "mask[]", "mask_image"])

        return {
            "image": _normalize_media_field(image_value),
            "prompt": body.get("prompt"),
            "mask": _normalize_media_field(mask_value),
            "model": body.get("model") or "gemini-3-flash",
            "n": _parse_int_field(body.get("n"), "n", 1),
            "size": body.get("size") or "1024x1024",
            "response_format": body.get("response_format") or "url",
            "user": body.get("user"),
            "_request_meta": {
                "content_type": content_type,
                "keys": sorted(list(body.keys())),
                "field_summaries": {
                    key: _summarize_payload_value(body.get(key))
                    for key in sorted(list(body.keys()))
                },
            },
        }

    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid form payload") from exc

    image_value = (
        form.get("image")
        or form.get("image[]")
        or form.get("images")
        or form.get("image_url")
        or form.get("input_image")
        or form.get("file")
        or form.get("file[]")
    )
    mask_value = form.get("mask") or form.get("mask[]") or form.get("mask_image")

    return {
        "image": _normalize_media_field(image_value),
        "prompt": form.get("prompt"),
        "mask": _normalize_media_field(mask_value),
        "model": form.get("model") or "gemini-3-flash",
        "n": _parse_int_field(form.get("n"), "n", 1),
        "size": form.get("size") or "1024x1024",
        "response_format": form.get("response_format") or "url",
        "user": form.get("user"),
        "_request_meta": {
            "content_type": content_type,
            "keys": sorted(list(form.keys())),
            "field_summaries": {
                key: _summarize_payload_value(form.get(key))
                for key in sorted(list(form.keys()))
            },
        },
    }


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

    all_images = _collect_output_images(output)

    if not all_images:
        raise HTTPException(
            status_code=500,
            detail="No image was generated. Try rephrasing your prompt.",
        )

    requested_count = max(1, min(payload.n, len(all_images)))
    data_items = []
    for image in all_images[:requested_count]:
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
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    parsed = await _parse_image_request_body(request)
    image = parsed.get("image")
    prompt = parsed.get("prompt")
    mask = parsed.get("mask")
    model = str(parsed.get("model") or "gemini-3-flash")
    n = _parse_int_field(parsed.get("n"), "n", 1)
    size = str(parsed.get("size") or "1024x1024")
    response_format = str(parsed.get("response_format") or "url")
    user = parsed.get("user")
    request_meta = parsed.get("_request_meta") or {}

    _debug_dump_request(
        "POST /v1/images/edits (parsed)",
        {
            "content_type": request_meta.get("content_type"),
            "keys": request_meta.get("keys"),
            "field_summaries": request_meta.get("field_summaries"),
            "has_image": image is not None,
            "image_type": type(image).__name__ if image is not None else None,
            "image_summary": _summarize_payload_value(image),
            "has_mask": mask is not None,
            "mask_type": type(mask).__name__ if mask is not None else None,
            "mask_summary": _summarize_payload_value(mask),
        },
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Missing required field: image")
    if not isinstance(image, str) and not _is_upload_file_like(image):
        raise HTTPException(status_code=400, detail="Invalid image field")
    if mask is not None and not isinstance(mask, str) and not _is_upload_file_like(mask):
        raise HTTPException(status_code=400, detail="Invalid mask field")

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
    generation_timeout = _generation_timeout_seconds()

    _debug_log(
        f"Image edit request: prompt='{prompt[:50] if prompt else None}...', model={model}, "
        f"n={n}, response_format={response_format}, has_mask={mask is not None}"
    )

    image_bytes, image_ext = await _read_image_payload(image, "image")

    if mask is not None:
        # Validate mask input for better client compatibility, even if not used by Gemini.
        await _read_image_payload(mask, "mask")

    edit_prompt = _build_image_edit_prompt(prompt)
    temp_image_path = _save_temp_image_file(image_bytes, image_ext)
    all_images: list[Any] = []

    try:
        output = None
        direct_exc: Exception | None = None

        try:
            output = await asyncio.wait_for(
                client.generate_content(
                    prompt=edit_prompt,
                    model=model,
                    files=[temp_image_path],
                    temporary=use_temporary_chats,
                ),
                timeout=generation_timeout,
            )
            _debug_log("Image edit direct generation succeeded")
        except Exception as exc:
            direct_exc = exc
            _debug_log(
                f"Image edit direct generation failed: {type(exc).__name__}: {exc}. "
                "Trying two-step chat fallback."
            )

        direct_images = _collect_output_images(output) if output is not None else []
        if output is not None and not direct_images:
            _debug_candidate_text_preview(output, "Image edit direct")

        if output is None or not direct_images:
            try:
                chat = client.start_chat(model=model)
                await asyncio.wait_for(
                    chat.send_message(
                        "Analyze this image for editing.",
                        files=[temp_image_path],
                        temporary=use_temporary_chats,
                    ),
                    timeout=generation_timeout,
                )
                output = await asyncio.wait_for(
                    chat.send_message(
                        edit_prompt,
                        temporary=use_temporary_chats,
                    ),
                    timeout=generation_timeout,
                )
                _debug_log("Image edit chat fallback succeeded")
                fallback_images = _collect_output_images(output)
                if not fallback_images:
                    _debug_candidate_text_preview(output, "Image edit chat fallback")
            except Exception as fallback_exc:
                _debug_log(
                    f"Image edit chat fallback failed: {type(fallback_exc).__name__}: {fallback_exc}"
                )
                if direct_exc is not None:
                    _debug_log(
                        f"Image edit direct failure that triggered fallback: {type(direct_exc).__name__}: {direct_exc}"
                    )
                raise HTTPException(status_code=502, detail=_upstream_failure_message()) from fallback_exc
        all_images = _collect_output_images(output)

        if not all_images:
            _debug_log("Image edit produced no images; trying final forced-image retry")
            forced_prompt = (
                "Generate only one edited image for the attached file. "
                "Do not return explanation text. "
                f"Apply this edit: {(prompt or 'Edit this image').strip()}"
            )
            try:
                output = await asyncio.wait_for(
                    client.generate_content(
                        prompt=forced_prompt,
                        model=model,
                        files=[temp_image_path],
                        temporary=use_temporary_chats,
                    ),
                    timeout=generation_timeout,
                )
                all_images = _collect_output_images(output)
                if not all_images:
                    _debug_candidate_text_preview(output, "Image edit forced retry")
            except Exception as force_exc:
                _debug_log(
                    f"Image edit forced-image retry failed: {type(force_exc).__name__}: {force_exc}"
                )
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not all_images:
        data_urls = _extract_data_urls_from_output_text(output)
        if data_urls:
            _debug_log(f"Image edit fallback recovered inline data URLs: count={len(data_urls)}")
            requested_count = max(1, min(n, len(data_urls)))
            data_items = [
                _openai_item_from_data_url(
                    data_url,
                    response_format=response_format,
                    revised_prompt=prompt or "",
                )
                for data_url in data_urls[:requested_count]
            ]
            response_payload = {
                "created": _unix_ts(),
                "data": data_items,
            }
            _debug_dump_response("POST /v1/images/edits", response_payload)
            return response_payload

        raise HTTPException(
            status_code=502,
            detail="No edited image was generated. Try rephrasing your prompt.",
        )

    requested_count = max(1, min(n, len(all_images)))
    data_items = []
    for generated in all_images[:requested_count]:
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
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    parsed = await _parse_image_request_body(request)
    image = parsed.get("image")
    model = str(parsed.get("model") or "gemini-3-flash")
    n = _parse_int_field(parsed.get("n"), "n", 1)
    response_format = str(parsed.get("response_format") or "url")
    size = str(parsed.get("size") or "1024x1024")
    user = parsed.get("user")
    request_meta = parsed.get("_request_meta") or {}

    _debug_dump_request(
        "POST /v1/images/variations (parsed)",
        {
            "content_type": request_meta.get("content_type"),
            "keys": request_meta.get("keys"),
            "field_summaries": request_meta.get("field_summaries"),
            "has_image": image is not None,
            "image_type": type(image).__name__ if image is not None else None,
            "image_summary": _summarize_payload_value(image),
        },
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Missing required field: image")
    if not isinstance(image, str) and not _is_upload_file_like(image):
        raise HTTPException(status_code=400, detail="Invalid image field")

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
    generation_timeout = _generation_timeout_seconds()

    _debug_log(
        f"Image variations request: model={model}, n={n}, response_format={response_format}"
    )

    image_bytes, image_ext = await _read_image_payload(image, "image")

    variation_prompt = "Create image variations while preserving the main subject and style."
    temp_image_path = _save_temp_image_file(image_bytes, image_ext)

    try:
        output = await asyncio.wait_for(
            client.generate_content(
                prompt=variation_prompt,
                model=model,
                files=[temp_image_path],
                temporary=use_temporary_chats,
            ),
            timeout=generation_timeout,
        )
    finally:
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass

    all_images = _collect_output_images(output)

    if not all_images:
        data_urls = _extract_data_urls_from_output_text(output)
        if data_urls:
            _debug_log(f"Image variations fallback recovered inline data URLs: count={len(data_urls)}")
            requested_count = max(1, min(n, len(data_urls)))
            data_items = [
                _openai_item_from_data_url(
                    data_url,
                    response_format=response_format,
                )
                for data_url in data_urls[:requested_count]
            ]
            response_payload = {
                "created": _unix_ts(),
                "data": data_items,
            }
            _debug_dump_response("POST /v1/images/variations", response_payload)
            return response_payload

        raise HTTPException(
            status_code=502,
            detail="No image variations were generated.",
        )

    requested_count = max(1, min(n, len(all_images)))
    data_items = []
    for generated in all_images[:requested_count]:
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
