import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import File, Form, Header, HTTPException, UploadFile

from app import app
from models import ImageGenerationRequest
from state import state
from utils import _debug_dump_request, _debug_dump_response, _debug_log, _require_auth, _unix_ts


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    image: str | UploadFile = File(...),
    prompt: str | None = Form(None),
    mask: str | UploadFile | None = File(None),
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
    image: str | UploadFile = File(...),
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
