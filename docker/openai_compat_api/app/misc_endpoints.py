import time
import uuid
from typing import Any

from fastapi import File, Form, Header, HTTPException, UploadFile

from app import app
from models import EmbeddingRequest, ModerationRequest
from state import state
from utils import _estimate_cost, _require_auth


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
                temporary=True,
            )

            # Simple heuristic: if response contains "unsafe" or similar, mark as flagged
            response_text = response.text.lower()
            is_flagged = "unsafe" in response_text or "inappropriate" in response_text or "harmful" in response_text

            results.append(
                {
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
                    "category_scores": {
                        cat: 0.0
                        for cat in [
                            "hate",
                            "hate/threatening",
                            "self-harm",
                            "sexual",
                            "sexual/minors",
                            "violence",
                            "violence/graphic",
                        ]
                    },
                }
            )

        except Exception:
            # If moderation fails, assume safe
            results.append(
                {
                    "flagged": False,
                    "categories": {
                        cat: False
                        for cat in [
                            "hate",
                            "hate/threatening",
                            "self-harm",
                            "sexual",
                            "sexual/minors",
                            "violence",
                            "violence/graphic",
                        ]
                    },
                    "category_scores": {
                        cat: 0.0
                        for cat in [
                            "hate",
                            "hate/threatening",
                            "self-harm",
                            "sexual",
                            "sexual/minors",
                            "violence",
                            "violence/graphic",
                        ]
                    },
                }
            )

    # Track usage for moderation requests (approximate)
    total_moderation_tokens = sum(len(text) // 4 for text in inputs)  # Rough token estimate
    moderation_cost = _estimate_cost(payload.model, total_moderation_tokens, total_moderation_tokens)
    state.usage_tracker.track_request(payload.model, total_moderation_tokens, total_moderation_tokens, moderation_cost)

    return {
        "id": f"modr-{uuid.uuid4().hex}",
        "model": payload.model,
        "results": results,
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
        detail="Embeddings are not supported by Gemini. Consider using a different embedding service.",
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
        detail="Audio transcription is not supported by Gemini. Consider using a different transcription service.",
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
        detail="Audio translation is not supported by Gemini. Consider using a different translation service.",
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
        "created_at": int(time.time()),
    }

    return {
        "id": file_id,
        "object": "file",
        "bytes": len(content),
        "created_at": int(time.time()),
        "filename": file.filename,
        "purpose": purpose,
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
            files.append(
                {
                    "id": file_id,
                    "object": "file",
                    "bytes": file_info["bytes"],
                    "created_at": file_info["created_at"],
                    "filename": file_info["filename"],
                    "purpose": file_info["purpose"],
                }
            )

    return {
        "object": "list",
        "data": files,
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
        "is_default": True,
    }


@app.get("/v1/usage")
async def get_usage(
    start_date: str | None = None,
    end_date: str | None = None,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    return state.usage_tracker.get_usage(start_date, end_date)
