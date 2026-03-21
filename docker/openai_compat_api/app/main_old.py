import os
import time
import uuid
import json
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import orjson
import tiktoken

from gemini_webapi import GeminiClient


class UsageTracker:
    def __init__(self, storage_file: str = "usage_data.json"):
        self.storage_file = Path(storage_file)
        self.usage_data = self._load_usage_data()
    
    def _load_usage_data(self) -> dict[str, Any]:
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"daily_usage": {}, "total_usage": {}}
    
    def _save_usage_data(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)
    
    def track_request(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
        today = time.strftime("%Y-%m-%d")
        
        if today not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][today] = {}
        
        if model not in self.usage_data["daily_usage"][today]:
            self.usage_data["daily_usage"][today][model] = {
                "total_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "requests": 0,
                "cost": 0.0
            }
        
        daily_model = self.usage_data["daily_usage"][today][model]
        daily_model["total_tokens"] += prompt_tokens + completion_tokens
        daily_model["completion_tokens"] += completion_tokens
        daily_model["prompt_tokens"] += prompt_tokens
        daily_model["requests"] += 1
        daily_model["cost"] += cost
        
        # Update totals
        if model not in self.usage_data["total_usage"]:
            self.usage_data["total_usage"][model] = {
                "total_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "requests": 0,
                "cost": 0.0
            }
        
        total_model = self.usage_data["total_usage"][model]
        total_model["total_tokens"] += prompt_tokens + completion_tokens
        total_model["completion_tokens"] += completion_tokens
        total_model["prompt_tokens"] += prompt_tokens
        total_model["requests"] += 1
        total_model["cost"] += cost
        
        self._save_usage_data()
    
    def get_usage(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        data = []
        total_usage = 0
        total_cost = 0.0
        
        for date, models in self.usage_data["daily_usage"].items():
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue
                
            for model, stats in models.items():
                data.append({
                    "object": "usage",
                    "date": date,
                    "model": model,
                    "total_tokens": stats["total_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "requests": stats["requests"],
                    "cost": {
                        "amount": stats["cost"],
                        "currency": "usd"
                    }
                })
                total_usage += stats["total_tokens"]
                total_cost += stats["cost"]
        
        return {
            "object": "list",
            "data": data,
            "total_usage": total_usage,
            "total_cost": {
                "amount": total_cost,
                "currency": "usd"
            }
        }


def _unix_ts() -> int:
    return int(time.time())


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost estimation disabled - always return 0"""
    return 0.0


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ModerationRequest(BaseModel):
    input: str | list[str]
    model: str = "text-moderation-latest"


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "text-embedding-ada-002"
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    user: str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    gem: str | None = None  # Gemini Gem ID for system prompt


class ModerationRequest(BaseModel):
    input: str | list[str]
    model: str = "gemini-3-flash"  # Model to use for moderation


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "gemini-3-flash"  # Default Gemini model for image generation
    n: int = 1  # Number of images (Gemini typically generates 1)
    size: str = "1024x1024"  # Ignored for Gemini, but kept for compatibility
    quality: str = "standard"  # Ignored for Gemini
    style: str = "natural"  # Ignored for Gemini


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken with GPT-3.5-turbo encoding as approximation."""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimation of ~4 characters per token
        return len(text) // 4


class AppState:
    client: GeminiClient | None = None
    uploaded_files: dict[str, dict[str, Any]] = {}
    usage_tracker: UsageTracker = UsageTracker()


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

    # Expanded list of available Gemini models
    data = [
        {"id": "gemini-3-flash", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-3-pro", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-3-flash-thinking", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-2-flash", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-2-pro", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-1-5-flash", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
        {"id": "gemini-1-5-pro", "object": "model", "created": 0, "owned_by": "gemini-webapi"},
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
    use_temporary_chats = os.getenv("OPENAI_COMPAT_USE_TEMPORARY_CHATS", "true").lower() == "true"

    if payload.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = _unix_ts()

        async def stream_events():
            full_response_text = ""  # Accumula il testo completo
            full_response_thoughts = ""  # Accumula i pensieri completi
            async for chunk in client.generate_content_stream(
                prompt=prompt,
                model=model,
                temporary=use_temporary_chats,
            ):
                delta = chunk.text_delta
                thoughts_delta = chunk.thoughts_delta or ""
                
                if delta:
                    full_response_text += delta  # Accumula testo
                if thoughts_delta:
                    full_response_thoughts += thoughts_delta  # Accumula pensieri

                payload_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta} if delta else {},
                            "finish_reason": None,
                        }
                    ],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            # Combina pensieri e testo per il contenuto finale
            full_content = ""
            if full_response_thoughts:
                full_content = f"<thinking>\n{full_response_thoughts}\n</thinking>\n\n{full_response_text}"
            else:
                full_content = full_response_text
            
            # Get images from a quick non-stream call (to avoid complexity in streaming)
            try:
                quick_output = await client.generate_content(prompt=prompt, model=model, temporary=use_temporary_chats)
                if quick_output.images:
                    image_markdown = "\n\n".join([
                        f"![Generated Image]({img.url})" if hasattr(img, 'url') and img.url else 
                        f"![Generated Image](data:{getattr(img, 'mime_type', 'image/png')};base64,{getattr(img, 'data', '')})"
                        for img in quick_output.images
                    ])
                    full_content += f"\n\n{image_markdown}"
            except:
                pass  # Ignore errors in image fetching for streaming

            # Chunk finale con token calcolati
            prompt_tokens = _estimate_tokens(prompt)
            completion_tokens = _estimate_tokens(full_content)
            total_tokens = prompt_tokens + completion_tokens

            # Track usage
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
                },
            }
            yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    output = await client.generate_content(prompt=prompt, model=model, temporary=use_temporary_chats)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = _unix_ts()
    
    # Combine thoughts and text for thinking models
    thoughts = output.thoughts
    text = output.text
    full_content = ""
    
    if thoughts:
        full_content = f"<thinking>\n{thoughts}\n</thinking>\n\n{text}"
    else:
        full_content = text
    
    # Add generated images to the response if any
    if output.images:
        image_markdown = "\n\n".join([
            f"![Generated Image]({img.url})" if hasattr(img, 'url') and img.url else 
            f"![Generated Image](data:{getattr(img, 'mime_type', 'image/png')};base64,{getattr(img, 'data', '')})"
            for img in output.images
        ])
        full_content += f"\n\n{image_markdown}"

    # Estimate token usage
    prompt_tokens = _estimate_tokens(prompt)
    completion_tokens = _estimate_tokens(full_content)
    total_tokens = prompt_tokens + completion_tokens

    # Track usage
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

    return {
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
    use_temporary_chats = os.getenv("OPENAI_COMPAT_USE_TEMPORARY_CHATS", "true").lower() == "true"
    
    if payload.stream:
        completion_id = f"cmpl-{uuid.uuid4().hex}"
        created = _unix_ts()

        async def stream_events():
            full_response_text = ""  # Accumula il testo completo
            full_response_thoughts = ""  # Accumula i pensieri completi
            async for chunk in client.generate_content_stream(
                prompt=payload.prompt,
                model=model,
                temporary=use_temporary_chats,
            ):
                delta = chunk.text_delta
                thoughts_delta = chunk.thoughts_delta or ""
                
                if delta:
                    full_response_text += delta  # Accumula testo
                if thoughts_delta:
                    full_response_thoughts += thoughts_delta  # Accumula pensieri

                payload_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                }
                yield b"data: " + orjson.dumps(payload_chunk) + b"\n\n"

            # Combina pensieri e testo per il contenuto finale
            full_content = ""
            if full_response_thoughts:
                full_content = f"<thinking>\n{full_response_thoughts}\n</thinking>\n\n{full_response_text}"
            else:
                full_content = full_response_text
            
            # Get images from a quick non-stream call (to avoid complexity in streaming)
            try:
                quick_output = await client.generate_content(prompt=payload.prompt, model=model, temporary=use_temporary_chats)
                if quick_output.images:
                    image_markdown = "\n\n".join([
                        f"![Generated Image]({img.url})" if hasattr(img, 'url') and img.url else 
                        f"![Generated Image](data:{getattr(img, 'mime_type', 'image/png')};base64,{getattr(img, 'data', '')})"
                        for img in quick_output.images
                    ])
                    full_content += f"\n\n{image_markdown}"
            except:
                pass  # Ignore errors in image fetching for streaming

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
            yield b"data: " + orjson.dumps(final_chunk) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    output = await client.generate_content(prompt=payload.prompt, model=model, temporary=use_temporary_chats)

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = _unix_ts()

    # Combine thoughts and text for thinking models
    thoughts = output.thoughts
    text = output.text
    full_content = ""
    
    if thoughts:
        full_content = f"<thinking>\n{thoughts}\n</thinking>\n\n{text}"
    else:
        full_content = text
    
    # Add generated images to the response if any
    if output.images:
        image_markdown = "\n\n".join([
            f"![Generated Image]({img.url})" if hasattr(img, 'url') and img.url else 
            f"![Generated Image](data:{getattr(img, 'mime_type', 'image/png')};base64,{getattr(img, 'data', '')})"
            for img in output.images
        ])
        full_content += f"\n\n{image_markdown}"

    # Estimate token usage
    prompt_tokens = _estimate_tokens(payload.prompt)
    completion_tokens = _estimate_tokens(full_content)
    total_tokens = prompt_tokens + completion_tokens

    # Track usage
    cost = _estimate_cost(model, prompt_tokens, completion_tokens)
    state.usage_tracker.track_request(model, prompt_tokens, completion_tokens, cost)

    return {
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


@app.post("/v1/images/generations")
async def create_image(
    payload: ImageGenerationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)

    if state.client is None:
        raise HTTPException(status_code=503, detail="Gemini client is not initialized")
    client = state.client

    # Use Gemini to generate image with a prompt that encourages image generation
    image_prompt = f"Generate an image based on this description: {payload.prompt}"
    
    try:
        output = await client.generate_content(
            prompt=image_prompt,
            model=payload.model,
            temporary=True  # Use temporary chats for image generation
        )
        
        # Check if Gemini generated any images
        if not output.images:
            raise HTTPException(
                status_code=500, 
                detail="No image was generated. Try rephrasing your prompt."
            )
        
        # Convert Gemini images to OpenAI-compatible format
        image_data = []
        for img in output.images:
            # Gemini returns images as data URLs or similar
            # For OpenAI compatibility, we return the image data
            image_data.append({
                "url": img.data_url if hasattr(img, 'data_url') else f"data:{img.mime_type};base64,{img.data}",
                "revised_prompt": payload.prompt  # Gemini doesn't revise prompts like DALL-E
            })
        
        return {
            "created": _unix_ts(),
            "data": image_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


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
async def health() -> dict[str, str]:
    return {"status": "ok"}
