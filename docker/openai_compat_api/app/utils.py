import os
import time
import json
import tiktoken
from typing import Any
from fastapi import Header, HTTPException

from gemini_webapi import GeminiClient


def _debug_log(message: str, *args, **kwargs):
    """Log debug messages if OPENAI_COMPAT_DEBUG is enabled"""
    if os.getenv("OPENAI_COMPAT_DEBUG", "false").strip().lower() in ("true", "1", "yes", "on"):
        print(f"[DEBUG] {message}", *args, **kwargs)


def _debug_enabled() -> bool:
    return os.getenv("OPENAI_COMPAT_DEBUG", "false").strip().lower() in ("true", "1", "yes", "on")


import re

def _debug_dump_request(endpoint: str, payload: Any) -> None:
    """Dump full request payload for debugging when enabled."""
    if not _debug_enabled():
        return

    normalized = payload
    if hasattr(payload, "model_dump"):
        normalized = payload.model_dump()

    try:
        serialized = json.dumps(normalized, ensure_ascii=True, default=str)
    except Exception:
        serialized = str(normalized)
        
    serialized = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', 'data:image/...;base64,...[truncated]', serialized)
    print(f"[DEBUG][REQUEST] {endpoint} {serialized}")


def _debug_dump_response(endpoint: str, payload: Any) -> None:
    """Dump full response payload for debugging when enabled."""
    if not _debug_enabled():
        return

    normalized = payload
    if hasattr(payload, "model_dump"):
        normalized = payload.model_dump()

    try:
        serialized = json.dumps(normalized, ensure_ascii=True, default=str)
    except Exception:
        serialized = str(normalized)

    serialized = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', 'data:image/...;base64,...[truncated]', serialized)
    print(f"[DEBUG][RESPONSE] {endpoint} {serialized}")


def _unix_ts() -> int:
    return int(time.time())


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost estimation disabled - always return 0"""
    return 0.0


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate of 4 chars per token
        return len(text) // 4


def _require_auth(authorization: str | None = Header(default=None)):
    """Require authentication via Bearer token"""
    expected = os.getenv("OPENAI_COMPAT_API_KEY", "your-api-key-here")
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")

    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


import base64
import tempfile
from pathlib import Path

def _messages_to_prompt(messages: list) -> tuple[str, list[Path]]:
    rendered: list[str] = []
    files: list[Path] = []
    
    for msg in messages:
        role = msg.role.lower()
        if role not in {"system", "user", "assistant"}:
            continue

        if isinstance(msg.content, str):
            text = msg.content
        else:
            # OpenAI-style content array support: [{"type":"text","text":"..."}, {"type":"image_url", "image_url": {"url": "..."}}]
            text_parts = []
            for part in msg.content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(str(part.get("text", "")))
                elif ptype == "image_url":
                    img_url_obj = part.get("image_url", {})
                    url = img_url_obj.get("url", "")
                    if url.startswith("data:"):
                        try:
                            if "," in url:
                                mime_part, b64 = url.split(",", 1)
                                ext = ".png" # default
                                if "image/jpeg" in mime_part:
                                    ext = ".jpg"
                                elif "image/webp" in mime_part:
                                    ext = ".webp"
                                elif "image/gif" in mime_part:
                                    ext = ".gif"
                                
                                img_bytes = base64.b64decode(b64, validate=True)
                                
                                # Write to a temporary file locally so Gemini API can upload it
                                tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=ext, prefix="openai_compat_usr_img_")
                                with os.fdopen(tmp_fd, 'wb') as f:
                                    f.write(img_bytes)
                                
                                files.append(Path(tmp_path_str))
                        except Exception as e:
                            _debug_log(f"Ignoring unparseable base64 image: {e}")
                            pass
                    elif url.startswith("http://") or url.startswith("https://"):
                        # Just an informational approach for unsupported URLs: we can insert it in the text.
                        text_parts.append(f"[Image URL: {url}]")
                        
            text = "".join(text_parts)

        if role == "system":
            rendered.append(f"system: {text}")
        elif role == "user":
            rendered.append(f"user: {text}")
        elif role == "assistant":
            rendered.append(f"assistant: {text}")
            
    return "\n".join(rendered), files