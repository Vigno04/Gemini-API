import os
import time
import tiktoken
from typing import Any
from fastapi import Header, HTTPException

from gemini_webapi import GeminiClient


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


def _messages_to_prompt(messages: list) -> str:
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
            text = "".join(text_parts)

        if role == "system":
            rendered.append(f"system: {text}")
        elif role == "user":
            rendered.append(f"user: {text}")
        elif role == "assistant":
            rendered.append(f"assistant: {text}")
    return "\n".join(rendered)