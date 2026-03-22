from pydantic import BaseModel, Field
from typing import Any


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


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "gemini-3-flash"  # Default Gemini model for image generation
    n: int = 1  # Number of images (Gemini typically generates 1)
    size: str = "1024x1024"  # Ignored for Gemini, but kept for compatibility
    quality: str = "standard"  # Ignored for Gemini
    style: str = "natural"  # Ignored for Gemini
    response_format: str = "url"  # OpenAI-compatible: url | b64_json
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