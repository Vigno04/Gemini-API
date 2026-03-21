# Gemini OpenAI-Compatible API

A FastAPI-based wrapper around Gemini WebAPI that provides OpenAI-compatible endpoints.

## Architecture

The codebase is organized into modular components for better maintainability:

- **`main.py`** - Application initialization and FastAPI app setup
- **`models.py`** - Pydantic models for API requests/responses
- **`utils.py`** - Utility functions (auth, token estimation, etc.)
- **`usage_tracker.py`** - Usage tracking and cost calculation
- **`endpoints.py`** - All API endpoint implementations

## Features

- ✅ OpenAI-compatible API endpoints
- ✅ Streaming and non-streaming responses
- ✅ Token usage tracking
- ✅ Image generation support
- ✅ Content moderation
- ✅ File upload/management
- ✅ Comprehensive error handling

## Supported Endpoints

### Core Endpoints
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /v1/models` - List available models
- `POST /v1/images/generations` - Image generation

### Additional Endpoints
- `POST /v1/moderations` - Content moderation
- `POST /v1/files` - File upload
- `GET /v1/files` - List files
- `GET /v1/usage` - Usage statistics
- `GET /v1/organization` - Organization info

### Placeholder Endpoints (501 Not Implemented)
- `/v1/embeddings` - Not supported by Gemini
- `/v1/audio/*` - Not supported by Gemini
- `/v1/assistants` - Not supported by Gemini
- `/v1/threads` - Not supported by Gemini
- `/v1/fine_tuning/*` - Not supported by Gemini
- `/v1/batches` - Not supported by Gemini
- `/v1/uploads` - Not supported by Gemini
- `/v1/vector_stores` - Not supported by Gemini

## Usage Tracking

The API automatically tracks usage statistics including:
- Token counts (prompt/completion/total)
- Request counts per model
- Daily and total usage aggregation
- Cost estimation (currently set to $0)

Usage data is persisted in `usage_data.json`.