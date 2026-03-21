# OpenAI-compatible Gemini wrapper (Docker)

This folder contains an isolated compatibility layer that exposes OpenAI-style endpoints while using `gemini-webapi` under the hood.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/images/variations`
- `POST /v1/moderations`
- `POST /v1/embeddings`

Notes:
- `stream=true` is supported for `/v1/chat/completions` and `/v1/completions` using SSE (`data:` chunks + `[DONE]`).
- Token usage fields are returned as `0` because Gemini web responses do not provide OpenAI token accounting.
- Set `OPENAI_COMPAT_DEBUG=true` to enable detailed request/response logging for debugging.

## Environment Variables

- `GEMINI_SECURE_1PSID` - Required: Your Gemini session token
- `GEMINI_SECURE_1PSIDTS` - Optional: Additional session token for some accounts  
- `GEMINI_PROXY` - Optional: Proxy URL for requests
- `OPENAI_COMPAT_API_KEY` - Optional: API key for authentication
- `OPENAI_COMPAT_DEFAULT_MODEL` - Optional: Default model (default: gemini-3-flash)
- `OPENAI_COMPAT_DEBUG` - Optional: Enable detailed logging (default: false)
- `OPENAI_COMPAT_USE_TEMPORARY_CHATS` - Optional: Use temporary chats (default: true)

## Quick start

1. Create environment file:

```powershell
Copy-Item .env.example .env
```

2. Edit `.env` and set at least:

- `GEMINI_SECURE_1PSID`
- optionally `GEMINI_SECURE_1PSIDTS`
- optionally `OPENAI_COMPAT_API_KEY`

3. Start the service:

```powershell
docker compose pull
docker compose up -d
```

4. Test:

```powershell
curl http://localhost:8080/health
```

## Example request

```powershell
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"gemini-3-flash","messages":[{"role":"user","content":"Say hello"}]}'
```
