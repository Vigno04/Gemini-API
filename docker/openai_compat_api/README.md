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

## Prebuilt image

Default image reference in compose:

- `ghcr.io/hanaokayuzu/gemini-api-openai-compat:develop`

Override it on your own server with `OPENAI_COMPAT_IMAGE`, for example:

```dotenv
OPENAI_COMPAT_IMAGE=ghcr.io/<your-org>/<your-image>:develop
```

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
