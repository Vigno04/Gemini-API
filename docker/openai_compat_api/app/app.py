from fastapi import FastAPI

app = FastAPI(title="Gemini OpenAI-compatible API")

# Import all endpoints to register them with the app
import endpoints