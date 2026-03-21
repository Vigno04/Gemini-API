from fastapi import FastAPI

app = FastAPI(title="Gemini OpenAI-compatible API")

# Import all endpoints to register them with the app
import endpoints

# Import main to set up the lifespan context manager
import main