from app import app

# Import route modules to register all handlers on the shared FastAPI app.
import chat_endpoints  # noqa: F401
import image_endpoints  # noqa: F401
import misc_endpoints  # noqa: F401
