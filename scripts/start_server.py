"""
Startup script for the Advanced RAG Application.
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings


def main():
    """Start the FastAPI server."""
    settings = get_settings()
    
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Debug mode: {settings.DEBUG}")
    print(f"Host: {settings.HOST}")
    print(f"Port: {settings.PORT}")
    print(f"API docs: http://{settings.HOST}:{settings.PORT}/docs")
    
    # Check for required environment variables
    if not settings.GOOGLE_API_KEY and not settings.OPENAI_API_KEY:
        print("WARNING: No API keys found. Please set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
