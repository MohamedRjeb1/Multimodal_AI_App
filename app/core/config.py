"""
Configuration settings for the Advanced RAG Application.
"""
import os
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    APP_NAME: str = Field(default="Advanced RAG", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API settings
    API_V1_STR: str = "/api/v1"
    HOST: str = Field(default="0.0.0.0", description="Host to bind to")
    PORT: int = Field(default=8000, description="Port to bind to")
    
    # Data paths
    DATA_DIR: str = Field(default="data", description="Data directory")
    AUDIO_DIR: str = Field(default="data/audio", description="Audio files directory")
    TRANSCRIPT_DIR: str = Field(default="data/transcripts", description="Transcripts directory")
    VECTORSTORE_DIR: str = Field(default="data/vectorstore", description="Vector store directory")
    COURSES_DIR: str = Field(default="data/courses", description="Courses directory")
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = Field(default=None, description="Google API key")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # LLM Settings
    DEFAULT_LLM_MODEL: str = Field(default="gemini-2.5-flash", description="Default LLM model")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="google", description="Default embedding model")
    
    # Whisper settings
    WHISPER_MODEL: str = Field(default="small", description="Whisper model size")
    WHISPER_LANGUAGE: str = Field(default="fr", description="Whisper language")
    
    # RAG settings
    CHUNK_SIZE: int = Field(default=1000, description="Text chunk size for processing")
    CHUNK_OVERLAP: int = Field(default=200, description="Overlap between chunks")
    TOP_K_RESULTS: int = Field(default=5, description="Number of top results to retrieve")
    
    # Vector store settings
    VECTORSTORE_TYPE: str = Field(default="docarray", description="Vector store type")
    EMBEDDING_DIMENSION: int = Field(default=768, description="Embedding dimension")
    
    # YouTube settings
    YOUTUBE_DOWNLOAD_FORMAT: str = Field(default="bestaudio", description="YouTube download format")
    YOUTUBE_AUDIO_QUALITY: str = Field(default="192", description="Audio quality")
    
    @validator("DATA_DIR", "AUDIO_DIR", "TRANSCRIPT_DIR", "VECTORSTORE_DIR", "COURSES_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator("GOOGLE_API_KEY", "OPENAI_API_KEY")
    def validate_api_keys(cls, v):
        """Validate that at least one API key is provided."""
        if not v:
            return None
        return v
    


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
