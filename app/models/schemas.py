"""
Pydantic schemas for API request/response models.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class YouTubeRequest(BaseModel):
    """Request model for YouTube URL processing."""
    url: HttpUrl = Field(..., description="YouTube video URL")
    language: Optional[str] = Field(default="fr", description="Transcription language")
    whisper_model: Optional[str] = Field(default="small", description="Whisper model size")


class ProcessingResponse(BaseModel):
    """Response model for processing status."""
    task_id: str = Field(..., description="Unique task identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class TranscriptResponse(BaseModel):
    """Response model for transcript data."""
    task_id: str = Field(..., description="Task identifier")
    video_title: str = Field(..., description="Video title")
    transcript: str = Field(..., description="Full transcript text")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    language: str = Field(..., description="Detected language")
    created_at: datetime = Field(default_factory=datetime.now)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="User question", min_length=1, max_length=1000)
    task_id: str = Field(..., description="Task identifier for the processed video")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Similarity threshold")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source chunks used for answer")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, str] = Field(..., description="Dependent services status")
