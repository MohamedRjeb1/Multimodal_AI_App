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


# ===== FORMATION MULTI-VIDÉOS MODELS =====

class VideoInfo(BaseModel):
    """Information about a single video in a course."""
    video_id: str = Field(..., description="Unique video identifier")
    url: HttpUrl = Field(..., description="YouTube video URL")
    title: str = Field(..., description="Video title")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    order: int = Field(..., description="Order in the course sequence")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Processing status")
    transcript: Optional[str] = Field(None, description="Video transcript")
    created_at: datetime = Field(default_factory=datetime.now)


class CourseRequest(BaseModel):
    """Request model for creating a multi-video course."""
    course_name: str = Field(..., description="Course name", min_length=1, max_length=200)
    course_description: Optional[str] = Field(None, description="Course description", max_length=1000)
    videos: List[HttpUrl] = Field(..., description="List of YouTube video URLs", min_items=1, max_items=20)
    language: Optional[str] = Field(default="fr", description="Transcription language")
    whisper_model: Optional[str] = Field(default="small", description="Whisper model size")
    processing_strategy: Optional[str] = Field(default="sequential", description="Processing strategy: sequential, parallel, or hybrid")


class CourseResponse(BaseModel):
    """Response model for course processing."""
    course_id: str = Field(..., description="Unique course identifier")
    course_name: str = Field(..., description="Course name")
    course_description: Optional[str] = Field(None, description="Course description")
    total_videos: int = Field(..., description="Total number of videos")
    processed_videos: int = Field(default=0, description="Number of processed videos")
    status: ProcessingStatus = Field(..., description="Overall course status")
    videos: List[VideoInfo] = Field(..., description="List of videos in the course")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class CourseQueryRequest(BaseModel):
    """Request model for querying a course."""
    query: str = Field(..., description="User question", min_length=1, max_length=1000)
    course_id: str = Field(..., description="Course identifier")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results")
    similarity_threshold: Optional[float] = Field(default=0.7, description="Similarity threshold")
    search_strategy: Optional[str] = Field(default="cross_video", description="Search strategy: cross_video, single_video, or hybrid")
    target_videos: Optional[List[str]] = Field(None, description="Specific video IDs to search in (optional)")


class CourseQueryResponse(BaseModel):
    """Response model for course queries."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source chunks with video context")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    video_context: Dict[str, Any] = Field(..., description="Context about which videos were used")
    cross_video_connections: List[Dict[str, Any]] = Field(default=[], description="Connections between videos")
    created_at: datetime = Field(default_factory=datetime.now)


class CourseStatistics(BaseModel):
    """Statistics for a course."""
    course_id: str = Field(..., description="Course identifier")
    total_videos: int = Field(..., description="Total number of videos")
    processed_videos: int = Field(..., description="Number of processed videos")
    total_transcript_length: int = Field(..., description="Total transcript length in characters")
    total_chunks: int = Field(..., description="Total number of chunks")
    average_chunk_size: float = Field(..., description="Average chunk size")
    processing_time: float = Field(..., description="Total processing time in seconds")
    created_at: datetime = Field(..., description="Course creation time")
    last_updated: datetime = Field(..., description="Last update time")