"""
API routes for the Advanced RAG Application.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.models.schemas import (
    YouTubeRequest, ProcessingResponse, TranscriptResponse, 
    QueryRequest, QueryResponse, ErrorResponse, HealthResponse
)
from app.services.rag_service import AdvancedRAGService
from app.models.schemas import ProcessingStatus

# Initialize router
router = APIRouter()

# Initialize RAG service
rag_service = AdvancedRAGService()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=get_settings().APP_VERSION,
        services={
            "youtube_service": "active",
            "transcription_service": "active", 
            "embedding_service": "active",
            "indexing_service": "active",
            "rag_service": "active"
        }
    )


@router.post("/process", response_model=ProcessingResponse)
async def process_video(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks
) -> ProcessingResponse:
    """
    Process a YouTube video for RAG queries.
    
    This endpoint:
    1. Downloads the video
    2. Transcribes the audio
    3. Generates embeddings
    4. Creates an index
    """
    try:
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Start processing in background
        background_tasks.add_task(
            rag_service.process_video,
            str(request.url),
            task_id,
            request.language,
            "semantic"  # Default strategy
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status=ProcessingStatus.PENDING,
            message="Video processing started"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video processing: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=ProcessingResponse)
async def get_processing_status(task_id: str) -> ProcessingResponse:
    """Get processing status for a task."""
    status = rag_service.get_processing_status(task_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return ProcessingResponse(
        task_id=task_id,
        status=status["status"],
        message=status["message"],
        updated_at=status["updated_at"]
    )


@router.get("/info/{task_id}", response_model=TranscriptResponse)
async def get_video_info(task_id: str) -> TranscriptResponse:
    """Get processed video information and transcript."""
    try:
        # Get transcript
        transcript = rag_service.transcription_service.get_transcript(task_id)
        if not transcript:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript found for task {task_id}"
            )
        
        # Get transcript info
        transcript_info = rag_service.transcription_service.get_transcript_info(task_id)
        if not transcript_info:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript info found for task {task_id}"
            )
        
        # Get index statistics
        index_stats = rag_service.indexing_service.get_index_statistics(task_id)
        
        return TranscriptResponse(
            task_id=task_id,
            video_title=index_stats.get("task_id", "Unknown") if index_stats else "Unknown",
            transcript=transcript,
            language="fr",  # Default, could be improved
            created_at=transcript_info["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video info: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest) -> QueryResponse:
    """
    Query a processed video using RAG.
    
    This endpoint:
    1. Searches for relevant chunks
    2. Generates an answer using LLM
    3. Returns answer with sources
    """
    try:
        result = rag_service.query_video(
            query=request.query,
            task_id=request.task_id,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["message"]
            )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            processing_time=result["processing_time"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.get("/video-info")
async def get_youtube_info(url: str) -> Dict[str, Any]:
    """Get YouTube video information without processing."""
    try:
        result = rag_service.get_video_info(url)
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video info: {str(e)}"
        )


@router.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str) -> Dict[str, Any]:
    """Clean up all files and data for a task."""
    try:
        success = rag_service.cleanup_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to cleanup task"
            )
        
        return {
            "message": f"Task {task_id} cleaned up successfully",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )


@router.get("/stats/{task_id}")
async def get_task_statistics(task_id: str) -> Dict[str, Any]:
    """Get detailed statistics for a processed task."""
    try:
        # Get index statistics
        index_stats = rag_service.indexing_service.get_index_statistics(task_id)
        
        if not index_stats:
            raise HTTPException(
                status_code=404,
                detail=f"No statistics found for task {task_id}"
            )
        
        # Get transcript info
        transcript_info = rag_service.transcription_service.get_transcript_info(task_id)
        
        return {
            "task_id": task_id,
            "index_statistics": index_stats,
            "transcript_info": transcript_info,
            "processing_status": rag_service.get_processing_status(task_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )
