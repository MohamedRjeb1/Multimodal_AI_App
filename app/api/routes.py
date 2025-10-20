"""
API routes for the Advanced RAG Application.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.models.schemas import (
    YouTubeRequest, ProcessingResponse, TranscriptResponse, 
    QueryRequest, QueryResponse, ErrorResponse, HealthResponse,
    CourseRequest, CourseResponse, CourseQueryRequest, CourseQueryResponse,
    CourseStatistics, VideoInfo
)
from app.services.rag_service import AdvancedRAGService
from app.services.course_service import AdvancedCourseService
from app.services.combined_rag_service import CombinedRAGService
from app.models.schemas import ProcessingStatus

# Initialize router
router = APIRouter()

# Initialize services
rag_service = AdvancedRAGService()
course_service = AdvancedCourseService()
combined_rag_service = CombinedRAGService()


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


# ===== FORMATION MULTI-VIDÉOS ROUTES =====

@router.post("/courses", response_model=CourseResponse)
async def create_course(request: CourseRequest) -> CourseResponse:
    """
    Create a new multi-video course.
    
    This endpoint allows users to create a formation with multiple videos
    that will be processed and linked together for intelligent querying.
    """
    try:
        course = course_service.create_course(request)
        return course
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create course: {str(e)}"
        )


@router.post("/courses/{course_id}/process")
async def process_course(
    course_id: str,
    processing_strategy: str = "sequential",
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """
    Process all videos in a course.
    
    Args:
        course_id: Course identifier
        processing_strategy: Processing strategy (sequential, parallel, hybrid)
    """
    try:
        if background_tasks:
            # Process in background
            background_tasks.add_task(
                course_service.process_course,
                course_id,
                processing_strategy
            )
            return {
                "message": f"Course {course_id} processing started",
                "course_id": course_id,
                "strategy": processing_strategy
            }
        else:
            # Process synchronously
            result = course_service.process_course(course_id, processing_strategy)
            return result
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process course: {str(e)}"
        )


@router.get("/courses/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str) -> CourseResponse:
    """Get course information."""
    try:
        if course_id not in course_service.courses:
            raise HTTPException(
                status_code=404,
                detail=f"Course {course_id} not found"
            )
        
        return course_service.courses[course_id]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get course: {str(e)}"
        )


@router.post("/courses/{course_id}/query", response_model=CourseQueryResponse)
async def query_course(
    course_id: str,
    request: CourseQueryRequest
) -> CourseQueryResponse:
    """
    Query a course with cross-video retrieval capabilities.
    
    This endpoint allows intelligent querying across multiple videos
    in a formation, with automatic cross-video context linking.
    """
    try:
        # Ensure course_id matches
        request.course_id = course_id
        
        result = course_service.query_course(request)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query course: {str(e)}"
        )


@router.get("/courses/{course_id}/statistics", response_model=CourseStatistics)
async def get_course_statistics(course_id: str) -> CourseStatistics:
    """Get detailed statistics for a course."""
    try:
        stats = course_service.get_course_statistics(course_id)
        return stats
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get course statistics: {str(e)}"
        )


@router.get("/courses/{course_id}/videos/{video_id}")
async def get_video_info(course_id: str, video_id: str) -> Dict[str, Any]:
    """Get information about a specific video in a course."""
    try:
        if course_id not in course_service.courses:
            raise HTTPException(
                status_code=404,
                detail=f"Course {course_id} not found"
            )
        
        course = course_service.courses[course_id]
        video = next((v for v in course.videos if v.video_id == video_id), None)
        
        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} not found in course {course_id}"
            )
        
        return {
            "video_id": video.video_id,
            "title": video.title,
            "url": str(video.url),
            "duration": video.duration,
            "order": video.order,
            "status": video.status,
            "transcript_length": len(video.transcript) if video.transcript else 0,
            "created_at": video.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video info: {str(e)}"
        )


@router.get("/courses")
async def list_courses() -> Dict[str, Any]:
    """List all courses."""
    try:
        courses = []
        for course_id, course in course_service.courses.items():
            courses.append({
                "course_id": course_id,
                "course_name": course.course_name,
                "course_description": course.course_description,
                "total_videos": course.total_videos,
                "processed_videos": course.processed_videos,
                "status": course.status,
                "created_at": course.created_at
            })
        
        return {
            "courses": courses,
            "total_courses": len(courses)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list courses: {str(e)}"
        )


@router.delete("/courses/{course_id}")
async def delete_course(course_id: str) -> Dict[str, Any]:
    """Delete a course and all its associated data."""
    try:
        if course_id not in course_service.courses:
            raise HTTPException(
                status_code=404,
                detail=f"Course {course_id} not found"
            )
        
        # Remove from memory
        del course_service.courses[course_id]
        
        # Clean up files (implementation would depend on storage strategy)
        # course_service.cleanup_course(course_id)
        
        return {
            "message": f"Course {course_id} deleted successfully",
            "course_id": course_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete course: {str(e)}"
        )


# ===== STRATÉGIE COMBINATOIRE ROUTES =====

@router.post("/combined/process", response_model=ProcessingResponse)
async def process_video_combined(
    request: YouTubeRequest,
    background_tasks: BackgroundTasks
) -> ProcessingResponse:
    """
    Traite une vidéo YouTube avec la stratégie combinatoire avancée.
    
    Cette stratégie intègre :
    - Chunking sémantique dynamique
    - Indexation par résumé avec modèle fine-tuné
    - Clustering sémantique
    - Memory RAG avec contexte historique
    - Corrective RAG avec grading et refinement
    """
    try:
        # Générer un ID de tâche unique
        import uuid
        task_id = str(uuid.uuid4())
        
        # Démarrer le traitement en arrière-plan
        background_tasks.add_task(
            combined_rag_service.process_video_with_combined_strategy,
            str(request.url),
            task_id,
            request.language
        )
        
        return ProcessingResponse(
            task_id=task_id,
            status=ProcessingStatus.PENDING,
            message="Traitement combinatoire démarré"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec du démarrage du traitement combinatoire: {str(e)}"
        )


@router.post("/combined/query")
async def query_video_combined(
    request: QueryRequest,
    user_id: str = None,
    session_id: str = None
) -> Dict[str, Any]:
    """
    Interroge une vidéo traitée avec la stratégie combinatoire.
    
    Cette stratégie utilise :
    - Memory RAG pour enrichir le contexte avec l'historique
    - Corrective RAG pour grader et raffiner les connaissances
    - Génération de réponse avec contexte enrichi
    """
    try:
        result = combined_rag_service.query_with_combined_strategy(
            query=request.query,
            task_id=request.task_id,
            user_id=user_id,
            session_id=session_id,
            max_results=request.max_results
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["message"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec du traitement de la requête combinatoire: {str(e)}"
        )


@router.get("/combined/stats/{task_id}")
async def get_combined_statistics(task_id: str) -> Dict[str, Any]:
    """Obtient les statistiques complètes de la stratégie combinatoire."""
    try:
        stats = combined_rag_service.get_combined_strategy_statistics(task_id)
        
        if "error" in stats:
            raise HTTPException(
                status_code=404,
                detail=stats["error"]
            )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec de l'obtention des statistiques combinatoires: {str(e)}"
        )


@router.delete("/combined/cleanup/{task_id}")
async def cleanup_combined_task(task_id: str) -> Dict[str, Any]:
    """Nettoie toutes les données d'une tâche pour la stratégie combinatoire."""
    try:
        success = combined_rag_service.cleanup_combined_task(task_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Échec du nettoyage de la tâche combinatoire"
            )
        
        return {
            "message": f"Tâche combinatoire {task_id} nettoyée avec succès",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec du nettoyage combinatoire: {str(e)}"
        )


@router.get("/combined/memory/history")
async def get_memory_history(
    user_id: str = None,
    session_id: str = None
) -> Dict[str, Any]:
    """Obtient l'historique Memory RAG."""
    try:
        stats = combined_rag_service.memory_rag.get_history_statistics(
            user_id=user_id,
            session_id=session_id
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec de l'obtention de l'historique Memory RAG: {str(e)}"
        )


@router.delete("/combined/memory/clear")
async def clear_memory_history(
    user_id: str = None,
    session_id: str = None
) -> Dict[str, Any]:
    """Efface l'historique Memory RAG."""
    try:
        combined_rag_service.memory_rag.clear_history(
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "message": "Historique Memory RAG effacé avec succès",
            "user_id": user_id,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Échec de l'effacement de l'historique Memory RAG: {str(e)}"
        )
