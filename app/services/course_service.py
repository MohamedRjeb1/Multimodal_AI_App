"""
Advanced Course Service for Multi-Video Formations.
"""
import os
import json
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from app.core.config import get_settings
from app.models.schemas import (
    CourseRequest, CourseResponse, VideoInfo, CourseQueryRequest, 
    CourseQueryResponse, CourseStatistics, ProcessingStatus
)
from app.services.youtube_service import YouTubeService
from app.services.transcription_service import TranscriptionService
from app.services.embedding_service import AdvancedEmbeddingService
from app.services.indexing_service import AdvancedIndexingService
from app.services.rag_service import AdvancedRAGService


class AdvancedCourseService:
    """
    Advanced Course Service for managing multi-video formations:
    - Sequential and parallel video processing
    - Cross-video semantic linking
    - Intelligent course-level retrieval
    - Formation context awareness
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.courses_dir = os.path.join(self.settings.DATA_DIR, "courses")
        os.makedirs(self.courses_dir, exist_ok=True)
        
        # Initialize services
        self.youtube_service = YouTubeService()
        self.transcription_service = TranscriptionService()
        self.embedding_service = AdvancedEmbeddingService()
        self.indexing_service = AdvancedIndexingService()
        self.rag_service = AdvancedRAGService()
        
        # Course storage
        self.courses = {}
        self.course_metadata = {}
    
    def create_course(self, request: CourseRequest) -> CourseResponse:
        """
        Create a new multi-video course.
        
        Args:
            request: Course creation request
            
        Returns:
            Course response with video information
        """
        course_id = str(uuid.uuid4())
        
        # Create video info objects
        videos = []
        for i, url in enumerate(request.videos):
            video_id = str(uuid.uuid4())
            video_info = VideoInfo(
                video_id=video_id,
                url=url,
                title=f"Video {i+1}",  # Will be updated after processing
                order=i,
                status=ProcessingStatus.PENDING
            )
            videos.append(video_info)
        
        # Create course response
        course_response = CourseResponse(
            course_id=course_id,
            course_name=request.course_name,
            course_description=request.course_description,
            total_videos=len(videos),
            processed_videos=0,
            status=ProcessingStatus.PENDING,
            videos=videos
        )
        
        # Store course metadata
        self.courses[course_id] = course_response
        self._save_course_metadata(course_id, course_response)
        
        return course_response
    
    def process_course(self, course_id: str, processing_strategy: str = "sequential") -> Dict[str, Any]:
        """
        Process all videos in a course.
        
        Args:
            course_id: Course identifier
            processing_strategy: Processing strategy (sequential, parallel, hybrid)
            
        Returns:
            Processing result
        """
        if course_id not in self.courses:
            return {"error": "Course not found"}
        
        course = self.courses[course_id]
        course.status = ProcessingStatus.DOWNLOADING
        
        try:
            if processing_strategy == "sequential":
                return self._process_sequential(course_id)
            elif processing_strategy == "parallel":
                return self._process_parallel(course_id)
            elif processing_strategy == "hybrid":
                return self._process_hybrid(course_id)
            else:
                return {"error": "Invalid processing strategy"}
                
        except Exception as e:
            course.status = ProcessingStatus.FAILED
            return {"error": str(e)}
    
    def _process_sequential(self, course_id: str) -> Dict[str, Any]:
        """Process videos sequentially."""
        course = self.courses[course_id]
        results = []
        
        for video in course.videos:
            # Process each video one by one
            result = self._process_single_video(video, course_id)
            results.append(result)
            
            if result["status"] == ProcessingStatus.FAILED:
                course.status = ProcessingStatus.FAILED
                return {"error": f"Failed to process video {video.video_id}"}
        
        # Create cross-video connections
        self._create_cross_video_connections(course_id)
        
        course.status = ProcessingStatus.COMPLETED
        course.processed_videos = len(course.videos)
        self._save_course_metadata(course_id, course)
        
        return {
            "status": "completed",
            "processed_videos": len(course.videos),
            "results": results
        }
    
    def _process_parallel(self, course_id: str) -> Dict[str, Any]:
        """Process videos in parallel."""
        course = self.courses[course_id]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all video processing tasks
            future_to_video = {
                executor.submit(self._process_single_video, video, course_id): video
                for video in course.videos
            }
            
            results = []
            for future in as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == ProcessingStatus.FAILED:
                        course.status = ProcessingStatus.FAILED
                        return {"error": f"Failed to process video {video.video_id}"}
                        
                except Exception as e:
                    course.status = ProcessingStatus.FAILED
                    return {"error": f"Error processing video {video.video_id}: {str(e)}"}
        
        # Create cross-video connections
        self._create_cross_video_connections(course_id)
        
        course.status = ProcessingStatus.COMPLETED
        course.processed_videos = len(course.videos)
        self._save_course_metadata(course_id, course)
        
        return {
            "status": "completed",
            "processed_videos": len(course.videos),
            "results": results
        }
    
    def _process_hybrid(self, course_id: str) -> Dict[str, Any]:
        """Process videos with hybrid strategy (first video sequential, then parallel)."""
        course = self.courses[course_id]
        
        # Process first video sequentially to establish context
        if course.videos:
            first_video = course.videos[0]
            result = self._process_single_video(first_video, course_id)
            
            if result["status"] == ProcessingStatus.FAILED:
                course.status = ProcessingStatus.FAILED
                return {"error": f"Failed to process first video {first_video.video_id}"}
        
        # Process remaining videos in parallel
        if len(course.videos) > 1:
            remaining_videos = course.videos[1:]
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_video = {
                    executor.submit(self._process_single_video, video, course_id): video
                    for video in remaining_videos
                }
                
                for future in as_completed(future_to_video):
                    video = future_to_video[future]
                    try:
                        result = future.result()
                        if result["status"] == ProcessingStatus.FAILED:
                            course.status = ProcessingStatus.FAILED
                            return {"error": f"Failed to process video {video.video_id}"}
                            
                    except Exception as e:
                        course.status = ProcessingStatus.FAILED
                        return {"error": f"Error processing video {video.video_id}: {str(e)}"}
        
        # Create cross-video connections
        self._create_cross_video_connections(course_id)
        
        course.status = ProcessingStatus.COMPLETED
        course.processed_videos = len(course.videos)
        self._save_course_metadata(course_id, course)
        
        return {
            "status": "completed",
            "processed_videos": len(course.videos)
        }
    
    def _process_single_video(self, video: VideoInfo, course_id: str) -> Dict[str, Any]:
        """Process a single video within a course context."""
        try:
            # Update video status
            video.status = ProcessingStatus.DOWNLOADING
            
            # Download video
            download_result = self.youtube_service.download_video(str(video.url), video.video_id)
            if download_result["status"] == ProcessingStatus.FAILED:
                video.status = ProcessingStatus.FAILED
                return {"status": ProcessingStatus.FAILED, "error": download_result.get("error")}
            
            # Update video title
            video.title = download_result.get("title", f"Video {video.order + 1}")
            video.duration = download_result.get("duration")
            
            # Transcribe audio
            video.status = ProcessingStatus.TRANSCRIBING
            audio_file = download_result["audio_file"]
            transcription_result = self.transcription_service.transcribe_audio(
                audio_file, video.video_id, "fr"
            )
            
            if transcription_result["status"] == ProcessingStatus.FAILED:
                video.status = ProcessingStatus.FAILED
                return {"status": ProcessingStatus.FAILED, "error": transcription_result.get("error")}
            
            # Store transcript
            video.transcript = transcription_result["transcript"]
            
            # Generate embeddings with course context
            video.status = ProcessingStatus.EMBEDDING
            embedding_result = self.embedding_service.process_transcript(
                video.transcript, video.video_id, "semantic"
            )
            
            if embedding_result["status"] == ProcessingStatus.FAILED:
                video.status = ProcessingStatus.FAILED
                return {"status": ProcessingStatus.FAILED, "error": embedding_result.get("error")}
            
            # Create index with course metadata
            video.status = ProcessingStatus.INDEXING
            index_result = self.indexing_service.create_hybrid_index(
                embedding_result["embeddings"], video.video_id
            )
            
            # Add course context to index
            self._add_course_context_to_index(video.video_id, course_id, video.order)
            
            video.status = ProcessingStatus.COMPLETED
            return {"status": ProcessingStatus.COMPLETED, "video_id": video.video_id}
            
        except Exception as e:
            video.status = ProcessingStatus.FAILED
            return {"status": ProcessingStatus.FAILED, "error": str(e)}
    
    def _create_cross_video_connections(self, course_id: str) -> None:
        """Create semantic connections between videos in the course."""
        course = self.courses[course_id]
        
        # Get all video transcripts
        transcripts = []
        for video in course.videos:
            if video.transcript:
                transcripts.append({
                    "video_id": video.video_id,
                    "title": video.title,
                    "order": video.order,
                    "transcript": video.transcript
                })
        
        # Create cross-video semantic links
        connections = self._analyze_cross_video_semantics(transcripts)
        
        # Store connections
        self._save_cross_video_connections(course_id, connections)
    
    def _analyze_cross_video_semantics(self, transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze semantic connections between video transcripts."""
        connections = []
        
        # Simple keyword-based connection analysis
        # In a real implementation, this would use more sophisticated NLP
        for i, video1 in enumerate(transcripts):
            for j, video2 in enumerate(transcripts[i+1:], i+1):
                # Find common topics/concepts
                common_concepts = self._find_common_concepts(
                    video1["transcript"], video2["transcript"]
                )
                
                if common_concepts:
                    connections.append({
                        "video1_id": video1["video_id"],
                        "video2_id": video2["video_id"],
                        "video1_title": video1["title"],
                        "video2_title": video2["title"],
                        "common_concepts": common_concepts,
                        "connection_strength": len(common_concepts) / 10.0,  # Normalized
                        "connection_type": "semantic"
                    })
        
        return connections
    
    def _find_common_concepts(self, text1: str, text2: str) -> List[str]:
        """Find common concepts between two texts."""
        # Simple implementation - in reality, use more sophisticated NLP
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Filter out common words
        common_words = {"le", "la", "les", "de", "du", "des", "et", "ou", "mais", "donc", "car", "que", "qui", "quoi", "où", "quand", "comment", "pourquoi"}
        
        common_concepts = list(words1.intersection(words2) - common_words)
        return common_concepts[:5]  # Return top 5 common concepts
    
    def query_course(self, request: CourseQueryRequest) -> CourseQueryResponse:
        """
        Query a course with cross-video retrieval capabilities.
        
        Args:
            request: Course query request
            
        Returns:
            Course query response with cross-video context
        """
        if request.course_id not in self.courses:
            return CourseQueryResponse(
                answer="Course not found",
                sources=[],
                confidence=0.0,
                processing_time=0.0,
                video_context={}
            )
        
        course = self.courses[request.course_id]
        start_time = time.time()
        
        # Determine which videos to search
        target_videos = request.target_videos or [v.video_id for v in course.videos if v.status == ProcessingStatus.COMPLETED]
        
        # Perform cross-video search
        if request.search_strategy == "cross_video":
            results = self._cross_video_search(request.query, target_videos, request.max_results, request.similarity_threshold)
        elif request.search_strategy == "single_video":
            results = self._single_video_search(request.query, target_videos, request.max_results, request.similarity_threshold)
        else:  # hybrid
            results = self._hybrid_search(request.query, target_videos, request.max_results, request.similarity_threshold)
        
        # Generate answer with cross-video context
        answer, confidence = self._generate_cross_video_answer(request.query, results, course)
        
        # Get cross-video connections
        connections = self._get_cross_video_connections(request.course_id)
        
        processing_time = time.time() - start_time
        
        return CourseQueryResponse(
            answer=answer,
            sources=results["sources"],
            confidence=confidence,
            processing_time=processing_time,
            video_context=results["video_context"],
            cross_video_connections=connections
        )
    
    def _cross_video_search(self, query: str, target_videos: List[str], max_results: int, threshold: float) -> Dict[str, Any]:
        """Perform cross-video semantic search."""
        all_sources = []
        video_context = {}
        
        for video_id in target_videos:
            # Search in each video
            video_results = self.rag_service.query_video(
                query=query,
                task_id=video_id,
                max_results=max_results // len(target_videos) + 1,
                similarity_threshold=threshold
            )
            
            if "sources" in video_results:
                for source in video_results["sources"]:
                    source["video_id"] = video_id
                    source["video_context"] = self._get_video_context(video_id)
                    all_sources.append(source)
                
                video_context[video_id] = {
                    "results_count": len(video_results["sources"]),
                    "confidence": video_results.get("confidence", 0.0)
                }
        
        # Sort by similarity score
        all_sources.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return {
            "sources": all_sources[:max_results],
            "video_context": video_context
        }
    
    def _single_video_search(self, query: str, target_videos: List[str], max_results: int, threshold: float) -> Dict[str, Any]:
        """Search in the most relevant single video."""
        best_video = None
        best_score = 0
        
        # Find the most relevant video
        for video_id in target_videos:
            # Simple relevance scoring based on video metadata
            video_context = self._get_video_context(video_id)
            score = self._calculate_video_relevance(query, video_context)
            
            if score > best_score:
                best_score = score
                best_video = video_id
        
        if best_video:
            results = self.rag_service.query_video(
                query=query,
                task_id=best_video,
                max_results=max_results,
                similarity_threshold=threshold
            )
            
            if "sources" in results:
                for source in results["sources"]:
                    source["video_id"] = best_video
                    source["video_context"] = self._get_video_context(best_video)
            
            return {
                "sources": results.get("sources", []),
                "video_context": {best_video: {"results_count": len(results.get("sources", [])), "confidence": results.get("confidence", 0.0)}}
            }
        
        return {"sources": [], "video_context": {}}
    
    def _hybrid_search(self, query: str, target_videos: List[str], max_results: int, threshold: float) -> Dict[str, Any]:
        """Combine cross-video and single-video search strategies."""
        # Get cross-video results
        cross_video_results = self._cross_video_search(query, target_videos, max_results // 2, threshold)
        
        # Get single-video results
        single_video_results = self._single_video_search(query, target_videos, max_results // 2, threshold)
        
        # Combine and deduplicate
        combined_sources = cross_video_results["sources"] + single_video_results["sources"]
        
        # Remove duplicates based on content similarity
        unique_sources = self._deduplicate_sources(combined_sources)
        
        # Merge video contexts
        combined_video_context = {**cross_video_results["video_context"], **single_video_results["video_context"]}
        
        return {
            "sources": unique_sources[:max_results],
            "video_context": combined_video_context
        }
    
    def _generate_cross_video_answer(self, query: str, results: Dict[str, Any], course) -> Tuple[str, float]:
        """Generate answer with cross-video context."""
        sources = results["sources"]
        video_context = results["video_context"]
        
        if not sources:
            return "Aucune information pertinente trouvée dans cette formation.", 0.0
        
        # Group sources by video
        sources_by_video = {}
        for source in sources:
            video_id = source.get("video_id")
            if video_id not in sources_by_video:
                sources_by_video[video_id] = []
            sources_by_video[video_id].append(source)
        
        # Generate answer with video context
        answer_parts = []
        confidence_scores = []
        
        for video_id, video_sources in sources_by_video.items():
            video_info = next((v for v in course.videos if v.video_id == video_id), None)
            if video_info:
                video_title = video_info.title
                answer_parts.append(f"Dans la vidéo '{video_title}':")
                
                for source in video_sources[:2]:  # Limit to top 2 sources per video
                    answer_parts.append(f"- {source.get('content', '')}")
                    confidence_scores.append(source.get('similarity', 0.0))
        
        answer = "\n\n".join(answer_parts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return answer, avg_confidence
    
    def _get_video_context(self, video_id: str) -> Dict[str, Any]:
        """Get context information for a video."""
        # This would typically load from stored metadata
        return {
            "video_id": video_id,
            "title": f"Video {video_id[:8]}",
            "duration": 0,
            "transcript_length": 0
        }
    
    def _calculate_video_relevance(self, query: str, video_context: Dict[str, Any]) -> float:
        """Calculate relevance score for a video."""
        # Simple keyword matching - in reality, use more sophisticated methods
        query_words = set(query.lower().split())
        title_words = set(video_context.get("title", "").lower().split())
        
        common_words = len(query_words.intersection(title_words))
        return common_words / len(query_words) if query_words else 0.0
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on content similarity."""
        unique_sources = []
        seen_contents = set()
        
        for source in sources:
            content = source.get("content", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_sources.append(source)
        
        return unique_sources
    
    def get_course_statistics(self, course_id: str) -> CourseStatistics:
        """Get detailed statistics for a course."""
        if course_id not in self.courses:
            raise ValueError("Course not found")
        
        course = self.courses[course_id]
        
        # Calculate statistics
        total_transcript_length = sum(len(v.transcript or "") for v in course.videos)
        total_chunks = 0  # Would be calculated from actual index data
        
        return CourseStatistics(
            course_id=course_id,
            total_videos=course.total_videos,
            processed_videos=course.processed_videos,
            total_transcript_length=total_transcript_length,
            total_chunks=total_chunks,
            average_chunk_size=total_transcript_length / max(total_chunks, 1),
            processing_time=0.0,  # Would be calculated from actual processing time
            created_at=course.created_at,
            last_updated=course.updated_at or course.created_at
        )
    
    def _save_course_metadata(self, course_id: str, course: CourseResponse) -> None:
        """Save course metadata to disk."""
        metadata_file = os.path.join(self.courses_dir, f"{course_id}_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(course.dict(), f, indent=2, default=str)
    
    def _save_cross_video_connections(self, course_id: str, connections: List[Dict[str, Any]]) -> None:
        """Save cross-video connections to disk."""
        connections_file = os.path.join(self.courses_dir, f"{course_id}_connections.json")
        with open(connections_file, 'w', encoding='utf-8') as f:
            json.dump(connections, f, indent=2, default=str)
    
    def _get_cross_video_connections(self, course_id: str) -> List[Dict[str, Any]]:
        """Load cross-video connections from disk."""
        connections_file = os.path.join(self.courses_dir, f"{course_id}_connections.json")
        if os.path.exists(connections_file):
            with open(connections_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _add_course_context_to_index(self, video_id: str, course_id: str, video_order: int) -> None:
        """Add course context to video index."""
        # This would add course metadata to the video's vector index
        # Implementation depends on the specific vector store being used
        pass
