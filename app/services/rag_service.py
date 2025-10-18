"""
Advanced RAG service orchestrating the complete pipeline.
"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus
from app.services.youtube_service import YouTubeService
from app.services.transcription_service import TranscriptionService
from app.services.embedding_service import AdvancedEmbeddingService
from app.services.indexing_service import AdvancedIndexingService


class AdvancedRAGService:
    """
    Advanced RAG service that orchestrates the complete pipeline:
    - YouTube video processing
    - Audio transcription
    - Advanced embedding generation
    - Intelligent indexing
    - Query processing and response generation
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize services
        self.youtube_service = YouTubeService()
        self.transcription_service = TranscriptionService()
        self.embedding_service = AdvancedEmbeddingService()
        self.indexing_service = AdvancedIndexingService()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Processing status tracking
        self.processing_status = {}
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        if self.settings.DEFAULT_LLM_MODEL.startswith("gemini"):
            if not self.settings.GOOGLE_API_KEY:
                raise ValueError("Google API key is required for Gemini")
            return ChatGoogleGenerativeAI(
                model=self.settings.DEFAULT_LLM_MODEL,
                google_api_key=self.settings.GOOGLE_API_KEY,
                temperature=0.1
            )
        elif self.settings.DEFAULT_LLM_MODEL.startswith("gpt"):
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for GPT")
            return ChatOpenAI(
                model=self.settings.DEFAULT_LLM_MODEL,
                openai_api_key=self.settings.OPENAI_API_KEY,
                temperature=0.1
            )
        else:
            raise ValueError(f"Unsupported LLM model: {self.settings.DEFAULT_LLM_MODEL}")
    
    def _update_status(self, task_id: str, status: ProcessingStatus, message: str):
        """Update processing status for a task."""
        self.processing_status[task_id] = {
            "status": status,
            "message": message,
            "updated_at": datetime.now()
        }
    
    def process_video(self, url: str, task_id: str, language: Optional[str] = None, 
                     embedding_strategy: str = "semantic") -> Dict[str, Any]:
        """
        Complete video processing pipeline.
        
        Args:
            url: YouTube video URL
            task_id: Unique task identifier
            language: Transcription language
            embedding_strategy: Embedding strategy ("semantic" or "hierarchical")
            
        Returns:
            Complete processing result
        """
        start_time = time.time()
        
        try:
            # Step 1: Download video
            self._update_status(task_id, ProcessingStatus.DOWNLOADING, "Downloading video...")
            download_result = self.youtube_service.download_video(url, task_id)
            
            if download_result["status"] == ProcessingStatus.FAILED:
                return download_result
            
            # Step 2: Transcribe audio
            self._update_status(task_id, ProcessingStatus.TRANSCRIBING, "Transcribing audio...")
            audio_file = download_result["audio_file"]
            transcription_result = self.transcription_service.transcribe_audio(
                audio_file, task_id, language
            )
            
            if transcription_result["status"] == ProcessingStatus.FAILED:
                return transcription_result
            
            # Step 3: Generate embeddings
            self._update_status(task_id, ProcessingStatus.EMBEDDING, "Generating embeddings...")
            transcript = transcription_result["transcript"]
            embedding_result = self.embedding_service.process_transcript(
                transcript, task_id, embedding_strategy
            )
            
            if embedding_result["status"] == ProcessingStatus.FAILED:
                return embedding_result
            
            # Step 4: Create index
            self._update_status(task_id, ProcessingStatus.INDEXING, "Creating index...")
            # Load embeddings for indexing
            embeddings_data = self.embedding_service.load_embeddings(task_id)
            if not embeddings_data:
                return {
                    "task_id": task_id,
                    "status": ProcessingStatus.FAILED,
                    "message": "Failed to load embeddings for indexing"
                }
            
            # Convert to embedding results format
            embedding_results = []
            for emb_data in embeddings_data["embeddings"]:
                from langchain.schema import Document
                doc = Document(
                    page_content=emb_data["content"],
                    metadata=emb_data["metadata"]
                )
                embedding_results.append({
                    "document": doc,
                    "embedding": emb_data["embedding"]
                })
            
            indexing_result = self.indexing_service.create_hybrid_index(embedding_results, task_id)
            
            if indexing_result["status"] == ProcessingStatus.FAILED:
                return indexing_result
            
            # Step 5: Complete
            processing_time = time.time() - start_time
            self._update_status(task_id, ProcessingStatus.COMPLETED, "Processing completed successfully")
            
            return {
                "task_id": task_id,
                "status": ProcessingStatus.COMPLETED,
                "message": "Video processing completed successfully",
                "processing_time": processing_time,
                "video_title": download_result.get("video_title", "Unknown"),
                "duration": download_result.get("duration", 0),
                "transcript_length": len(transcript),
                "total_chunks": embedding_result.get("total_chunks", 0),
                "clusters": indexing_result.get("clusters", 0),
                "silhouette_score": indexing_result.get("silhouette_score", 0),
                "embedding_strategy": embedding_strategy,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._update_status(task_id, ProcessingStatus.FAILED, f"Processing failed: {str(e)}")
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Processing failed: {str(e)}",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def query_video(self, query: str, task_id: str, max_results: int = 5, 
                   similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Query processed video using RAG.
        
        Args:
            query: User question
            task_id: Task identifier
            max_results: Maximum number of results
            similarity_threshold: Similarity threshold
            
        Returns:
            RAG query result
        """
        start_time = time.time()
        
        try:
            # Check if index exists
            if not self.indexing_service.load_index(task_id):
                return {
                    "error": "Index not found",
                    "message": f"No index found for task {task_id}. Please process the video first."
                }
            
            # Search for relevant chunks
            similar_chunks = self.indexing_service.search_similar(
                query, task_id, k=max_results, similarity_threshold=similarity_threshold
            )
            
            if not similar_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start_time
                }
            
            # Prepare context from similar chunks
            context = "\n\n".join([chunk["content"] for chunk in similar_chunks])
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Vous êtes un assistant IA spécialisé dans l'analyse de contenu vidéo. 
                Utilisez le contexte fourni pour répondre à la question de l'utilisateur.
                
                Contexte:
                {context}
                
                Question: {question}
                
                Réponse: Fournissez une réponse claire et précise basée sur le contexte. 
                Si le contexte ne contient pas suffisamment d'informations, indiquez-le clairement.
                """
            )
            
            # Generate answer
            formatted_prompt = prompt_template.format(context=context, question=query)
            response = self.llm.invoke(formatted_prompt)
            
            # Calculate confidence based on similarity scores
            avg_similarity = sum(chunk["similarity_score"] for chunk in similar_chunks) / len(similar_chunks)
            confidence = min(avg_similarity, 1.0)
            
            # Prepare sources
            sources = []
            for i, chunk in enumerate(similar_chunks):
                sources.append({
                    "chunk_id": chunk["metadata"].get("chunk_id", f"chunk_{i}"),
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "similarity_score": chunk["similarity_score"],
                    "metadata": chunk["metadata"]
                })
            
            return {
                "answer": response.content if hasattr(response, 'content') else str(response),
                "sources": sources,
                "confidence": confidence,
                "processing_time": time.time() - start_time,
                "total_sources": len(similar_chunks),
                "query": query,
                "task_id": task_id
            }
            
        except Exception as e:
            return {
                "error": "Query processing failed",
                "message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_processing_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current processing status for a task."""
        return self.processing_status.get(task_id)
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without processing."""
        try:
            return self.youtube_service.get_video_info(url)
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_task(self, task_id: str) -> bool:
        """Clean up all files and data for a task."""
        try:
            # Clean up YouTube service
            self.youtube_service.cleanup_audio_file(task_id)
            
            # Clean up transcription service
            self.transcription_service.delete_transcript(task_id)
            
            # Clean up indexing service
            self.indexing_service.delete_index(task_id)
            
            # Remove from processing status
            if task_id in self.processing_status:
                del self.processing_status[task_id]
            
            return True
            
        except Exception as e:
            print(f"Error cleaning up task {task_id}: {str(e)}")
            return False
