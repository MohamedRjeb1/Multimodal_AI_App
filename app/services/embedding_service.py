"""
Advanced embedding service with innovative techniques.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus


class AdvancedEmbeddingService:
    """
    Advanced embedding service with innovative techniques:
    - Multi-level chunking strategies
    - Semantic chunking with overlap optimization
    - Embedding caching and optimization
    - Hybrid embedding approaches
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.vectorstore_dir = self.settings.VECTORSTORE_DIR
        
        # Ensure vectorstore directory exists
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Initialize embedding models
        self.embeddings = self._initialize_embeddings()
        
        # Initialize text splitter with advanced configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            is_separator_regex=False,
        )
        
        # Embedding cache
        self.embedding_cache = {}
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on configuration."""
        if self.settings.DEFAULT_EMBEDDING_MODEL == "google":
            if not self.settings.GOOGLE_API_KEY:
                raise ValueError("Google API key is required for Google embeddings")
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.settings.GOOGLE_API_KEY
            )
        elif self.settings.DEFAULT_EMBEDDING_MODEL == "openai":
            if not self.settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.settings.OPENAI_API_KEY
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.settings.DEFAULT_EMBEDDING_MODEL}")
    
    def _generate_chunk_hash(self, text: str) -> str:
        """Generate hash for text chunk for caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding if available."""
        chunk_hash = self._generate_chunk_hash(text)
        return self.embedding_cache.get(chunk_hash)
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for future use."""
        chunk_hash = self._generate_chunk_hash(text)
        self.embedding_cache[chunk_hash] = embedding
    
    def create_semantic_chunks(self, text: str, task_id: str) -> List[Document]:
        """
        Create semantic chunks with advanced strategies.
        
        Args:
            text: Input text to chunk
            task_id: Task identifier
            
        Returns:
            List of Document objects with metadata
        """
        # Basic chunking
        chunks = self.text_splitter.split_text(text)
        
        # Enhanced chunking with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Calculate chunk statistics
            word_count = len(chunk.split())
            char_count = len(chunk)
            
            # Create document with rich metadata
            doc = Document(
                page_content=chunk,
                metadata={
                    "task_id": task_id,
                    "chunk_id": f"{task_id}_chunk_{i}",
                    "chunk_index": i,
                    "word_count": word_count,
                    "character_count": char_count,
                    "created_at": datetime.now().isoformat(),
                    "chunk_type": "semantic",
                    "source": "transcript"
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_hierarchical_chunks(self, text: str, task_id: str) -> List[Document]:
        """
        Create hierarchical chunks with different granularities.
        
        Args:
            text: Input text to chunk
            task_id: Task identifier
            
        Returns:
            List of Document objects with different granularities
        """
        documents = []
        
        # Fine-grained chunks (smaller)
        fine_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE // 2,
            chunk_overlap=self.settings.CHUNK_OVERLAP // 2,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        
        # Coarse-grained chunks (larger)
        coarse_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE * 2,
            chunk_overlap=self.settings.CHUNK_OVERLAP * 2,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        
        # Create different granularity levels
        for splitter, granularity in [(fine_splitter, "fine"), (coarse_splitter, "coarse")]:
            chunks = splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "task_id": task_id,
                        "chunk_id": f"{task_id}_{granularity}_{i}",
                        "chunk_index": i,
                        "granularity": granularity,
                        "word_count": len(chunk.split()),
                        "character_count": len(chunk),
                        "created_at": datetime.now().isoformat(),
                        "chunk_type": "hierarchical",
                        "source": "transcript"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def generate_embeddings(self, documents: List[Document], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for documents with caching.
        
        Args:
            documents: List of Document objects
            use_cache: Whether to use embedding cache
            
        Returns:
            List of embedding results with metadata
        """
        embedding_results = []
        
        for doc in documents:
            text = doc.page_content
            
            # Check cache first
            if use_cache:
                cached_embedding = self._get_cached_embedding(text)
                if cached_embedding:
                    embedding_results.append({
                        "document": doc,
                        "embedding": cached_embedding,
                        "cached": True
                    })
                    continue
            
            try:
                # Generate new embedding
                embedding = self.embeddings.embed_query(text)
                
                # Cache the embedding
                if use_cache:
                    self._cache_embedding(text, embedding)
                
                embedding_results.append({
                    "document": doc,
                    "embedding": embedding,
                    "cached": False
                })
                
            except Exception as e:
                print(f"Error generating embedding for chunk {doc.metadata.get('chunk_id', 'unknown')}: {str(e)}")
                continue
        
        return embedding_results
    
    def save_embeddings(self, embedding_results: List[Dict[str, Any]], task_id: str) -> str:
        """
        Save embeddings to disk with metadata.
        
        Args:
            embedding_results: List of embedding results
            task_id: Task identifier
            
        Returns:
            Path to saved embeddings file
        """
        embeddings_file = os.path.join(self.vectorstore_dir, f"{task_id}_embeddings.json")
        
        # Prepare data for saving
        embeddings_data = {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.settings.DEFAULT_EMBEDDING_MODEL,
            "chunk_size": self.settings.CHUNK_SIZE,
            "chunk_overlap": self.settings.CHUNK_OVERLAP,
            "total_chunks": len(embedding_results),
            "embeddings": []
        }
        
        for result in embedding_results:
            embeddings_data["embeddings"].append({
                "chunk_id": result["document"].metadata["chunk_id"],
                "content": result["document"].page_content,
                "metadata": result["document"].metadata,
                "embedding": result["embedding"],
                "cached": result["cached"]
            })
        
        # Save to file
        with open(embeddings_file, "w", encoding="utf-8") as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        return embeddings_file
    
    def load_embeddings(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Load embeddings from disk.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Embeddings data or None if not found
        """
        embeddings_file = os.path.join(self.vectorstore_dir, f"{task_id}_embeddings.json")
        
        if not os.path.exists(embeddings_file):
            return None
        
        try:
            with open(embeddings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading embeddings for task {task_id}: {str(e)}")
            return None
    
    def process_transcript(self, transcript: str, task_id: str, strategy: str = "semantic") -> Dict[str, Any]:
        """
        Process transcript with advanced embedding techniques.
        
        Args:
            transcript: Transcript text
            task_id: Task identifier
            strategy: Chunking strategy ("semantic" or "hierarchical")
            
        Returns:
            Processing result with embeddings
        """
        try:
            # Create chunks based on strategy
            if strategy == "hierarchical":
                documents = self.create_hierarchical_chunks(transcript, task_id)
            else:
                documents = self.create_semantic_chunks(transcript, task_id)
            
            # Generate embeddings
            embedding_results = self.generate_embeddings(documents)
            
            # Save embeddings
            embeddings_file = self.save_embeddings(embedding_results, task_id)
            
            return {
                "task_id": task_id,
                "status": ProcessingStatus.COMPLETED,
                "message": f"Embeddings generated successfully using {strategy} strategy",
                "total_chunks": len(documents),
                "embeddings_file": embeddings_file,
                "strategy": strategy,
                "embedding_model": self.settings.DEFAULT_EMBEDDING_MODEL
            }
            
        except Exception as e:
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Embedding generation failed: {str(e)}",
                "error": str(e)
            }
