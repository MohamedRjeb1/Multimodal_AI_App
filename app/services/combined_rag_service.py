"""
Service RAG combinatoire intégrant toutes les stratégies avancées.
"""
import time
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.config import get_settings
from app.models.schemas import ProcessingStatus

# Import des nouveaux services
from app.services.semantic_chunking_service import SemanticChunkingService
from app.services.summarization_service import SummarizationService
from app.services.memory_rag_service import MemoryRAGService
from app.services.corrective_rag_service import CorrectiveRAGService

# Import des services existants
from app.services.youtube_service import YouTubeService
from app.services.transcription_service import TranscriptionService
from app.services.embedding_service import AdvancedEmbeddingService
from app.services.indexing_service import AdvancedIndexingService


class CombinedRAGService:
    """
    Service RAG combinatoire qui orchestre :
    - Chunking sémantique dynamique
    - Indexation par résumé avec modèle fine-tuné
    - Memory RAG avec contexte historique
    - Corrective RAG avec grading et refinement
    - Clustering sémantique pour l'organisation
    - Cache optimisé pour la performance
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialiser tous les services
        self.youtube_service = YouTubeService()
        self.transcription_service = TranscriptionService()
        self.embedding_service = AdvancedEmbeddingService()
        self.indexing_service = AdvancedIndexingService()
        
        # Nouveaux services avancés
        self.semantic_chunker = SemanticChunkingService()
        self.summarizer = SummarizationService()
        self.memory_rag = MemoryRAGService()
        self.corrective_rag = CorrectiveRAGService()
        
        # Suivi du statut de traitement
        self.processing_status = {}
        
        # Configuration de la stratégie combinatoire
        self.strategy_config = {
            "chunking": {
                "strategy": "semantic_dynamic",
                "similarity_threshold": 0.65,
                "min_chunk_size": 100,
                "max_chunk_size": 2000
            },
            "summarization": {
                "strategy": "semantic_chunk",
                "max_summary_length": 200,
                "quality_threshold": 0.8
            },
            "memory_rag": {
                "similarity_threshold": 0.65,
                "context_window": 5,
                "max_history_length": 100
            },
            "corrective_rag": {
                "relevance_threshold": 0.7,
                "min_strips_for_response": 2,
                "strip_size": 200
            }
        }
    
    def _update_status(self, task_id: str, status: ProcessingStatus, message: str):
        """Met à jour le statut de traitement pour une tâche."""
        self.processing_status[task_id] = {
            "status": status,
            "message": message,
            "updated_at": datetime.now()
        }
    
    def process_video_with_combined_strategy(self, url: str, task_id: str, 
                                           language: Optional[str] = None) -> Dict[str, Any]:
        """
        Pipeline complet de traitement vidéo avec la stratégie combinatoire.
        
        Args:
            url: URL de la vidéo YouTube
            task_id: Identifiant unique de la tâche
            language: Langue de transcription (optionnel)
            
        Returns:
            Résultat complet du traitement
        """
        start_time = time.time()
        
        try:
            print(f"Début du traitement combinatoire pour la tâche {task_id}")
            
            # ÉTAPE 1: Téléchargement de la vidéo
            self._update_status(task_id, ProcessingStatus.DOWNLOADING, "Téléchargement de la vidéo...")
            download_result = self.youtube_service.download_video(url, task_id)
            
            if download_result["status"] == ProcessingStatus.FAILED:
                return download_result
            
            # ÉTAPE 2: Transcription audio
            self._update_status(task_id, ProcessingStatus.TRANSCRIBING, "Transcription audio...")
            audio_file = download_result["audio_file"]
            transcription_result = self.transcription_service.transcribe_audio(
                audio_file, task_id, language
            )
            
            if transcription_result["status"] == ProcessingStatus.FAILED:
                return transcription_result
            
            transcript = transcription_result["transcript"]
            
            # ÉTAPE 3: Chunking sémantique dynamique
            self._update_status(task_id, ProcessingStatus.EMBEDDING, "Chunking sémantique dynamique...")
            semantic_chunks = self.semantic_chunker.create_semantic_chunks(transcript, task_id)
            
            if not semantic_chunks:
                return {
                    "task_id": task_id,
                    "status": ProcessingStatus.FAILED,
                    "message": "Échec du chunking sémantique"
                }
            
            chunking_stats = self.semantic_chunker.get_chunking_statistics(semantic_chunks)
            
            # ÉTAPE 4: Résumé des chunks pour l'indexation
            self._update_status(task_id, ProcessingStatus.INDEXING, "Génération des résumés pour l'indexation...")
            summarized_chunks = self.summarizer.summarize_chunks(
                semantic_chunks, 
                summary_type="semantic_chunk"
            )
            
            # Sauvegarder les résumés
            summaries_file = self.summarizer.save_summaries(summarized_chunks, task_id)
            
            # ÉTAPE 5: Génération des embeddings pour les résumés
            self._update_status(task_id, ProcessingStatus.INDEXING, "Génération des embeddings des résumés...")
            
            # Créer les documents pour l'embedding
            from langchain.schema import Document
            documents = []
            embedding_results = []
            
            for chunk in summarized_chunks:
                # Utiliser le résumé comme contenu principal
                doc = Document(
                    page_content=chunk["summary"],
                    metadata={
                        **chunk["metadata"],
                        "original_content": chunk["original_chunk"]["content"],
                        "summary_metadata": chunk["summary_metadata"]
                    }
                )
                documents.append(doc)
                
                # Générer l'embedding du résumé
                try:
                    embedding = self.embedding_service.embeddings.embed_query(chunk["summary"])
                    embedding_results.append({
                        "document": doc,
                        "embedding": embedding,
                        "original_chunk": chunk["original_chunk"],
                        "summary": chunk["summary"]
                    })
                except Exception as e:
                    print(f"Erreur lors de la génération d'embedding pour le chunk {chunk['metadata']['chunk_id']}: {str(e)}")
                    continue
            
            # ÉTAPE 6: Indexation hybride avec clustering
            self._update_status(task_id, ProcessingStatus.INDEXING, "Création de l'index hybride avec clustering...")
            indexing_result = self.indexing_service.create_hybrid_index(embedding_results, task_id)
            
            if indexing_result["status"] == ProcessingStatus.FAILED:
                return indexing_result
            
            # ÉTAPE 7: Finalisation
            processing_time = time.time() - start_time
            self._update_status(task_id, ProcessingStatus.COMPLETED, "Traitement combinatoire terminé avec succès")
            
            return {
                "task_id": task_id,
                "status": ProcessingStatus.COMPLETED,
                "message": "Traitement combinatoire terminé avec succès",
                "processing_time": processing_time,
                "video_title": download_result.get("video_title", "Unknown"),
                "duration": download_result.get("duration", 0),
                "transcript_length": len(transcript),
                
                # Statistiques du chunking sémantique
                "semantic_chunking_stats": chunking_stats,
                
                # Statistiques du résumé
                "summarization_stats": {
                    "total_chunks_summarized": len(summarized_chunks),
                    "avg_summary_quality": sum(
                        chunk["summary_metadata"]["quality_score"] 
                        for chunk in summarized_chunks
                    ) / len(summarized_chunks) if summarized_chunks else 0.0,
                    "summaries_file": summaries_file
                },
                
                # Statistiques de l'indexation
                "indexing_stats": {
                    "total_chunks_indexed": indexing_result.get("total_chunks", 0),
                    "clusters_created": indexing_result.get("clusters", 0),
                    "silhouette_score": indexing_result.get("silhouette_score", 0),
                    "index_type": "hybrid_with_summaries"
                },
                
                "strategy_used": "combined_semantic_summarization_clustering",
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._update_status(task_id, ProcessingStatus.FAILED, f"Traitement échoué: {str(e)}")
            return {
                "task_id": task_id,
                "status": ProcessingStatus.FAILED,
                "message": f"Traitement combinatoire échoué: {str(e)}",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def query_with_combined_strategy(self, query: str, task_id: str, user_id: Optional[str] = None,
                                   session_id: Optional[str] = None, max_results: int = 5) -> Dict[str, Any]:
        """
        Traitement de requête avec la stratégie combinatoire Memory RAG + Corrective RAG.
        
        Args:
            query: Requête utilisateur
            task_id: Identifiant de la tâche
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
            max_results: Nombre maximum de résultats
            
        Returns:
            Résultat de la requête avec contexte enrichi
        """
        start_time = time.time()
        
        try:
            print(f"Début de la requête combinatoire pour: {query[:100]}...")
            
            # ÉTAPE 1: Memory RAG - Récupération du contexte historique
            print("Étape 1: Récupération du contexte historique avec Memory RAG...")
            memory_context = self.memory_rag.retrieve_memory_context(
                query=query,
                user_id=user_id,
                session_id=session_id
            )
            
            print(f"Contexte mémoire récupéré: {memory_context['similar_queries_count']} requêtes similaires")
            
            # ÉTAPE 2: Récupération des documents avec contexte enrichi
            print("Étape 2: Récupération des documents...")
            if not self.indexing_service.load_index(task_id):
                return {
                    "error": "Index not found",
                    "message": f"Aucun index trouvé pour la tâche {task_id}. Veuillez traiter la vidéo d'abord."
                }
            
            # Recherche avec seuil de similarité
            similar_chunks = self.indexing_service.search_similar(
                query=query,
                task_id=task_id,
                k=max_results,
                similarity_threshold=0.7
            )
            
            if not similar_chunks:
                return {
                    "answer": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                    "sources": [],
                    "confidence": 0.0,
                    "memory_context": memory_context,
                    "processing_time": time.time() - start_time
                }
            
            # ÉTAPE 3: Corrective RAG - Grading et refinement
            print("Étape 3: Traitement avec Corrective RAG...")
            corrective_result = self.corrective_rag.process_corrective_rag(
                query=query,
                documents=similar_chunks,
                memory_context=memory_context
            )
            
            # ÉTAPE 4: Mise à jour de l'historique Memory RAG
            print("Étape 4: Mise à jour de l'historique...")
            self.memory_rag.add_query_to_history(
                query=query,
                response=corrective_result["answer"],
                user_id=user_id,
                session_id=session_id,
                context={
                    "task_id": task_id,
                    "sources_count": len(corrective_result["sources"]),
                    "confidence": corrective_result["confidence"],
                    "memory_context_used": memory_context["has_context"]
                }
            )
            
            # ÉTAPE 5: Préparation du résultat final
            processing_time = time.time() - start_time
            
            result = {
                "answer": corrective_result["answer"],
                "sources": corrective_result["sources"],
                "confidence": corrective_result["confidence"],
                
                # Contexte mémoire utilisé
                "memory_context": {
                    "used": memory_context["has_context"],
                    "similar_queries_count": memory_context["similar_queries_count"],
                    "avg_similarity": memory_context["avg_similarity"],
                    "context_sources": memory_context["context_sources"]
                },
                
                # Statistiques du Corrective RAG
                "corrective_rag_stats": {
                    "relevant_strips_count": corrective_result["relevant_strips_count"],
                    "total_strips_evaluated": corrective_result["total_strips_evaluated"],
                    "memory_context_used": corrective_result["memory_context_used"]
                },
                
                "processing_time": processing_time,
                "query": query,
                "task_id": task_id,
                "user_id": user_id,
                "session_id": session_id,
                "strategy_used": "combined_memory_corrective_rag",
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Requête combinatoire terminée en {processing_time:.2f}s - Confiance: {corrective_result['confidence']:.2f}")
            return result
            
        except Exception as e:
            print(f"Erreur lors de la requête combinatoire: {str(e)}")
            return {
                "error": "Erreur lors du traitement de la requête",
                "message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def get_combined_strategy_statistics(self, task_id: str) -> Dict[str, Any]:
        """
        Obtient les statistiques complètes de la stratégie combinatoire.
        
        Args:
            task_id: Identifiant de la tâche
            
        Returns:
            Statistiques complètes
        """
        try:
            stats = {
                "task_id": task_id,
                "strategy_config": self.strategy_config,
                "timestamp": datetime.now().isoformat()
            }
            
            # Statistiques de l'index
            index_stats = self.indexing_service.get_index_statistics(task_id)
            if index_stats:
                stats["index_statistics"] = index_stats
            
            # Statistiques Memory RAG
            memory_stats = self.memory_rag.get_history_statistics()
            stats["memory_rag_statistics"] = memory_stats
            
            # Statistiques des résumés
            summaries_data = self.summarizer.load_summaries(task_id)
            if summaries_data:
                stats["summarization_statistics"] = {
                    "total_chunks": summaries_data["total_chunks"],
                    "avg_quality_score": summaries_data["avg_quality_score"],
                    "summarization_model": summaries_data["summarization_model"]
                }
            
            return stats
            
        except Exception as e:
            return {
                "error": f"Erreur lors de l'obtention des statistiques: {str(e)}",
                "task_id": task_id
            }
    
    def cleanup_combined_task(self, task_id: str) -> bool:
        """
        Nettoie toutes les données d'une tâche pour la stratégie combinatoire.
        
        Args:
            task_id: Identifiant de la tâche
            
        Returns:
            True si le nettoyage a réussi
        """
        try:
            # Nettoyer les services existants
            self.youtube_service.cleanup_audio_file(task_id)
            self.transcription_service.delete_transcript(task_id)
            self.indexing_service.delete_index(task_id)
            
            # Nettoyer les nouveaux fichiers
            summaries_file = os.path.join(self.settings.VECTORSTORE_DIR, f"{task_id}_summaries.json")
            if os.path.exists(summaries_file):
                os.remove(summaries_file)
            
            # Supprimer du statut de traitement
            if task_id in self.processing_status:
                del self.processing_status[task_id]
            
            print(f"Nettoyage combinatoire terminé pour la tâche {task_id}")
            return True
            
        except Exception as e:
            print(f"Erreur lors du nettoyage combinatoire: {str(e)}")
            return False
