"""
Service Memory RAG avec recherche de contexte historique.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import get_settings
from app.services.embedding_service import AdvancedEmbeddingService


class MemoryRAGService:
    """
    Service Memory RAG qui :
    - Maintient un historique des requêtes et réponses
    - Recherche des requêtes similaires dans l'historique
    - Enrichit le contexte de retrieval avec l'historique souhaitable
    - Utilise un seuil de similarité de 65% pour la recherche
    - Supporte le contexte utilisateur et de session
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = AdvancedEmbeddingService()
        
        # Configuration du Memory RAG
        self.similarity_threshold = 0.65  # Seuil de similarité pour recherche historique
        self.max_history_length = 100  # Longueur maximale de l'historique
        self.context_window = 5  # Nombre de requêtes similaires à considérer
        
        # Stockage de l'historique
        self.query_history = []  # Historique global
        self.user_histories = defaultdict(list)  # Historique par utilisateur
        self.session_histories = defaultdict(list)  # Historique par session
        
        # Fichier de persistance
        self.history_file = os.path.join(self.settings.VECTORSTORE_DIR, "memory_rag_history.json")
        
        # Charger l'historique existant
        self._load_history()
    
    def _load_history(self):
        """Charge l'historique depuis le fichier de persistance."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                
                self.query_history = history_data.get("global_history", [])
                self.user_histories = defaultdict(list, history_data.get("user_histories", {}))
                self.session_histories = defaultdict(list, history_data.get("session_histories", {}))
                
                print(f"Historique Memory RAG chargé: {len(self.query_history)} requêtes globales")
                
        except Exception as e:
            print(f"Erreur lors du chargement de l'historique: {str(e)}")
            self.query_history = []
            self.user_histories = defaultdict(list)
            self.session_histories = defaultdict(list)
    
    def _save_history(self):
        """Sauvegarde l'historique dans le fichier de persistance."""
        try:
            history_data = {
                "global_history": self.query_history,
                "user_histories": dict(self.user_histories),
                "session_histories": dict(self.session_histories),
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calcule la similarité cosinus entre deux embeddings."""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Erreur lors du calcul de similarité: {str(e)}")
            return 0.0
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Génère l'embedding d'une requête."""
        try:
            return self.embedding_service.embeddings.embed_query(query)
        except Exception as e:
            print(f"Erreur lors de la génération d'embedding: {str(e)}")
            return []
    
    def add_query_to_history(self, query: str, response: str, user_id: Optional[str] = None, 
                           session_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Ajoute une requête et sa réponse à l'historique.
        
        Args:
            query: Requête utilisateur
            response: Réponse générée
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
            context: Contexte supplémentaire (optionnel)
        """
        try:
            # Générer l'embedding de la requête
            query_embedding = self._get_query_embedding(query)
            
            if not query_embedding:
                return
            
            # Créer l'entrée d'historique
            history_entry = {
                "query": query,
                "response": response,
                "query_embedding": query_embedding,
                "user_id": user_id,
                "session_id": session_id,
                "context": context or {},
                "timestamp": datetime.now().isoformat(),
                "query_length": len(query.split()),
                "response_length": len(response.split())
            }
            
            # Ajouter à l'historique global
            self.query_history.append(history_entry)
            
            # Ajouter aux historiques spécifiques
            if user_id:
                self.user_histories[user_id].append(history_entry)
            
            if session_id:
                self.session_histories[session_id].append(history_entry)
            
            # Limiter la taille de l'historique
            self._trim_history()
            
            # Sauvegarder
            self._save_history()
            
        except Exception as e:
            print(f"Erreur lors de l'ajout à l'historique: {str(e)}")
    
    def _trim_history(self):
        """Limite la taille de l'historique pour éviter la surcharge mémoire."""
        # Trier par timestamp (plus récent en premier)
        self.query_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Garder seulement les entrées les plus récentes
        if len(self.query_history) > self.max_history_length:
            self.query_history = self.query_history[:self.max_history_length]
        
        # Faire de même pour les historiques utilisateur et session
        for user_id in self.user_histories:
            self.user_histories[user_id].sort(key=lambda x: x["timestamp"], reverse=True)
            if len(self.user_histories[user_id]) > self.max_history_length // 2:
                self.user_histories[user_id] = self.user_histories[user_id][:self.max_history_length // 2]
        
        for session_id in self.session_histories:
            self.session_histories[session_id].sort(key=lambda x: x["timestamp"], reverse=True)
            if len(self.session_histories[session_id]) > self.max_history_length // 4:
                self.session_histories[session_id] = self.session_histories[session_id][:self.max_history_length // 4]
    
    def find_similar_queries(self, query: str, user_id: Optional[str] = None, 
                           session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Trouve des requêtes similaires dans l'historique.
        
        Args:
            query: Requête actuelle
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
            
        Returns:
            Liste des requêtes similaires avec scores
        """
        try:
            # Générer l'embedding de la requête actuelle
            query_embedding = self._get_query_embedding(query)
            
            if not query_embedding:
                return []
            
            similar_queries = []
            
            # Rechercher dans l'historique global
            for entry in self.query_history:
                similarity = self._calculate_similarity(query_embedding, entry["query_embedding"])
                
                if similarity >= self.similarity_threshold:
                    similar_queries.append({
                        "query": entry["query"],
                        "response": entry["response"],
                        "similarity": similarity,
                        "timestamp": entry["timestamp"],
                        "source": "global",
                        "user_id": entry["user_id"],
                        "session_id": entry["session_id"],
                        "context": entry["context"]
                    })
            
            # Rechercher dans l'historique utilisateur si spécifié
            if user_id and user_id in self.user_histories:
                for entry in self.user_histories[user_id]:
                    similarity = self._calculate_similarity(query_embedding, entry["query_embedding"])
                    
                    if similarity >= self.similarity_threshold:
                        similar_queries.append({
                            "query": entry["query"],
                            "response": entry["response"],
                            "similarity": similarity,
                            "timestamp": entry["timestamp"],
                            "source": "user",
                            "user_id": entry["user_id"],
                            "session_id": entry["session_id"],
                            "context": entry["context"]
                        })
            
            # Rechercher dans l'historique de session si spécifié
            if session_id and session_id in self.session_histories:
                for entry in self.session_histories[session_id]:
                    similarity = self._calculate_similarity(query_embedding, entry["query_embedding"])
                    
                    if similarity >= self.similarity_threshold:
                        similar_queries.append({
                            "query": entry["query"],
                            "response": entry["response"],
                            "similarity": similarity,
                            "timestamp": entry["timestamp"],
                            "source": "session",
                            "user_id": entry["user_id"],
                            "session_id": entry["session_id"],
                            "context": entry["context"]
                        })
            
            # Dédupliquer et trier par similarité
            seen_queries = set()
            unique_similar_queries = []
            
            for sq in similar_queries:
                query_key = sq["query"]
                if query_key not in seen_queries:
                    seen_queries.add(query_key)
                    unique_similar_queries.append(sq)
            
            # Trier par similarité (décroissant)
            unique_similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limiter au nombre de contextes souhaités
            return unique_similar_queries[:self.context_window]
            
        except Exception as e:
            print(f"Erreur lors de la recherche de requêtes similaires: {str(e)}")
            return []
    
    def retrieve_memory_context(self, query: str, user_id: Optional[str] = None, 
                              session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère le contexte de mémoire pour enrichir la requête.
        
        Args:
            query: Requête actuelle
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
            
        Returns:
            Contexte de mémoire enrichi
        """
        try:
            # Trouver les requêtes similaires
            similar_queries = self.find_similar_queries(query, user_id, session_id)
            
            if not similar_queries:
                return {
                    "has_context": False,
                    "similar_queries": [],
                    "enriched_context": "",
                    "context_sources": [],
                    "avg_similarity": 0.0
                }
            
            # Construire le contexte enrichi
            enriched_context_parts = []
            context_sources = []
            
            for sq in similar_queries:
                # Ajouter la requête et réponse similaires au contexte
                context_part = f"Requête similaire: {sq['query']}\nRéponse: {sq['response']}"
                enriched_context_parts.append(context_part)
                context_sources.append(sq["source"])
            
            enriched_context = "\n\n".join(enriched_context_parts)
            
            # Calculer la similarité moyenne
            avg_similarity = sum(sq["similarity"] for sq in similar_queries) / len(similar_queries)
            
            return {
                "has_context": True,
                "similar_queries": similar_queries,
                "enriched_context": enriched_context,
                "context_sources": list(set(context_sources)),
                "avg_similarity": avg_similarity,
                "context_length": len(enriched_context.split()),
                "similar_queries_count": len(similar_queries)
            }
            
        except Exception as e:
            print(f"Erreur lors de la récupération du contexte mémoire: {str(e)}")
            return {
                "has_context": False,
                "similar_queries": [],
                "enriched_context": "",
                "context_sources": [],
                "avg_similarity": 0.0,
                "error": str(e)
            }
    
    def get_history_statistics(self, user_id: Optional[str] = None, 
                             session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtient les statistiques de l'historique.
        
        Args:
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
            
        Returns:
            Statistiques de l'historique
        """
        try:
            stats = {
                "global_history_count": len(self.query_history),
                "total_users": len(self.user_histories),
                "total_sessions": len(self.session_histories),
                "similarity_threshold": self.similarity_threshold,
                "max_history_length": self.max_history_length
            }
            
            if user_id and user_id in self.user_histories:
                stats["user_history_count"] = len(self.user_histories[user_id])
                stats["user_id"] = user_id
            
            if session_id and session_id in self.session_histories:
                stats["session_history_count"] = len(self.session_histories[session_id])
                stats["session_id"] = session_id
            
            return stats
            
        except Exception as e:
            print(f"Erreur lors de l'obtention des statistiques: {str(e)}")
            return {"error": str(e)}
    
    def clear_history(self, user_id: Optional[str] = None, session_id: Optional[str] = None):
        """
        Efface l'historique.
        
        Args:
            user_id: Identifiant utilisateur (optionnel)
            session_id: Identifiant de session (optionnel)
        """
        try:
            if user_id:
                if user_id in self.user_histories:
                    del self.user_histories[user_id]
                print(f"Historique utilisateur {user_id} effacé")
            
            elif session_id:
                if session_id in self.session_histories:
                    del self.session_histories[session_id]
                print(f"Historique session {session_id} effacé")
            
            else:
                # Effacer tout l'historique
                self.query_history = []
                self.user_histories = defaultdict(list)
                self.session_histories = defaultdict(list)
                print("Tout l'historique Memory RAG effacé")
            
            self._save_history()
            
        except Exception as e:
            print(f"Erreur lors de l'effacement de l'historique: {str(e)}")
