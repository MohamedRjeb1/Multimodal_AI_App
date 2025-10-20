"""
Service Corrective RAG avec grading et refinement intelligent.
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from app.core.config import get_settings
from app.services.embedding_service import AdvancedEmbeddingService


class CorrectiveRAGService:
    """
    Service Corrective RAG qui :
    - Récupère des documents et évalue leur pertinence
    - Décompose les documents en "knowledge strips"
    - Grade chaque strip selon la pertinence
    - Filtre les strips non pertinents
    - Recherche des informations supplémentaires si nécessaire
    - Génère une réponse finale basée sur les informations les plus pertinentes
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = AdvancedEmbeddingService()
        self.llm = self._initialize_llm()
        
        # Configuration du Corrective RAG
        self.relevance_threshold = 0.7  # Seuil de pertinence pour les knowledge strips
        self.min_strips_for_response = 2  # Nombre minimum de strips pertinents
        self.max_strips_per_document = 10  # Nombre maximum de strips par document
        self.strip_size = 200  # Taille approximative d'un knowledge strip
        
        # Templates de prompts pour le grading et la génération
        self.prompts = self._initialize_prompts()
    
    def _initialize_llm(self):
        """Initialise le modèle LLM pour le grading et la génération."""
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
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialise les templates de prompts pour le Corrective RAG."""
        return {
            "knowledge_stripping": """
            Analysez le document suivant et décomposez-le en "knowledge strips" - des sections d'information distinctes et cohérentes.
            
            Document:
            {document}
            
            Instructions:
            - Divisez le document en strips de maximum {strip_size} mots chacun
            - Chaque strip doit contenir une information complète et cohérente
            - Identifiez le sujet principal de chaque strip
            - Numérotez les strips de manière séquentielle
            
            Format de réponse:
            Strip 1: [contenu] - Sujet: [sujet principal]
            Strip 2: [contenu] - Sujet: [sujet principal]
            ...
            """,
            
            "relevance_grading": """
            Évaluez la pertinence de ce knowledge strip par rapport à la requête utilisateur.
            
            Requête: {query}
            Knowledge Strip: {strip}
            Sujet du strip: {strip_subject}
            
            Instructions:
            - Évaluez la pertinence de 0.0 à 1.0
            - 1.0 = parfaitement pertinent et répond directement à la requête
            - 0.5 = quelque peu pertinent mais pas directement lié
            - 0.0 = non pertinent ou hors sujet
            
            Justifiez votre évaluation en une phrase.
            
            Score de pertinence (0.0-1.0): 
            Justification:
            """,
            
            "web_search_fallback": """
            La requête suivante nécessite des informations plus récentes ou supplémentaires qui ne sont pas disponibles dans la base de connaissances actuelle.
            
            Requête: {query}
            Contexte disponible: {context}
            
            Instructions:
            - Recherchez des informations récentes et pertinentes
            - Intégrez les nouvelles informations avec le contexte existant
            - Fournissez une réponse complète et à jour
            
            Réponse avec informations supplémentaires:
            """,
            
            "final_generation": """
            Générez une réponse finale basée sur les knowledge strips les plus pertinents.
            
            Requête: {query}
            Contexte mémoire: {memory_context}
            Knowledge Strips Pertinents:
            {relevant_strips}
            
            Instructions:
            - Utilisez uniquement les informations des strips pertinents
            - Intégrez le contexte mémoire si disponible
            - Fournissez une réponse claire, précise et complète
            - Citez les sources quand approprié
            - Si les strips ne contiennent pas suffisamment d'informations, indiquez-le clairement
            
            Réponse finale:
            """
        }
    
    def break_into_knowledge_strips(self, documents: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Décompose les documents en knowledge strips.
        
        Args:
            documents: Liste des documents récupérés
            query: Requête utilisateur
            
        Returns:
            Liste des knowledge strips
        """
        knowledge_strips = []
        
        print(f"Décomposition de {len(documents)} documents en knowledge strips...")
        
        for doc_idx, document in enumerate(documents):
            try:
                content = document.get("content", "")
                if not content:
                    continue
                
                # Utiliser l'LLM pour décomposer en strips
                prompt = PromptTemplate(
                    input_variables=["document", "strip_size"],
                    template=self.prompts["knowledge_stripping"]
                )
                
                formatted_prompt = prompt.format(
                    document=content,
                    strip_size=self.strip_size
                )
                
                response = self.llm.invoke(formatted_prompt)
                strips_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parser les strips générés
                strips = self._parse_strips(strips_text, doc_idx, document)
                knowledge_strips.extend(strips)
                
                print(f"Document {doc_idx + 1}: {len(strips)} strips créés")
                
            except Exception as e:
                print(f"Erreur lors de la décomposition du document {doc_idx}: {str(e)}")
                continue
        
        print(f"Total: {len(knowledge_strips)} knowledge strips créés")
        return knowledge_strips
    
    def _parse_strips(self, strips_text: str, doc_idx: int, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse les strips générés par l'LLM."""
        strips = []
        lines = strips_text.strip().split('\n')
        
        current_strip = None
        for line in lines:
            line = line.strip()
            if line.startswith('Strip'):
                # Nouveau strip
                if current_strip:
                    strips.append(current_strip)
                
                # Extraire le contenu et le sujet
                parts = line.split(' - Sujet: ', 1)
                if len(parts) == 2:
                    strip_content = parts[0].split(':', 1)[1].strip() if ':' in parts[0] else parts[0].strip()
                    strip_subject = parts[1].strip()
                    
                    current_strip = {
                        "content": strip_content,
                        "subject": strip_subject,
                        "document_index": doc_idx,
                        "document_metadata": document.get("metadata", {}),
                        "strip_id": f"doc_{doc_idx}_strip_{len(strips)}"
                    }
                else:
                    current_strip = {
                        "content": line,
                        "subject": "Non spécifié",
                        "document_index": doc_idx,
                        "document_metadata": document.get("metadata", {}),
                        "strip_id": f"doc_{doc_idx}_strip_{len(strips)}"
                    }
            elif current_strip:
                # Continuer le strip actuel
                current_strip["content"] += " " + line
        
        # Ajouter le dernier strip
        if current_strip:
            strips.append(current_strip)
        
        return strips
    
    def grade_relevance(self, knowledge_strips: List[Dict[str, Any]], query: str, 
                       memory_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Grade la pertinence des knowledge strips par rapport à la requête.
        
        Args:
            knowledge_strips: Liste des knowledge strips
            query: Requête utilisateur
            memory_context: Contexte mémoire (optionnel)
            
        Returns:
            Liste des strips avec scores de pertinence
        """
        graded_strips = []
        
        print(f"Évaluation de la pertinence de {len(knowledge_strips)} knowledge strips...")
        
        for strip_idx, strip in enumerate(knowledge_strips):
            try:
                # Utiliser l'LLM pour évaluer la pertinence
                prompt = PromptTemplate(
                    input_variables=["query", "strip", "strip_subject"],
                    template=self.prompts["relevance_grading"]
                )
                
                formatted_prompt = prompt.format(
                    query=query,
                    strip=strip["content"],
                    strip_subject=strip["subject"]
                )
                
                response = self.llm.invoke(formatted_prompt)
                grading_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parser le score et la justification
                relevance_score, justification = self._parse_grading(grading_text)
                
                # Calculer un score de similarité cosinus comme backup
                cosine_score = self._calculate_cosine_similarity(query, strip["content"])
                
                # Score composite (moyenne pondérée)
                composite_score = (relevance_score * 0.7) + (cosine_score * 0.3)
                
                graded_strip = {
                    **strip,
                    "relevance_score": relevance_score,
                    "cosine_score": cosine_score,
                    "composite_score": composite_score,
                    "justification": justification,
                    "is_relevant": composite_score >= self.relevance_threshold
                }
                
                graded_strips.append(graded_strip)
                
                # Progress indicator
                if (strip_idx + 1) % 10 == 0:
                    print(f"Strips évalués: {strip_idx + 1}/{len(knowledge_strips)}")
                
            except Exception as e:
                print(f"Erreur lors de l'évaluation du strip {strip_idx}: {str(e)}")
                # Ajouter le strip avec un score par défaut
                graded_strip = {
                    **strip,
                    "relevance_score": 0.0,
                    "cosine_score": 0.0,
                    "composite_score": 0.0,
                    "justification": f"Erreur d'évaluation: {str(e)}",
                    "is_relevant": False
                }
                graded_strips.append(graded_strip)
        
        # Trier par score composite (décroissant)
        graded_strips.sort(key=lambda x: x["composite_score"], reverse=True)
        
        print(f"Évaluation terminée: {sum(1 for s in graded_strips if s['is_relevant'])} strips pertinents")
        return graded_strips
    
    def _parse_grading(self, grading_text: str) -> Tuple[float, str]:
        """Parse le texte de grading pour extraire le score et la justification."""
        try:
            lines = grading_text.strip().split('\n')
            score = 0.0
            justification = "Non spécifié"
            
            for line in lines:
                if line.startswith('Score de pertinence'):
                    # Extraire le score
                    score_part = line.split(':')[1].strip() if ':' in line else ""
                    try:
                        score = float(score_part)
                    except ValueError:
                        score = 0.0
                
                elif line.startswith('Justification'):
                    # Extraire la justification
                    justification = line.split(':', 1)[1].strip() if ':' in line else "Non spécifié"
            
            return score, justification
            
        except Exception as e:
            print(f"Erreur lors du parsing du grading: {str(e)}")
            return 0.0, "Erreur de parsing"
    
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité cosinus entre deux textes."""
        try:
            embedding1 = self.embedding_service.embeddings.embed_query(text1)
            embedding2 = self.embedding_service.embeddings.embed_query(text2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"Erreur lors du calcul de similarité cosinus: {str(e)}")
            return 0.0
    
    def filter_relevant_strips(self, graded_strips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtre les strips pertinents selon le seuil défini.
        
        Args:
            graded_strips: Liste des strips avec scores
            
        Returns:
            Liste des strips pertinents
        """
        relevant_strips = [strip for strip in graded_strips if strip["is_relevant"]]
        
        print(f"Filtrage: {len(relevant_strips)}/{len(graded_strips)} strips pertinents")
        return relevant_strips
    
    def check_sufficiency(self, relevant_strips: List[Dict[str, Any]], query: str) -> bool:
        """
        Vérifie si les strips pertinents sont suffisants pour répondre à la requête.
        
        Args:
            relevant_strips: Liste des strips pertinents
            query: Requête utilisateur
            
        Returns:
            True si suffisant, False sinon
        """
        if len(relevant_strips) < self.min_strips_for_response:
            return False
        
        # Vérifier la qualité moyenne des strips
        avg_score = sum(strip["composite_score"] for strip in relevant_strips) / len(relevant_strips)
        if avg_score < self.relevance_threshold:
            return False
        
        return True
    
    def generate_final_response(self, query: str, relevant_strips: List[Dict[str, Any]], 
                              memory_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Génère la réponse finale basée sur les strips pertinents.
        
        Args:
            query: Requête utilisateur
            relevant_strips: Liste des strips pertinents
            memory_context: Contexte mémoire (optionnel)
            
        Returns:
            Réponse finale avec métadonnées
        """
        try:
            # Préparer le contexte mémoire
            memory_context_text = ""
            if memory_context and memory_context.get("has_context"):
                memory_context_text = memory_context["enriched_context"]
            
            # Préparer les strips pertinents
            strips_text = ""
            for i, strip in enumerate(relevant_strips):
                strips_text += f"Strip {i+1}: {strip['content']}\n"
                strips_text += f"Score: {strip['composite_score']:.2f} - {strip['justification']}\n\n"
            
            # Générer la réponse
            prompt = PromptTemplate(
                input_variables=["query", "memory_context", "relevant_strips"],
                template=self.prompts["final_generation"]
            )
            
            formatted_prompt = prompt.format(
                query=query,
                memory_context=memory_context_text,
                relevant_strips=strips_text
            )
            
            response = self.llm.invoke(formatted_prompt)
            final_answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculer la confiance basée sur les scores des strips
            confidence = self._calculate_confidence(relevant_strips)
            
            # Préparer les sources
            sources = []
            for strip in relevant_strips:
                sources.append({
                    "strip_id": strip["strip_id"],
                    "content_preview": strip["content"][:200] + "..." if len(strip["content"]) > 200 else strip["content"],
                    "subject": strip["subject"],
                    "relevance_score": strip["composite_score"],
                    "justification": strip["justification"],
                    "document_metadata": strip["document_metadata"]
                })
            
            return {
                "answer": final_answer,
                "sources": sources,
                "confidence": confidence,
                "relevant_strips_count": len(relevant_strips),
                "total_strips_evaluated": len(relevant_strips),  # Sera mis à jour par l'appelant
                "memory_context_used": memory_context is not None,
                "generation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse finale: {str(e)}")
            return {
                "answer": f"Erreur lors de la génération de la réponse: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _calculate_confidence(self, relevant_strips: List[Dict[str, Any]]) -> float:
        """Calcule le score de confiance basé sur les strips pertinents."""
        if not relevant_strips:
            return 0.0
        
        # Moyenne des scores des strips
        avg_score = sum(strip["composite_score"] for strip in relevant_strips) / len(relevant_strips)
        
        # Bonus pour le nombre de strips (plus de sources = plus de confiance)
        count_bonus = min(len(relevant_strips) / 10, 0.2)  # Bonus max de 0.2
        
        confidence = min(avg_score + count_bonus, 1.0)
        return confidence
    
    def process_corrective_rag(self, query: str, documents: List[Dict[str, Any]], 
                             memory_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pipeline complet du Corrective RAG.
        
        Args:
            query: Requête utilisateur
            documents: Documents récupérés
            memory_context: Contexte mémoire (optionnel)
            
        Returns:
            Résultat complet du Corrective RAG
        """
        try:
            print(f"Début du Corrective RAG pour la requête: {query[:100]}...")
            
            # Étape 1: Knowledge Stripping
            knowledge_strips = self.break_into_knowledge_strips(documents, query)
            
            if not knowledge_strips:
                return {
                    "answer": "Aucune information pertinente trouvée dans les documents.",
                    "sources": [],
                    "confidence": 0.0,
                    "error": "No knowledge strips created"
                }
            
            # Étape 2: Relevance Grading
            graded_strips = self.grade_relevance(knowledge_strips, query, memory_context)
            
            # Étape 3: Filter Relevant Strips
            relevant_strips = self.filter_relevant_strips(graded_strips)
            
            # Étape 4: Check Sufficiency
            if not self.check_sufficiency(relevant_strips, query):
                # Fallback: utiliser les meilleurs strips même s'ils ne sont pas parfaitement pertinents
                print("Strips insuffisants, utilisation des meilleurs strips disponibles")
                relevant_strips = graded_strips[:self.min_strips_for_response]
            
            # Étape 5: Generate Final Response
            result = self.generate_final_response(query, relevant_strips, memory_context)
            result["total_strips_evaluated"] = len(knowledge_strips)
            
            print(f"Corrective RAG terminé: {result['relevant_strips_count']} strips utilisés, confiance: {result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"Erreur dans le pipeline Corrective RAG: {str(e)}")
            return {
                "answer": f"Erreur lors du traitement: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "error": str(e)
            }
