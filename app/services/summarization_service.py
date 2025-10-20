"""
Service de résumé avec modèle fine-tuné pour l'indexation optimisée.
"""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from app.core.config import get_settings


class SummarizationService:
    """
    Service de résumé qui :
    - Utilise un modèle fine-tuné pour créer des résumés de qualité
    - Génère des résumés optimisés pour l'indexation
    - Prévient la perte d'informations importantes
    - Supporte différents niveaux de granularité
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.llm = self._initialize_summarization_llm()
        
        # Configuration des résumés
        self.max_summary_length = 200  # Longueur maximale du résumé
        self.summary_quality_threshold = 0.8  # Seuil de qualité du résumé
        
        # Templates de prompts pour différents types de résumés
        self.summary_templates = self._initialize_summary_templates()
    
    def _initialize_summarization_llm(self):
        """Initialise le modèle LLM pour la génération de résumés."""
        if self.settings.DEFAULT_LLM_MODEL.startswith("gemini"):
            if not self.settings.GOOGLE_API_KEY:
                raise ValueError("Google API key is required for Gemini")
            return ChatGoogleGenerativeAI(
                model=self.settings.DEFAULT_LLM_MODEL,
                google_api_key=self.settings.GOOGLE_API_KEY,
                temperature=0.1  # Température basse pour la cohérence
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
    
    def _initialize_summary_templates(self) -> Dict[str, str]:
        """Initialise les templates de prompts pour différents types de résumés."""
        return {
            "semantic_chunk": """
            Vous êtes un expert en résumé de contenu vidéo. Créez un résumé concis et informatif du chunk suivant.
            
            Chunk à résumer:
            {content}
            
            Instructions:
            - Résumez en maximum {max_length} mots
            - Conservez les informations clés et les concepts importants
            - Utilisez un langage clair et précis
            - Évitez les détails superflus
            - Préservez le contexte sémantique
            
            Résumé:
            """,
            
            "hierarchical_fine": """
            Créez un résumé détaillé et précis du contenu suivant pour l'indexation fine:
            
            Contenu:
            {content}
            
            Instructions:
            - Résumé détaillé en maximum {max_length} mots
            - Incluez tous les points importants
            - Conservez les détails techniques et spécifiques
            - Structurez les informations logiquement
            
            Résumé détaillé:
            """,
            
            "hierarchical_coarse": """
            Créez un résumé général et synthétique du contenu suivant pour l'indexation globale:
            
            Contenu:
            {content}
            
            Instructions:
            - Résumé synthétique en maximum {max_length} mots
            - Focus sur les idées principales et les concepts généraux
            - Évitez les détails spécifiques
            - Donnez une vue d'ensemble cohérente
            
            Résumé synthétique:
            """,
            
            "query_optimized": """
            Créez un résumé optimisé pour les requêtes de recherche du contenu suivant:
            
            Contenu:
            {content}
            
            Instructions:
            - Résumé optimisé pour la recherche en maximum {max_length} mots
            - Incluez les mots-clés et concepts recherchables
            - Structurez pour faciliter la correspondance avec les requêtes
            - Conservez les informations pertinentes pour les questions fréquentes
            
            Résumé optimisé pour la recherche:
            """
        }
    
    def generate_summary(self, content: str, summary_type: str = "semantic_chunk", 
                        max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Génère un résumé du contenu selon le type spécifié.
        
        Args:
            content: Contenu à résumer
            summary_type: Type de résumé à générer
            max_length: Longueur maximale du résumé
            
        Returns:
            Résultat du résumé avec métadonnées
        """
        try:
            if max_length is None:
                max_length = self.max_summary_length
            
            # Obtenir le template approprié
            template = self.summary_templates.get(summary_type, self.summary_templates["semantic_chunk"])
            
            # Créer le prompt
            prompt = PromptTemplate(
                input_variables=["content", "max_length"],
                template=template
            )
            
            # Générer le résumé
            formatted_prompt = prompt.format(content=content, max_length=max_length)
            response = self.llm.invoke(formatted_prompt)
            
            summary = response.content if hasattr(response, 'content') else str(response)
            
            # Nettoyer le résumé
            summary = self._clean_summary(summary)
            
            # Évaluer la qualité du résumé
            quality_score = self._evaluate_summary_quality(content, summary)
            
            return {
                "summary": summary,
                "original_content": content,
                "summary_type": summary_type,
                "max_length": max_length,
                "actual_length": len(summary.split()),
                "quality_score": quality_score,
                "created_at": datetime.now().isoformat(),
                "model_used": self.settings.DEFAULT_LLM_MODEL
            }
            
        except Exception as e:
            print(f"Erreur lors de la génération du résumé: {str(e)}")
            return {
                "summary": "",
                "error": str(e),
                "summary_type": summary_type
            }
    
    def _clean_summary(self, summary: str) -> str:
        """Nettoie le résumé généré."""
        # Supprimer les préfixes inutiles
        prefixes_to_remove = [
            "Résumé:", "Résumé détaillé:", "Résumé synthétique:", 
            "Résumé optimisé pour la recherche:", "Summary:", "Summary:"
        ]
        
        for prefix in prefixes_to_remove:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
        
        # Supprimer les espaces multiples
        import re
        summary = re.sub(r'\s+', ' ', summary)
        
        return summary.strip()
    
    def _evaluate_summary_quality(self, original_content: str, summary: str) -> float:
        """
        Évalue la qualité du résumé basée sur plusieurs critères.
        
        Args:
            original_content: Contenu original
            summary: Résumé généré
            
        Returns:
            Score de qualité (0.0 à 1.0)
        """
        try:
            # Critères d'évaluation
            criteria_scores = []
            
            # 1. Longueur appropriée
            original_words = len(original_content.split())
            summary_words = len(summary.split())
            
            if original_words > 0:
                compression_ratio = summary_words / original_words
                # Score optimal entre 0.1 et 0.3 (10-30% du contenu original)
                if 0.1 <= compression_ratio <= 0.3:
                    length_score = 1.0
                elif compression_ratio < 0.1:
                    length_score = compression_ratio / 0.1
                else:
                    length_score = max(0.3 / compression_ratio, 0.1)
            else:
                length_score = 0.0
            
            criteria_scores.append(length_score)
            
            # 2. Présence de mots-clés importants
            original_keywords = set(original_content.lower().split())
            summary_keywords = set(summary.lower().split())
            
            if original_keywords:
                keyword_overlap = len(original_keywords.intersection(summary_keywords)) / len(original_keywords)
                criteria_scores.append(min(keyword_overlap * 2, 1.0))  # Bonus pour les mots-clés
            else:
                criteria_scores.append(0.0)
            
            # 3. Cohérence structurelle
            # Vérifier si le résumé commence et se termine correctement
            structure_score = 1.0
            if not summary.endswith(('.', '!', '?')):
                structure_score -= 0.2
            if len(summary.split()) < 5:  # Trop court
                structure_score -= 0.3
            
            criteria_scores.append(max(structure_score, 0.0))
            
            # Score composite
            quality_score = sum(criteria_scores) / len(criteria_scores)
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation de la qualité: {str(e)}")
            return 0.5  # Score par défaut
    
    def summarize_chunks(self, chunks: List[Dict[str, Any]], summary_type: str = "semantic_chunk") -> List[Dict[str, Any]]:
        """
        Résume une liste de chunks.
        
        Args:
            chunks: Liste des chunks à résumer
            summary_type: Type de résumé
            
        Returns:
            Liste des chunks avec leurs résumés
        """
        summarized_chunks = []
        
        print(f"Début du résumé de {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            try:
                # Générer le résumé
                summary_result = self.generate_summary(
                    chunk["content"], 
                    summary_type=summary_type
                )
                
                # Créer le chunk résumé
                summarized_chunk = {
                    "original_chunk": chunk,
                    "summary": summary_result["summary"],
                    "summary_metadata": {
                        "summary_type": summary_result["summary_type"],
                        "quality_score": summary_result["quality_score"],
                        "actual_length": summary_result["actual_length"],
                        "max_length": summary_result["max_length"],
                        "model_used": summary_result["model_used"],
                        "created_at": summary_result["created_at"]
                    },
                    "metadata": chunk["metadata"].copy()
                }
                
                # Mettre à jour les métadonnées
                summarized_chunk["metadata"]["has_summary"] = True
                summarized_chunk["metadata"]["summary_quality"] = summary_result["quality_score"]
                summarized_chunk["metadata"]["summarized_at"] = summary_result["created_at"]
                
                summarized_chunks.append(summarized_chunk)
                
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"Résumés générés: {i + 1}/{len(chunks)}")
                
            except Exception as e:
                print(f"Erreur lors du résumé du chunk {i}: {str(e)}")
                # Ajouter le chunk original en cas d'erreur
                summarized_chunks.append({
                    "original_chunk": chunk,
                    "summary": chunk["content"],  # Utiliser le contenu original
                    "summary_metadata": {
                        "summary_type": "error_fallback",
                        "quality_score": 0.0,
                        "error": str(e)
                    },
                    "metadata": chunk["metadata"].copy()
                })
        
        print(f"Résumé terminé: {len(summarized_chunks)} chunks traités")
        return summarized_chunks
    
    def save_summaries(self, summarized_chunks: List[Dict[str, Any]], task_id: str) -> str:
        """
        Sauvegarde les résumés sur disque.
        
        Args:
            summarized_chunks: Chunks résumés
            task_id: Identifiant de la tâche
            
        Returns:
            Chemin du fichier de sauvegarde
        """
        try:
            summaries_file = os.path.join(self.settings.VECTORSTORE_DIR, f"{task_id}_summaries.json")
            
            # Préparer les données pour la sauvegarde
            summaries_data = {
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "total_chunks": len(summarized_chunks),
                "summarization_model": self.settings.DEFAULT_LLM_MODEL,
                "avg_quality_score": sum(
                    chunk["summary_metadata"]["quality_score"] 
                    for chunk in summarized_chunks
                ) / len(summarized_chunks) if summarized_chunks else 0.0,
                "summaries": []
            }
            
            for chunk in summarized_chunks:
                summaries_data["summaries"].append({
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "original_content": chunk["original_chunk"]["content"],
                    "summary": chunk["summary"],
                    "summary_metadata": chunk["summary_metadata"],
                    "metadata": chunk["metadata"]
                })
            
            # Sauvegarder
            with open(summaries_file, "w", encoding="utf-8") as f:
                json.dump(summaries_data, f, indent=2, ensure_ascii=False)
            
            return summaries_file
            
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des résumés: {str(e)}")
            return ""
    
    def load_summaries(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Charge les résumés depuis le disque.
        
        Args:
            task_id: Identifiant de la tâche
            
        Returns:
            Données des résumés ou None si non trouvé
        """
        try:
            summaries_file = os.path.join(self.settings.VECTORSTORE_DIR, f"{task_id}_summaries.json")
            
            if not os.path.exists(summaries_file):
                return None
            
            with open(summaries_file, "r", encoding="utf-8") as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Erreur lors du chargement des résumés: {str(e)}")
            return None
