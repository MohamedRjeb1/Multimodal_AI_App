# Guide de Démarrage Rapide - Advanced RAG

## Installation et Configuration

### 1. Prérequis

- Python 3.10 ou supérieur
- pip (gestionnaire de paquets Python)
- Clés API Google AI ou OpenAI

### 2. Installation

```bash
# Cloner le repository (si applicable)
git clone <repository-url>
cd langchain

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copier le fichier de configuration
cp env.example .env

# Éditer le fichier .env avec vos clés API
# Au minimum, vous devez configurer:
GOOGLE_API_KEY=your_google_api_key_here
# OU
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Démarrage du serveur

```bash
# Option 1: Utiliser le script de démarrage
python scripts/start_server.py

# Option 2: Utiliser uvicorn directement
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Vérification

Ouvrez votre navigateur et allez à:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

## Utilisation de l'API

### 1. Traitement d'une vidéo YouTube

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "language": "fr",
    "whisper_model": "small"
  }'
```

**Réponse:**
```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Video processing started"
}
```

### 2. Vérifier le statut

```bash
curl "http://localhost:8000/api/v1/status/TASK_ID"
```

### 3. Poser une question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "De quoi parle cette vidéo ?",
    "task_id": "TASK_ID",
    "max_results": 5,
    "similarity_threshold": 0.7
  }'
```

## Test Automatique

```bash
# Exécuter le script de test
python scripts/test_api.py
```

## Fonctionnalités Avancées

### Stratégies d'Embedding

L'API supporte deux stratégies d'embedding:

1. **Semantic Chunking** (par défaut): Crée des chunks basés sur les frontières sémantiques
2. **Hierarchical Chunking**: Crée plusieurs niveaux de granularité

### Techniques d'Indexing

- **Hybrid Indexing**: Combine plusieurs stratégies d'indexation
- **Semantic Clustering**: Groupe les chunks similaires avec K-means
- **Multi-vector Search**: Supporte différentes dimensions d'embedding

### Modèles Supportés

**LLM:**
- Google Gemini 2.5 Flash (par défaut)
- OpenAI GPT-4, GPT-3.5

**Embedding:**
- Google Generative AI Embeddings
- OpenAI Embeddings

**Transcription:**
- OpenAI Whisper (tiny, base, small, medium, large)

## Structure du Projet

```
langchain/
├── app/                    # Application principale
│   ├── api/               # Routes et endpoints API
│   ├── core/              # Configuration et paramètres
│   ├── models/            # Modèles de données
│   ├── services/          # Logique métier
│   └── utils/             # Utilitaires
├── data/                  # Données persistantes
│   ├── audio/            # Fichiers audio téléchargés
│   ├── transcripts/      # Transcriptions
│   └── vectorstore/      # Base de données vectorielle
├── scripts/              # Scripts utilitaires
├── docs/                 # Documentation
└── tests/                # Tests unitaires
```

## Dépannage

### Erreurs Communes

1. **"No API key found"**
   - Vérifiez que vos clés API sont correctement configurées dans le fichier `.env`

2. **"Module not found"**
   - Vérifiez que l'environnement virtuel est activé
   - Réinstallez les dépendances: `pip install -r requirements.txt`

3. **"Port already in use"**
   - Changez le port dans le fichier `.env` ou arrêtez le processus utilisant le port 8000

4. **"Download failed"**
   - Vérifiez que l'URL YouTube est valide
   - Assurez-vous que la vidéo n'est pas privée ou restreinte

### Logs

Les logs sont affichés dans la console. Pour plus de détails, consultez la documentation de l'API à http://localhost:8000/docs

## Prochaines Étapes

1. **Personnalisation**: Modifiez les paramètres dans `app/core/config.py`
2. **Tests**: Ajoutez vos propres tests dans le dossier `tests/`
3. **Déploiement**: Consultez la documentation de déploiement
4. **Monitoring**: Implémentez des outils de monitoring pour la production

## Support

Pour plus d'informations:
- Consultez la documentation complète de l'API: `docs/API.md`
- Vérifiez les exemples dans `scripts/test_api.py`
- Consultez le README principal pour l'architecture détaillée
