# Advanced RAG Application

Une application RAG (Retrieval-Augmented Generation) avancée qui permet de télécharger des vidéos YouTube, les transcrire, et répondre aux questions des utilisateurs en utilisant des techniques d'embedding et d'indexing innovantes.

## Architecture

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
├── tests/                # Tests unitaires et d'intégration
├── docs/                 # Documentation
└── scripts/              # Scripts utilitaires
```

## Fonctionnalités

- [x] Téléchargement de vidéos YouTube
- [x] Transcription audio avec Whisper
- [ ] Embedding avancé avec techniques innovantes
- [ ] Indexing optimisé pour la recherche sémantique
- [ ] Interface API REST avec FastAPI
- [ ] Gestion d'erreurs robuste
- [ ] Tests automatisés

## Technologies

- **Backend**: FastAPI, Python 3.10+
- **LLM**: Google Gemini, OpenAI
- **Embedding**: Google Generative AI, OpenAI
- **Vector Store**: DocArray, ChromaDB (à implémenter)
- **Audio Processing**: Whisper, yt-dlp
- **Database**: SQLite/PostgreSQL (à implémenter)

## Installation

```bash
# Cloner le repository
git clone <repository-url>
cd langchain

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

## Utilisation

```bash
# Démarrer l'application
uvicorn app.main:app --reload

# Accéder à la documentation API
http://localhost:8000/docs
```

## Prochaines étapes

1. ✅ Téléchargement et transcription de vidéos
2. 🔄 **En cours**: Restructuration de l'architecture
3. ⏳ Embedding avancé avec techniques innovantes
4. ⏳ Indexing optimisé et recherche sémantique
5. ⏳ Interface utilisateur web
6. ⏳ Tests et déploiement
