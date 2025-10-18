# Advanced RAG API Documentation

## Overview

The Advanced RAG API provides endpoints for processing YouTube videos and performing intelligent queries using advanced embedding and indexing techniques.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

Currently, the API does not require authentication. In production, implement proper authentication mechanisms.

## Endpoints

### Health Check

**GET** `/health`

Check the health status of the API and its services.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-18T18:19:00.000Z",
  "services": {
    "youtube_service": "active",
    "transcription_service": "active",
    "embedding_service": "active",
    "indexing_service": "active",
    "rag_service": "active"
  }
}
```

### Process Video

**POST** `/process`

Start processing a YouTube video for RAG queries.

**Request Body:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "language": "fr",
  "whisper_model": "small"
}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Video processing started",
  "created_at": "2025-01-18T18:19:00.000Z"
}
```

### Get Processing Status

**GET** `/status/{task_id}`

Get the current processing status for a task.

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "message": "Processing completed successfully",
  "updated_at": "2025-01-18T18:19:00.000Z"
}
```

**Status Values:**
- `pending`: Task is queued
- `downloading`: Downloading video
- `transcribing`: Transcribing audio
- `embedding`: Generating embeddings
- `indexing`: Creating index
- `completed`: Processing complete
- `failed`: Processing failed

### Get Video Information

**GET** `/info/{task_id}`

Get processed video information and transcript.

**Response:**
```json
{
  "task_id": "uuid-string",
  "video_title": "Video Title",
  "transcript": "Full transcript text...",
  "duration": 300.5,
  "language": "fr",
  "created_at": "2025-01-18T18:19:00.000Z"
}
```

### Query Video

**POST** `/query`

Query a processed video using RAG.

**Request Body:**
```json
{
  "query": "What is this video about?",
  "task_id": "uuid-string",
  "max_results": 5,
  "similarity_threshold": 0.7
}
```

**Response:**
```json
{
  "answer": "This video is about...",
  "sources": [
    {
      "chunk_id": "chunk_0",
      "content": "Relevant text snippet...",
      "similarity_score": 0.85,
      "metadata": {...}
    }
  ],
  "confidence": 0.82,
  "processing_time": 1.5,
  "created_at": "2025-01-18T18:19:00.000Z"
}
```

### Get YouTube Video Info

**GET** `/video-info?url={youtube_url}`

Get YouTube video information without processing.

**Response:**
```json
{
  "title": "Video Title",
  "duration": 300,
  "uploader": "Channel Name",
  "upload_date": "20250101",
  "view_count": 1000,
  "description": "Video description...",
  "thumbnail": "https://...",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

### Get Task Statistics

**GET** `/stats/{task_id}`

Get detailed statistics for a processed task.

**Response:**
```json
{
  "task_id": "uuid-string",
  "index_statistics": {
    "total_chunks": 25,
    "embedding_model": "google",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "n_clusters": 5,
    "silhouette_score": 0.75,
    "created_at": "2025-01-18T18:19:00.000Z",
    "index_type": "hybrid"
  },
  "transcript_info": {
    "file_size": 15000,
    "word_count": 500,
    "character_count": 3000,
    "line_count": 50
  },
  "processing_status": {
    "status": "completed",
    "message": "Processing completed successfully"
  }
}
```

### Cleanup Task

**DELETE** `/cleanup/{task_id}`

Clean up all files and data for a task.

**Response:**
```json
{
  "message": "Task uuid-string cleaned up successfully",
  "task_id": "uuid-string"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error Type",
  "message": "Detailed error message",
  "details": {
    "additional": "error information"
  },
  "timestamp": "2025-01-18T18:19:00.000Z"
}
```

**Common HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Advanced Features

### Embedding Strategies

The API supports two embedding strategies:

1. **Semantic Chunking**: Creates chunks based on semantic boundaries
2. **Hierarchical Chunking**: Creates multiple granularity levels (fine and coarse)

### Indexing Techniques

- **Hybrid Indexing**: Combines multiple indexing strategies
- **Semantic Clustering**: Groups similar chunks using K-means clustering
- **Multi-vector Search**: Supports different embedding dimensions

### Query Processing

- **Similarity Search**: Uses cosine similarity for chunk retrieval
- **Confidence Scoring**: Provides confidence scores for answers
- **Source Attribution**: Returns source chunks with metadata

## Rate Limits

Currently, no rate limits are implemented. In production, implement appropriate rate limiting.

## Examples

### Complete Workflow

1. **Process a video:**
```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "language": "fr"}'
```

2. **Check status:**
```bash
curl "http://localhost:8000/api/v1/status/TASK_ID"
```

3. **Query the video:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this video about?", "task_id": "TASK_ID"}'
```

4. **Clean up:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/cleanup/TASK_ID"
```

## Configuration

The API can be configured using environment variables. See `env.example` for all available options.

Key configuration options:
- `GOOGLE_API_KEY`: Google AI API key
- `OPENAI_API_KEY`: OpenAI API key
- `DEFAULT_LLM_MODEL`: Default LLM model
- `DEFAULT_EMBEDDING_MODEL`: Default embedding model
- `CHUNK_SIZE`: Text chunk size
- `CHUNK_OVERLAP`: Overlap between chunks
