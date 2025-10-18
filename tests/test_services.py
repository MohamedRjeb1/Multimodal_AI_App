"""
Unit tests for the Advanced RAG services.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch

from app.services.youtube_service import YouTubeService
from app.services.transcription_service import TranscriptionService
from app.services.embedding_service import AdvancedEmbeddingService
from app.services.indexing_service import AdvancedIndexingService
from app.core.config import get_settings


class TestYouTubeService:
    """Test cases for YouTube service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = YouTubeService()
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.settings is not None
        assert os.path.exists(self.service.output_dir)
    
    @patch('yt_dlp.YoutubeDL')
    def test_get_video_info(self, mock_ydl):
        """Test getting video information."""
        # Mock video info
        mock_info = {
            'title': 'Test Video',
            'duration': 300,
            'uploader': 'Test Channel',
            'upload_date': '20250101',
            'view_count': 1000,
            'description': 'Test description',
            'thumbnail': 'https://example.com/thumb.jpg'
        }
        
        mock_ydl_instance = Mock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl.return_value.__enter__.return_value = mock_ydl_instance
        
        result = self.service.get_video_info("https://www.youtube.com/watch?v=test")
        
        assert result['title'] == 'Test Video'
        assert result['duration'] == 300
        assert result['uploader'] == 'Test Channel'


class TestTranscriptionService:
    """Test cases for transcription service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = TranscriptionService()
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.settings is not None
        assert os.path.exists(self.service.transcript_dir)
        assert self.service.model is not None
    
    def test_get_transcript_nonexistent(self):
        """Test getting transcript for non-existent task."""
        result = self.service.get_transcript("nonexistent_task")
        assert result is None
    
    def test_get_transcript_info_nonexistent(self):
        """Test getting transcript info for non-existent task."""
        result = self.service.get_transcript_info("nonexistent_task")
        assert result is None


class TestEmbeddingService:
    """Test cases for embedding service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the embedding initialization to avoid API calls
        with patch('app.services.embedding_service.GoogleGenerativeAIEmbeddings'):
            self.service = AdvancedEmbeddingService()
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.settings is not None
        assert os.path.exists(self.service.vectorstore_dir)
        assert self.service.text_splitter is not None
    
    def test_create_semantic_chunks(self):
        """Test creating semantic chunks."""
        test_text = "This is a test text. It has multiple sentences. Each sentence should be processed separately."
        task_id = "test_task"
        
        documents = self.service.create_semantic_chunks(test_text, task_id)
        
        assert len(documents) > 0
        assert all(doc.metadata["task_id"] == task_id for doc in documents)
        assert all(doc.metadata["chunk_type"] == "semantic" for doc in documents)
    
    def test_create_hierarchical_chunks(self):
        """Test creating hierarchical chunks."""
        test_text = "This is a test text. It has multiple sentences. Each sentence should be processed separately."
        task_id = "test_task"
        
        documents = self.service.create_hierarchical_chunks(test_text, task_id)
        
        assert len(documents) > 0
        assert all(doc.metadata["task_id"] == task_id for doc in documents)
        assert all(doc.metadata["chunk_type"] == "hierarchical" for doc in documents)
        
        # Should have both fine and coarse granularities
        granularities = set(doc.metadata["granularity"] for doc in documents)
        assert "fine" in granularities
        assert "coarse" in granularities


class TestIndexingService:
    """Test cases for indexing service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = AdvancedIndexingService()
    
    def test_initialization(self):
        """Test service initialization."""
        assert self.service.settings is not None
        assert os.path.exists(self.service.vectorstore_dir)
        assert isinstance(self.service.indices, dict)
        assert isinstance(self.service.index_metadata, dict)
    
    def test_get_index_statistics_nonexistent(self):
        """Test getting index statistics for non-existent task."""
        result = self.service.get_index_statistics("nonexistent_task")
        assert result is None
    
    def test_delete_index_nonexistent(self):
        """Test deleting non-existent index."""
        result = self.service.delete_index("nonexistent_task")
        assert result is False


class TestConfiguration:
    """Test cases for configuration."""
    
    def test_settings_loading(self):
        """Test that settings are loaded correctly."""
        settings = get_settings()
        
        assert settings.APP_NAME is not None
        assert settings.APP_VERSION is not None
        assert settings.DATA_DIR is not None
        assert settings.AUDIO_DIR is not None
        assert settings.TRANSCRIPT_DIR is not None
        assert settings.VECTORSTORE_DIR is not None
    
    def test_directory_creation(self):
        """Test that directories are created automatically."""
        settings = get_settings()
        
        assert os.path.exists(settings.DATA_DIR)
        assert os.path.exists(settings.AUDIO_DIR)
        assert os.path.exists(settings.TRANSCRIPT_DIR)
        assert os.path.exists(settings.VECTORSTORE_DIR)


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.mark.skip(reason="Requires API keys and external services")
    def test_complete_pipeline(self):
        """Test the complete RAG pipeline."""
        # This test would require actual API keys and external services
        # It's marked as skip for now but can be enabled for integration testing
        pass


if __name__ == "__main__":
    pytest.main([__file__])
