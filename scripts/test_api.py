"""
Test script for the Advanced RAG API.
"""
import requests
import time
import json
from typing import Dict, Any


class RAGAPITester:
    """Test client for the Advanced RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint."""
        print("Testing health endpoint...")
        response = requests.get(f"{self.api_url}/health")
        return response.json()
    
    def test_video_info(self, url: str) -> Dict[str, Any]:
        """Test video info endpoint."""
        print(f"Testing video info for: {url}")
        response = requests.get(f"{self.api_url}/video-info", params={"url": url})
        return response.json()
    
    def test_process_video(self, url: str) -> str:
        """Test video processing."""
        print(f"Processing video: {url}")
        payload = {
            "url": url,
            "language": "fr",
            "whisper_model": "small"
        }
        response = requests.post(f"{self.api_url}/process", json=payload)
        result = response.json()
        return result["task_id"]
    
    def wait_for_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for task completion."""
        print(f"Waiting for task {task_id} to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(f"{self.api_url}/status/{task_id}")
            status = response.json()
            
            print(f"Status: {status['status']} - {status['message']}")
            
            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Task failed: {status['message']}")
            
            time.sleep(5)
        
        raise Exception("Task timeout")
    
    def test_query(self, task_id: str, query: str) -> Dict[str, Any]:
        """Test RAG query."""
        print(f"Testing query: {query}")
        payload = {
            "query": query,
            "task_id": task_id,
            "max_results": 5,
            "similarity_threshold": 0.7
        }
        response = requests.post(f"{self.api_url}/query", json=payload)
        return response.json()
    
    def test_statistics(self, task_id: str) -> Dict[str, Any]:
        """Test statistics endpoint."""
        print(f"Getting statistics for task {task_id}")
        response = requests.get(f"{self.api_url}/stats/{task_id}")
        return response.json()
    
    def test_cleanup(self, task_id: str) -> Dict[str, Any]:
        """Test cleanup endpoint."""
        print(f"Cleaning up task {task_id}")
        response = requests.delete(f"{self.api_url}/cleanup/{task_id}")
        return response.json()


def main():
    """Main test function."""
    # Test URL (replace with your test video)
    test_url = "https://youtu.be/rjsbLZhO8Jo?si=-TqdVDCcNszn59Ec"  # Example URL
    
    tester = RAGAPITester()
    
    try:
        # Test 1: Health check
        print("=" * 50)
        print("TEST 1: Health Check")
        print("=" * 50)
        health = tester.test_health()
        print(json.dumps(health, indent=2))
        
        # Test 2: Video info
        print("\n" + "=" * 50)
        print("TEST 2: Video Info")
        print("=" * 50)
        video_info = tester.test_video_info(test_url)
        print(json.dumps(video_info, indent=2))
        
        # Test 3: Process video
        print("\n" + "=" * 50)
        print("TEST 3: Process Video")
        print("=" * 50)
        task_id = tester.test_process_video(test_url)
        print(f"Task ID: {task_id}")
        
        # Test 4: Wait for completion
        print("\n" + "=" * 50)
        print("TEST 4: Wait for Completion")
        print("=" * 50)
        final_status = tester.wait_for_completion(task_id)
        print(json.dumps(final_status, indent=2))
        
        # Test 5: Query
        print("\n" + "=" * 50)
        print("TEST 5: RAG Query")
        print("=" * 50)
        query_result = tester.test_query(task_id, "What is this video about?")
        print(json.dumps(query_result, indent=2))
        
        # Test 6: Statistics
        print("\n" + "=" * 50)
        print("TEST 6: Statistics")
        print("=" * 50)
        stats = tester.test_statistics(task_id)
        print(json.dumps(stats, indent=2))
        
        # Test 7: Cleanup
        print("\n" + "=" * 50)
        print("TEST 7: Cleanup")
        print("=" * 50)
        cleanup_result = tester.test_cleanup(task_id)
        print(json.dumps(cleanup_result, indent=2))
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    main()
