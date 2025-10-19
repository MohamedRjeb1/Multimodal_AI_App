"""
Test script for the Course API (Multi-Video Formations).
"""
import requests
import time
import json
from typing import Dict, Any, List


class CourseAPITester:
    """Test client for the Course API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def test_create_course(self, course_name: str, videos: List[str], course_description: str = None) -> Dict[str, Any]:
        """Test course creation."""
        print(f"Creating course: {course_name}")
        
        payload = {
            "course_name": course_name,
            "course_description": course_description,
            "videos": videos,
            "language": "fr",
            "whisper_model": "small",
            "processing_strategy": "sequential"
        }
        
        response = requests.post(f"{self.api_url}/courses", json=payload)
        return response.json()
    
    def test_process_course(self, course_id: str, strategy: str = "sequential") -> Dict[str, Any]:
        """Test course processing."""
        print(f"Processing course {course_id} with strategy: {strategy}")
        
        response = requests.post(
            f"{self.api_url}/courses/{course_id}/process",
            params={"processing_strategy": strategy}
        )
        return response.json()
    
    def test_get_course(self, course_id: str) -> Dict[str, Any]:
        """Test getting course information."""
        print(f"Getting course information: {course_id}")
        
        response = requests.get(f"{self.api_url}/courses/{course_id}")
        return response.json()
    
    def test_query_course(self, course_id: str, query: str, strategy: str = "cross_video") -> Dict[str, Any]:
        """Test course querying."""
        print(f"Querying course {course_id}: {query}")
        
        payload = {
            "query": query,
            "course_id": course_id,
            "max_results": 10,
            "similarity_threshold": 0.7,
            "search_strategy": strategy
        }
        
        response = requests.post(f"{self.api_url}/courses/{course_id}/query", json=payload)
        return response.json()
    
    def test_get_course_statistics(self, course_id: str) -> Dict[str, Any]:
        """Test getting course statistics."""
        print(f"Getting course statistics: {course_id}")
        
        response = requests.get(f"{self.api_url}/courses/{course_id}/statistics")
        return response.json()
    
    def test_list_courses(self) -> Dict[str, Any]:
        """Test listing all courses."""
        print("Listing all courses")
        
        response = requests.get(f"{self.api_url}/courses")
        return response.json()
    
    def test_get_video_info(self, course_id: str, video_id: str) -> Dict[str, Any]:
        """Test getting video information."""
        print(f"Getting video info: {video_id} in course {course_id}")
        
        response = requests.get(f"{self.api_url}/courses/{course_id}/videos/{video_id}")
        return response.json()
    
    def wait_for_course_completion(self, course_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for course processing to complete."""
        print(f"Waiting for course {course_id} to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            course_info = self.test_get_course(course_id)
            
            print(f"Course status: {course_info['status']} - Processed: {course_info['processed_videos']}/{course_info['total_videos']}")
            
            if course_info["status"] == "completed":
                return course_info
            elif course_info["status"] == "failed":
                raise Exception(f"Course processing failed")
            
            time.sleep(10)
        
        raise Exception("Course processing timeout")


def main():
    """Main test function."""
    # Test URLs for a sample course
    test_videos = [
        "https://youtu.be/rjsbLZhO8Jo?si=-TqdVDCcNszn59Ec",  # Learning Technologies
        "https://youtu.be/dQw4w9WgXcQ",  # Rick Roll (for testing)
        "https://youtu.be/9bZkp7q19f0"   # PSY - GANGNAM STYLE
    ]
    
    tester = CourseAPITester()
    
    try:
        # Test 1: Create Course
        print("=" * 60)
        print("TEST 1: Create Course")
        print("=" * 60)
        course = tester.test_create_course(
            course_name="Formation Test Multi-Vidéos",
            course_description="Une formation de test avec plusieurs vidéos pour tester les fonctionnalités cross-vidéos",
            videos=test_videos
        )
        print(json.dumps(course, indent=2))
        course_id = course["course_id"]
        
        # Test 2: Process Course
        print("\n" + "=" * 60)
        print("TEST 2: Process Course")
        print("=" * 60)
        process_result = tester.test_process_course(course_id, "sequential")
        print(json.dumps(process_result, indent=2))
        
        # Test 3: Wait for Completion
        print("\n" + "=" * 60)
        print("TEST 3: Wait for Completion")
        print("=" * 60)
        final_course = tester.wait_for_course_completion(course_id)
        print(json.dumps(final_course, indent=2))
        
        # Test 4: Query Course (Cross-Video)
        print("\n" + "=" * 60)
        print("TEST 4: Query Course (Cross-Video)")
        print("=" * 60)
        query_result = tester.test_query_course(
            course_id,
            "Qu'est-ce que cette formation enseigne ?",
            "cross_video"
        )
        print(json.dumps(query_result, indent=2))
        
        # Test 5: Query Course (Single Video)
        print("\n" + "=" * 60)
        print("TEST 5: Query Course (Single Video)")
        print("=" * 60)
        single_video_query = tester.test_query_course(
            course_id,
            "Parlez-moi de la première vidéo",
            "single_video"
        )
        print(json.dumps(single_video_query, indent=2))
        
        # Test 6: Course Statistics
        print("\n" + "=" * 60)
        print("TEST 6: Course Statistics")
        print("=" * 60)
        stats = tester.test_get_course_statistics(course_id)
        print(json.dumps(stats, indent=2))
        
        # Test 7: List Courses
        print("\n" + "=" * 60)
        print("TEST 7: List Courses")
        print("=" * 60)
        courses_list = tester.test_list_courses()
        print(json.dumps(courses_list, indent=2))
        
        # Test 8: Get Video Info
        print("\n" + "=" * 60)
        print("TEST 8: Get Video Info")
        print("=" * 60)
        if final_course["videos"]:
            first_video = final_course["videos"][0]
            video_info = tester.test_get_video_info(course_id, first_video["video_id"])
            print(json.dumps(video_info, indent=2))
        
        print("\n" + "=" * 60)
        print("ALL COURSE TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    main()
