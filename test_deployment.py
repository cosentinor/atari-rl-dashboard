#!/usr/bin/env python3
"""
Deployment Test Suite
Tests all new features and endpoints
"""

import requests
import time
from datetime import date, timedelta
from db_manager import TrainingDatabase

BASE_URL = "http://localhost:5001"
TEST_VISITOR_UUID = None
TEST_VISITOR_ID = None

def print_test(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")
    if details:
        print(f"  Details: {details}")

def test_database_init():
    """Test database initialization"""
    print("\n=== Testing Database Initialization ===")
    try:
        db = TrainingDatabase()
        stats = db.get_database_stats()
        print_test("Database initialized", True, f"Tables: {stats}")

        visitor_stats = db.get_visitor_stats()
        print_test("Visitor tables initialized", True, f"Visitors: {visitor_stats.get('total_visitors', 0)}")
        
        # Test training stats write/read
        db.record_training_activity(
            game_id="ALE/Pong-v5",
            duration_seconds=90,
            sessions=1,
            episodes=1,
            steps=100
        )
        leaderboard = db.get_leaderboard(limit=1)
        print_test("Training stats recorded", True, f"Leaderboard rows: {len(leaderboard)}")
        return True
    except Exception as e:
        print_test("Database initialization", False, str(e))
        return False

def test_analytics_register():
    """Test analytics register endpoint"""
    print("\n=== Testing Analytics Registration ===")
    global TEST_VISITOR_UUID, TEST_VISITOR_ID
    try:
        visitor_uuid = f"visitor_test_{int(time.time())}"
        response = requests.post(
            f"{BASE_URL}/api/analytics/register",
            json={
                "visitor_id": visitor_uuid,
                "user_agent": "test-suite",
                "screen_resolution": "1920x1080",
                "referrer": "test",
            }
        )
        data = response.json()
        success = data.get('success', False)
        if success:
            TEST_VISITOR_UUID = data.get('visitor_uuid') or visitor_uuid
            TEST_VISITOR_ID = data.get('visitor_id')
        print_test("Analytics register", success, f"Visitor: {TEST_VISITOR_UUID}")
        return success
    except Exception as e:
        print_test("Analytics register", False, str(e))
        return False

def test_analytics_batch():
    """Test analytics batch endpoint"""
    print("\n=== Testing Analytics Batch ===")
    try:
        visitor_id = TEST_VISITOR_UUID or f"visitor_test_{int(time.time())}"
        events = [
            {
                "event_type": "page_view",
                "event_data": {"page": "/test"},
                "visitor_id": visitor_id
            },
            {
                "event_type": "training_start",
                "event_data": {"game_id": "ALE/Pong-v5"},
                "visitor_id": visitor_id
            }
        ]
        response = requests.post(
            f"{BASE_URL}/api/analytics/batch",
            json={"events": events}
        )
        data = response.json()
        print_test("Analytics batch logging",
                   data.get('success', False),
                   f"Logged {len(events)} events")
        return data.get('success', False)
    except Exception as e:
        print_test("Analytics batch", False, str(e))
        return False

def test_feedback_submission():
    """Test feedback endpoint"""
    print("\n=== Testing Feedback System ===")
    try:
        visitor_id = TEST_VISITOR_UUID or TEST_VISITOR_ID
        response = requests.post(
            f"{BASE_URL}/api/feedback",
            json={
                "visitor_id": visitor_id,
                "category": "test",
                "rating": 5,
                "message": "Test feedback"
            }
        )
        data = response.json()
        print_test("Feedback submission",
                   data.get('success', False),
                   f"Feedback ID: {data.get('feedback_id', 'none')}")

        response = requests.get(f"{BASE_URL}/api/feedback/stats")
        data = response.json()
        print_test("Feedback stats retrieval",
                   data.get('success', False),
                   f"Total: {data.get('stats', {}).get('total_feedback', 0)}")
        return True
    except Exception as e:
        print_test("Feedback system", False, str(e))
        return False

def test_public_stats():
    """Test public stats endpoint"""
    print("\n=== Testing Public Stats ===")
    try:
        response = requests.get(f"{BASE_URL}/api/stats/public")
        data = response.json()
        stats = data.get('stats', {})
        print_test("Public stats endpoint",
                   data.get('success', False),
                   f"Visitors: {stats.get('visitors', 0)}, Sessions: {stats.get('sessions', 0)}")
        return True
    except Exception as e:
        print_test("Public stats", False, str(e))
        return False

def test_challenges_endpoint():
    """Test challenges endpoint"""
    print("\n=== Testing Challenges ===")
    try:
        db = TrainingDatabase()
        challenge_id = db.create_challenge(
            game_id='ALE/MsPacman-v5',
            challenge_type='test',
            target_value=1000,
            description='Test challenge',
            start_date=date.today().isoformat(),
            end_date=(date.today() + timedelta(days=1)).isoformat()
        )
        print_test("Challenge creation", True, f"Challenge ID: {challenge_id}")

        visitor_id = TEST_VISITOR_UUID or TEST_VISITOR_ID or ""
        response = requests.get(f"{BASE_URL}/api/challenges?visitor_id={visitor_id}")
        data = response.json()
        print_test("Challenges endpoint",
                   data.get('success', False),
                   f"Challenges: {len(data.get('challenges', []))}")
        return True
    except Exception as e:
        print_test("Challenges", False, str(e))
        return False

def test_leaderboard_endpoint():
    """Test leaderboard endpoint"""
    print("\n=== Testing Leaderboard ===")
    try:
        response = requests.get(f"{BASE_URL}/api/leaderboard")
        data = response.json()
        print_test(
            "Leaderboard endpoint",
            data.get('success', False),
            f"Entries: {len(data.get('leaderboard', []))}"
        )
        return True
    except Exception as e:
        print_test("Leaderboard", False, str(e))
        return False

def test_queue_status():
    """Test queue status endpoint"""
    print("\n=== Testing Queue System ===")
    try:
        response = requests.get(f"{BASE_URL}/api/queue/status")
        data = response.json()
        print_test("Queue status endpoint", 
                   data.get('success', False),
                   f"Active: {data.get('active_sessions', 0)}, Max: {data.get('max_sessions', 0)}")
        return True
    except Exception as e:
        print_test("Queue system", False, str(e))
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("ATARI RL DASHBOARD - DEPLOYMENT TEST SUITE")
    print("=" * 60)
    
    # Wait for server to be ready
    print("\nWaiting for server...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/api/device", timeout=1)
            print("Server ready!")
            break
        except:
            time.sleep(1)
            if i == 9:
                print("✗ FAIL: Server not responding")
                return
    
    results = []
    results.append(test_database_init())
    results.append(test_analytics_register())
    results.append(test_analytics_batch())
    results.append(test_feedback_submission())
    results.append(test_public_stats())
    results.append(test_challenges_endpoint())
    results.append(test_leaderboard_endpoint())
    results.append(test_queue_status())
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} test groups passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED - Ready for deployment!")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
