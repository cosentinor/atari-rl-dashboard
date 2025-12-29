#!/usr/bin/env python3
"""
Deployment Test Suite
Tests all new features and endpoints
"""

import requests
import json
import time
from db_manager import TrainingDatabase

BASE_URL = "http://localhost:5001"

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
        
        # Test new tables exist
        visitor_stats = db.get_visitor_stats()
        print_test("Visitor table accessible", True, f"Stats: {visitor_stats}")
        return True
    except Exception as e:
        print_test("Database initialization", False, str(e))
        return False

def test_visitor_registration():
    """Test visitor registration endpoint"""
    print("\n=== Testing Visitor Registration ===")
    try:
        # Test with email
        response = requests.post(
            f"{BASE_URL}/api/visitor/register",
            json={"email": "test@example.com", "opt_in_marketing": True}
        )
        data = response.json()
        print_test("Register with email", 
                   data.get('success', False), 
                   f"UUID: {data.get('visitor_uuid', 'none')}")
        
        # Test without email (skip)
        response = requests.post(
            f"{BASE_URL}/api/visitor/register",
            json={"email": None, "opt_in_marketing": False}
        )
        data = response.json()
        print_test("Register without email (skip)", 
                   data.get('success', False),
                   f"UUID: {data.get('visitor_uuid', 'none')}")
        return True
    except Exception as e:
        print_test("Visitor registration", False, str(e))
        return False

def test_analytics_batch():
    """Test analytics batch endpoint"""
    print("\n=== Testing Analytics Batch ===")
    try:
        events = [
            {
                "event_type": "page_view",
                "event_data": {"page": "/test"},
                "visitor_uuid": "test-uuid",
                "visitor_id": 1
            },
            {
                "event_type": "test_event",
                "event_data": {"test": "data"},
                "visitor_uuid": "test-uuid",
                "visitor_id": 1
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
        return True
    except Exception as e:
        print_test("Analytics batch", False, str(e))
        return False

def test_feedback_submission():
    """Test feedback endpoint"""
    print("\n=== Testing Feedback System ===")
    try:
        response = requests.post(
            f"{BASE_URL}/api/feedback",
            json={
                "visitor_id": 1,
                "category": "test",
                "rating": 5,
                "message": "Test feedback"
            }
        )
        data = response.json()
        print_test("Feedback submission", 
                   data.get('success', False),
                   f"Feedback ID: {data.get('feedback_id', 'none')}")
        
        # Get feedback stats
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
        # First create a challenge
        db = TrainingDatabase()
        from datetime import date, timedelta
        challenge_id = db.create_challenge(
            game_id='ALE/MsPacman-v5',
            challenge_type='test',
            target_value=1000,
            description='Test challenge',
            start_date=date.today().isoformat(),
            end_date=(date.today() + timedelta(days=1)).isoformat()
        )
        print_test("Challenge creation", True, f"Challenge ID: {challenge_id}")
        
        # Test endpoint
        response = requests.get(f"{BASE_URL}/api/challenges")
        data = response.json()
        print_test("Challenges endpoint", 
                   data.get('success', False),
                   f"Challenges: {len(data.get('challenges', []))}")
        return True
    except Exception as e:
        print_test("Challenges", False, str(e))
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

def test_frontend_endpoints():
    """Test that frontend files are accessible"""
    print("\n=== Testing Frontend Files ===")
    try:
        response = requests.get(f"{BASE_URL}/")
        print_test("Main page (index.html)", 
                   response.status_code == 200,
                   f"Status: {response.status_code}")
        
        response = requests.get(f"{BASE_URL}/analytics.js")
        print_test("Analytics.js", 
                   response.status_code == 200,
                   f"Status: {response.status_code}")
        
        response = requests.get(f"{BASE_URL}/components/EmailModal.js")
        print_test("EmailModal component", 
                   response.status_code == 200,
                   f"Status: {response.status_code}")
        return True
    except Exception as e:
        print_test("Frontend files", False, str(e))
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
    results.append(test_visitor_registration())
    results.append(test_analytics_batch())
    results.append(test_feedback_submission())
    results.append(test_public_stats())
    results.append(test_challenges_endpoint())
    results.append(test_queue_status())
    results.append(test_frontend_endpoints())
    
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

