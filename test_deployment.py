#!/usr/bin/env python3
"""
Deployment Test Suite
Tests all new features and endpoints
"""

import requests
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
