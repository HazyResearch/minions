#!/usr/bin/env python3
"""
Test runner for A2A-Minions integration tests.

This script runs the integration test suite that requires a running A2A-Minions server.
Make sure the server is running at http://localhost:8001 before running tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Run integration tests."""
    
    # Change to test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Available test files
    test_files = {
        "minion_query": "test_minion_query.py",
        "minions_query": "test_minions_query.py",
        "all": "both"
    }
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_choice = sys.argv[1].lower()
    else:
        print("🧪 A2A-Minions Integration Test Runner")
        print("=" * 50)
        print()
        print("❗ PREREQUISITE: A2A-Minions server must be running at http://localhost:8001")
        print()
        print("Available tests:")
        for key, file in test_files.items():
            if key != "all":
                print(f"  {key}: {file}")
        print("  all: Run all tests")
        print()
        
        test_choice = input("Select test to run (or 'all'): ").lower().strip()
    
    if test_choice not in test_files:
        print(f"❌ Invalid test choice: {test_choice}")
        print(f"Available options: {', '.join(test_files.keys())}")
        sys.exit(1)
    
    # Check if server is running
    print("🔍 Checking server availability...")
    try:
        import httpx
        with httpx.Client() as client:
            response = client.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and healthy")
            else:
                print(f"❌ Server returned status {response.status_code}")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Start the server with: python run_server.py")
        sys.exit(1)
    
    # Run tests
    if test_choice == "all":
        test_files_to_run = ["test_minion_query.py", "test_minions_query.py"]
    else:
        test_files_to_run = [test_files[test_choice]]
    
    print(f"🚀 Running tests: {', '.join(test_files_to_run)}")
    print("-" * 50)
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files_to_run:
        print(f"\n📁 Running {test_file}...")
        print("-" * 30)
        
        # Run the Python script directly
        cmd = [sys.executable, test_file]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print(f"✅ {test_file} passed")
            total_passed += 1
        else:
            print(f"❌ {test_file} failed (exit code {result.returncode})")
            total_failed += 1
    
    # Final summary
    print("\n" + "=" * 50)
    print(f"📊 Overall Results: {total_passed} passed, {total_failed} failed")
    
    if total_failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️ {total_failed} test file(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 