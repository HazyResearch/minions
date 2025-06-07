#!/usr/bin/env python3
"""
Run all unit tests for A2A-Minions.
"""

import sys
import os
import unittest
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all test modules
from unit.test_models import *
from unit.test_config import *
from unit.test_agent_cards import *
from unit.test_auth import *
from unit.test_client_factory import *
from unit.test_converters import *
from unit.test_metrics import *
from unit.test_server import *


def run_tests():
    """Run all unit tests and display results."""
    print("=" * 70)
    print("A2A-Minions Unit Test Suite")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    test_modules = [
        'unit.test_models',
        'unit.test_config',
        'unit.test_agent_cards',
        'unit.test_auth',
        'unit.test_client_factory',
        'unit.test_converters',
        'unit.test_metrics',
        'unit.test_server'
    ]
    
    for module_name in test_modules:
        try:
            module = sys.modules[module_name]
            suite.addTests(loader.loadTestsFromModule(module))
        except Exception as e:
            print(f"Warning: Failed to load tests from {module_name}: {e}")
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    print(f"Running {suite.countTestCases()} tests...")
    print()
    
    start_time = time.time()
    result = runner.run(suite)
    duration = time.time() - start_time
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Duration: {duration:.2f} seconds")
    print()
    
    if result.failures:
        print("Failed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("Tests with errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)