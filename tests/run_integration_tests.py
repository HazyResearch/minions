#!/usr/bin/env python
"""
Integration test runner for minions clients - zero mocking, real API calls only.
"""

import unittest
import sys
import os
import warnings

# Add the parent directory to the path so we can import minions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the tests directory to the path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.env_checker import APIKeyChecker


class IntegrationTestResult:
    """Track test results per service"""
    def __init__(self):
        self.services_tested = []
        self.services_skipped = []
        self.services_skipped_reasons = {}  # service -> reason mapping
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0


def run_service_tests(service_name: str, result_tracker: IntegrationTestResult) -> bool:
    """Run tests for a specific service"""
    # Special mappings for services with different naming
    service_to_test_name = {
        'azure': 'azure_openai',
        'llama_api': 'llama_api',
        'mlx_parallm_model': 'mlx_parallm_model',
        'cartesia': 'cartesia_mlx',
    }
    
    test_name = service_to_test_name.get(service_name, service_name)
    test_module = f"client_tests.test_{test_name}_client_integration"
    
    try:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_module)
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
            test_result = runner.run(suite)
            
            # Check if tests were skipped due to missing API key or dependencies
            skip_reasons = []
            for warning in w:
                warning_msg = str(warning.message)
                if "API key" in warning_msg or "not available" in warning_msg:
                    skip_reasons.append(warning_msg)
                elif "not installed" in warning_msg or "dependencies" in warning_msg:
                    skip_reasons.append(warning_msg)
                elif "not supported" in warning_msg:
                    skip_reasons.append(warning_msg)
            
            # If we have skip reasons, mark as skipped
            if skip_reasons:
                result_tracker.services_skipped.append(service_name)
                result_tracker.services_skipped_reasons[service_name] = skip_reasons[0]  # Use first reason
                print(f"⚠️  {service_name}: {skip_reasons[0]}")
                return True  # Not a failure, just skipped
        
        # Update tracking
        result_tracker.total_tests += test_result.testsRun
        result_tracker.passed_tests += test_result.testsRun - len(test_result.failures) - len(test_result.errors) - len(test_result.skipped)
        result_tracker.failed_tests += len(test_result.failures) + len(test_result.errors)
        result_tracker.skipped_tests += len(test_result.skipped)
        
        # Only add to tested services if tests actually ran (not all skipped)
        if test_result.testsRun > 0 and (test_result.testsRun - len(test_result.skipped)) > 0:
            result_tracker.services_tested.append(service_name)
        elif test_result.testsRun == 0 or test_result.testsRun == len(test_result.skipped):
            # All tests were skipped, move to skipped services if not already there
            if service_name not in result_tracker.services_skipped:
                result_tracker.services_skipped.append(service_name)
                result_tracker.services_skipped_reasons[service_name] = "All tests skipped (likely missing API key or dependencies)"
        
        return test_result.wasSuccessful()
        
    except ModuleNotFoundError:
        result_tracker.services_skipped.append(service_name)
        result_tracker.services_skipped_reasons[service_name] = "Test file not found"
        print(f"⚠️  No tests found for {service_name}")
        return True


def print_summary(result_tracker: IntegrationTestResult):
    """Print test summary"""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    
    print(f"Total Tests Run: {result_tracker.total_tests}")
    print(f"Passed: {result_tracker.passed_tests}")
    print(f"Failed: {result_tracker.failed_tests}")
    print(f"Skipped: {result_tracker.skipped_tests}")
    print()
    
    if result_tracker.services_tested:
        print("✅ Services Tested Successfully:")
        for service in result_tracker.services_tested:
            print(f"   - {service}")
        print()
    
    if result_tracker.services_skipped:
        print("⚠️  Services Skipped:")
        for service in result_tracker.services_skipped:
            reason = result_tracker.services_skipped_reasons.get(service, "Unknown reason")
            print(f"   - {service}: {reason}")
        print()
    
    # Show available vs missing API keys
    available = APIKeyChecker.get_available_services()
    all_services = list(APIKeyChecker.REQUIRED_KEYS.keys())
    missing = [s for s in all_services if s not in available]
    
    print("API Key Status:")
    print(f"  Available: {len(available)}/{len(all_services)} services")
    if missing:
        print(f"  Missing keys for: {', '.join(missing)}")
    print()
    
    # Show skip reasons summary
    if result_tracker.services_skipped_reasons:
        print("Skip Reasons Summary:")
        skip_categories = {}
        for service, reason in result_tracker.services_skipped_reasons.items():
            if "API key" in reason:
                category = "Missing API Keys"
            elif "not installed" in reason or "dependencies" in reason:
                category = "Missing Dependencies"
            elif "not supported" in reason:
                category = "Platform Not Supported"
            elif "Test file not found" in reason:
                category = "Missing Test Files"
            else:
                category = "Other"
            
            if category not in skip_categories:
                skip_categories[category] = []
            skip_categories[category].append(service)
        
        for category, services in skip_categories.items():
            print(f"  {category}: {', '.join(services)}")
        print()


def main():
    """Main test runner"""
    # Print API key status first
    print("Minions Integration Tests (Real API Calls)")
    print("=" * 50)
    APIKeyChecker.print_status()
    
    # Determine which tests to run
    if len(sys.argv) > 1:
        # Run specific service
        service = sys.argv[1].lower()
        services_to_test = [service]
    else:
        # Run all available services
        services_to_test = [
            'openai', 'anthropic', 'mistral', 'groq', 'gemini',
            'together', 'perplexity', 'sambanova', 'deepseek',
            'huggingface', 'grok', 'sarvam', 'azure', 'llama_api',
            'cartesia', 'openrouter', 'tokasaurus',
            'ollama',
            'llamacpp', 'mlx_audio', 'mlx_lm', 'mlx_omni',
            'mlx_parallm_model', 'secure', 'transformers'
        ]
    
    result_tracker = IntegrationTestResult()
    all_success = True
    
    for service in services_to_test:
        print(f"\n{'='*20} Testing {service.upper()} {'='*20}")
        
        success = run_service_tests(service, result_tracker)
        all_success = all_success and success
    
    # Print final summary
    print_summary(result_tracker)
    
    if result_tracker.failed_tests > 0:
        print("❌ Some tests failed!")
        return False
    elif result_tracker.total_tests == 0:
        print("⚠️  No tests were run (all skipped due to missing API keys)")
        return True
    else:
        print("✅ All tests passed!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)