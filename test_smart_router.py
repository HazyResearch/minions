"""
Test script for Smart Router implementation

This script demonstrates the smart routing functionality and shows
how different query types are classified and routed.
"""

from minions.utils.smart_router import SmartRouter


def test_query_classification():
    """Test the query classification functionality"""
    print("\n" + "="*60)
    print("Testing Query Classification")
    print("="*60)

    router = SmartRouter()

    test_queries = [
        ("What is a cartridge", "lookup"),
        ("Define machine learning", "lookup"),
        ("Calculate 15 + 27", "math"),
        ("Analyze the implications of quantum computing on cryptography", "multi-hop"),
        ("Extract all email addresses from the document", "extract"),
        ("Write a function to sort a list", "code"),
        ("Why did the stock market crash?", "multi-hop"),
        ("What's the weather like?", "open-ended"),
    ]

    for query, expected_type in test_queries:
        classified_type = router.classify_query_type(query)
        status = "✓" if classified_type == expected_type else "✗"
        print(f"\n{status} Query: '{query}'")
        print(f"  Expected: {expected_type}, Got: {classified_type}")


def test_threshold_logic():
    """Test the threshold logic for different query types"""
    print("\n" + "="*60)
    print("Testing Threshold Logic")
    print("="*60)

    router = SmartRouter()

    print("\nQuery Type Thresholds (higher = more conservative, tries local first):")
    for query_type, threshold in router.query_type_thresholds.items():
        print(f"  {query_type:15} : {threshold:.1f}")

    print("\nInterpretation:")
    print("  - 'lookup' (0.8): Very conservative, strongly prefers local")
    print("  - 'math' (0.7): Prefers local for math operations")
    print("  - 'multi-hop' (0.3): Escalates quickly to remote for complex reasoning")
    print("  - 'open-ended' (0.4): Leans toward remote for ambiguous queries")


def test_confidence_check():
    """Test the confidence checking logic"""
    print("\n" + "="*60)
    print("Testing Response Confidence Check")
    print("="*60)

    router = SmartRouter()

    test_responses = [
        ("A cartridge is a container that holds a substance or material.", True),
        ("I don't know what a cartridge is.", False),
        ("I'm not sure, but it might be...", False),
        ("Unclear from the context provided.", False),
        ("Hi", False),  # Too short
        ("Machine learning is a subset of artificial intelligence that enables systems to learn from data.", True),
    ]

    for response, expected_confident in test_responses:
        is_confident = router.is_response_confident(response)
        status = "✓" if is_confident == expected_confident else "✗"
        print(f"\n{status} Response: '{response[:60]}...'")
        print(f"  Expected confident: {expected_confident}, Got: {is_confident}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SMART ROUTER TEST SUITE")
    print("="*60)

    test_query_classification()
    test_threshold_logic()
    test_confidence_check()

    print("\n" + "="*60)
    print("Expected Behavior for 'What is a cartridge':")
    print("="*60)
    print("""
1. Query Classification: 'lookup' ✓
2. Threshold: 0.8 (very conservative - tries local)
3. Uncertainty Measurement: LOW (definition queries have high agreement)
4. Decision: Should NOT escalate (uncertainty < 0.8)
5. Expected routing: Local-only
6. Expected time: ~1-2 seconds (vs 123 seconds for full protocol)
7. Cost savings: ~$0 vs remote API costs
    """)

    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
