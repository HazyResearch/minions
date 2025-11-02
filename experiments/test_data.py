"""
Synthetic Test Dataset for Smart Router Experiments

This module provides test queries categorized by type with expected behaviors
and ground truth answers for evaluation.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestQuery:
    """A test query with metadata for evaluation"""
    query: str
    query_type: str
    expected_route: str  # "local" or "remote"
    ground_truth: str  # Expected answer for accuracy evaluation
    context: List[str]  # Optional context documents
    difficulty: str  # "easy", "medium", "hard"


# Lookup queries - should route to local
LOOKUP_QUERIES = [
    TestQuery(
        query="What is a cartridge",
        query_type="lookup",
        expected_route="local",
        ground_truth="A cartridge is a container or case that holds a substance, device, or material. Common types include ink cartridges for printers, game cartridges for video game consoles, and ammunition cartridges for firearms.",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Define machine learning",
        query_type="lookup",
        expected_route="local",
        ground_truth="Machine learning is a subset of artificial intelligence that enables computer systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions or predictions.",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the meaning of blockchain",
        query_type="lookup",
        expected_route="local",
        ground_truth="Blockchain is a distributed digital ledger technology that records transactions across multiple computers in a way that makes the records difficult to alter retroactively. Each block contains transaction data and is cryptographically linked to the previous block.",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Explain photosynthesis",
        query_type="lookup",
        expected_route="local",
        ground_truth="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
        context=[],
        difficulty="easy"
    ),
]

# Math queries - should route to local
MATH_QUERIES = [
    TestQuery(
        query="Calculate 15 + 27",
        query_type="math",
        expected_route="local",
        ground_truth="42",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="What is the average of 10, 20, and 30",
        query_type="math",
        expected_route="local",
        ground_truth="20",
        context=[],
        difficulty="easy"
    ),
    TestQuery(
        query="Compute 144 divided by 12",
        query_type="math",
        expected_route="local",
        ground_truth="12",
        context=[],
        difficulty="easy"
    ),
]

# Extract queries - moderate difficulty
EXTRACT_QUERIES = [
    TestQuery(
        query="Extract all numbers from the text",
        query_type="extract",
        expected_route="local",
        ground_truth="Depends on context",
        context=["The project has 15 members, completed 8 tasks, and has a budget of $50,000."],
        difficulty="medium"
    ),
]

# Multi-hop reasoning queries - should route to remote
MULTIHOP_QUERIES = [
    TestQuery(
        query="Analyze the economic implications of widespread AI adoption and recommend policy interventions",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Analysis should cover: job displacement concerns, productivity gains, wealth inequality, need for education reform, social safety net adjustments, and AI governance frameworks.",
        context=[],
        difficulty="hard"
    ),
    TestQuery(
        query="Compare the advantages and disadvantages of renewable vs fossil fuel energy sources",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Comparison should include: cost, environmental impact, reliability, scalability, infrastructure requirements, and geopolitical implications.",
        context=[],
        difficulty="hard"
    ),
    TestQuery(
        query="Why did the Roman Empire fall",
        query_type="multi-hop",
        expected_route="remote",
        ground_truth="Multiple interconnected factors including: military overextension, economic troubles, political instability, barbarian invasions, and administrative challenges.",
        context=[],
        difficulty="hard"
    ),
]

# Code queries - moderate
CODE_QUERIES = [
    TestQuery(
        query="Write a Python function to check if a number is prime",
        query_type="code",
        expected_route="local",
        ground_truth="Should include: checking if n <= 1, iterating up to sqrt(n), checking divisibility",
        context=[],
        difficulty="medium"
    ),
]

# Open-ended queries - prefer remote
OPENENDED_QUERIES = [
    TestQuery(
        query="What should I do this weekend",
        query_type="open-ended",
        expected_route="remote",
        ground_truth="Depends on personal preferences, should offer varied suggestions",
        context=[],
        difficulty="medium"
    ),
]


class TestDataset:
    """Container for all test queries"""

    def __init__(self):
        self.all_queries = (
            LOOKUP_QUERIES +
            MATH_QUERIES +
            EXTRACT_QUERIES +
            MULTIHOP_QUERIES +
            CODE_QUERIES +
            OPENENDED_QUERIES
        )

        # Group by type
        self.queries_by_type = {
            "lookup": LOOKUP_QUERIES,
            "math": MATH_QUERIES,
            "extract": EXTRACT_QUERIES,
            "multi-hop": MULTIHOP_QUERIES,
            "code": CODE_QUERIES,
            "open-ended": OPENENDED_QUERIES,
        }

        # Group by expected route
        self.queries_by_route = {
            "local": [q for q in self.all_queries if q.expected_route == "local"],
            "remote": [q for q in self.all_queries if q.expected_route == "remote"],
        }

        # Group by difficulty
        self.queries_by_difficulty = {
            "easy": [q for q in self.all_queries if q.difficulty == "easy"],
            "medium": [q for q in self.all_queries if q.difficulty == "medium"],
            "hard": [q for q in self.all_queries if q.difficulty == "hard"],
        }

    def get_all(self) -> List[TestQuery]:
        """Get all test queries"""
        return self.all_queries

    def get_by_type(self, query_type: str) -> List[TestQuery]:
        """Get queries of a specific type"""
        return self.queries_by_type.get(query_type, [])

    def get_by_route(self, expected_route: str) -> List[TestQuery]:
        """Get queries expected to route a certain way"""
        return self.queries_by_route.get(expected_route, [])

    def get_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """Get queries of a specific difficulty"""
        return self.queries_by_difficulty.get(difficulty, [])

    def summary(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            "total_queries": len(self.all_queries),
            "by_type": {k: len(v) for k, v in self.queries_by_type.items()},
            "by_route": {k: len(v) for k, v in self.queries_by_route.items()},
            "by_difficulty": {k: len(v) for k, v in self.queries_by_difficulty.items()},
        }


if __name__ == "__main__":
    # Demo
    dataset = TestDataset()
    print("Test Dataset Summary")
    print("=" * 60)

    summary = dataset.summary()
    print(f"Total queries: {summary['total_queries']}")
    print(f"\nBy type: {summary['by_type']}")
    print(f"By expected route: {summary['by_route']}")
    print(f"By difficulty: {summary['by_difficulty']}")

    print("\nSample queries:")
    for query_type in ["lookup", "math", "multi-hop"]:
        queries = dataset.get_by_type(query_type)
        if queries:
            print(f"\n{query_type.upper()}: {queries[0].query}")
