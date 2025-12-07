"""
Enhanced Smart Router with Formalized Complexity Scoring

This module implements an advanced smart routing system that uses:
1. Complexity scoring based on reasoning theory
2. Adaptive thresholds based on query characteristics
3. Self-consistency for uncertainty measurement
4. Multi-dimensional routing decisions

Key improvements over basic smart router:
- Theoretically grounded complexity analysis
- Erotetic logic for question understanding
- Bloom's taxonomy for cognitive demand
- Information-theoretic measures
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time

from minions.utils.complexity_scorer import ComplexityScorer, ComplexityScore, EroteticType


@dataclass
class RoutingDecision:
    """
    Decision made by the enhanced router
    """
    route: str  # "local" or "remote"
    confidence: float  # 0-1, confidence in decision
    complexity_score: ComplexityScore
    uncertainty_score: float  # Self-consistency uncertainty
    reasoning: str  # Human-readable explanation

    def __repr__(self):
        return f"""RoutingDecision(
    route={self.route.upper()},
    confidence={self.confidence:.3f},
    complexity={self.complexity_score.overall:.3f},
    uncertainty={self.uncertainty_score:.3f},
    reasoning="{self.reasoning}"
)"""


class EnhancedSmartRouter:
    """
    Enhanced smart router using complexity theory and adaptive thresholds

    This router improves upon basic pattern matching by:
    1. Using formalized complexity scoring
    2. Combining complexity with uncertainty measurement
    3. Adaptive decision-making based on multiple factors
    4. Explainable routing decisions
    """

    def __init__(
        self,
        local_client,
        remote_client=None,
        base_threshold: float = 0.5,
        uncertainty_weight: float = 0.4,
        complexity_weight: float = 0.6,
        self_consistency_k: int = 3
    ):
        """
        Initialize enhanced smart router

        Args:
            local_client: Local LLM client
            remote_client: Remote LLM client (optional)
            base_threshold: Base complexity threshold for routing
            uncertainty_weight: Weight for uncertainty in decision (0-1)
            complexity_weight: Weight for complexity in decision (0-1)
            self_consistency_k: Number of samples for self-consistency check
        """
        self.local_client = local_client
        self.remote_client = remote_client
        self.base_threshold = base_threshold
        self.uncertainty_weight = uncertainty_weight
        self.complexity_weight = complexity_weight
        self.self_consistency_k = self_consistency_k

        # Initialize complexity scorer
        self.complexity_scorer = ComplexityScorer()

        # Adaptive thresholds by erotetic type
        # These thresholds are adjusted based on query type characteristics
        self.erotetic_thresholds = {
            EroteticType.FACTUAL: 0.7,       # Higher threshold - prefer local
            EroteticType.CONCEPTUAL: 0.6,    # Moderate-high - usually local
            EroteticType.PROCEDURAL: 0.55,   # Moderate - depends on complexity
            EroteticType.CAUSAL: 0.5,        # Base threshold
            EroteticType.COMPARATIVE: 0.45,  # Moderate-low - often need remote
            EroteticType.EVALUATIVE: 0.4,    # Lower - prefer remote
            EroteticType.SYNTHETIC: 0.3,     # Much lower - usually remote
            EroteticType.METACOGNITIVE: 0.4  # Lower - prefer remote
        }

    def route(
        self,
        query: str,
        context: Optional[List[str]] = None,
        explain: bool = True
    ) -> RoutingDecision:
        """
        Make routing decision for a query

        Args:
            query: The query to route
            context: Optional context documents
            explain: Whether to provide detailed reasoning

        Returns:
            RoutingDecision with route, confidence, and explanation
        """
        # Step 1: Compute complexity score
        complexity_score = self.complexity_scorer.score(query, context)

        # Step 2: Measure uncertainty via self-consistency
        uncertainty_score = self._measure_uncertainty(query, context)

        # Step 3: Get adaptive threshold for this query type
        threshold = self.erotetic_thresholds[complexity_score.erotetic_type]

        # Step 4: Make routing decision
        route, confidence = self._make_decision(
            complexity_score,
            uncertainty_score,
            threshold
        )

        # Step 5: Generate explanation
        reasoning = self._explain_decision(
            route,
            complexity_score,
            uncertainty_score,
            threshold,
            confidence
        ) if explain else ""

        return RoutingDecision(
            route=route,
            confidence=confidence,
            complexity_score=complexity_score,
            uncertainty_score=uncertainty_score,
            reasoning=reasoning
        )

    def _measure_uncertainty(
        self,
        query: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 200
    ) -> float:
        """
        Measure uncertainty via self-consistency

        Generate k responses and measure diversity as uncertainty signal.
        High diversity (many unique responses) = high uncertainty

        Based on Wang et al. (2023): Self-Consistency Improves Chain of Thought Reasoning
        """
        if self.self_consistency_k < 2:
            return 0.0  # No uncertainty measurement

        responses = []

        try:
            # Generate k responses
            for _ in range(self.self_consistency_k):
                # Create messages
                messages = [{"role": "user", "content": query}]

                if context:
                    context_str = "\n\n".join(context)
                    messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

                # Get response from local model
                result = self.local_client.chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7  # Some randomness for diversity
                )

                # Handle different return formats
                if isinstance(result, tuple):
                    response = result[0][0] if isinstance(result[0], list) else result[0]
                else:
                    response = result[0] if isinstance(result, list) else result

                responses.append(response.strip().lower())

            # Compute diversity as uncertainty measure
            unique_responses = len(set(responses))
            uncertainty = unique_responses / len(responses)

            return uncertainty

        except Exception as e:
            print(f"Warning: Error in uncertainty measurement: {e}")
            # On error, assume moderate uncertainty
            return 0.5

    def _make_decision(
        self,
        complexity_score: ComplexityScore,
        uncertainty_score: float,
        threshold: float
    ) -> Tuple[str, float]:
        """
        Make routing decision combining complexity and uncertainty

        Args:
            complexity_score: ComplexityScore object
            uncertainty_score: Uncertainty from self-consistency (0-1)
            threshold: Decision threshold for this query type

        Returns:
            (route, confidence) where route is "local" or "remote"
        """
        # Combine complexity and uncertainty
        # Higher weight on what matters more for decision
        combined_score = (
            self.complexity_weight * complexity_score.overall +
            self.uncertainty_weight * uncertainty_score
        )

        # Decision: compare combined score to threshold
        if combined_score < threshold:
            route = "local"
            # Confidence: how far below threshold
            confidence = min((threshold - combined_score) / threshold, 1.0)
        else:
            route = "remote"
            # Confidence: how far above threshold
            confidence = min((combined_score - threshold) / (1.0 - threshold), 1.0)

        return route, confidence

    def _explain_decision(
        self,
        route: str,
        complexity_score: ComplexityScore,
        uncertainty_score: float,
        threshold: float,
        confidence: float
    ) -> str:
        """
        Generate human-readable explanation of routing decision
        """
        lines = []

        # Overall decision
        lines.append(f"Routing to {route.upper()} with {confidence:.1%} confidence")

        # Complexity analysis
        lines.append(f"Complexity: {complexity_score.overall:.3f} (threshold: {threshold:.3f})")
        lines.append(f"  - Question type: {complexity_score.erotetic_type.value}")
        lines.append(f"  - Cognitive level: {complexity_score.bloom_level.name} (L{complexity_score.bloom_level.value})")
        lines.append(f"  - Reasoning depth: {complexity_score.reasoning_depth} steps")

        # Uncertainty analysis
        lines.append(f"Uncertainty: {uncertainty_score:.3f}")
        if uncertainty_score < 0.4:
            lines.append("  → Local model is confident (consistent responses)")
        elif uncertainty_score < 0.7:
            lines.append("  → Moderate uncertainty")
        else:
            lines.append("  → High uncertainty (inconsistent responses)")

        # Key factors in decision
        if route == "local":
            reasons = []
            if complexity_score.overall < 0.4:
                reasons.append("low complexity")
            if complexity_score.reasoning_depth <= 2:
                reasons.append("shallow reasoning")
            if uncertainty_score < 0.5:
                reasons.append("high confidence")
            if reasons:
                lines.append(f"Key factors: {', '.join(reasons)}")
        else:
            reasons = []
            if complexity_score.overall > 0.6:
                reasons.append("high complexity")
            if complexity_score.reasoning_depth >= 3:
                reasons.append("multi-hop reasoning required")
            if uncertainty_score > 0.6:
                reasons.append("local model uncertain")
            if reasons:
                lines.append(f"Key factors: {', '.join(reasons)}")

        return "\n".join(lines)

    def process_query(
        self,
        query: str,
        context: Optional[List[str]] = None,
        max_tokens: int = 2048,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Process query end-to-end with routing

        Args:
            query: Query to process
            context: Optional context documents
            max_tokens: Max tokens for response
            verbose: Print routing decision

        Returns:
            Dict with answer, routing info, timing
        """
        start_time = time.time()

        # Make routing decision
        decision = self.route(query, context, explain=True)

        if verbose:
            print("\n" + "="*80)
            print("ROUTING DECISION")
            print("="*80)
            print(decision.reasoning)
            print("="*80 + "\n")

        # Execute based on routing
        if decision.route == "local":
            # Use local model
            messages = [{"role": "user", "content": query}]
            if context:
                context_str = "\n\n".join(context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

            result = self.local_client.chat(messages=messages, max_tokens=max_tokens)
            if isinstance(result, tuple):
                answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                answer = result[0] if isinstance(result, list) else result
        else:
            # Use remote model (or full protocol if available)
            if self.remote_client:
                messages = [{"role": "user", "content": query}]
                if context:
                    context_str = "\n\n".join(context)
                    messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

                result = self.remote_client.chat(messages=messages, max_tokens=max_tokens)
                if isinstance(result, tuple):
                    answer = result[0][0] if isinstance(result[0], list) else result[0]
                else:
                    answer = result[0] if isinstance(result, list) else result
            else:
                answer = "Remote client not configured"

        elapsed = time.time() - start_time

        return {
            "answer": answer,
            "routing": {
                "decision": decision.route,
                "confidence": decision.confidence,
                "complexity": decision.complexity_score.overall,
                "uncertainty": decision.uncertainty_score,
                "erotetic_type": decision.complexity_score.erotetic_type.value,
                "bloom_level": decision.complexity_score.bloom_level.name,
                "reasoning_depth": decision.complexity_score.reasoning_depth
            },
            "time_seconds": elapsed
        }


def demo():
    """Demo of enhanced smart router"""
    print("\n" + "="*80)
    print("ENHANCED SMART ROUTER DEMO")
    print("="*80)
    print("\nThis demo shows routing decisions without actually calling models.")
    print("To use with real models, initialize with local_client and remote_client.")
    print("="*80 + "\n")

    # Create mock client for demo
    class MockClient:
        def chat(self, messages, max_tokens=100, temperature=0.7):
            return ["Mock response"], None

    router = EnhancedSmartRouter(
        local_client=MockClient(),
        remote_client=MockClient(),
        self_consistency_k=0  # Skip uncertainty for demo
    )

    test_queries = [
        "What is the capital of France?",
        "Explain how neural networks work",
        "Why did the Roman Empire fall?",
        "Compare renewable vs fossil fuel energy and recommend policy changes",
        "Analyze the economic implications of AI adoption"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*80)
        decision = router.route(query, explain=True)
        print(decision.reasoning)
        print("\n" + "="*80)


if __name__ == "__main__":
    demo()
