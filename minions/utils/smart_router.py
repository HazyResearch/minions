"""
Smart Router for Minions Protocol

This module provides intelligent query routing to optimize performance by
determining whether queries should be handled by local models alone or
escalated to the full distributed protocol with remote supervision.
"""

from typing import Optional, List


class SmartRouter:
    """
    Intelligent query router that classifies queries and makes routing decisions
    based on query type and local model uncertainty.
    """

    def __init__(self):
        """
        Initialize the SmartRouter with query type thresholds.

        Thresholds determine when to escalate to remote based on uncertainty:
        - Higher threshold = more conservative, tries local first
        - Lower threshold = escalates to remote more quickly
        """
        self.query_type_thresholds = {
            'lookup': 0.8,      # Very conservative - try local first
            'math': 0.7,        # Local good at math
            'extract': 0.6,     # Moderate
            'multi-hop': 0.3,   # Escalate quickly for complex reasoning
            'code': 0.5,        # Moderate
            'open-ended': 0.4   # Lean toward remote
        }

    def classify_query_type(self, query: str) -> str:
        """
        Classify the query into predefined categories based on pattern matching.

        Args:
            query: The user's query string

        Returns:
            Query type as a string: 'lookup', 'math', 'extract', 'multi-hop',
            'code', or 'open-ended'
        """
        query_lower = query.lower().strip()

        # Lookup patterns (route to local first)
        if any(pattern in query_lower for pattern in [
            'what is', 'define', 'definition of', 'meaning of',
            'what are', 'explain', 'describe', 'who is', 'who are'
        ]):
            return 'lookup'

        # Math patterns (route to local first)
        elif any(pattern in query_lower for pattern in [
            'calculate', 'compute', '+', '-', '*', '/', 'sum',
            'average', 'mean', 'total', 'count'
        ]):
            return 'math'

        # Extract patterns (moderate)
        elif any(pattern in query_lower for pattern in [
            'extract', 'find all', 'list all', 'get all',
            'show me', 'give me'
        ]):
            return 'extract'

        # Code patterns (moderate)
        elif any(pattern in query_lower for pattern in [
            'code', 'function', 'class', 'method', 'implement',
            'write a', 'create a', 'debug'
        ]):
            return 'code'

        # Complex reasoning (route to remote)
        elif any(pattern in query_lower for pattern in [
            'analyze', 'compare', 'evaluate', 'why', 'how does',
            'implications', 'strategy', 'recommend', 'assess',
            'predict', 'forecast'
        ]):
            return 'multi-hop'

        return 'open-ended'

    def measure_local_uncertainty(
        self,
        query: str,
        local_client,
        k: int = 3
    ) -> float:
        """
        Measure uncertainty of local model using self-consistency.

        Generates multiple samples from the local model and measures
        disagreement as a proxy for uncertainty.

        Args:
            query: The query to test
            local_client: The local LLM client
            k: Number of samples to generate (default: 3)

        Returns:
            Uncertainty score from 0 to 1, where higher means more uncertain
        """
        samples = []

        # Generate k samples from the local model
        for i in range(k):
            try:
                response, _, _ = local_client.chat([{
                    "role": "user",
                    "content": query
                }])
                # Normalize the response for comparison
                samples.append(response[0].strip().lower())
            except Exception as e:
                print(f"Error generating sample {i+1}: {e}")
                # If we can't get samples, assume high uncertainty
                return 1.0

        # Simple agreement scoring: more unique responses = higher uncertainty
        unique_responses = len(set(samples))
        uncertainty = unique_responses / k

        return uncertainty

    def should_escalate_to_remote(
        self,
        query: str,
        local_client,
        query_type: str
    ) -> bool:
        """
        Determine whether to escalate to remote based on query type and uncertainty.

        Args:
            query: The user's query
            local_client: The local LLM client
            query_type: The classified query type

        Returns:
            True if should escalate to remote, False if can handle locally
        """
        # Get threshold for this query type (default to 0.5 if unknown)
        threshold = self.query_type_thresholds.get(query_type, 0.5)

        # Measure uncertainty using self-consistency
        uncertainty = self.measure_local_uncertainty(query, local_client)

        print(f"Smart Router: Query type={query_type}, Uncertainty={uncertainty:.2f}, Threshold={threshold}")

        # Escalate if uncertainty exceeds threshold
        return uncertainty > threshold

    def is_response_confident(self, response: str) -> bool:
        """
        Check if a response seems complete and confident.

        Args:
            response: The model's response string

        Returns:
            True if response appears confident, False otherwise
        """
        response_lower = response.lower().strip()

        # Basic checks for confident response
        if len(response.strip()) < 10:
            return False

        # Check for uncertainty markers
        uncertainty_markers = [
            "i don't know",
            "i'm not sure",
            "unclear",
            "cannot determine",
            "insufficient information",
            "not enough context",
            "unable to answer"
        ]

        for marker in uncertainty_markers:
            if marker in response_lower:
                return False

        return True

    def get_local_only_response(
        self,
        query: str,
        local_client,
        query_type: str
    ) -> Optional[dict]:
        """
        Try to get a response from local model only.

        Args:
            query: The user's query
            local_client: The local LLM client
            query_type: The classified query type

        Returns:
            Dict with response info if successful, None if should escalate
        """
        # Check if we should try local-only
        should_escalate = self.should_escalate_to_remote(
            query, local_client, query_type
        )

        if should_escalate:
            return None

        # Try local-only response
        try:
            local_response, usage, _ = local_client.chat([
                {"role": "user", "content": f"Answer concisely: {query}"}
            ])

            response_text = local_response[0]

            # Check if response is confident
            if self.is_response_confident(response_text):
                return {
                    "final_answer": response_text,
                    "routing_decision": f"Local-only ({query_type})",
                    "time_saved": "~99%",
                    "usage": usage.to_dict() if hasattr(usage, 'to_dict') else {}
                }
            else:
                print("Smart Router: Local response not confident, escalating to full protocol")
                return None

        except Exception as e:
            print(f"Smart Router: Error getting local response: {e}")
            return None
