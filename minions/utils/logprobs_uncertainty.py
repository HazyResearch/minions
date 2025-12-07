"""
Logprobs-based Uncertainty Estimation

This module implements uncertainty measurement using log probabilities (logprobs)
from language models, inspired by OpenAI's cookbook and recent research.

Key advantages of logprobs over self-consistency:
1. No extra forward passes needed (much faster)
2. Direct model confidence signal (not sampled diversity)
3. Token-level uncertainty (granular analysis)
4. Works with single generation (no k=3 samples)

References:
- OpenAI Cookbook: Using Logprobs (https://cookbook.openai.com/examples/using_logprobs)
- Kadavath et al. (2022): "Language Models (Mostly) Know What They Know"
- Kuhn et al. (2023): "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation"
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LogprobsAnalysis:
    """
    Analysis of logprobs for uncertainty estimation
    """
    # Overall uncertainty score (0-1, higher = more uncertain)
    uncertainty: float

    # Average log probability across tokens
    avg_logprob: float

    # Entropy of token distribution
    entropy: float

    # Perplexity (exp of negative avg logprob)
    perplexity: float

    # Number of low-confidence tokens (logprob < threshold)
    num_low_confidence_tokens: int

    # Fraction of low-confidence tokens
    low_confidence_ratio: float

    # Variance in logprobs (higher = more uncertainty)
    logprob_variance: float

    # Minimum logprob seen (lowest confidence token)
    min_logprob: float

    # Maximum logprob seen (highest confidence token)
    max_logprob: float

    def __repr__(self):
        return f"""LogprobsAnalysis(
    uncertainty={self.uncertainty:.3f},
    avg_logprob={self.avg_logprob:.3f},
    entropy={self.entropy:.3f},
    perplexity={self.perplexity:.2f},
    low_confidence_ratio={self.low_confidence_ratio:.2%},
    logprob_variance={self.logprob_variance:.3f}
)"""


class LogprobsUncertaintyEstimator:
    """
    Estimates uncertainty using log probabilities from language models

    Theory:
    -------
    Logprobs represent the model's confidence in its predictions.

    - High logprob (close to 0): Model is confident (e.g., -0.01)
    - Low logprob (very negative): Model is uncertain (e.g., -5.0)

    Entropy measures the spread of probability distribution:
    - Low entropy: Peaked distribution (one choice dominates)
    - High entropy: Flat distribution (many choices equally likely)

    Connection to uncertainty:
    - If model assigns low probability to generated tokens → uncertain
    - If distribution has high entropy → many plausible alternatives → uncertain
    """

    def __init__(
        self,
        low_confidence_threshold: float = -2.0,
        uncertainty_method: str = "normalized_entropy"
    ):
        """
        Initialize logprobs uncertainty estimator

        Args:
            low_confidence_threshold: Logprob threshold for "low confidence" tokens
                                     (default -2.0 means p < 0.135)
            uncertainty_method: Method for computing uncertainty score
                               Options: "normalized_entropy", "avg_logprob", "combined"
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.uncertainty_method = uncertainty_method

    def estimate_uncertainty(
        self,
        logprobs: List[float],
        normalize: bool = True
    ) -> LogprobsAnalysis:
        """
        Estimate uncertainty from sequence of token logprobs

        Args:
            logprobs: List of log probabilities for generated tokens
            normalize: Whether to normalize uncertainty to [0, 1]

        Returns:
            LogprobsAnalysis with comprehensive uncertainty metrics
        """
        if not logprobs or len(logprobs) == 0:
            return self._empty_analysis()

        # Convert to numpy for easier computation
        logprobs_array = np.array(logprobs)

        # Basic statistics
        avg_logprob = float(np.mean(logprobs_array))
        min_logprob = float(np.min(logprobs_array))
        max_logprob = float(np.max(logprobs_array))
        logprob_variance = float(np.var(logprobs_array))

        # Perplexity: exp(-avg_logprob)
        perplexity = math.exp(-avg_logprob)

        # Entropy from logprobs
        # H = -sum(p * log(p)) where p = exp(logprob)
        probs = np.exp(logprobs_array)
        # Clip to avoid numerical issues
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)))

        # Count low-confidence tokens
        num_low_confidence = int(np.sum(logprobs_array < self.low_confidence_threshold))
        low_confidence_ratio = num_low_confidence / len(logprobs)

        # Compute overall uncertainty score
        uncertainty = self._compute_uncertainty_score(
            avg_logprob=avg_logprob,
            entropy=entropy,
            logprob_variance=logprob_variance,
            low_confidence_ratio=low_confidence_ratio,
            normalize=normalize
        )

        return LogprobsAnalysis(
            uncertainty=uncertainty,
            avg_logprob=avg_logprob,
            entropy=entropy,
            perplexity=perplexity,
            num_low_confidence_tokens=num_low_confidence,
            low_confidence_ratio=low_confidence_ratio,
            logprob_variance=logprob_variance,
            min_logprob=min_logprob,
            max_logprob=max_logprob
        )

    def estimate_from_tokens(
        self,
        tokens_with_logprobs: List[Dict[str, any]]
    ) -> LogprobsAnalysis:
        """
        Estimate uncertainty from tokens with logprobs (OpenAI API format)

        Args:
            tokens_with_logprobs: List of dicts with 'token' and 'logprob' keys
                                 Example: [{'token': 'The', 'logprob': -0.5}, ...]

        Returns:
            LogprobsAnalysis
        """
        logprobs = [t['logprob'] for t in tokens_with_logprobs]
        return self.estimate_uncertainty(logprobs)

    def _compute_uncertainty_score(
        self,
        avg_logprob: float,
        entropy: float,
        logprob_variance: float,
        low_confidence_ratio: float,
        normalize: bool = True
    ) -> float:
        """
        Compute overall uncertainty score from multiple signals

        Different methods combine these signals differently
        """
        if self.uncertainty_method == "avg_logprob":
            # Use average logprob as uncertainty proxy
            # More negative = more uncertain
            # Normalize to [0, 1]: map [-10, 0] to [1, 0]
            if normalize:
                uncertainty = np.clip((-avg_logprob) / 10.0, 0, 1)
            else:
                uncertainty = -avg_logprob

        elif self.uncertainty_method == "normalized_entropy":
            # Use entropy normalized to reasonable range
            # Typical entropy range: [0, 3]
            if normalize:
                uncertainty = np.clip(entropy / 3.0, 0, 1)
            else:
                uncertainty = entropy

        elif self.uncertainty_method == "combined":
            # Combine multiple signals with weights
            # Normalized to [0, 1]
            logprob_component = np.clip((-avg_logprob) / 10.0, 0, 1)
            entropy_component = np.clip(entropy / 3.0, 0, 1)
            variance_component = np.clip(logprob_variance / 5.0, 0, 1)

            uncertainty = (
                0.4 * logprob_component +
                0.3 * entropy_component +
                0.2 * low_confidence_ratio +
                0.1 * variance_component
            )

        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")

        return float(uncertainty)

    def _empty_analysis(self) -> LogprobsAnalysis:
        """Return empty analysis for edge cases"""
        return LogprobsAnalysis(
            uncertainty=0.5,  # Moderate default
            avg_logprob=0.0,
            entropy=0.0,
            perplexity=1.0,
            num_low_confidence_tokens=0,
            low_confidence_ratio=0.0,
            logprob_variance=0.0,
            min_logprob=0.0,
            max_logprob=0.0
        )

    def compare_with_self_consistency(
        self,
        logprobs_uncertainty: float,
        self_consistency_uncertainty: float
    ) -> Dict[str, any]:
        """
        Compare logprobs-based vs self-consistency uncertainty

        Args:
            logprobs_uncertainty: Uncertainty from logprobs (0-1)
            self_consistency_uncertainty: Uncertainty from self-consistency (0-1)

        Returns:
            Dict with comparison metrics
        """
        agreement = 1.0 - abs(logprobs_uncertainty - self_consistency_uncertainty)

        # Both high = definitely uncertain
        both_high = (logprobs_uncertainty > 0.7 and self_consistency_uncertainty > 0.7)

        # Both low = definitely confident
        both_low = (logprobs_uncertainty < 0.3 and self_consistency_uncertainty < 0.3)

        # Disagreement cases
        logprobs_high_sc_low = (logprobs_uncertainty > 0.7 and self_consistency_uncertainty < 0.3)
        logprobs_low_sc_high = (logprobs_uncertainty < 0.3 and self_consistency_uncertainty > 0.7)

        interpretation = ""
        if both_high:
            interpretation = "Both signals indicate HIGH uncertainty - escalate to remote"
        elif both_low:
            interpretation = "Both signals indicate LOW uncertainty - safe for local"
        elif logprobs_high_sc_low:
            interpretation = "Model internally uncertain but externally consistent - borderline case"
        elif logprobs_low_sc_high:
            interpretation = "Model internally confident but outputs vary - check for ambiguity"
        else:
            interpretation = "Moderate uncertainty from both signals"

        return {
            "agreement": agreement,
            "both_high": both_high,
            "both_low": both_low,
            "interpretation": interpretation,
            "recommended_route": "remote" if (both_high or logprobs_high_sc_low) else "local"
        }


def demo():
    """Demonstrate logprobs uncertainty estimation"""

    print("="*80)
    print("LOGPROBS UNCERTAINTY ESTIMATION DEMO")
    print("="*80)
    print()

    estimator = LogprobsUncertaintyEstimator(uncertainty_method="combined")

    # Example 1: High confidence sequence
    print("Example 1: High Confidence (Simple Factual)")
    print("-"*80)
    print("Query: 'What is 2 + 2?'")
    print("Response: 'The answer is 4'")
    print()

    # Simulated logprobs for "The answer is 4"
    # High confidence = logprobs close to 0
    high_confidence_logprobs = [-0.01, -0.02, -0.01, -0.03]

    analysis = estimator.estimate_uncertainty(high_confidence_logprobs)
    print(analysis)
    print(f"Interpretation: LOW uncertainty ({analysis.uncertainty:.3f}) → Use LOCAL model")
    print()

    # Example 2: Low confidence sequence
    print("\nExample 2: Low Confidence (Uncertain Answer)")
    print("-"*80)
    print("Query: 'Predict the stock market in 2025'")
    print("Response: 'The market might possibly perhaps...'")
    print()

    # Simulated logprobs for uncertain words
    # Low confidence = very negative logprobs
    low_confidence_logprobs = [-3.5, -4.2, -2.8, -5.1, -3.9]

    analysis = estimator.estimate_uncertainty(low_confidence_logprobs)
    print(analysis)
    print(f"Interpretation: HIGH uncertainty ({analysis.uncertainty:.3f}) → Use REMOTE model")
    print()

    # Example 3: Mixed confidence
    print("\nExample 3: Mixed Confidence")
    print("-"*80)
    print("Query: 'Explain quantum computing'")
    print("Response: 'Quantum computing uses qubits which...'")
    print()

    # Mixed: some confident tokens, some uncertain
    mixed_logprobs = [-0.5, -0.3, -2.5, -0.4, -3.0, -0.6, -1.8]

    analysis = estimator.estimate_uncertainty(mixed_logprobs)
    print(analysis)
    print(f"Interpretation: MODERATE uncertainty ({analysis.uncertainty:.3f}) → Depends on threshold")
    print()

    # Example 4: Comparison with self-consistency
    print("\nExample 4: Comparing Logprobs vs Self-Consistency")
    print("-"*80)

    # Case 1: Both agree - high uncertainty
    print("\nCase 1: Both Uncertain")
    logprobs_unc = 0.8
    sc_unc = 0.9
    comparison = estimator.compare_with_self_consistency(logprobs_unc, sc_unc)
    print(f"Logprobs uncertainty: {logprobs_unc:.2f}")
    print(f"Self-consistency uncertainty: {sc_unc:.2f}")
    print(f"Agreement: {comparison['agreement']:.2%}")
    print(f"Interpretation: {comparison['interpretation']}")
    print(f"Recommended route: {comparison['recommended_route'].upper()}")

    # Case 2: Both agree - low uncertainty
    print("\nCase 2: Both Confident")
    logprobs_unc = 0.2
    sc_unc = 0.25
    comparison = estimator.compare_with_self_consistency(logprobs_unc, sc_unc)
    print(f"Logprobs uncertainty: {logprobs_unc:.2f}")
    print(f"Self-consistency uncertainty: {sc_unc:.2f}")
    print(f"Agreement: {comparison['agreement']:.2%}")
    print(f"Interpretation: {comparison['interpretation']}")
    print(f"Recommended route: {comparison['recommended_route'].upper()}")

    # Case 3: Disagreement
    print("\nCase 3: Disagreement (Logprobs high, SC low)")
    logprobs_unc = 0.8
    sc_unc = 0.2
    comparison = estimator.compare_with_self_consistency(logprobs_unc, sc_unc)
    print(f"Logprobs uncertainty: {logprobs_unc:.2f}")
    print(f"Self-consistency uncertainty: {sc_unc:.2f}")
    print(f"Agreement: {comparison['agreement']:.2%}")
    print(f"Interpretation: {comparison['interpretation']}")
    print(f"Recommended route: {comparison['recommended_route'].upper()}")

    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("1. Logprobs provide direct model confidence (no extra samples needed)")
    print("2. Low logprobs (< -2.0) indicate uncertainty")
    print("3. High entropy indicates many plausible alternatives")
    print("4. Combining with self-consistency provides robust uncertainty estimate")
    print("5. When both signals agree, routing decision is more reliable")
    print("="*80)


if __name__ == "__main__":
    demo()
