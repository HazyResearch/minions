"""
Query Complexity Scorer for Adaptive Routing

This module implements a formalized complexity scoring system for queries,
drawing from:
1. Erotetic Logic (Wiśniewski, 2013) - theory of questions and reasoning
2. Question Taxonomies (Graesser et al., 1994) - cognitive question classification
3. Bloom's Taxonomy (Anderson & Krathwohl, 2001) - cognitive complexity levels
4. Information Theory (Shannon, 1948) - entropy and information content

References:
- Wiśniewski, A. (2013). Questions, Inferences, and Scenarios. College Publications.
- Graesser, A. C., Person, N., & Huber, J. (1992). Mechanisms that generate questions.
  In Questions and information systems (pp. 167-187).
- Anderson, L. W., & Krathwohl, D. R. (2001). A taxonomy for learning, teaching, and
  assessing: A revision of Bloom's taxonomy of educational objectives.
- Lenat, D. B. (1983). The role of heuristics in learning by discovery.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class EroteticType(Enum):
    """
    Question types based on Erotetic Logic (Wiśniewski, 2013)

    Erotetic logic studies the logical structure of questions and their role
    in reasoning. Different question types impose different cognitive demands.
    """
    # Factual questions - seek simple facts
    FACTUAL = "factual"              # "What is X?", "When did X happen?"

    # Conceptual questions - seek definitions/explanations
    CONCEPTUAL = "conceptual"        # "What does X mean?", "How does X work?"

    # Procedural questions - seek methods/procedures
    PROCEDURAL = "procedural"        # "How to do X?", "What are steps for X?"

    # Causal questions - seek causes/effects
    CAUSAL = "causal"                # "Why does X happen?", "What causes X?"

    # Comparative questions - seek comparisons
    COMPARATIVE = "comparative"      # "How does X compare to Y?"

    # Evaluative questions - seek judgments/recommendations
    EVALUATIVE = "evaluative"        # "Should I do X?", "Which is better?"

    # Synthetic questions - require synthesis of multiple concepts
    SYNTHETIC = "synthetic"          # "Analyze X", "Evaluate the impact of X"

    # Meta-cognitive questions - questions about thinking/reasoning
    METACOGNITIVE = "metacognitive"  # "How should I approach X?"


class BloomLevel(Enum):
    """
    Cognitive complexity levels from Bloom's Taxonomy (revised, 2001)

    Each level represents increasing cognitive demand:
    1 (Remember) → 6 (Create)
    """
    REMEMBER = 1     # Recall facts and basic concepts
    UNDERSTAND = 2   # Explain ideas or concepts
    APPLY = 3        # Use information in new situations
    ANALYZE = 4      # Draw connections among ideas
    EVALUATE = 5     # Justify a decision or course of action
    CREATE = 6       # Produce new or original work


@dataclass
class ComplexityScore:
    """
    Multi-dimensional complexity score for a query
    """
    # Overall complexity (0-1 scale, higher = more complex)
    overall: float

    # Erotetic type
    erotetic_type: EroteticType

    # Bloom's taxonomy level (1-6)
    bloom_level: BloomLevel

    # Reasoning depth (number of inference steps required)
    reasoning_depth: int

    # Semantic entropy (information-theoretic complexity)
    semantic_entropy: float

    # Syntactic complexity (structural complexity)
    syntactic_complexity: float

    # Domain specificity (0-1, higher = more specialized knowledge needed)
    domain_specificity: float

    # Breakdown for interpretability
    breakdown: Dict[str, float]

    def __repr__(self):
        return f"""ComplexityScore(
    overall={self.overall:.3f},
    erotetic_type={self.erotetic_type.value},
    bloom_level={self.bloom_level.name} (L{self.bloom_level.value}),
    reasoning_depth={self.reasoning_depth},
    semantic_entropy={self.semantic_entropy:.3f},
    syntactic={self.syntactic_complexity:.3f},
    domain={self.domain_specificity:.3f}
)"""


class ComplexityScorer:
    """
    Formalized complexity scorer using reasoning theory
    """

    def __init__(self):
        self._init_patterns()
        self._init_weights()

    def _init_patterns(self):
        """Initialize patterns for erotetic classification"""

        # Factual question patterns
        self.factual_patterns = [
            r'\bwhat is\b', r'\bwho is\b', r'\bwhen\b', r'\bwhere\b',
            r'\bdefine\b', r'\blist\b', r'\bname\b'
        ]

        # Conceptual question patterns
        self.conceptual_patterns = [
            r'\bexplain\b', r'\bdescribe\b', r'\bwhat does\b',
            r'\bhow does .+ work\b', r'\bwhat are .+ of\b'
        ]

        # Procedural question patterns
        self.procedural_patterns = [
            r'\bhow to\b', r'\bhow do i\b', r'\bhow can i\b',
            r'\bsteps to\b', r'\bwrite .+function\b', r'\bimplement\b',
            r'\bcreate\b', r'\bbuild\b'
        ]

        # Causal question patterns
        self.causal_patterns = [
            r'\bwhy\b', r'\bwhat causes\b', r'\bhow come\b',
            r'\breason for\b', r'\bdue to\b'
        ]

        # Comparative question patterns
        self.comparative_patterns = [
            r'\bcompare\b', r'\bvs\b', r'\bversus\b', r'\bdifference between\b',
            r'\bbetter than\b', r'\bcontrast\b'
        ]

        # Evaluative question patterns
        self.evaluative_patterns = [
            r'\bshould i\b', r'\bshould we\b', r'\brecommend\b',
            r'\bwhich is best\b', r'\bevaluate\b', r'\bassess\b'
        ]

        # Synthetic question patterns
        self.synthetic_patterns = [
            r'\banalyze\b', r'\bsynthesize\b', r'\bdiscuss\b',
            r'\bimplications of\b', r'\bimpact of\b', r'\brelationship between\b',
            r'\bconsider .+ and\b'
        ]

        # Meta-cognitive patterns
        self.metacognitive_patterns = [
            r'\bhow should i think about\b', r'\bapproach to\b',
            r'\bstrategy for\b', r'\bframework for\b'
        ]

    def _init_weights(self):
        """
        Initialize weights for complexity score calculation

        Based on empirical analysis and theoretical grounding from:
        - Erotetic logic: question type impacts reasoning difficulty
        - Information theory: entropy correlates with processing difficulty
        - Cognitive psychology: working memory constraints
        """
        self.weights = {
            'erotetic': 0.25,      # Question type intrinsic complexity
            'bloom': 0.20,         # Cognitive level required
            'reasoning_depth': 0.20,  # Multi-hop reasoning
            'semantic_entropy': 0.15,  # Information content
            'syntactic': 0.10,     # Structural complexity
            'domain': 0.10         # Specialized knowledge
        }

    def score(self, query: str, context: Optional[List[str]] = None) -> ComplexityScore:
        """
        Compute comprehensive complexity score for a query

        Args:
            query: The question/query to score
            context: Optional context documents

        Returns:
            ComplexityScore with multi-dimensional analysis
        """
        query_lower = query.lower()

        # 1. Erotetic type classification
        erotetic_type = self._classify_erotetic_type(query_lower)

        # 2. Bloom's taxonomy level
        bloom_level = self._infer_bloom_level(query_lower, erotetic_type)

        # 3. Reasoning depth (number of inference steps)
        reasoning_depth = self._estimate_reasoning_depth(query_lower, erotetic_type)

        # 4. Semantic entropy (information-theoretic complexity)
        semantic_entropy = self._compute_semantic_entropy(query)

        # 5. Syntactic complexity
        syntactic_complexity = self._compute_syntactic_complexity(query)

        # 6. Domain specificity
        domain_specificity = self._estimate_domain_specificity(query_lower)

        # Compute component scores (normalized to 0-1)
        erotetic_score = self._erotetic_to_score(erotetic_type)
        bloom_score = (bloom_level.value - 1) / 5.0  # Normalize to 0-1
        reasoning_score = min(reasoning_depth / 5.0, 1.0)  # Cap at 5 steps

        # Build breakdown
        breakdown = {
            'erotetic': erotetic_score,
            'bloom': bloom_score,
            'reasoning_depth': reasoning_score,
            'semantic_entropy': semantic_entropy,
            'syntactic': syntactic_complexity,
            'domain': domain_specificity
        }

        # Weighted overall score
        overall = sum(
            breakdown[key] * self.weights[key]
            for key in self.weights
        )

        return ComplexityScore(
            overall=overall,
            erotetic_type=erotetic_type,
            bloom_level=bloom_level,
            reasoning_depth=reasoning_depth,
            semantic_entropy=semantic_entropy,
            syntactic_complexity=syntactic_complexity,
            domain_specificity=domain_specificity,
            breakdown=breakdown
        )

    def _classify_erotetic_type(self, query: str) -> EroteticType:
        """
        Classify question type using erotetic logic framework

        Based on Wiśniewski's taxonomy of interrogative sentences
        """
        # Check patterns in order of specificity (most specific first)
        if any(re.search(p, query) for p in self.synthetic_patterns):
            return EroteticType.SYNTHETIC

        if any(re.search(p, query) for p in self.metacognitive_patterns):
            return EroteticType.METACOGNITIVE

        if any(re.search(p, query) for p in self.evaluative_patterns):
            return EroteticType.EVALUATIVE

        if any(re.search(p, query) for p in self.comparative_patterns):
            return EroteticType.COMPARATIVE

        if any(re.search(p, query) for p in self.causal_patterns):
            return EroteticType.CAUSAL

        if any(re.search(p, query) for p in self.procedural_patterns):
            return EroteticType.PROCEDURAL

        if any(re.search(p, query) for p in self.conceptual_patterns):
            return EroteticType.CONCEPTUAL

        if any(re.search(p, query) for p in self.factual_patterns):
            return EroteticType.FACTUAL

        # Default: treat as conceptual
        return EroteticType.CONCEPTUAL

    def _infer_bloom_level(self, query: str, erotetic_type: EroteticType) -> BloomLevel:
        """
        Infer Bloom's taxonomy level from question structure

        Maps erotetic types to cognitive complexity levels
        """
        # Mapping based on cognitive demand
        erotetic_to_bloom = {
            EroteticType.FACTUAL: BloomLevel.REMEMBER,
            EroteticType.CONCEPTUAL: BloomLevel.UNDERSTAND,
            EroteticType.PROCEDURAL: BloomLevel.APPLY,
            EroteticType.CAUSAL: BloomLevel.ANALYZE,
            EroteticType.COMPARATIVE: BloomLevel.ANALYZE,
            EroteticType.EVALUATIVE: BloomLevel.EVALUATE,
            EroteticType.SYNTHETIC: BloomLevel.CREATE,
            EroteticType.METACOGNITIVE: BloomLevel.EVALUATE
        }

        base_level = erotetic_to_bloom[erotetic_type]

        # Adjust based on query complexity indicators
        if re.search(r'\b(analyze|synthesize|evaluate)\b', query):
            # Explicit high-level cognitive verbs
            if base_level.value < BloomLevel.ANALYZE.value:
                return BloomLevel.ANALYZE
            return base_level

        if re.search(r'\b(and|or|but|however|although)\b', query):
            # Conjunctions suggest multi-faceted reasoning
            current_value = base_level.value
            upgraded_value = min(current_value + 1, 6)
            return BloomLevel(upgraded_value)

        return base_level

    def _estimate_reasoning_depth(self, query: str, erotetic_type: EroteticType) -> int:
        """
        Estimate number of reasoning steps required (multi-hop reasoning)

        Based on:
        - Yang et al. (2018): HotpotQA multi-hop reasoning
        - Question decomposition theory
        """
        depth = 1  # Minimum depth

        # Base depth from erotetic type
        type_depths = {
            EroteticType.FACTUAL: 1,
            EroteticType.CONCEPTUAL: 1,
            EroteticType.PROCEDURAL: 2,
            EroteticType.CAUSAL: 2,
            EroteticType.COMPARATIVE: 3,
            EroteticType.EVALUATIVE: 3,
            EroteticType.SYNTHETIC: 4,
            EroteticType.METACOGNITIVE: 3
        }
        depth = type_depths[erotetic_type]

        # Indicators of multi-hop reasoning
        multi_hop_indicators = [
            (r'\b(and|or)\b', 1),  # Conjunctions suggest multiple sub-questions
            (r'\b(compare|contrast)\b', 1),  # Comparison requires 2+ retrievals
            (r'\b(analyze|synthesize)\b', 2),  # Synthesis requires 3+ steps
            (r'\b(implications?|impact|effects?)\b', 1),  # Causal chains
            (r'\brecommend\b', 1),  # Evaluation requires multi-step reasoning
        ]

        for pattern, additional_depth in multi_hop_indicators:
            if re.search(pattern, query):
                depth += additional_depth

        # Count clauses (more clauses = more reasoning steps)
        clause_markers = len(re.findall(r'[,;]', query))
        depth += min(clause_markers, 2)  # Cap clause contribution

        return min(depth, 10)  # Cap at 10 reasoning steps

    def _compute_semantic_entropy(self, query: str) -> float:
        """
        Compute semantic entropy (information-theoretic complexity)

        Based on Shannon entropy: H(X) = -Σ p(x) log p(x)

        Higher entropy = more information content = more complex to process
        """
        # Tokenize (simple word-based)
        words = re.findall(r'\b\w+\b', query.lower())

        if not words:
            return 0.0

        # Compute word frequency distribution
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Compute entropy
        total_words = len(words)
        entropy = 0.0

        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum possible entropy (log2 of vocabulary size)
        max_entropy = math.log2(len(word_counts)) if len(word_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return normalized_entropy

    def _compute_syntactic_complexity(self, query: str) -> float:
        """
        Compute syntactic (structural) complexity

        Factors:
        - Query length (longer = more complex)
        - Nested structures (subordinate clauses)
        - Question words (multiple = more complex)
        """
        # Length component (normalize by typical query length ~15 words)
        words = len(re.findall(r'\b\w+\b', query))
        length_score = min(words / 30.0, 1.0)  # Cap at 30 words

        # Nesting component (subordinate clauses, parentheticals)
        nesting_markers = len(re.findall(r'[(),]', query))
        nesting_score = min(nesting_markers / 5.0, 1.0)  # Cap at 5 markers

        # Multiple question words suggest compound questions
        question_words = len(re.findall(
            r'\b(what|who|when|where|why|how|which)\b',
            query.lower()
        ))
        question_score = min(question_words / 3.0, 1.0)  # Cap at 3

        # Combined syntactic complexity
        syntactic = (length_score + nesting_score + question_score) / 3.0

        return syntactic

    def _estimate_domain_specificity(self, query: str) -> float:
        """
        Estimate domain-specific knowledge required

        Higher domain specificity = more specialized knowledge needed
        """
        # Domain-specific indicators (technical terms, jargon)
        domain_indicators = [
            # Technical/scientific
            r'\b(algorithm|neural|quantum|protein|genome|theorem)\b',
            # Business/economics
            r'\b(gdp|inflation|equity|roi|revenue|market)\b',
            # Academic
            r'\b(hypothesis|methodology|paradigm|epistemology)\b',
            # Medical
            r'\b(diagnosis|treatment|symptom|syndrome|pathology)\b',
            # Legal
            r'\b(statute|jurisdiction|plaintiff|defendant|liability)\b',
        ]

        domain_matches = sum(
            1 for pattern in domain_indicators
            if re.search(pattern, query)
        )

        # Normalize by number of indicator categories
        domain_score = min(domain_matches / len(domain_indicators), 1.0)

        return domain_score

    def _erotetic_to_score(self, erotetic_type: EroteticType) -> float:
        """
        Map erotetic type to complexity score (0-1)

        Based on cognitive demand of question type
        """
        type_scores = {
            EroteticType.FACTUAL: 0.1,
            EroteticType.CONCEPTUAL: 0.2,
            EroteticType.PROCEDURAL: 0.4,
            EroteticType.CAUSAL: 0.5,
            EroteticType.COMPARATIVE: 0.6,
            EroteticType.EVALUATIVE: 0.7,
            EroteticType.SYNTHETIC: 0.9,
            EroteticType.METACOGNITIVE: 0.8
        }
        return type_scores[erotetic_type]

    def recommend_routing(self, complexity_score: ComplexityScore, threshold: float = 0.5) -> str:
        """
        Recommend routing decision based on complexity score

        Args:
            complexity_score: ComplexityScore object
            threshold: Complexity threshold for routing (default 0.5)

        Returns:
            "local" or "remote"
        """
        if complexity_score.overall < threshold:
            return "local"
        else:
            return "remote"


def demo():
    """Demonstrate complexity scorer on example queries"""
    scorer = ComplexityScorer()

    test_queries = [
        "What is the capital of France?",
        "Explain how neural networks work",
        "Write a Python function to sort a list",
        "Why did the Roman Empire fall?",
        "Compare the advantages of renewable vs fossil fuel energy",
        "Should I invest in stocks or bonds?",
        "Analyze the economic implications of widespread AI adoption and recommend policy interventions",
    ]

    print("="*80)
    print("COMPLEXITY SCORING DEMONSTRATION")
    print("="*80)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-"*80)
        score = scorer.score(query)
        print(score)
        routing = scorer.recommend_routing(score)
        print(f"Recommended routing: {routing.upper()}")
        print()


if __name__ == "__main__":
    demo()
