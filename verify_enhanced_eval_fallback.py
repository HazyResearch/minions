import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append("/Users/eric/Documents/minions")

from experiments.enhanced_routing_eval import EnhancedRoutingEvaluator
from minions.clients.ollama import LowLogProbError, OllamaClient

class TestEnhancedEvalFallback(unittest.TestCase):
    def setUp(self):
        self.local_client = MagicMock()
        self.remote_client = MagicMock()
        self.judge_client = MagicMock()
        
        # Mock clients to avoid network
        self.evaluator = EnhancedRoutingEvaluator(
            local_client=self.local_client,
            remote_client=self.remote_client,
            judge_client=self.judge_client,
            logprob_threshold=-0.5
        )
        
        # Mock complexity scorer to avoid overhead
        self.evaluator.complexity_scorer = MagicMock()
        self.evaluator.complexity_scorer.score.return_value = MagicMock(
            overall=0.1, 
            erotetic_type=MagicMock(value="factual"),
            bloom_level=MagicMock(name="REMEMBER", value=1),
            reasoning_depth=1
        )
        # Mock SC
        self.evaluator._measure_self_consistency = MagicMock(return_value=0.0)

    @patch("experiments.enhanced_routing_eval.Minions")
    def test_fallback_logic(self, mock_minions_cls):
        print("\nTesting Enhanced Eval Fallback...")
        
        # Setup local client to raise LowLogProbError
        self.local_client.chat.side_effect = LowLogProbError("Low prob")
        
        # Setup Minions fallback to return a result
        mock_minions_instance = mock_minions_cls.return_value
        mock_minions_instance.return_value = {"final_answer": "Fallback Answer"}
        
        # Create a dummy query
        from experiments.test_data import TestQuery
        query = TestQuery(
            query="Test", 
            query_type="factual", 
            expected_route="local", 
            context=[],
            ground_truth="GT",
            difficulty="easy"
        )
        
        # Run experiment
        result = self.evaluator.run_experiment(query)
        
        # Verify local_answer contains fallback result
        self.assertEqual(result.local_answer, "Fallback Answer")
        print("✓ Local answer matches fallback result")
        
        # Verify Minions was initialized
        # It should be called TWICE:
        # 1. Inside the "Full Minions Protocol" block (standard part of experiment)
        # 2. Inside the "Local Response" block (fallback triggered)
        self.assertEqual(mock_minions_cls.call_count, 2)
        print("✓ Minions protocol was instantiated twice (Full run + Fallback)")

if __name__ == "__main__":
    unittest.main()
