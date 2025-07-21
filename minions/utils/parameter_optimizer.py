import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from minions.clients.openai import OpenAIClient
from minions.clients.ollama import OllamaClient
from minions.minions import Minions
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None


class OptimizationResult(BaseModel):
    """Results from parameter optimization"""
    parameter_name: str
    optimal_value: Union[int, float]
    all_results: List[Dict[str, Any]]
    evaluation_metric: str
    optimization_time: float


class ParameterOptimizer:
    """
    Class for optimizing Minions parameters by running actual experiments.
    Currently supports optimization of num_tasks_per_round.
    
    Note: max_chunk_size optimization is not supported because the chunking 
    function signature with default parameters is hardcoded in the supervisor's
    generated code, making it difficult to control chunk size dynamically.
    """
    
    def __init__(
        self,
        local_client=None,
        remote_client=None,
        log_dir="optimization_logs",
        evaluation_metric="efficiency_score",  # efficiency_score, accuracy, speed, cost
        max_workers=2,  # Number of parallel experiments
        verbose=True
    ):
        """
        Initialize the parameter optimizer
        
        Args:
            local_client: Local client for Minions (will create OllamaClient if None)
            remote_client: Remote client for Minions (will create OpenAIClient if None)
            log_dir: Directory to save optimization logs
            evaluation_metric: Metric to optimize for
            max_workers: Maximum number of parallel experiments
            verbose: Whether to print progress
        """
        
        self.log_dir = log_dir
        self.evaluation_metric = evaluation_metric
        self.max_workers = max_workers
        self.verbose = verbose
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients if not provided
        if local_client is None:
            self.logger.info("Initializing default OllamaClient")
            local_client = OllamaClient(
                model_name="llama3.2:3b",
                temperature=0.0,
                structured_output_schema=StructuredLocalOutput,
            )
        
        if remote_client is None:
            self.logger.info("Initializing default OpenAIClient")
            remote_client = OpenAIClient(
                model_name="gpt-4o",
                temperature=0.0
            )
            
        self.local_client = local_client
        self.remote_client = remote_client
        
    def _evaluate_minions_run(self, result: Dict[str, Any]) -> float:
        """
        Evaluate a single minions run based on the specified metric
        
        Args:
            result: Result dictionary from minions run
            
        Returns:
            Score for the run (higher is better)
        """
        timing = result.get('timing', {})
        usage = result.get('local_usage', {})
        remote_usage = result.get('remote_usage', {})
        final_answer = result.get('final_answer', '')
        
        # Calculate different metrics
        total_time = timing.get('total_time', 0)
        total_tokens = usage.get('total_tokens', 0) + remote_usage.get('total_tokens', 0)
        
        # Simple answer quality heuristic (length and completeness)
        answer_quality = min(len(final_answer) / 100, 1.0) if final_answer and final_answer != "No answer found." else 0.0
        
        if self.evaluation_metric == "speed":
            # Optimize for speed (lower time is better, so return negative)
            return -total_time if total_time > 0 else -999
            
        elif self.evaluation_metric == "cost":
            # Optimize for cost (fewer tokens is better)
            return -total_tokens if total_tokens > 0 else -999
            
        elif self.evaluation_metric == "accuracy":
            # Optimize for answer quality
            return answer_quality
            
        else:  # efficiency_score (default)
            # Combined metric: answer quality / (time * token_cost_factor)
            if total_time <= 0:
                return 0
            
            # Normalize factors
            time_factor = max(1, total_time / 60)  # Normalize to minutes
            token_factor = max(1, total_tokens / 1000)  # Normalize to thousands
            
            efficiency = answer_quality / (time_factor * token_factor)
            return efficiency
    
    def _run_minions_experiment(
        self, 
        task: str,
        context: List[str],
        doc_metadata: str,
        experiment_params: Dict[str, Any],
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Run a single minions experiment with given parameters
        
        Args:
            task: Task for minions to perform
            context: Context for the task
            doc_metadata: Document metadata
            experiment_params: Parameters for this experiment
            experiment_id: Unique identifier for this experiment
            
        Returns:
            Dictionary with experiment results
        """
        
        self.logger.info(f"Starting experiment {experiment_id} with params: {experiment_params}")
        
        start_time = time.time()
        
        try:
            # Create Minions instance
            minions = Minions(
                local_client=self.local_client,
                remote_client=self.remote_client,
                max_rounds=experiment_params.get('max_rounds', 3),
                log_dir=os.path.join(self.log_dir, f"exp_{experiment_id}")
            )
            
            # Run minions
            result = minions(
                task=task,
                doc_metadata=doc_metadata,
                context=context,
                **experiment_params
            )
            
            experiment_time = time.time() - start_time
            
            # Calculate score
            score = self._evaluate_minions_run(result)
            
            experiment_result = {
                'experiment_id': experiment_id,
                'parameters': experiment_params,
                'score': score,
                'experiment_time': experiment_time,
                'final_answer': result.get('final_answer', ''),
                'total_time': result.get('timing', {}).get('total_time', 0),
                'local_usage': result.get('local_usage', {}),
                'remote_usage': result.get('remote_usage', {}),
                'success': True,
                'error': None
            }
            
            self.logger.info(f"Experiment {experiment_id} completed. Score: {score:.4f}, Time: {experiment_time:.2f}s")
            
            return experiment_result
            
        except Exception as e:
            experiment_time = time.time() - start_time
            self.logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            
            return {
                'experiment_id': experiment_id,
                'parameters': experiment_params,
                'score': -999,  # Very low score for failed experiments
                'experiment_time': experiment_time,
                'final_answer': '',
                'total_time': 0,
                'local_usage': {},
                'remote_usage': {},
                'success': False,
                'error': str(e)
            }
    
    def optimize_num_tasks_per_round(
        self,
        task: str,
        context: List[str],
        doc_metadata: str,
        candidate_values: Optional[List[int]] = None,
        max_rounds: int = 3,
        **other_params
    ) -> OptimizationResult:
        """
        Find the optimal num_tasks_per_round value by running experiments
        
        Args:
            task: Task for minions to perform
            context: Context for the task
            doc_metadata: Document metadata
            candidate_values: List of num_tasks_per_round values to test
            max_rounds: Max rounds for each experiment
            **other_params: Other parameters to pass to minions
            
        Returns:
            OptimizationResult with optimal value and all experiment data
        """
        
        if candidate_values is None:
            candidate_values = [1, 2, 3, 4, 5, 6, 8, 10]
        
        self.logger.info(f"Optimizing num_tasks_per_round with values: {candidate_values}")
        
        start_time = time.time()
        
        # Prepare experiments
        experiments = []
        for i, num_tasks in enumerate(candidate_values):
            experiment_params = {
                'num_tasks_per_round': num_tasks,
                'max_rounds': max_rounds,
                **other_params
            }
            experiments.append({
                'task': task,
                'context': context,
                'doc_metadata': doc_metadata,
                'experiment_params': experiment_params,
                'experiment_id': f"num_tasks_{num_tasks}_{i}"
            })
        
        # Run experiments in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_exp = {
                executor.submit(
                    self._run_minions_experiment,
                    exp['task'],
                    exp['context'],
                    exp['doc_metadata'],
                    exp['experiment_params'],
                    exp['experiment_id']
                ): exp for exp in experiments
            }
            
            for future in as_completed(future_to_exp):
                result = future.result()
                results.append(result)
        
        # Find optimal value
        successful_results = [r for r in results if r['success']]
        if not successful_results:
            self.logger.error("All experiments failed!")
            raise RuntimeError("All optimization experiments failed")
        
        optimal_result = max(successful_results, key=lambda x: x['score'])
        optimal_value = optimal_result['parameters']['num_tasks_per_round']
        
        optimization_time = time.time() - start_time
        
        # Save results
        results_file = os.path.join(self.log_dir, "num_tasks_per_round_optimization.json")
        with open(results_file, 'w') as f:
            json.dump({
                'optimal_value': optimal_value,
                'all_results': results,
                'evaluation_metric': self.evaluation_metric,
                'optimization_time': optimization_time
            }, f, indent=2)
        
        self.logger.info(f"Optimization complete! Optimal num_tasks_per_round: {optimal_value}")
        self.logger.info(f"Best score: {optimal_result['score']:.4f}")
        
        return OptimizationResult(
            parameter_name="num_tasks_per_round",
            optimal_value=optimal_value,
            all_results=results,
            evaluation_metric=self.evaluation_metric,
            optimization_time=optimization_time
        )


def find_optimal_num_tasks_per_round(
    task: str,
    context: List[str],
    doc_metadata: str = "Document",
    candidate_values: Optional[List[int]] = None,
    evaluation_metric: str = "efficiency_score",
    max_workers: int = 2,
    **minions_params
) -> int:
    """
    Convenience function to find optimal num_tasks_per_round
    
    Args:
        task: Task for minions to perform
        context: Context for the task
        doc_metadata: Document metadata  
        candidate_values: List of values to test (default: [1,2,3,4,5,6,8,10])
        evaluation_metric: Metric to optimize for
        max_workers: Number of parallel experiments
        **minions_params: Additional parameters for minions
        
    Returns:
        Optimal num_tasks_per_round value
    """
    
    optimizer = ParameterOptimizer(
        evaluation_metric=evaluation_metric,
        max_workers=max_workers
    )
    
    result = optimizer.optimize_num_tasks_per_round(
        task=task,
        context=context,
        doc_metadata=doc_metadata,
        candidate_values=candidate_values,
        **minions_params
    )
    
    return result.optimal_value


# Example usage and testing
if __name__ == "__main__":
    # Example context and task for testing
    sample_context = ["""
    Artificial Intelligence (AI) has transformed numerous industries over the past decade. 
    Machine learning algorithms have become increasingly sophisticated, enabling computers 
    to perform tasks that were once thought to be uniquely human. Deep learning, a subset 
    of machine learning, has been particularly revolutionary in fields such as computer vision, 
    natural language processing, and speech recognition.
    
    The development of transformer architectures has been a major breakthrough in AI research. 
    Models like GPT (Generative Pre-trained Transformer) have demonstrated remarkable capabilities 
    in generating human-like text, answering questions, and even writing code. These models are 
    trained on vast amounts of text data and can understand context and generate coherent responses.
    
    However, AI development also comes with challenges. Issues such as bias in training data, 
    explainability of AI decisions, and the computational resources required for training large 
    models are ongoing concerns. Researchers are actively working on addressing these challenges 
    to make AI more reliable, fair, and accessible.
    
    The future of AI looks promising, with potential applications in healthcare, education, 
    scientific research, and many other domains. As AI technology continues to advance, 
    it will be crucial to ensure that its development and deployment are guided by ethical 
    considerations and benefit society as a whole.
    """]
    
    sample_task = "What are the main challenges in AI development mentioned in this document?"
    
    print("Testing num_tasks_per_round optimization...")
    
    # Test num_tasks_per_round optimization (quick test with fewer values)
    print("\n🔍 Testing num_tasks_per_round optimization...")
    optimal_tasks = find_optimal_num_tasks_per_round(
        task=sample_task,
        context=sample_context,
        candidate_values=[1, 2, 3],  # Quick test
        max_workers=1  # Sequential for testing
    )
    print(f"✅ Optimal num_tasks_per_round: {optimal_tasks}")
    
    print("\n🎉 Parameter optimization testing complete!")
    print(f"📊 Result: Optimal num_tasks_per_round = {optimal_tasks}")
    print("\n💡 Usage:")
    print("   from minions.utils.parameter_optimizer import find_optimal_num_tasks_per_round")
    print("   optimal_tasks = find_optimal_num_tasks_per_round(task, context)") 