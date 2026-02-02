"""
FinanceBench Evaluation Script
Replicates the FinanceBench benchmark experiments from the MinionS paper.

This script evaluates multiple protocols (MINION, MINIONS, local-only, remote-only)
on the FinanceBench dataset and computes accuracy and cost metrics.

NOTE: This evaluator ALWAYS loads full PDF documents from the pdf_dir.
Pre-extracted snippets or evidence from the benchmark JSON are ignored.
"""

import os
import sys
import json
import time
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

import fitz  # PyMuPDF for PDF text extraction
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.clients.sglang import SGLangClient
from minions.minion import Minion
from minions.minions import Minions
from minions.usage import Usage

# Configuration loading
from evaluate.kconfig_loader import KconfigLoader, EvaluatorConfig, load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# GPT-4o pricing (January 2025 rates from paper)
GPT4O_INPUT_PRICE_PER_MILLION = 2.50
GPT4O_OUTPUT_PRICE_PER_MILLION = 10.00


def load_logit_processor(path: str) -> Optional[Any]:
    """
    Dynamically load a logit processor from a Python file.
    
    Args:
        path: Path to the logit processor Python file
        
    Returns:
        The processor class (e.g., LearnedBloatAxisProcessor), or None if loading fails
    """
    if not path:
        return None
    
    path = Path(path)
    if not path.exists():
        logger.warning(f"Logit processor file not found: {path}")
        return None
    
    try:
        spec = importlib.util.spec_from_file_location("learned_logit_processor", path)
        if spec is None or spec.loader is None:
            logger.warning(f"Failed to load spec for logit processor: {path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for the processor class
        if hasattr(module, 'LearnedBloatAxisProcessor'):
            logger.info(f"Loaded logit processor from {path}")
            return module.LearnedBloatAxisProcessor
        else:
            logger.warning(f"No LearnedBloatAxisProcessor class found in {path}")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to load logit processor from {path}: {e}")
        return None


def load_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Concatenated text from all pages, separated by double newlines
        
    Raises:
        FileNotFoundError: If PDF file does not exist
        Exception: If PDF parsing fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


def generate_cache_dir_name(
    protocol: str,
    remote_model: str,
    local_model: str,
    remote_temp: float,
    max_rounds: int,
    num_samples_per_task: int = 1,
    max_samples: Optional[int] = None
) -> str:
    """
    Generate a human-readable cache directory name from configuration.
    
    Format: {protocol}_{remote_model}_{local_model}_t{temp}_r{rounds}_s{samples_per_task}[_n{max_samples}]
    Example: minions_gpt-4o_llama3.2-3b_t0.0_r2_s1
    """
    # Clean model names for use in directory names
    def clean_model_name(name: str) -> str:
        return name.replace(':', '-').replace('/', '-').replace('\\', '-')
    
    clean_remote = clean_model_name(remote_model)
    clean_local = clean_model_name(local_model)
    
    # Format temperature (remove trailing zeros)
    temp_str = f"{remote_temp:.1f}".rstrip('0').rstrip('.')
    if temp_str == '':
        temp_str = '0'
    
    # Build directory name
    parts = [
        protocol,
        clean_remote,
        clean_local,
        f"t{temp_str}",
        f"r{max_rounds}",
        f"s{num_samples_per_task}"
    ]
    
    if max_samples is not None:
        parts.append(f"n{max_samples}")
    
    return "_".join(parts)


def generate_cache_dir_name_from_config(config: EvaluatorConfig) -> str:
    """
    Generate cache directory name from specific configuration options.
    
    The hash is computed from:
    - Dataset path
    - Local model (name, temperature, context)
    - Remote model (name, temperature)
    - Protocol settings (active, max_rounds, samples_per_task)
    - MINIONS settings (tasks_per_round, chunk_fn, max_chunk_size, pages_per_chunk)
    - Logit processor path (if configured)
    
    Returns:
        Directory name in format: <prefix>_<protocols>_<chunk_fn>[_pages:<N>]_<rounds>[_<logit_processor>]_<hash>
        Example: myprefix_minions_on-multiple-pages_pages:5_r5_learned_logit_processor_a1b2c3d4
    """
    import hashlib
    
    # Build hash from specific options
    hash_data = {
        'path': config.dataset.path,
        'local_model': config.models.local.name,
        'local_temp': config.models.local.temperature,
        'local_ctx': config.models.local.num_ctx,
        'remote_model': config.models.remote.name,
        'remote_temp': config.models.remote.temperature,
        'protocols': sorted(config.protocols.active),
        'max_rounds': config.protocols.common.max_rounds,
        'samples_per_task': config.protocols.common.num_samples_per_task,
        'tasks_per_round': config.protocols.minions.num_tasks_per_round,
        'chunk_fn': config.protocols.minions.chunk_fn,
        'max_chunk_size': config.protocols.minions.max_chunk_size,
    }
    
    # Include pages_per_chunk only if using multi-page chunking
    if config.protocols.minions.chunk_fn == 'chunk_on_multiple_pages':
        hash_data['pages_per_chunk'] = config.protocols.minions.pages_per_chunk
    
    # Include logit processor path in hash if configured
    logit_processor_path = getattr(config.models.local, 'logit_processor_path', None)
    if logit_processor_path:
        hash_data['logit_processor_path'] = logit_processor_path
    
    # Generate short hash from the specific options
    content_hash = hashlib.sha256(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:8]
    
    # Build directory name parts
    parts = []
    
    # Add prefix only if set
    prefix = config.global_config.cache_prefix
    if prefix:
        parts.append(prefix)
    
    # Protocol(s)
    protocols_str = "+".join(sorted(config.protocols.active))
    parts.append(protocols_str)
    
    # Shorten chunk function name for readability
    chunk_fn_short = config.protocols.minions.chunk_fn.replace('chunk_', '').replace('_', '-')
    parts.append(chunk_fn_short)
    
    # Add pages_per_chunk only if using multi-page chunking
    if config.protocols.minions.chunk_fn == 'chunk_on_multiple_pages':
        parts.append(f"pages:{config.protocols.minions.pages_per_chunk}")
    
    # Add number of rounds
    parts.append(f"r{config.protocols.common.max_rounds}")
    
    # Add logit processor filename (without extension) if configured
    if logit_processor_path:
        lp_name = Path(logit_processor_path).stem
        parts.append(lp_name)
    
    # Add hash
    parts.append(content_hash)
    
    return "_".join(parts)


@dataclass
class FinanceBenchSample:
    """Represents a single FinanceBench sample."""
    question: str
    answer: List[str]  # List of possible correct answers
    document_text: str  # Full PDF document text (loaded from pdf_dir, NOT from benchmark)
    doc_name: str  # SEC filing document name (e.g., "3M_2018_10K")
    pdf_path: Optional[str] = None  # Relative path to PDF file for cache matching
    sample_id: Optional[str] = None
    question_reasoning: Optional[str] = None  # Reasoning type from dataset


@dataclass
class EvaluationResult:
    """Results for a single evaluation run."""
    sample_id: str
    protocol: str
    question: str
    predicted_answer: str
    ground_truth: List[str]
    cost_usd: float
    input_tokens: int
    output_tokens: int
    execution_time: float
    is_correct: Optional[bool] = None
    error: Optional[str] = None


class FinanceBenchDataset:
    """
    Loads and filters FinanceBench dataset.
    
    IMPORTANT: This class ALWAYS loads full PDF documents from pdf_dir.
    Pre-extracted 'evidence' or 'context' fields in the benchmark JSON are ignored.
    """
    
    def __init__(self, dataset_path: str, pdf_dir: str, filter_numerical: bool = False, max_samples: Optional[int] = None, sample_indices: Optional[List[int]] = None):
        """
        Initialize FinanceBench dataset loader.
        
        Args:
            dataset_path: Path to FinanceBench repository
            pdf_dir: Path to directory containing SEC filing PDFs
            filter_numerical: Whether to filter to numerical reasoning questions (default: False)
            max_samples: Maximum number of samples to load (None for all)
            sample_indices: List of 1-based line indices to include (None for all)
        """
        self.dataset_path = Path(dataset_path)
        self.pdf_dir = Path(pdf_dir)
        self.filter_numerical = filter_numerical
        self.max_samples = max_samples
        self.sample_indices = sample_indices
        self.samples: List[FinanceBenchSample] = []
        
        # Validate PDF directory exists
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
    
    def _check_file_has_questions(self, file_path: Path) -> Tuple[bool, bool]:
        """Check if a file contains questions and doc_name fields."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for i, line in enumerate(f):
                        if i >= 10:
                            break
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            has_question = 'question' in data or 'query' in data or 'q' in data
                            has_doc_name = 'doc_name' in data
                            if has_question or has_doc_name:
                                return (has_question, has_doc_name)
                        except (json.JSONDecodeError, KeyError):
                            continue
                else:
                    try:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            data = data[0]
                        elif isinstance(data, dict):
                            for key in ['samples', 'data', 'test', 'train', 'dev']:
                                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                                    data = data[key][0]
                                    break
                        
                        has_question = 'question' in data or 'query' in data or 'q' in data
                        has_doc_name = 'doc_name' in data
                        return (has_question, has_doc_name)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
        except Exception as e:
            logger.debug(f"Error checking file {file_path}: {e}")
        
        return (False, False)
    
    def load(self) -> List[FinanceBenchSample]:
        """Load FinanceBench dataset."""
        logger.info(f"Loading FinanceBench dataset from {self.dataset_path}")
        logger.info(f"PDF documents will be loaded from: {self.pdf_dir}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Find all JSON/JSONL files
        all_files = []
        for ext in ['*.json', '*.jsonl']:
            all_files.extend(list(self.dataset_path.glob(ext)))
            data_dir = self.dataset_path / 'data'
            if data_dir.exists() and data_dir.is_dir():
                all_files.extend(list(data_dir.glob(ext)))
        
        if not all_files:
            raise FileNotFoundError(
                f"No data files found in {self.dataset_path}. "
                "Expected JSON/JSONL files with question and doc_name fields."
            )
        
        logger.info(f"Found {len(all_files)} potential data file(s)")
        
        # Categorize files
        files_with_both = []
        files_with_question_only = []
        
        for file_path in all_files:
            has_question, has_doc_name = self._check_file_has_questions(file_path)
            if has_question and has_doc_name:
                files_with_both.append(file_path)
            elif has_question:
                files_with_question_only.append(file_path)
        
        # Prioritize files
        prioritized_files = []
        for file_path in files_with_both:
            if 'data' in file_path.parts:
                prioritized_files.append(file_path)
        for file_path in files_with_both:
            if 'data' not in file_path.parts:
                prioritized_files.append(file_path)
        prioritized_files.extend(files_with_question_only)
        
        if not prioritized_files:
            raise FileNotFoundError(
                f"No valid dataset files found in {self.dataset_path}. "
                "Expected files with 'question' and 'doc_name' fields."
            )
        
        logger.info(f"Loading from {len(prioritized_files)} prioritized file(s)")
        
        # Load samples
        all_samples = []
        for data_file in prioritized_files:
            logger.info(f"Loading from {data_file}")
            samples = self._load_from_file(data_file)
            all_samples.extend(samples)
            logger.info(f"  Loaded {len(samples)} samples from {data_file.name}")
            
            if self.max_samples and len(all_samples) >= self.max_samples:
                break
        
        # Filter by specific indices if provided
        if self.sample_indices:
            logger.info(f"Filtering to {len(self.sample_indices)} specific sample indices...")
            indexed_samples = []
            for idx in self.sample_indices:
                if 1 <= idx <= len(all_samples):
                    indexed_samples.append(all_samples[idx - 1])
                else:
                    logger.warning(f"Sample index {idx} out of range (1-{len(all_samples)})")
            all_samples = indexed_samples
        
        # Filter to numerical reasoning if requested
        if self.filter_numerical:
            logger.info("Filtering to numerical reasoning questions...")
            numerical_samples = [s for s in all_samples if self._is_numerical_question(s)]
            all_samples = numerical_samples
            logger.info(f"Filtered to {len(all_samples)} numerical reasoning samples")
        
        # Limit to max_samples
        if self.max_samples:
            all_samples = all_samples[:self.max_samples]
        
        self.samples = all_samples
        logger.info(f"Loaded {len(self.samples)} samples (full PDF documents)")
        
        return self.samples
    
    def _load_from_file(self, file_path: Path) -> List[FinanceBenchSample]:
        """Load samples from a JSON/JSONL file."""
        samples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                # Use unified line_<N> format for sample IDs
                                sample = self._parse_sample(data, f"line_{line_num}")
                                if sample:
                                    samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        for idx, item in enumerate(data):
                            # Use unified line_<N> format (1-indexed for consistency)
                            sample = self._parse_sample(item, f"line_{idx + 1}")
                            if sample:
                                samples.append(sample)
                    elif isinstance(data, dict):
                        for key in ['samples', 'data', 'test', 'train', 'dev']:
                            if key in data and isinstance(data[key], list):
                                for idx, item in enumerate(data[key]):
                                    # Use unified line_<N> format (1-indexed for consistency)
                                    sample = self._parse_sample(item, f"line_{idx + 1}")
                                    if sample:
                                        samples.append(sample)
                                break
                        else:
                            sample = self._parse_sample(data, f"line_1")
                            if sample:
                                samples.append(sample)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
        
        return samples
    
    def _parse_sample(self, data: Dict[str, Any], sample_id: str) -> Optional[FinanceBenchSample]:
        """
        Parse a single sample from JSON data and load full PDF document.
        
        NOTE: Pre-extracted 'evidence' or 'context' fields in the JSON are IGNORED.
        We always load the full PDF document from pdf_dir.
        """
        try:
            # Get question
            question = data.get('question') or data.get('query') or data.get('q')
            if not question:
                logger.warning(f"Sample {sample_id} missing question field")
                return None
            
            # Get answer
            answer_data = data.get('answer') or data.get('answers') or data.get('gold_answer') or data.get('ground_truth')
            if isinstance(answer_data, str):
                answers = [answer_data]
            elif isinstance(answer_data, list):
                answers = answer_data
            elif isinstance(answer_data, dict):
                answers = [str(answer_data.get('text', answer_data))]
            else:
                answers = [str(answer_data)] if answer_data else []
            
            if not answers:
                logger.warning(f"Sample {sample_id} missing answer field")
                return None
            
            # Get doc_name for PDF loading (REQUIRED)
            doc_name = data.get('doc_name')
            if not doc_name:
                logger.warning(f"Sample {sample_id} missing doc_name field - cannot load PDF")
                return None
            
            # Load FULL PDF document (ignore any pre-extracted evidence/context)
            pdf_path = self.pdf_dir / f"{doc_name}.pdf"
            if not pdf_path.exists():
                logger.warning(f"Sample {sample_id}: PDF not found at {pdf_path}")
                return None
            
            try:
                document_text = load_pdf_text(str(pdf_path))
                logger.info(f"Sample {sample_id}: Loaded full PDF {doc_name} ({len(document_text)} chars)")
            except Exception as e:
                logger.warning(f"Sample {sample_id}: Failed to load PDF {pdf_path}: {e}")
                return None
            
            question_reasoning = data.get('question_reasoning')
            pdf_path_str = os.path.basename(pdf_path)
            
            return FinanceBenchSample(
                question=question,
                answer=answers,
                document_text=document_text,
                doc_name=doc_name,
                pdf_path=pdf_path_str,
                sample_id=sample_id,
                question_reasoning=question_reasoning
            )
        except Exception as e:
            logger.warning(f"Error parsing sample {sample_id}: {e}")
            return None
    
    def _is_numerical_question(self, sample: FinanceBenchSample) -> bool:
        """Check if a question is a numerical reasoning question based on FinanceBench labels."""
        # Only use the FinanceBench question_reasoning label - no fallbacks
        return bool(sample.question_reasoning and 'numerical' in sample.question_reasoning.lower())


def _normalize_usage(usage: Any) -> Usage:
    """Convert dict to Usage object if needed."""
    if isinstance(usage, dict):
        return Usage(
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0)
        )
    elif isinstance(usage, Usage):
        return usage
    else:
        return Usage()


class MetricsCalculator:
    """Calculates accuracy and cost metrics."""
    
    @staticmethod
    def calculate_accuracy(predictions: List[str], ground_truths: List[List[str]]) -> float:
        """Calculate accuracy by comparing predictions to ground truth."""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground_truths must have same length")
        
        correct = 0
        for pred, gt_list in zip(predictions, ground_truths):
            if MetricsCalculator._is_answer_correct(pred, gt_list):
                correct += 1
        
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def _is_answer_correct(predicted: str, ground_truth_list: List[str], remote_client=None, question: Optional[str] = None, context: Optional[str] = None) -> bool:
        """Check if predicted answer matches any ground truth answer."""
        if not predicted or predicted.strip().lower() in ['none', 'null', '']:
            return False
        
        predicted = predicted.strip()
        
        # Use LLM-as-a-judge if remote_client is provided
        if remote_client is not None:
            truncated_context = context
            if context:
                try:
                    import tiktoken
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                    tokens = encoding.encode(context)
                    MAX_CONTEXT_TOKENS = 122000
                    if len(tokens) > MAX_CONTEXT_TOKENS:
                        truncated_tokens = tokens[:MAX_CONTEXT_TOKENS]
                        truncated_context = encoding.decode(truncated_tokens)
                        logging.getLogger(__name__).info(f"LLM-as-a-judge: truncated context from {len(tokens)} to {MAX_CONTEXT_TOKENS} tokens")
                except ImportError:
                    max_chars = 122000 * 4
                    if len(context) > max_chars:
                        truncated_context = context[:max_chars]
            
            for gt in ground_truth_list:
                gt = gt.strip()
                try:
                    prompt = f"""You are evaluating whether two answers are equivalent. Follow these steps STRICTLY in order:

STEP 1: Check for explicit contradictions
- Does one say "Yes" and the other "No" (or vice versa)? → UNEQUAL
- Does one say "increased" and the other "decreased"? → UNEQUAL
- Does one say "positive" and the other "negative"? → UNEQUAL
- If ANY contradiction found → Answer: unequal

STEP 2: Check for different entities
- Do they name different segments/companies/people? → UNEQUAL
- Do they reference different time periods? → UNEQUAL
- If different entities → Answer: unequal

STEP 3: Check for opposite conclusions
- Do they reach opposite conclusions about the same thing? → UNEQUAL
- If opposite conclusions → Answer: unequal

STEP 4: Check semantic equivalence (ONLY if no contradictions found)
- Are they saying the same thing in different words? → EQUAL
- For numerical answers, are the values the same or very close (within 1%)? → EQUAL

QUESTION:
{question if question else "N/A"}

CONTEXT:
{truncated_context if truncated_context else "N/A"}

PREDICTED ANSWER:
{predicted}

GROUND TRUTH ANSWER:
{gt}

CRITICAL: If you found ANY contradiction in Steps 1-3, respond with "unequal". 
Only respond with "equal" if the answers are truly equivalent.

Respond with only one word: "equal" or "unequal"."""
                    
                    messages = [{"role": "user", "content": prompt}]
                    chat_result = remote_client.chat(messages)
                    log = logging.getLogger(__name__)
                    
                    response = None
                    if chat_result is None or not hasattr(chat_result, '__len__'):
                        continue
                    elif len(chat_result) == 2:
                        response, _ = chat_result
                    elif len(chat_result) >= 3:
                        response = chat_result[0]
                    else:
                        continue
                    
                    if response is None or not isinstance(response, list) or len(response) == 0:
                        continue
                    
                    response_text = str(response[0]).strip().lower()
                    
                    if "equal" in response_text and "unequal" not in response_text:
                        return True
                    elif "unequal" in response_text:
                        continue
                except Exception as e:
                    log = logging.getLogger(__name__)
                    log.warning(f"LLM-as-a-judge failed: {e}")
        
        # Rule-based matching fallback
        for gt in ground_truth_list:
            gt = gt.strip()
            
            if predicted.lower() == gt.lower():
                return True
            
            pred_numbers = MetricsCalculator._extract_numbers(predicted)
            gt_numbers = MetricsCalculator._extract_numbers(gt)
            
            if pred_numbers and gt_numbers:
                if len(pred_numbers) == len(gt_numbers):
                    if all(abs(p - g) < 0.01 for p, g in zip(pred_numbers, gt_numbers)):
                        return True
            
            if gt.lower() in predicted.lower() or predicted.lower() in gt.lower():
                return True
        
        return False
    
    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extract numerical values from text."""
        pattern = r'\d+[.,]?\d*'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            cleaned = match.replace(',', '')
            try:
                numbers.append(float(cleaned))
            except ValueError:
                continue
        
        return numbers
    
    @staticmethod
    def calculate_cost(usage: Any) -> float:
        """Calculate cost in USD based on token usage."""
        normalized_usage = _normalize_usage(usage)
        
        input_tokens = normalized_usage.prompt_tokens
        output_tokens = normalized_usage.completion_tokens
        
        input_cost = (input_tokens / 1_000_000) * GPT4O_INPUT_PRICE_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * GPT4O_OUTPUT_PRICE_PER_MILLION
        
        return input_cost + output_cost
    
    @staticmethod
    def aggregate_results(results: List[EvaluationResult], skip_accuracy: bool = False) -> Dict[str, Any]:
        """Aggregate evaluation results into summary statistics."""
        if not results:
            return {
                'accuracy': None if skip_accuracy else 0.0,
                'total_cost': 0.0,
                'avg_cost_per_query': 0.0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'avg_input_tokens': 0.0,
                'avg_output_tokens': 0.0,
                'total_samples': 0,
                'successful_samples': 0,
                'failed_samples': 0
            }
        
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        if skip_accuracy:
            accuracy = None
        else:
            accuracy = sum(1 for r in successful_results if r.is_correct) / len(successful_results) if successful_results else 0.0
        
        total_cost = sum(r.cost_usd for r in successful_results)
        total_input_tokens = sum(r.input_tokens for r in successful_results)
        total_output_tokens = sum(r.output_tokens for r in successful_results)
        
        return {
            'accuracy': accuracy,
            'total_cost': total_cost,
            'avg_cost_per_query': total_cost / len(successful_results) if successful_results else 0.0,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'avg_input_tokens': total_input_tokens / len(successful_results) if successful_results else 0.0,
            'avg_output_tokens': total_output_tokens / len(successful_results) if successful_results else 0.0,
            'total_samples': len(results),
            'successful_samples': len(successful_results),
            'failed_samples': len(failed_results),
            'avg_execution_time': sum(r.execution_time for r in successful_results) / len(successful_results) if successful_results else 0.0
        }


class ProtocolRunner:
    """Runs different protocols for evaluation."""
    
    def __init__(self, local_model: str, remote_model: str, local_temp: float = 0.2, remote_temp: float = 0.0, num_ctx: int = 4096, local_backend: str = "ollama", sglang_base_url: str = "http://localhost:8000/v1", logit_processor_path: Optional[str] = None):
        """Initialize protocol runner with model configurations."""
        self.local_model = local_model
        self.remote_model = remote_model
        self.local_temp = local_temp
        self.remote_temp = remote_temp
        self.num_ctx = num_ctx
        self.local_backend = local_backend
        self.sglang_base_url = sglang_base_url
        
        # Logit processor settings
        self.logit_processor_path = logit_processor_path
        self.logit_processor_class = load_logit_processor(logit_processor_path) if logit_processor_path else None
        
        self._local_client = None
        self._remote_client = None
        self._minion = None
        self._minions = None
    
    def _get_local_client(self):
        """Get or create local client."""
        if self._local_client is None:
            if self.local_backend == "sglang":
                self._local_client = SGLangClient(
                    model_name=self.local_model,
                    base_url=self.sglang_base_url,
                    temperature=self.local_temp,
                )
            else:
                self._local_client = OllamaClient(
                    model_name=self.local_model,
                    temperature=self.local_temp,
                    num_ctx=self.num_ctx
                )
        return self._local_client
    
    def _get_remote_client(self) -> OpenAIClient:
        """Get or create remote client."""
        if self._remote_client is None:
            self._remote_client = OpenAIClient(
                model_name=self.remote_model,
                temperature=self.remote_temp
            )
        return self._remote_client
    
    def _truncate_to_tokens(self, text: str, max_tokens: int, model: str = "gpt-4o") -> str:
        """Truncate text to fit within max_tokens using tiktoken."""
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            truncated_text = encoding.decode(truncated_tokens)
            logger.info(f"Truncated context from {len(tokens)} to {max_tokens} tokens")
            return truncated_text
        except ImportError:
            logger.warning("tiktoken not installed, using character-based truncation")
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]
    
    def run_local_only(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """Run local-only baseline."""
        start_time = time.time()
        
        local_client = self._get_local_client()
        
        messages = [{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nPlease answer the question based on the context."
        }]
        
        try:
            chat_result = local_client.chat(messages)
            if len(chat_result) == 2:
                response, usage = chat_result
            elif len(chat_result) == 3:
                response, usage, _ = chat_result
            else:
                response, usage = chat_result[0], Usage()
            
            return {
                'final_answer': response[0] if response else "No answer",
                'remote_usage': Usage(),
                'local_usage': usage,
                'timing': {'total_time': time.time() - start_time},
                'error': None
            }
        except Exception as e:
            logger.error(f"Error in local-only run: {e}")
            return {
                'final_answer': None,
                'remote_usage': Usage(),
                'local_usage': Usage(),
                'timing': {'total_time': time.time() - start_time},
                'error': str(e)
            }
    
    def run_remote_only(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """Run remote-only baseline."""
        start_time = time.time()
        
        remote_client = self._get_remote_client()
        
        if isinstance(context, list):
            context_str = "\n\n".join(context)
        else:
            context_str = context
        
        MAX_CONTEXT_TOKENS = 122000
        context_str = self._truncate_to_tokens(context_str, MAX_CONTEXT_TOKENS)
        
        messages = [{
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {question}\n\nPlease answer the question based on the context."
        }]
        
        try:
            chat_result = remote_client.chat(messages)
            if len(chat_result) == 2:
                response, usage = chat_result
            elif len(chat_result) >= 3:
                response, usage = chat_result[0], chat_result[1]
            else:
                response, usage = chat_result[0] if chat_result else [], Usage()
            
            return {
                'final_answer': response[0] if response else "No answer",
                'remote_usage': usage,
                'local_usage': Usage(),
                'timing': {'total_time': time.time() - start_time},
                'error': None
            }
        except Exception as e:
            logger.error(f"Error in remote-only run: {e}")
            return {
                'final_answer': None,
                'remote_usage': Usage(),
                'local_usage': Usage(),
                'timing': {'total_time': time.time() - start_time},
                'error': str(e)
            }
    
    def run_minion(self, question: str, context: str, max_rounds: int = 3, **kwargs) -> Dict[str, Any]:
        """Run MINION protocol."""
        if self._minion is None:
            self._minion = Minion(
                local_client=self._get_local_client(),
                remote_client=self._get_remote_client(),
                max_rounds=max_rounds
            )
        
        try:
            result = self._minion(
                task=question,
                context=[context],
                max_rounds=max_rounds
            )
            
            return {
                'final_answer': result.get('final_answer', 'No answer'),
                'remote_usage': result.get('remote_usage', Usage()),
                'local_usage': result.get('local_usage', Usage()),
                'timing': result.get('timing', {}),
                'error': None
            }
        except Exception as e:
            logger.error(f"Error in MINION run: {e}")
            return {
                'final_answer': None,
                'remote_usage': Usage(),
                'local_usage': Usage(),
                'timing': {'total_time': 0.0},
                'error': str(e)
            }
    
    def run_minions(self, question: str, context: str, max_rounds: int = 2, **kwargs) -> Dict[str, Any]:
        """Run MINIONS protocol.
        
        Args:
            question: The question to answer
            context: Document context
            max_rounds: Maximum rounds of interaction
            **kwargs: Additional arguments including:
                - prompt_overrides: Dict of prompt name -> value for evolution
                - logging_id: Unified identifier for trace file naming
                - run_output_dir: Directory for output files
                - max_chunk_size, pages_per_chunk, etc.
        """
        from pydantic import BaseModel
        
        run_output_dir = kwargs.get('run_output_dir', '.')
        minions_log_dir = os.path.join(run_output_dir, 'minions_logs')
        
        # Get prompt overrides for evolution experiments
        prompt_overrides = kwargs.get('prompt_overrides', None)
        
        # Get logging_id for unified trace file naming
        logging_id = kwargs.get('logging_id', None)
        
        class StructuredLocalOutput(BaseModel):
            explanation: str
            citation: str | None
            answer: str | None
        
        if self.local_backend == "sglang":
            local_client = SGLangClient(
                model_name=self.local_model,
                base_url=self.sglang_base_url,
                temperature=self.local_temp,
                structured_output_schema=StructuredLocalOutput,
                logit_processor_class=self.logit_processor_class,
            )
        else:
            local_client = OllamaClient(
                model_name=self.local_model,
                temperature=self.local_temp,
                structured_output_schema=StructuredLocalOutput,
                num_ctx=self.num_ctx
            )
        
        minions_instance = Minions(
            local_client=local_client,
            remote_client=self._get_remote_client(),
            max_rounds=max_rounds,
            log_dir=minions_log_dir,
            max_chunk_size=kwargs.get('max_chunk_size', 3000),
            pages_per_chunk=kwargs.get('pages_per_chunk', 5),
            prompt_overrides=prompt_overrides,
        )
        
        try:
            doc_metadata = f"Financial document. Total length: {len(context)} characters"
            
            result = minions_instance(
                task=question,
                doc_metadata=doc_metadata,
                context=[context],
                max_rounds=max_rounds,
                num_tasks_per_round=kwargs.get('num_tasks_per_round', 3),
                num_samples_per_task=kwargs.get('num_samples_per_task', 1),
                chunk_fn=kwargs.get('chunk_fn', 'chunk_by_section'),
                use_retrieval=kwargs.get('use_retrieval', None),
                max_jobs_per_round=kwargs.get('max_jobs_per_round', None),
                retrieval_model=kwargs.get('retrieval_model', None),
                mcp_tools_info=kwargs.get('mcp_tools_info', None),
                logging_id=logging_id,
            )
            
            return {
                'final_answer': result.get('final_answer', 'No answer'),
                'remote_usage': result.get('remote_usage', Usage()),
                'local_usage': result.get('local_usage', Usage()),
                'timing': result.get('timing', {}),
                'error': None
            }
        except Exception as e:
            logger.error(f"Error in MINIONS run: {e}")
            return {
                'final_answer': None,
                'remote_usage': Usage(),
                'local_usage': Usage(),
                'timing': {'total_time': 0.0},
                'error': str(e)
            }


class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(
        self,
        dataset: FinanceBenchDataset,
        protocol_runner: ProtocolRunner,
        protocols: List[str],
        output_dir: str = "evaluate/results",
        num_samples: Optional[int] = None,
        minions_kwargs: Optional[Dict[str, Any]] = None,
        skip_accuracy: bool = False,
        command_line: Optional[str] = None,
        use_cache: bool = True,
        cache_dir_name: Optional[str] = None,
        all_args: Optional[Dict[str, Any]] = None
    ):
        """Initialize evaluator."""
        self.dataset = dataset
        self.protocol_runner = protocol_runner
        self.protocols = protocols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.minions_kwargs = minions_kwargs or {}
        self.skip_accuracy = skip_accuracy
        self.command_line = command_line
        self.use_cache = use_cache
        self.all_args = all_args or {}
        
        if cache_dir_name and use_cache:
            self.run_output_dir = self.output_dir / cache_dir_name
            self.run_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using cache directory: {self.run_output_dir}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            if num_samples is None:
                num_samples = len(dataset.samples) if dataset.samples else 0
            protocols_str = "_".join(self.protocols)
            dir_name = f"{timestamp}_{protocols_str}_samples#{num_samples}"
            self.run_output_dir = self.output_dir / dir_name
            self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_logs_dir = self.run_output_dir / "sample_logs"
        self.sample_logs_dir.mkdir(parents=True, exist_ok=True)
        
        self._save_config_files()
        self.results: Dict[str, List[EvaluationResult]] = {}
    
    def _save_config_files(self):
        """Save config.json and command.txt on first run."""
        config_path = self.run_output_dir / "config.json"
        command_path = self.run_output_dir / "command.txt"
        
        if not config_path.exists():
            config_data = {
                "protocols": self.protocols,
                "skip_accuracy": self.skip_accuracy,
                "use_cache": self.use_cache,
                "minions_kwargs": self.minions_kwargs,
                "all_args": self.all_args,
                "created_at": datetime.now().isoformat()
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        if not command_path.exists() and self.command_line:
            with open(command_path, 'w', encoding='utf-8') as f:
                f.write(self.command_line + "\n")
    
    def _get_sample_cache_path(self, sample_id: str, protocol: str) -> Path:
        """Get the cache file path for a sample.
        
        Uses unified naming: minions_line_<N>.json for easy correlation with trace logs.
        """
        safe_id = sample_id.replace('/', '_').replace('\\', '_').replace(':', '_')[:100]
        # Unified naming: minions_line_<N>.json (protocol prefix + sample_id)
        return self.sample_logs_dir / f"{protocol}_{safe_id}.json"
    
    def _get_sample_log_path(self, sample_id: str, protocol: str) -> Path:
        """Get the log file path for a sample.
        
        Uses unified naming: minions_line_<N>.log for easy correlation with trace logs.
        """
        safe_id = sample_id.replace('/', '_').replace('\\', '_').replace(':', '_')[:100]
        # Unified naming: minions_line_<N>.log (protocol prefix + sample_id)
        return self.sample_logs_dir / f"{protocol}_{safe_id}.log"
    
    def _get_unified_logging_id(self, sample_id: str, protocol: str) -> str:
        """Get the unified logging ID for minions trace files.
        
        Returns: ID in format 'minions_line_<N>' for use as logging_id parameter.
        """
        safe_id = sample_id.replace('/', '_').replace('\\', '_').replace(':', '_')[:100]
        return f"{protocol}_{safe_id}"
    
    def _load_cached_result(self, sample_id: str, protocol: str) -> Optional[EvaluationResult]:
        """Load cached result for a sample if it exists."""
        cache_path = self._get_sample_cache_path(sample_id, protocol)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure predicted_answer is always a string (cache may have numeric values)
            raw_answer = data['predicted_answer']
            predicted_answer = str(raw_answer) if raw_answer is not None else ''
            
            return EvaluationResult(
                sample_id=data['sample_id'],
                protocol=data['protocol'],
                question=data['question'],
                predicted_answer=predicted_answer,
                ground_truth=data['ground_truth'],
                cost_usd=data['cost_usd'],
                input_tokens=data['input_tokens'],
                output_tokens=data['output_tokens'],
                execution_time=data['execution_time'],
                is_correct=data.get('is_correct'),
                error=data.get('error')
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cached result for {sample_id}: {e}")
            return None
    
    def _save_sample_result(self, result: EvaluationResult, log_content: str = ""):
        """Save a sample result to cache and optionally git commit."""
        cache_path = self._get_sample_cache_path(result.sample_id, result.protocol)
        log_path = self._get_sample_log_path(result.sample_id, result.protocol)
        
        result_data = asdict(result)
        result_data['cached_at'] = datetime.now().isoformat()
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample: {result.sample_id}\n")
            f.write(f"Protocol: {result.protocol}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Execution time: {result.execution_time:.2f}s\n")
            f.write(f"Cost: ${result.cost_usd:.4f}\n")
            f.write(f"Is correct: {result.is_correct}\n")
            f.write(f"Error: {result.error}\n")
            f.write("\n--- Question ---\n")
            f.write(result.question)
            f.write("\n\n--- Predicted Answer ---\n")
            f.write(str(result.predicted_answer) if result.predicted_answer is not None else "N/A")
            f.write("\n\n--- Ground Truth ---\n")
            f.write(str(result.ground_truth))
            if log_content:
                f.write("\n\n--- Additional Log ---\n")
                f.write(log_content)
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation across all protocols and samples."""
        self._start_time = time.time()
        self._total_runtime = 0  # Initialize for incremental updates
        logger.info(f"Starting evaluation with protocols: {self.protocols}")
        
        samples = self.dataset.samples
        if not samples:
            raise ValueError("No samples loaded from dataset")
        
        # Generate initial LaTeX report at the start
        self._update_latex_report()
        
        for protocol in self.protocols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating protocol: {protocol}")
            logger.info(f"{'='*60}")
            
            protocol_results = []
            cached_count = 0
            evaluated_count = 0
            
            for idx, sample in enumerate(samples, 1):
                if self.use_cache:
                    cached_result = self._load_cached_result(sample.sample_id, protocol)
                    if cached_result is not None:
                        protocol_results.append(cached_result)
                        self.results[protocol] = protocol_results  # Update results for report
                        cached_count += 1
                        is_correct_str = "✓" if cached_result.is_correct else "✗" if cached_result.is_correct is not None else "?"
                        print(f"\n\033[1m[{idx} / {len(samples)}]\033[0m \033[90m(cached)\033[0m")
                        logger.info(f"Loaded cached result for {sample.sample_id}: {is_correct_str}")
                        # Update LaTeX report with cached result
                        self._update_latex_report()
                        continue
                
                print(f"\n\033[1m[{idx} / {len(samples)}]\033[0m")
                logger.info(f"Processing sample: {sample.sample_id}")
                logger.info(f"Question: {sample.question[:100]}...")
                
                try:
                    result = self._evaluate_sample(sample, protocol)
                    protocol_results.append(result)
                    self.results[protocol] = protocol_results  # Update results for report
                    evaluated_count += 1
                    
                    if self.use_cache:
                        self._save_sample_result(result)
                    
                    if result.error:
                        logger.warning(f"  Error: {result.error}")
                    else:
                        is_correct_str = "✓" if result.is_correct else "✗" if result.is_correct is not None else "?"
                        logger.info(f"  Result: {is_correct_str} | Cost: ${result.cost_usd:.4f} | Time: {result.execution_time:.2f}s")
                    
                    # Update LaTeX report after each completed query
                    self._update_latex_report()
                
                except Exception as e:
                    logger.error(f"  Unexpected error: {e}")
                    error_result = EvaluationResult(
                        sample_id=sample.sample_id,
                        protocol=protocol,
                        question=sample.question,
                        predicted_answer="",
                        ground_truth=sample.answer,
                        cost_usd=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        execution_time=0.0,
                        error=str(e)
                    )
                    protocol_results.append(error_result)
                    self.results[protocol] = protocol_results  # Update results for report
                    if self.use_cache:
                        self._save_sample_result(error_result)
                    # Update LaTeX report even on errors
                    self._update_latex_report()
            
            self.results[protocol] = protocol_results
            
            if self.use_cache:
                logger.info(f"Cache stats for {protocol}: {cached_count} cached, {evaluated_count} evaluated")
        
        self._total_runtime = time.time() - self._start_time
        return self._aggregate_all_results()
    
    def _evaluate_sample(self, sample: FinanceBenchSample, protocol: str) -> EvaluationResult:
        """Evaluate a single sample with a given protocol."""
        start_time = time.time()
        
        protocol_method = {
            'local_only': self.protocol_runner.run_local_only,
            'remote_only': self.protocol_runner.run_remote_only,
            'minion': self.protocol_runner.run_minion,
            'minions': self.protocol_runner.run_minions,
        }.get(protocol.lower())
        
        if not protocol_method:
            raise ValueError(f"Unknown protocol: {protocol}")
        
        if protocol.lower() == 'minions':
            minions_kwargs_with_dir = self.minions_kwargs.copy()
            minions_kwargs_with_dir['run_output_dir'] = str(self.run_output_dir)
            # Use unified logging_id for trace file naming (matches sample_logs naming)
            minions_kwargs_with_dir['logging_id'] = self._get_unified_logging_id(sample.sample_id, protocol)
            result_dict = protocol_method(
                question=sample.question,
                context=sample.document_text,
                **minions_kwargs_with_dir
            )
        else:
            result_dict = protocol_method(
                question=sample.question,
                context=sample.document_text
            )
        
        execution_time = time.time() - start_time
        raw_answer = result_dict.get('final_answer', '')
        predicted_answer = str(raw_answer) if raw_answer is not None else ''
        
        if protocol.lower() == 'remote_only':
            print(f"\n--- Remote-Only Answer Log ---")
            print(f"Question: {sample.question[:100]}...")
            print(f"Ground Truth: {sample.answer}")
            print(f"Predicted: {predicted_answer[:500]}{'...' if len(predicted_answer) > 500 else ''}")
            print(f"-------------------------------\n")
        
        remote_usage = _normalize_usage(result_dict.get('remote_usage', Usage()))
        cost_usd = MetricsCalculator.calculate_cost(remote_usage)
        
        if self.skip_accuracy:
            is_correct = None
        else:
            remote_client = self.protocol_runner._get_remote_client()
            is_correct = MetricsCalculator._is_answer_correct(
                predicted_answer, 
                sample.answer, 
                remote_client=remote_client, 
                question=sample.question,
                context=sample.document_text
            )
        
        return EvaluationResult(
            sample_id=sample.sample_id or 'unknown',
            protocol=protocol,
            question=sample.question,
            predicted_answer=predicted_answer,
            ground_truth=sample.answer,
            is_correct=is_correct,
            cost_usd=cost_usd,
            input_tokens=remote_usage.prompt_tokens,
            output_tokens=remote_usage.completion_tokens,
            execution_time=execution_time,
            error=result_dict.get('error')
        )
    
    def _aggregate_all_results(self) -> Dict[str, Any]:
        """Aggregate results from all protocols."""
        summary = {}
        for protocol, results in self.results.items():
            summary[protocol] = MetricsCalculator.aggregate_results(results, skip_accuracy=self.skip_accuracy)
        return summary
    
    def save_results(self, filename_prefix: str = "financebench_results") -> Tuple[Path, Path]:
        """Save results to JSON and CSV files."""
        json_path = self.run_output_dir / f"{filename_prefix}.json"
        csv_path = self.run_output_dir / f"{filename_prefix}.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        output_data = {
            'timestamp': timestamp,
            'protocols': self.protocols,
            'total_samples': len(self.dataset.samples),
            'results': {
                protocol: [
                    {k: v for k, v in asdict(r).items() if v is not None}
                    for r in results
                ]
                for protocol, results in self.results.items()
            },
            'summary': self._aggregate_all_results()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved detailed results to {json_path}")
        
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if self.skip_accuracy:
                writer.writerow([
                    'Protocol', 'Avg Cost ($)', 'Total Cost ($)',
                    'Avg Input Tokens (1k)', 'Avg Output Tokens (1k)',
                    'Total Samples', 'Successful', 'Failed', 'Avg Time (s)'
                ])
            else:
                writer.writerow([
                    'Protocol', 'Accuracy', 'Avg Cost ($)', 'Total Cost ($)',
                    'Avg Input Tokens (1k)', 'Avg Output Tokens (1k)',
                    'Total Samples', 'Successful', 'Failed', 'Avg Time (s)'
                ])
            
            for protocol, summary in output_data['summary'].items():
                if self.skip_accuracy:
                    writer.writerow([
                        protocol,
                        f"{summary['avg_cost_per_query']:.4f}",
                        f"{summary['total_cost']:.4f}",
                        f"{summary['avg_input_tokens']/1000:.2f}",
                        f"{summary['avg_output_tokens']/1000:.2f}",
                        summary['total_samples'],
                        summary['successful_samples'],
                        summary['failed_samples'],
                        f"{summary['avg_execution_time']:.2f}"
                    ])
                else:
                    accuracy = summary.get('accuracy', 0)
                    accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                    writer.writerow([
                        protocol,
                        accuracy_str,
                        f"{summary['avg_cost_per_query']:.4f}",
                        f"{summary['total_cost']:.4f}",
                        f"{summary['avg_input_tokens']/1000:.2f}",
                        f"{summary['avg_output_tokens']/1000:.2f}",
                        summary['total_samples'],
                        summary['successful_samples'],
                        summary['failed_samples'],
                        f"{summary['avg_execution_time']:.2f}"
                    ])
        
        logger.info(f"Saved summary to {csv_path}")
        
        return json_path, csv_path
    
    def print_summary(self):
        """Print summary table to console."""
        summary = self._aggregate_all_results()
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        if self.skip_accuracy:
            print(f"{'Protocol':<15} {'Avg Cost ($)':<15} {'Input Tokens (1k)':<18} {'Output Tokens (1k)':<18}")
        else:
            print(f"{'Protocol':<15} {'Accuracy':<10} {'Avg Cost ($)':<15} {'Input Tokens (1k)':<18} {'Output Tokens (1k)':<18}")
        print("-"*80)
        
        for protocol in self.protocols:
            s = summary.get(protocol, {})
            if self.skip_accuracy:
                print(f"{protocol:<15} {s.get('avg_cost_per_query', 0):<15.4f} "
                      f"{s.get('avg_input_tokens', 0)/1000:<18.2f} {s.get('avg_output_tokens', 0)/1000:<18.2f}")
            else:
                accuracy = s.get('accuracy', 0)
                accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                print(f"{protocol:<15} {accuracy_str:<10} {s.get('avg_cost_per_query', 0):<15.4f} "
                      f"{s.get('avg_input_tokens', 0)/1000:<18.2f} {s.get('avg_output_tokens', 0)/1000:<18.2f}")
        
        print("="*80)
    
    def _format_config_summary(self) -> str:
        """Format configuration as a human-readable summary string."""
        lines = []
        config = self.all_args
        
        lines.append("Configuration:")
        
        lines.append("  Dataset:")
        dataset = config.get('dataset', {})
        lines.append(f"    path: {dataset.get('path', 'N/A')}")
        lines.append(f"    filter_numerical: {str(dataset.get('filter_numerical', False)).lower()}")
        
        sample_indices = dataset.get('sample_indices')
        max_samples = dataset.get('max_samples')
        
        if sample_indices:
            lines.append(f"    samples: {len(sample_indices)} specific indices")
            if len(sample_indices) > 10:
                indices_str = str(sample_indices[:10])[:-1] + ", ...]"
            else:
                indices_str = str(sample_indices)
            lines.append(f"    indices: {indices_str}")
        elif max_samples:
            lines.append(f"    max_samples: {max_samples}")
        else:
            lines.append(f"    samples: all")
        lines.append("")
        
        lines.append("  Models:")
        models = config.get('models', {})
        local = models.get('local', {})
        remote = models.get('remote', {})
        local_ctx = local.get('num_ctx', 4096)
        lines.append(f"    local: {local.get('name', 'N/A')} (temp={local.get('temperature', 0.0)}, ctx={local_ctx})")
        lines.append(f"    remote: {remote.get('name', 'N/A')} (temp={remote.get('temperature', 0.0)})")
        lines.append("")
        
        protocols_config = config.get('protocols', {})
        active = protocols_config.get('active', [])
        lines.append(f"  Protocols: {', '.join(active)}")
        
        common = protocols_config.get('common', {})
        lines.append(f"    max_rounds: {common.get('max_rounds', 2)}")
        lines.append(f"    num_samples_per_task: {common.get('num_samples_per_task', 1)}")
        lines.append("")
        
        if 'minions' in active:
            minions = protocols_config.get('minions', {})
            lines.append("  MINIONS Settings:")
            lines.append(f"    num_tasks_per_round: {minions.get('num_tasks_per_round', 3)}")
            lines.append(f"    chunk_fn: {minions.get('chunk_fn', 'chunk_by_section')}")
            lines.append(f"    max_chunk_size: {minions.get('max_chunk_size', 3000)}")
            if minions.get('chunk_fn') == 'chunk_on_multiple_pages':
                lines.append(f"    pages_per_chunk: {minions.get('pages_per_chunk', 5)}")
            if minions.get('use_retrieval'):
                lines.append(f"    use_retrieval: {minions.get('use_retrieval')}")
            lines.append("")
        
        lines.append("  Global:")
        global_config = config.get('global', {})
        lines.append(f"    output_dir: {global_config.get('output_dir', 'evaluate/results')}")
        lines.append(f"    skip_accuracy: {str(global_config.get('skip_accuracy', False)).lower()}")
        lines.append(f"    use_cache: {str(global_config.get('use_cache', True)).lower()}")
        
        return "\n".join(lines)
    
    def save_summary(self) -> Path:
        """Save summary table to summary.txt file with per-query details."""
        summary = self._aggregate_all_results()
        summary_path = self.run_output_dir / "summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Configuration summary
            if self.all_args:
                f.write(self._format_config_summary())
                f.write("\n\n")
            
            # Runtime
            if hasattr(self, '_total_runtime'):
                minutes, seconds = divmod(self._total_runtime, 60)
                hours, minutes = divmod(minutes, 60)
                if hours > 0:
                    f.write(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.1f}s\n")
                elif minutes > 0:
                    f.write(f"Total runtime: {int(minutes)}m {seconds:.1f}s\n")
                else:
                    f.write(f"Total runtime: {seconds:.1f}s\n")
                f.write("\n")
            
            # Per-query details section
            f.write("="*80 + "\n")
            f.write("PER-QUERY DETAILS\n")
            f.write("="*80 + "\n\n")
            
            for protocol in self.protocols:
                protocol_results = self.results.get(protocol, [])
                if not protocol_results:
                    continue
                
                f.write(f"{protocol.upper()}:\n")
                f.write("-"*90 + "\n")
                f.write(f"{'Sample ID':<45} {'Cost ($)':<12} {'Input Tok':<12} {'Output Tok':<12} {'Correct':<8}\n")
                f.write("-"*90 + "\n")
                
                # Sort results by sample_id for consistent ordering
                sorted_results = sorted(protocol_results, key=lambda r: r.sample_id)
                
                for result in sorted_results:
                    sample_id_display = result.sample_id[-45:] if len(result.sample_id) > 45 else result.sample_id
                    # Correct column will be filled by correctness.py later, leave as "---" for now
                    f.write(f"{sample_id_display:<45} {result.cost_usd:<12.4f} {result.input_tokens:<12} {result.output_tokens:<12} {'---':<8}\n")
                
                f.write("\n")
            
            # Aggregate summary section
            f.write("="*80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("="*80 + "\n")
            if self.skip_accuracy:
                f.write(f"{'Protocol':<15} {'Avg Cost ($)':<15} {'Input Tokens (1k)':<18} {'Output Tokens (1k)':<18}\n")
            else:
                f.write(f"{'Protocol':<15} {'Accuracy':<10} {'Avg Cost ($)':<15} {'Input Tokens (1k)':<18} {'Output Tokens (1k)':<18}\n")
            f.write("-"*80 + "\n")
            
            for protocol in self.protocols:
                s = summary.get(protocol, {})
                if self.skip_accuracy:
                    f.write(f"{protocol:<15} {s.get('avg_cost_per_query', 0):<15.4f} "
                           f"{s.get('avg_input_tokens', 0)/1000:<18.2f} {s.get('avg_output_tokens', 0)/1000:<18.2f}\n")
                else:
                    accuracy = s.get('accuracy', 0)
                    accuracy_str = f"{accuracy:.4f}" if accuracy is not None else "N/A"
                    f.write(f"{protocol:<15} {accuracy_str:<10} {s.get('avg_cost_per_query', 0):<15.4f} "
                           f"{s.get('avg_input_tokens', 0)/1000:<18.2f} {s.get('avg_output_tokens', 0)/1000:<18.2f}\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"Saved summary to {summary_path}")
        return summary_path
    
    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in text."""
        if text is None:
            return ""
        text = str(text)
        # Order matters: escape backslash first
        replacements = [
            ('\\', r'\textbackslash{}'),
            ('&', r'\&'),
            ('%', r'\%'),
            ('$', r'\$'),
            ('#', r'\#'),
            ('_', r'\_'),
            ('{', r'\{'),
            ('}', r'\}'),
            ('~', r'\textasciitilde{}'),
            ('^', r'\textasciicircum{}'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    
    def _compile_latex(self, tex_path: Path) -> Optional[Path]:
        """
        Compile LaTeX file to PDF using pdflatex.
        
        Args:
            tex_path: Path to the .tex file
            
        Returns:
            Path to the generated PDF, or None if compilation failed
        """
        pdf_path = tex_path.with_suffix('.pdf')
        
        try:
            # Check if pdflatex is available
            result = subprocess.run(
                ['which', 'pdflatex'],
                capture_output=True,
                check=False,
                timeout=10
            )
            if result.returncode != 0:
                logger.warning("pdflatex not found. Please install texlive to compile LaTeX reports.")
                return None
            
            # Run pdflatex twice for proper cross-references
            for run_num in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_path.name],
                    cwd=str(tex_path.parent),
                    capture_output=True,
                    check=False,
                    timeout=120
                )
                if result.returncode != 0:
                    logger.error(f"pdflatex compilation failed (run {run_num + 1})")
                    logger.error(f"stdout: {result.stdout.decode('utf-8', errors='replace')[-2000:]}")
                    logger.error(f"stderr: {result.stderr.decode('utf-8', errors='replace')[-500:]}")
                    return None
            
            # Clean up auxiliary files
            for ext in ['.aux', '.log', '.out', '.toc']:
                aux_file = tex_path.with_suffix(ext)
                if aux_file.exists():
                    try:
                        aux_file.unlink()
                    except Exception:
                        pass
            
            if pdf_path.exists():
                logger.info(f"Successfully compiled LaTeX to {pdf_path}")
                return pdf_path
            else:
                logger.error("PDF file not created despite successful pdflatex run")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("pdflatex compilation timed out")
            return None
        except Exception as e:
            logger.error(f"Error compiling LaTeX: {e}")
            return None
    
    def _update_latex_report(self):
        """
        Update the LaTeX report with current results.
        Called after each sample is processed to provide incremental updates.
        Errors are logged but don't interrupt the evaluation.
        """
        try:
            # Update runtime for the report
            self._total_runtime = time.time() - self._start_time
            # Generate and compile the report
            self.save_latex_report()
        except Exception as e:
            logger.warning(f"Failed to update LaTeX report: {e}")
    
    def save_latex_report(self, filename_prefix: str = "report") -> Tuple[Path, Optional[Path]]:
        """
        Generate a LaTeX report and compile it to PDF.
        
        Args:
            filename_prefix: Prefix for the output files
            
        Returns:
            Tuple of (tex_path, pdf_path). pdf_path may be None if compilation failed.
        """
        summary = self._aggregate_all_results()
        tex_path = self.run_output_dir / f"{filename_prefix}.tex"
        
        # Format runtime
        runtime_str = "N/A"
        if hasattr(self, '_total_runtime'):
            minutes, seconds = divmod(self._total_runtime, 60)
            hours, minutes = divmod(minutes, 60)
            if hours > 0:
                runtime_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
            elif minutes > 0:
                runtime_str = f"{int(minutes)}m {seconds:.1f}s"
            else:
                runtime_str = f"{seconds:.1f}s"
        
        # Get configuration details
        config = self.all_args
        dataset_config = config.get('dataset', {})
        models_config = config.get('models', {})
        protocols_config = config.get('protocols', {})
        global_config = config.get('global', {})
        
        local_model = models_config.get('local', {})
        remote_model = models_config.get('remote', {})
        minions_config = protocols_config.get('minions', {})
        common_config = protocols_config.get('common', {})
        
        # Escape all text for LaTeX
        esc = self._escape_latex
        
        # Build LaTeX document
        latex_content = r"""\documentclass[11pt,a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\usepackage{fancyhdr}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{array}
\usepackage{tabularx}

% Page geometry
\geometry{margin=1in, top=1.2in, bottom=1in}

% Colors
\definecolor{headerblue}{RGB}{41, 128, 185}
\definecolor{lightgray}{RGB}{245, 245, 245}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{headerblue}{\textbf{FinanceBench Evaluation Report}}}
\fancyhead[R]{\textcolor{gray}{""" + esc(str(self.run_output_dir.name)) + r"""}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% Hyperref setup
\hypersetup{
    colorlinks=true,
    linkcolor=headerblue,
    urlcolor=headerblue
}

% Title
\title{\vspace{-1cm}\textcolor{headerblue}{\textbf{FinanceBench Evaluation Report}}}
\author{Generated by financebench\_evaluator.py}
\date{""" + esc(datetime.now().strftime("%B %d, %Y at %H:%M")) + r"""}

\begin{document}

\maketitle

\vspace{0.5cm}

"""
        
        # Calculate progress
        total_samples = len(self.dataset.samples)
        processed_samples = sum(len(r) for r in self.results.values())
        progress_pct = (processed_samples / total_samples * 100) if total_samples > 0 else 0
        
        if processed_samples >= total_samples and total_samples > 0:
            status_text = r"\textcolor{green!60!black}{\textbf{COMPLETED}}"
        elif processed_samples > 0:
            status_text = r"\textcolor{orange}{\textbf{IN PROGRESS}}"
        else:
            status_text = r"\textcolor{gray}{\textbf{STARTING}}"
        
        latex_content += r"""
% Progress indicator
\begin{center}
\large """ + status_text + r""" --- """ + str(processed_samples) + r"""/""" + str(total_samples) + r""" samples (""" + f"{progress_pct:.1f}" + r"""\%)
\end{center}

\vspace{0.3cm}

% Configuration Section
\section*{Configuration}

\subsection*{Dataset}
\begin{tabular}{@{}ll@{}}
\textbf{Path:} & \texttt{""" + esc(str(dataset_config.get('path', 'N/A'))) + r"""} \\
\textbf{Filter Numerical:} & """ + ('Yes' if dataset_config.get('filter_numerical') else 'No') + r""" \\
\textbf{Total Samples:} & """ + str(len(self.dataset.samples)) + r""" \\
\end{tabular}

\subsection*{Models}
\begin{tabular}{@{}ll@{}}
\textbf{Local Model:} & \texttt{""" + esc(str(local_model.get('name', 'N/A'))) + r"""} \\
\textbf{Local Temperature:} & """ + str(local_model.get('temperature', 0.0)) + r""" \\
\textbf{Local Context:} & """ + str(local_model.get('num_ctx', 4096)) + r""" tokens \\
\textbf{Remote Model:} & \texttt{""" + esc(str(remote_model.get('name', 'N/A'))) + r"""} \\
\textbf{Remote Temperature:} & """ + str(remote_model.get('temperature', 0.0)) + r""" \\
\end{tabular}

"""
        
        # Add SGLang/Constraint Decoding settings if using SGLang backend
        if local_model.get('backend') == 'sglang':
            latex_content += r"""
\subsection*{SGLang / Constraint Decoding Settings}
\begin{tabular}{@{}ll@{}}
"""
            # Logit processor path
            lp_path = local_model.get('logit_processor_path')
            if lp_path:
                latex_content += r"\textbf{Logit Processor:} & \texttt{" + esc(str(lp_path)) + r"} \\" + "\n"
            else:
                latex_content += r"\textbf{Logit Processor:} & \textit{None} \\" + "\n"
            
            latex_content += r"\end{tabular}" + "\n\n"
        
        latex_content += r"""
\subsection*{Protocol Settings}
\begin{tabular}{@{}ll@{}}
\textbf{Active Protocols:} & """ + esc(', '.join(protocols_config.get('active', []))) + r""" \\
\textbf{\textcolor{red}{Max Rounds:}} & """ + str(common_config.get('max_rounds', 2)) + r""" \\
\textbf{Samples per Task:} & """ + str(common_config.get('num_samples_per_task', 1)) + r""" \\
"""
        
        # Add MINIONS-specific settings if minions protocol is active
        if 'minions' in protocols_config.get('active', []):
            latex_content += r"""\textbf{Tasks per Round:} & """ + str(minions_config.get('num_tasks_per_round', 3)) + r""" \\
\textbf{\textcolor{red}{Chunk Function:}} & \texttt{""" + esc(str(minions_config.get('chunk_fn', 'chunk_by_section'))) + r"""} \\
\textbf{Max Chunk Size:} & """ + str(minions_config.get('max_chunk_size', 3000)) + r""" \\
"""
        
        latex_content += r"""\end{tabular}

\vspace{1cm}

% Runtime
\noindent\textbf{Total Runtime:} """ + runtime_str + r"""

\vspace{1cm}

% Summary Section
\section*{Evaluation Summary}

\begin{table}[h!]
\centering
\begin{tabular}{l""" + ('cccc' if not self.skip_accuracy else 'ccc') + r"""}
\toprule
"""
        
        # Table header
        if self.skip_accuracy:
            latex_content += r"""\textbf{Protocol} & \textbf{Avg Cost (\$)} & \textbf{Input Tokens (k)} & \textbf{Output Tokens (k)} \\
"""
        else:
            latex_content += r"""\textbf{Protocol} & \textbf{Accuracy} & \textbf{Avg Cost (\$)} & \textbf{Input Tokens (k)} & \textbf{Output Tokens (k)} \\
"""
        
        latex_content += r"""\midrule
"""
        
        # Table rows
        for protocol in self.protocols:
            s = summary.get(protocol, {})
            avg_cost = s.get('avg_cost_per_query', 0)
            avg_input = s.get('avg_input_tokens', 0) / 1000
            avg_output = s.get('avg_output_tokens', 0) / 1000
            
            if self.skip_accuracy:
                latex_content += f"{esc(protocol)} & {avg_cost:.4f} & {avg_input:.2f} & {avg_output:.2f} \\\\\n"
            else:
                accuracy = s.get('accuracy', 0)
                if accuracy is not None:
                    accuracy_str = f"{accuracy*100:.2f}\\%"
                else:
                    accuracy_str = "N/A"
                latex_content += f"{esc(protocol)} & {accuracy_str} & {avg_cost:.4f} & {avg_input:.2f} & {avg_output:.2f} \\\\\n"
        
        latex_content += r"""\bottomrule
\end{tabular}
\caption{Evaluation metrics by protocol}
\end{table}

"""
        
        # Add per-protocol statistics
        for protocol in self.protocols:
            s = summary.get(protocol, {})
            latex_content += r"""
\subsection*{""" + esc(protocol.upper()) + r""" Statistics}
\begin{tabular}{@{}ll@{}}
\textbf{Total Samples:} & """ + str(s.get('total_samples', 0)) + r""" \\
\textbf{Successful:} & """ + str(s.get('successful_samples', 0)) + r""" \\
\textbf{Failed:} & """ + str(s.get('failed_samples', 0)) + r""" \\
\textbf{Total Cost:} & \$""" + f"{s.get('total_cost', 0):.4f}" + r""" \\
\textbf{Avg Execution Time:} & """ + f"{s.get('avg_execution_time', 0):.2f}" + r"""s \\
\textbf{Total Input Tokens:} & """ + f"{s.get('total_input_tokens', 0):,}" + r""" \\
\textbf{Total Output Tokens:} & """ + f"{s.get('total_output_tokens', 0):,}" + r""" \\
\end{tabular}

"""
        
        # Add sample results section (first 20 samples for readability)
        latex_content += r"""
\newpage
\section*{Sample Results}

\small
\begin{longtable}{p{3cm}p{5cm}p{2cm}p{2cm}p{1.5cm}}
\toprule
\textbf{Sample ID} & \textbf{Question (truncated)} & \textbf{Predicted} & \textbf{Ground Truth} & \textbf{Correct} \\
\midrule
\endhead
\bottomrule
\endfoot
"""
        
        # Add sample rows (limit to first 30 for PDF size)
        max_samples_to_show = 30
        samples_shown = 0
        
        for protocol, results in self.results.items():
            for result in results[:max_samples_to_show]:
                if samples_shown >= max_samples_to_show:
                    break
                
                # Truncate long text
                question_short = result.question[:80] + "..." if len(result.question) > 80 else result.question
                predicted_short = str(result.predicted_answer)[:40] + "..." if len(str(result.predicted_answer)) > 40 else str(result.predicted_answer)
                gt_short = str(result.ground_truth[0])[:40] + "..." if result.ground_truth and len(str(result.ground_truth[0])) > 40 else (str(result.ground_truth[0]) if result.ground_truth else "N/A")
                
                if result.is_correct is None:
                    correct_str = "---"
                elif result.is_correct:
                    correct_str = r"\textcolor{green!60!black}{Yes}"
                else:
                    correct_str = r"\textcolor{red}{No}"
                
                sample_id_short = result.sample_id[-25:] if len(result.sample_id) > 25 else result.sample_id
                
                latex_content += f"{esc(sample_id_short)} & {esc(question_short)} & {esc(predicted_short)} & {esc(gt_short)} & {correct_str} \\\\\n"
                samples_shown += 1
            
            if samples_shown >= max_samples_to_show:
                break
        
        total_samples = sum(len(r) for r in self.results.values())
        if total_samples > max_samples_to_show:
            latex_content += r"""\multicolumn{5}{c}{\textit{... """ + str(total_samples - max_samples_to_show) + r""" more samples not shown ...}} \\
"""
        
        latex_content += r"""
\end{longtable}

\vfill
\begin{center}
\small\textit{Generated by FinanceBench Evaluator}
\end{center}

\end{document}
"""
        
        # Write LaTeX file
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        logger.info(f"Saved LaTeX report to {tex_path}")
        
        # Compile to PDF
        pdf_path = self._compile_latex(tex_path)
        
        return tex_path, pdf_path


def main():
    """Main entry point using Kconfig .config file."""
    if len(sys.argv) != 2:
        print("Usage: python financebench_evaluator.py <.config>")
        print("\nExample:")
        print("  python evaluate/financebench_evaluator.py .config")
        print("\nTo create a configuration interactively:")
        print("  python evaluate/kmenuconfig.py")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Load dataset (always uses full PDF documents)
    dataset = FinanceBenchDataset(
        dataset_path=config.dataset.path,
        pdf_dir=config.dataset.pdf_dir,
        filter_numerical=config.dataset.filter_numerical,
        max_samples=config.dataset.max_samples,
        sample_indices=config.dataset.sample_indices
    )
    dataset.load()
    
    num_samples = len(dataset.samples)
    if config.dataset.max_samples and config.dataset.max_samples < num_samples:
        num_samples = config.dataset.max_samples
    
    protocol_runner = ProtocolRunner(
        local_model=config.models.local.name,
        remote_model=config.models.remote.name,
        local_temp=config.models.local.temperature,
        remote_temp=config.models.remote.temperature,
        num_ctx=config.models.local.num_ctx,
        local_backend=getattr(config.models.local, 'backend', 'ollama'),
        sglang_base_url=getattr(config.models.local, 'sglang_base_url', 'http://localhost:8000/v1'),
        logit_processor_path=getattr(config.models.local, 'logit_processor_path', None),
    )
    
    # Build minions_kwargs from config
    minions_kwargs = config.protocols.minions.to_dict()
    minions_kwargs.update({
        'max_rounds': config.protocols.common.max_rounds,
        'num_samples_per_task': config.protocols.common.num_samples_per_task
    })
    
    # Load prompt overrides for evolution experiments
    if config.global_config.prompt_set:
        prompt_set_path = config.global_config.prompt_set
        if os.path.exists(prompt_set_path):
            logger.info(f"Loading prompt overrides from: {prompt_set_path}")
            with open(prompt_set_path, 'r', encoding='utf-8') as f:
                prompt_overrides = json.load(f)
            minions_kwargs['prompt_overrides'] = prompt_overrides
            logger.info(f"Loaded {len(prompt_overrides)} prompt overrides: {list(prompt_overrides.keys())}")
        else:
            logger.warning(f"Prompt set file not found: {prompt_set_path}")
    
    # Generate cache directory name if caching is enabled
    cache_dir_name = None
    if config.global_config.use_cache:
        cache_dir_name = generate_cache_dir_name_from_config(config)
        logger.info(f"Cache directory name: {cache_dir_name}")
    
    config_dict = config.to_dict()
    
    evaluator = Evaluator(
        dataset=dataset,
        protocol_runner=protocol_runner,
        protocols=config.protocols.active,
        output_dir=config.global_config.output_dir,
        num_samples=num_samples,
        minions_kwargs=minions_kwargs,
        skip_accuracy=config.global_config.skip_accuracy,
        command_line=f"python {sys.argv[0]} {config_path}",
        use_cache=config.global_config.use_cache,
        cache_dir_name=cache_dir_name,
        all_args=config_dict
    )
    
    # Copy original .config file for reproducibility
    import shutil
    config_copy_path = evaluator.run_output_dir / "config_used.config"
    config_json_path = evaluator.run_output_dir / "config_parsed.json"
    try:
        shutil.copy2(config_path, config_copy_path)
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save configuration: {e}")
    
    try:
        evaluator.evaluate()
        json_path, csv_path = evaluator.save_results()
        summary_path = evaluator.save_summary()
        tex_path, pdf_path = evaluator.save_latex_report()
        evaluator.print_summary()
        
        print(f"\nResults saved to:")
        print(f"  Run directory: {evaluator.run_output_dir}")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print(f"  Summary: {summary_path}")
        print(f"  LaTeX: {tex_path}")
        if pdf_path:
            print(f"  PDF: {pdf_path}")
        else:
            print(f"  PDF: (compilation failed, check {tex_path} for errors)")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        if evaluator.results:
            evaluator.save_results(filename_prefix="financebench_results_partial")
            evaluator.save_summary()
            evaluator.save_latex_report(filename_prefix="report_partial")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
