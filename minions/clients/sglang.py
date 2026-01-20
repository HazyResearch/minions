"""
SGLang Client for MinionS with Length Penalty.

Uses SGLang's OpenAI-compatible API with logit_bias to implement a length
penalty that encourages shorter outputs. Higher beta values = stronger
preference for shorter responses (via EOS token boost).

SGLang provides RadixAttention which automatically shares KV cache across
parallel requests with common prefixes (document chunks).

Usage:
    client = SGLangClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000",
        temperature=0.2,
        beta_explanation=1.0,  # Length penalty for explanations
        beta_citation=2.0,     # Length penalty for citations  
        beta_answer=1.5,       # Length penalty for answers
    )
"""

import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

# AutoTokenizer is optional - used to get model-specific stop token IDs
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None

# CustomLogitProcessor for per-step length penalty
try:
    from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
    HAS_CUSTOM_LOGIT_PROCESSOR = True
except ImportError:
    HAS_CUSTOM_LOGIT_PROCESSOR = False
    CustomLogitProcessor = None

from minions.usage import Usage
from minions.clients.base import MinionsClient
from minions.prompts.minions import (
    WORKER_PROMPT_EXPLANATION,
    WORKER_PROMPT_CITATION,
    WORKER_PROMPT_ANSWER,
)


# Define CompressedNoStepsLogitProcessor if CustomLogitProcessor is available
if HAS_CUSTOM_LOGIT_PROCESSOR:
    import torch
    
    class CompressedNoStepsLogitProcessor(CustomLogitProcessor):
        """
        SGLang-compatible logit processor for compressed answers.
        
        Goals:
          - Compressed answers (penalize fluff/preamble)
          - No numbered lists / step-by-step framing
          - Avoid truncation: disallow EOS until min_new_tokens
        
        This follows SGLang's CustomLogitProcessor interface where:
        - __call__(logits, custom_param_list) is called at each step
        - custom_param_list contains per-request params with token IDs and penalties
        - The processor is serialized via .to_str() and passed to the server
        
        All token IDs are pre-computed on the client and passed via custom_params.
        """
        
        def __call__(self, logits, custom_param_list):
            """
            Apply compressed output penalties at each decoding step.
            
            Args:
                logits: (batch_size, vocab_size) tensor of next-token logits
                custom_param_list: List of dicts with per-request parameters
            """
            assert logits.shape[0] == len(custom_param_list)
            
            for i, p in enumerate(custom_param_list):
                # Extract parameters from custom_params
                eos_token_id = int(p.get("eos_token_id", 128001))
                min_new_tokens = int(p.get("min_new_tokens", 48))
                current_len = int(p.get("current_len", 0))
                
                # Penalty values (hardcoded defaults)
                eos_bonus_after_min = float(p.get("eos_bonus_after_min", 1.2))
                intro_window = int(p.get("intro_window", 48))
                intro_penalty = float(p.get("intro_penalty", 5.0))
                always_penalty = float(p.get("always_penalty", 0.5))
                list_penalty = float(p.get("list_penalty", 4.0))
                repeat_penalty = float(p.get("repeat_penalty", 0.6))
                repeat_last_k = int(p.get("repeat_last_k", 128))
                
                # Token ID sets (pre-computed on client, passed as lists)
                intro_token_ids = set(p.get("intro_token_ids", []))
                always_token_ids = set(p.get("always_token_ids", []))
                list_token_ids = set(p.get("list_token_ids", []))
                newline_token_ids = set(p.get("newline_token_ids", []))
                recent_token_ids = p.get("recent_token_ids", [])  # For repetition penalty
                last_token_id = p.get("last_token_id", None)  # For newline tail heuristic
                
                # 1) Anti-truncation: hard block EOS before min_new_tokens
                if current_len < min_new_tokens:
                    logits[i, eos_token_id] = -1e9
                else:
                    logits[i, eos_token_id] += eos_bonus_after_min
                
                # 2) Always-on light anti-verbosity penalty
                if always_token_ids and always_penalty > 0:
                    for tid in always_token_ids:
                        logits[i, tid] -= always_penalty
                
                # 3) Early-only strong anti-preamble ("foreplay")
                if current_len < intro_window and intro_token_ids and intro_penalty > 0:
                    for tid in intro_token_ids:
                        logits[i, tid] -= intro_penalty
                
                # 4) Strong anti-list / anti-steps penalty
                if list_token_ids and list_penalty > 0:
                    for tid in list_token_ids:
                        logits[i, tid] -= list_penalty
                
                # 5) Tail heuristic: if last token was a newline, extra-discourage list starters
                if last_token_id is not None and int(last_token_id) in newline_token_ids:
                    # Extra penalty for tokens that commonly start lists after newlines
                    for tid in list_token_ids:
                        logits[i, tid] -= (list_penalty + 1.0)
                
                # 6) Repetition penalty on recent tokens
                if recent_token_ids and repeat_penalty > 0:
                    unique_recent = set(recent_token_ids[-repeat_last_k:])
                    for tid in unique_recent:
                        logits[i, tid] -= repeat_penalty
            
            return logits
else:
    CompressedNoStepsLogitProcessor = None


def _extract_string_from_value(value: Any) -> str:
    """
    Extract a string from a value that might be a dict or other type.
    
    This handles cases where the model returns a dict instead of a string,
    e.g., {"FY2017": "None", "FY2018": "None"} instead of "1.9%"
    
    Args:
        value: The value to extract a string from
        
    Returns:
        A string representation of the value
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Try to extract meaningful content from dict
        # Filter out "None" string values
        meaningful_values = [
            str(v) for v in value.values() 
            if v is not None and str(v).lower() != "none" and str(v).strip()
        ]
        if meaningful_values:
            return "; ".join(meaningful_values)
        # If all values are None/empty, return empty string
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v is not None)
    return str(value)


def _postprocess_json_output(output_text: str) -> str:
    """
    Post-process JSON output to ensure fields are strings, not dicts.
    
    This handles the case where the model returns structured data for
    fields that should be plain strings, or returns an empty JSON object.
    
    Args:
        output_text: Raw JSON output from the model
        
    Returns:
        Corrected JSON string with all fields as strings
    """
    try:
        data = json.loads(output_text)
        if not isinstance(data, dict):
            return output_text
        
        # Handle empty JSON {} - fill with default values
        if not data:
            return json.dumps({
                "explanation": "No relevant information found in this chunk.",
                "citation": "",
                "answer": ""
            })
        
        # Fields that should be strings
        string_fields = ['explanation', 'citation', 'answer']
        
        corrected = {}
        for key, value in data.items():
            if key in string_fields:
                corrected[key] = _extract_string_from_value(value)
            else:
                corrected[key] = value
        
        # Ensure required fields exist
        if 'explanation' not in corrected:
            corrected['explanation'] = "No explanation provided."
        if 'citation' not in corrected:
            corrected['citation'] = ""
        if 'answer' not in corrected:
            corrected['answer'] = ""
        
        return json.dumps(corrected)
    except json.JSONDecodeError:
        return output_text
    except Exception:
        return output_text


class SGLangClient(MinionsClient):
    """
    Client for SGLang inference server with WFSA length prior.
    
    Uses SGLang's native /generate endpoint with custom logit processor
    to implement a WFSA-based length prior. Generates JobOutput fields
    (explanation, citation, answer) sequentially to maximize KV cache reuse.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_workers: int = 32,
        structured_output_schema=None,
        # Generation strategy
        generation_strategy: str = "sequential",  # "sequential" or "single_shot"
        # WFSA beta values per field (higher = shorter outputs)
        beta_explanation: float = 1.0,
        beta_citation: float = 2.0,
        beta_answer: float = 1.5,
        # Minimum tokens per field before EOS is allowed
        min_tokens_explanation: int = 10,
        min_tokens_citation: int = 5,
        min_tokens_answer: int = 3,
        # Max tokens per field
        max_tokens_explanation: int = 200,
        max_tokens_citation: int = 150,
        max_tokens_answer: int = 100,
        **kwargs
    ):
        """
        Initialize SGLang client with WFSA length prior.
        
        Args:
            model_name: Model name (should match what SGLang server loaded)
            base_url: SGLang server URL (default: http://localhost:8000)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum tokens to generate for full response (default: 2048)
            max_workers: Maximum parallel workers for batch requests (default: 32)
            structured_output_schema: Optional Pydantic model for structured JSON output
            generation_strategy: "sequential" (3 API calls) or "single_shot" (1 API call)
            beta_explanation: WFSA strength for explanation field (default: 1.0)
            beta_citation: WFSA strength for citation field (default: 2.0)
            beta_answer: WFSA strength for answer field (default: 1.5)
            min_tokens_explanation: Minimum tokens for explanation (default: 10)
            min_tokens_citation: Minimum tokens for citation (default: 5)
            min_tokens_answer: Minimum tokens for answer (default: 3)
            max_tokens_explanation: Maximum tokens for explanation (default: 200)
            max_tokens_citation: Maximum tokens for citation (default: 150)
            max_tokens_answer: Maximum tokens for answer (default: 100)
            **kwargs: Additional arguments passed to base class
        """
        # Normalize base_url - remove /v1 suffix if present
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=True,
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.base_url = base_url or os.getenv("SGLANG_BASE_URL", "http://localhost:8000")
        self.max_workers = max_workers
        self.structured_output_schema = structured_output_schema
        self._generation_strategy = generation_strategy
        
        # WFSA parameters per field
        self.beta_explanation = beta_explanation
        self.beta_citation = beta_citation
        self.beta_answer = beta_answer
        self.min_tokens_explanation = min_tokens_explanation
        self.min_tokens_citation = min_tokens_citation
        self.min_tokens_answer = min_tokens_answer
        self.max_tokens_explanation = max_tokens_explanation
        self.max_tokens_citation = max_tokens_citation
        self.max_tokens_answer = max_tokens_answer
        
        # Generate JSON schema for strict enforcement if Pydantic model provided
        self.json_schema = None
        if structured_output_schema is not None:
            try:
                # Get JSON schema from Pydantic model
                if hasattr(structured_output_schema, 'model_json_schema'):
                    self.json_schema = structured_output_schema.model_json_schema()
                    self.logger.info(f"Generated JSON schema for strict enforcement: {list(self.json_schema.get('properties', {}).keys())}")
            except Exception as e:
                self.logger.warning(f"Could not generate JSON schema: {e}")
        
        # Initialize stop token IDs for length penalty
        # These are the EOS/EOT tokens that get boosted via logit_bias
        self.stop_ids = [128001, 128009]  # Default: Llama-3 EOS and EOT tokens
        
        # Store tokenizer for token ID precomputation
        self._tokenizer = None
        
        if HAS_TRANSFORMERS:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.stop_ids = [self._tokenizer.eos_token_id]
                # Get additional stop tokens for Llama-3 instruct format
                eot_id = self._tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None and eot_id != self._tokenizer.unk_token_id:
                    self.stop_ids.append(eot_id)
                self.logger.debug(f"[SGLang] Loaded stop IDs from tokenizer: {self.stop_ids}")
            except Exception as e:
                self.logger.warning(f"[SGLang] Could not load tokenizer, using default Llama-3 stop IDs: {e}")
        
        # Pre-compute token IDs for CompressedNoStepsLogitProcessor
        self._precompute_penalty_token_ids()
        
        # Track whether server supports CustomLogitProcessor
        # Will be validated on first request
        self._custom_processor_supported = HAS_CUSTOM_LOGIT_PROCESSOR and CompressedNoStepsLogitProcessor is not None
        self._custom_processor_validated = False  # Not yet tested against server
        self._processor_lock = threading.Lock()  # Thread-safe validation
        
        # Create OpenAI client for fallback (uses /v1 endpoint)
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("SGLANG_API_KEY", "not-needed"),
            base_url=f"{self.base_url}/v1"
        )
        
        # Field delimiters for sequential generation
        self.field_stop_sequences = {
            'explanation': ['\n"citation":', '\ncitation:', '"citation":'],
            'citation': ['\n"answer":', '\nanswer:', '"answer":'],
            'answer': ['\n}', '}\n', '}'],
        }
        
        self.logger.info(
            f"[SGLang] Client initialized: {self.base_url}, model={model_name}"
        )
        self.logger.info(
            f"[SGLang] Generation strategy: {generation_strategy}, "
            f"betas: (exp={beta_explanation}, cit={beta_citation}, ans={beta_answer}), "
            f"EOS tokens={self.stop_ids}"
        )
        if self._custom_processor_supported:
            self.logger.info(
                "[SGLang] CustomLogitProcessor available - will validate on first request. "
                "Server needs --enable-custom-logit-processor flag."
            )
        else:
            self.logger.info(
                "[SGLang] CustomLogitProcessor not available - using logit_bias fallback"
            )
    
    def _precompute_penalty_token_ids(self):
        """
        Pre-compute token IDs for phrases to penalize in CompressedNoStepsLogitProcessor.
        
        This converts phrase strings to token IDs once at initialization,
        so they can be passed to the server via custom_params.
        """
        # Initialize empty sets (will be populated if tokenizer is available)
        self.intro_token_ids = set()
        self.always_token_ids = set()
        self.list_token_ids = set()
        self.newline_token_ids = set()
        
        if self._tokenizer is None:
            self.logger.warning("[SGLang] No tokenizer available - penalty token IDs not pre-computed")
            return
        
        def collect_token_ids(phrases):
            """Convert phrases into a set of token IDs by encoding them."""
            out = set()
            for p in phrases:
                try:
                    ids = self._tokenizer.encode(p, add_special_tokens=False)
                    for tid in ids:
                        out.add(int(tid))
                except Exception:
                    pass
            return out
        
        # Phrases to penalize early (preamble / "foreplay")
        intro_phrases = [
            # Existing
            "To address", "Additionally", "It's also important", "It is also important",
            "Pay attention to", "Look for", "focus on extracting", "This could include",
            "Please provide", "In this implementation",
            # Meta-commentary starters
            "The task requires", "I will focus", "I was able to", "I can provide",
            "Let me", "I'll", "We need to", "We will", "We can",
            "Based on", "In order to", "To find", "To extract", "To determine",
            # Hedging starters
            "Unfortunately", "However", "Nevertheless", "Therefore",
            # "Not found" verbose patterns
            "The required", "The provided", "The document", "The excerpt",
            "is not explicitly", "is not directly", "was not found",
            "does not contain", "does not include", "does not provide",
        ]
        
        # Discourse/verbosity tokens/phrases to penalize always
        always_phrases = [
            # Existing
            "basically", "actually", "really", "just", "overall", "generally",
            "in summary", "to summarize", "in conclusion", "of course", "certainly", "sure",
            "I think", "I can", "I will",
            # Hedging/transition words
            "however", "unfortunately", "nevertheless", "therefore", "furthermore",
            "additionally", "moreover", "consequently", "hence",
            # Filler phrases
            "it appears", "it seems", "we can infer", "we can assume",
            "might be", "could be", "may be", "likely", "possibly",
            "in other words", "that is to say", "specifically",
            # Empty qualifiers
            "explicitly", "directly", "specifically mentioned",
        ]
        
        # Step-by-step / numbered-list markers
        list_markers = [
            "\n1", "\n2", "\n3", "\n4", "\n5",
            "\n- ", "\n* ", "\n• ",
            "\nFirst", "\nSecond", "\nThird", "\nNext", "\nThen",
            "Step 1", "Step 2", "Step 3",
            "1.", "2.", "3.", "4.", "5.",
            "1)", "2)", "3)", "4)", "5)",
        ]
        
        # Newline markers for tail heuristic
        newline_markers = ["\n", "\n\n"]
        
        self.intro_token_ids = collect_token_ids(intro_phrases)
        self.always_token_ids = collect_token_ids(always_phrases)
        self.list_token_ids = collect_token_ids(list_markers)
        self.newline_token_ids = collect_token_ids(newline_markers)
        
        self.logger.debug(
            f"[SGLang] Pre-computed penalty token IDs: "
            f"intro={len(self.intro_token_ids)}, always={len(self.always_token_ids)}, "
            f"list={len(self.list_token_ids)}, newline={len(self.newline_token_ids)}"
        )
    
    def _generate_with_length_penalty(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        stop_sequences: Optional[List[str]] = None,
        min_tokens: int = 0,
    ) -> Tuple[str, int, int, str]:
        """
        Generate text with per-step length penalty.
        
        Uses CustomLogitProcessor if available (applies penalty each decoding step),
        otherwise falls back to logit_bias (static one-time boost).
        
        Args:
            text: Input text/prompt
            beta: Length penalty strength (EOS boost per step)
            max_new_tokens: Maximum tokens to generate
            stop_sequences: List of stop strings
            min_tokens: Minimum tokens to generate before allowing EOS (default: 0)
            
        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens, request_id)
        """
        # Try CustomLogitProcessor if supported and not yet proven unsupported
        if self._custom_processor_supported:
            result = self._generate_with_custom_processor(text, beta, max_new_tokens, stop_sequences, min_tokens)
            
            # Check if it actually worked
            if result[0] and not result[0].startswith("Error:"):
                if not self._custom_processor_validated:
                    with self._processor_lock:
                        if not self._custom_processor_validated:
                            self._custom_processor_validated = True
                            self.logger.debug("[SGLang] ✓ CustomLogitProcessor validated - per-step length penalty ACTIVE")
                return result
            
            # Custom processor failed - disable it for future requests (thread-safe)
            with self._processor_lock:
                if self._custom_processor_supported:  # Check again inside lock
                    self._custom_processor_supported = False
                    self.logger.warning(
                        "[SGLang] ✗ CustomLogitProcessor FAILED - server may not have --enable-custom-logit-processor. "
                        "Falling back to logit_bias for all future requests."
                    )
        
        # Fallback to logit_bias (static boost)
        return self._generate_with_logit_bias(text, beta, max_new_tokens, stop_sequences, min_tokens)
    
    def _generate_with_custom_processor(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        stop_sequences: Optional[List[str]] = None,
        min_tokens: int = 0,
    ) -> Tuple[str, int, int, str]:
        """
        Generate using CompressedNoStepsLogitProcessor for compressed output.
        
        This passes the serialized processor to SGLang via extra_body, which applies:
        - Anti-truncation (blocks EOS before min_new_tokens)
        - Anti-preamble penalties (early tokens)
        - Anti-verbosity penalties (always-on)
        - Anti-list/steps penalties
        - Repetition penalties
        
        Requires SGLang server started with --enable-custom-logit-processor flag.
        """
        try:
            # Build custom_params with all token IDs and penalties
            custom_params = {
                # Core parameters
                "eos_token_id": self.stop_ids[0] if self.stop_ids else 128001,
                "min_new_tokens": min_tokens,
                # Pre-computed token ID sets
                "intro_token_ids": list(self.intro_token_ids),
                "always_token_ids": list(self.always_token_ids),
                "list_token_ids": list(self.list_token_ids),
                "newline_token_ids": list(self.newline_token_ids),
                # Hardcoded penalty values
                "eos_bonus_after_min": 1.2,
                "intro_window": 32,
                "intro_penalty": 3.5,
                "always_penalty": 0.35,
                "list_penalty": 4.0,
                "repeat_penalty": 0.6,
                "repeat_last_k": 128,
            }
            
            extra_body = {
                "custom_logit_processor": CompressedNoStepsLogitProcessor().to_str(),
                "custom_params": custom_params,
            }
            
            # Add min_tokens to extra_body to prevent immediate EOS
            if min_tokens > 0:
                extra_body["min_tokens"] = min_tokens
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": text}],
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "extra_body": extra_body,
            }
            
            if stop_sequences:
                params["stop"] = stop_sequences
            
            self.logger.debug(f"[SGLang] Using CompressedNoStepsLogitProcessor with min_tokens={min_tokens}")
            response = self.openai_client.chat.completions.create(**params)
            
            generated_text = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            return generated_text, prompt_tokens, completion_tokens, str(uuid.uuid4())
            
        except Exception as e:
            self.logger.error(f"[SGLang] CustomLogitProcessor error: {e}")
            return f"Error: {str(e)}", 0, 0, ""
    
    def _generate_with_logit_bias(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        stop_sequences: Optional[List[str]] = None,
        min_tokens: int = 0,
    ) -> Tuple[str, int, int, str]:
        """Generate using OpenAI-compatible API with logit_bias for EOS boost."""
        try:
            # Build logit_bias dict: boost EOS tokens by beta
            # Scale beta aggressively (20x) to encourage shorter outputs
            # logit_bias range is typically -100 to +100
            logit_bias = {str(tid): min(int(beta * 20), 100) for tid in self.stop_ids}
            self.logger.debug(f"[SGLang] Using logit_bias for EOS boost: {logit_bias}, min_tokens={min_tokens}")
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": text}],
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "logit_bias": logit_bias,
            }
            
            # Add min_tokens via extra_body to prevent immediate EOS
            if min_tokens > 0:
                params["extra_body"] = {"min_tokens": min_tokens}
            
            if stop_sequences:
                params["stop"] = stop_sequences
            
            response = self.openai_client.chat.completions.create(**params)
            
            generated_text = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            
            return generated_text, prompt_tokens, completion_tokens, str(uuid.uuid4())
            
        except Exception as e:
            self.logger.error(f"OpenAI-compatible generate error: {e}")
            return f"Error: {str(e)}", 0, 0, ""
    
    def _generate_job_output_single_shot(
        self,
        prompt: str,
    ) -> Tuple[str, Usage, str]:
        """
        Generate JobOutput in a single API call.
        
        Uses CustomLogitProcessor if available for per-step length penalty,
        otherwise falls back to logit_bias.
        
        Args:
            prompt: The full worker prompt
            
        Returns:
            Tuple of (json_response, usage, done_reason)
        """
        try:
            avg_beta = (self.beta_explanation + self.beta_citation + self.beta_answer) / 3
            max_tokens = self.max_tokens_explanation + self.max_tokens_citation + self.max_tokens_answer
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            
            # Use CompressedNoStepsLogitProcessor if supported, otherwise logit_bias
            avg_min_tokens = (self.min_tokens_explanation + self.min_tokens_citation + self.min_tokens_answer) // 3
            
            if self._custom_processor_supported:
                # Build custom_params with all token IDs and penalties
                custom_params = {
                    # Core parameters
                    "eos_token_id": self.stop_ids[0] if self.stop_ids else 128001,
                    "min_new_tokens": avg_min_tokens,
                    # Pre-computed token ID sets
                    "intro_token_ids": list(self.intro_token_ids),
                    "always_token_ids": list(self.always_token_ids),
                    "list_token_ids": list(self.list_token_ids),
                    "newline_token_ids": list(self.newline_token_ids),
                    # Hardcoded penalty values
                    "eos_bonus_after_min": 1.2,
                    "intro_window": 32,
                    "intro_penalty": 3.5,
                    "always_penalty": 0.35,
                    "list_penalty": 4.0,
                    "repeat_penalty": 0.6,
                    "repeat_last_k": 128,
                }
                
                params["extra_body"] = {
                    "custom_logit_processor": CompressedNoStepsLogitProcessor().to_str(),
                    "custom_params": custom_params,
                    "min_tokens": avg_min_tokens,
                }
                self.logger.debug(f"[SGLang] Single-shot with CompressedNoStepsLogitProcessor, min_tokens={avg_min_tokens}")
            else:
                # Fallback to logit_bias (static boost)
                logit_bias = {str(tid): min(int(avg_beta * 15), 100) for tid in self.stop_ids}
                params["logit_bias"] = logit_bias
                if avg_min_tokens > 0:
                    params["extra_body"] = {"min_tokens": avg_min_tokens}
                self.logger.debug(f"[SGLang] Single-shot with logit_bias: {logit_bias}")
            
            response = self.openai_client.chat.completions.create(**params)
            
            json_response = response.choices[0].message.content or "{}"
            
            # Validate and normalize the JSON
            try:
                parsed = json.loads(json_response)
                result = {
                    "explanation": parsed.get("explanation", ""),
                    "citation": parsed.get("citation"),
                    "answer": parsed.get("answer"),
                }
                json_response = json.dumps(result)
            except json.JSONDecodeError:
                # If JSON is invalid, wrap the response
                result = {
                    "explanation": json_response[:200],
                    "citation": None,
                    "answer": None,
                }
                json_response = json.dumps(result)
            
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            
            return json_response, usage, "stop"
            
        except Exception as e:
            self.logger.error(f"Single-shot generation error: {e}")
            error_result = {
                "explanation": f"Error: {str(e)}",
                "citation": None,
                "answer": None,
            }
            return json.dumps(error_result), Usage(), "error"
    
    def _parse_worker_prompt(self, prompt: str) -> Tuple[str, str, str]:
        """
        Parse a worker prompt to extract context, task, and advice.
        
        The prompt follows the WORKER_PROMPT_SHORT format with sections
        separated by "--------------------------------".
        
        Args:
            prompt: The full worker prompt
            
        Returns:
            Tuple of (context, task, advice)
        """
        separator = "-" * 32
        
        # Default values
        context = ""
        task = ""
        advice = ""
        
        try:
            # Split by separator
            parts = prompt.split(separator)
            
            if len(parts) >= 1:
                # First part contains context
                context_part = parts[0]
                # Remove "Here is a document excerpt:" prefix if present
                if "document excerpt:" in context_part.lower():
                    context = context_part.split(":", 1)[-1].strip()
                else:
                    context = context_part.strip()
            
            if len(parts) >= 2:
                # Second part contains task
                task_part = parts[1]
                # Remove "And here is your task:" prefix if present
                if "task:" in task_part.lower():
                    task = task_part.split(":", 1)[-1].strip()
                else:
                    task = task_part.strip()
            
            if len(parts) >= 3:
                # Third part contains advice
                advice_part = parts[2]
                # Remove "And here is additional higher-level advice..." prefix if present
                if "advice" in advice_part.lower():
                    advice = advice_part.split(":", 1)[-1].strip()
                else:
                    advice = advice_part.strip()
            
        except Exception as e:
            self.logger.warning(f"[SGLang] Error parsing worker prompt: {e}")
            # Return the whole prompt as context if parsing fails
            context = prompt
        
        return context, task, advice
    
    def _generate_job_output_sequential(
        self,
        prompt: str,
    ) -> Tuple[str, Usage, str]:
        """
        Generate JobOutput using 3 sequential API calls for KV cache reuse.
        
        This generates explanation, citation, and answer sequentially, with each
        subsequent call building on the previous results. All calls share a common
        prefix (context + task + advice) which SGLang's RadixAttention caches.
        
        Each field uses its own beta value for per-step length penalty via
        CustomLogitProcessor (or logit_bias fallback).
        
        Args:
            prompt: The full worker prompt (will be parsed to extract context, task, advice)
            
        Returns:
            Tuple of (json_response, usage, done_reason)
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Parse the incoming prompt to extract components
        context, task, advice = self._parse_worker_prompt(prompt)
        
        self.logger.debug(f"[SGLang] Sequential generation: context={len(context)} chars, task={len(task)} chars")
        
        # Step 1: Generate explanation
        explanation_prompt = WORKER_PROMPT_EXPLANATION.format(
            context=context,
            task=task,
            advice=advice,
        )
        
        explanation, pt1, ct1, _ = self._generate_with_length_penalty(
            text=explanation_prompt,
            beta=self.beta_explanation,
            max_new_tokens=self.max_tokens_explanation,
            min_tokens=self.min_tokens_explanation,
        )
        total_prompt_tokens += pt1
        total_completion_tokens += ct1
        
        # Clean up explanation (remove "None" variations)
        explanation = explanation.strip()
        if explanation.lower() in ("none", "none.", '"none"', "'none'"):
            explanation = ""
        
        self.logger.debug(f"[SGLang] Generated explanation: {len(explanation)} chars")
        
        # Step 2: Generate citation (with explanation context)
        citation_prompt = WORKER_PROMPT_CITATION.format(
            context=context,
            task=task,
            advice=advice,
            explanation=explanation,
        )
        
        citation, pt2, ct2, _ = self._generate_with_length_penalty(
            text=citation_prompt,
            beta=self.beta_citation,
            max_new_tokens=self.max_tokens_citation,
            min_tokens=self.min_tokens_citation,
        )
        total_prompt_tokens += pt2
        total_completion_tokens += ct2
        
        # Clean up citation
        citation = citation.strip()
        if citation.lower() in ("none", "none.", '"none"', "'none'"):
            citation = None
        
        self.logger.debug(f"[SGLang] Generated citation: {len(citation) if citation else 0} chars")
        
        # Step 3: Generate answer (with explanation and citation context)
        answer_prompt = WORKER_PROMPT_ANSWER.format(
            context=context,
            task=task,
            advice=advice,
            explanation=explanation,
            citation=citation or "None",
        )
        
        answer, pt3, ct3, _ = self._generate_with_length_penalty(
            text=answer_prompt,
            beta=self.beta_answer,
            max_new_tokens=self.max_tokens_answer,
            min_tokens=self.min_tokens_answer,
        )
        total_prompt_tokens += pt3
        total_completion_tokens += ct3
        
        # Clean up answer
        answer = answer.strip()
        if answer.lower() in ("none", "none.", '"none"', "'none'"):
            answer = None
        
        self.logger.debug(f"[SGLang] Generated answer: {len(answer) if answer else 0} chars")
        
        # Combine into JSON response
        result = {
            "explanation": explanation or "No relevant information found.",
            "citation": citation,
            "answer": answer,
        }
        json_response = json.dumps(result)
        
        usage = Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
        )
        
        self.logger.debug(
            f"[SGLang] Sequential generation complete: "
            f"prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens}"
        )
        
        return json_response, usage, "stop"
    
    def _is_batch_of_single_prompts(self, messages) -> bool:
        """
        Check if messages is a list of individual prompts (batch mode).
        
        Batch mode: List of individual user message dicts, each to be processed separately.
        e.g., [{"role": "user", "content": "..."}, {"role": "user", "content": "..."}]
        
        This matches the format used by Minions when sending worker_chats.
        """
        if not isinstance(messages, list) or len(messages) <= 1:
            return False
        
        return all(
            isinstance(m, dict) and m.get('role') == 'user' and 'content' in m
            for m in messages
        )
    
    def _group_by_prefix(self, messages: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Group message indices by shared prefix (chunk content) for KV cache optimization.
        
        SGLang's RadixAttention automatically detects shared prefixes across concurrent
        requests and reuses KV cache. By grouping jobs that share the same chunk together,
        we maximize the KV cache hit rate.
        
        Args:
            messages: List of user message dicts
            
        Returns:
            List of groups, where each group is a list of message indices sharing the same prefix
        """
        prefix_groups = {}
        separator = '-' * 30
        
        for idx, msg in enumerate(messages):
            content = msg.get('content', '')
            if separator in content:
                prefix = content.split(separator)[0]
            else:
                prefix = content[:500]
            
            prefix_key = hash(prefix)
            if prefix_key not in prefix_groups:
                prefix_groups[prefix_key] = []
            prefix_groups[prefix_key].append(idx)
        
        return list(prefix_groups.values())
    
    def _single_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[str, Usage, str]:
        """
        Execute a single chat request with WFSA length prior.
        
        Generates JobOutput fields sequentially (explanation, citation, answer)
        with per-field beta values, reusing KV cache between generations.
        
        Args:
            messages: List of message dicts for the conversation
            **kwargs: Additional arguments for the API call
            
        Returns:
            Tuple of (response_text, usage, done_reason)
        """
        # Extract prompt from messages
        if len(messages) == 1 and messages[0].get('role') == 'user':
            prompt = messages[0].get('content', '')
        else:
            # Format as chat for multi-turn
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt += f"System: {content}\n\n"
                elif role == 'user':
                    prompt += f"User: {content}\n\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        # Generate structured output using the configured strategy
        if self.structured_output_schema is not None:
            if self._generation_strategy == "single_shot":
                # Single API call for all fields
                json_response, usage, done_reason = self._generate_job_output_single_shot(prompt)
            else:
                # Sequential: 3 separate API calls for explanation, citation, answer
                json_response, usage, done_reason = self._generate_job_output_sequential(prompt)
            # Post-process JSON output to ensure fields are strings, not dicts
            if json_response:
                json_response = _postprocess_json_output(json_response)
            return json_response, usage, done_reason
        else:
            # For non-structured output, use single generation with default beta
            text, pt, ct, _ = self._generate_with_length_penalty(
                text=prompt,
                beta=self.beta_answer,  # Use answer beta as default
                max_new_tokens=self.max_tokens,
                min_tokens=self.min_tokens_answer,  # Use answer min_tokens as default
            )
            usage = Usage(prompt_tokens=pt, completion_tokens=ct)
            return text, usage, "stop"
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions with WFSA length prior, supporting batch processing.
        
        For batch mode (list of single-message conversations), requests are
        sent in parallel via ThreadPoolExecutor. SGLang's RadixAttention
        automatically detects shared prefixes and reuses KV cache.
        
        Args:
            messages: Either a single conversation (list of message dicts)
                     or batch mode (list of single-message conversations)
            **kwargs: Additional arguments for the API call
            
        Returns:
            Tuple of (list of response strings, total usage, done_reasons)
        """
        # BATCH MODE: Handle list of individual prompts in parallel
        if self._is_batch_of_single_prompts(messages):
            self.logger.debug(f"[SGLang] Batch mode: Processing {len(messages)} prompts in parallel")
            
            responses = [None] * len(messages)
            usage_total = Usage()
            done_reasons = ["stop"] * len(messages)
            
            # Group messages by chunk prefix for optimal KV cache reuse
            prefix_groups = self._group_by_prefix(messages)
            self.logger.debug(f"[SGLang] Grouped into {len(prefix_groups)} prefix groups for KV cache optimization")
            
            max_workers = min(len(messages), self.max_workers)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                for group in prefix_groups:
                    for idx in group:
                        msg = messages[idx]
                        future = executor.submit(self._single_chat, [msg], **kwargs)
                        future_to_idx[future] = idx
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        response_text, usage, done_reason = future.result()
                        responses[idx] = response_text
                        usage_total += usage
                        done_reasons[idx] = done_reason
                    except Exception as e:
                        self.logger.error(f"Worker {idx} failed: {e}")
                        responses[idx] = json.dumps({
                            "explanation": f"Error: {str(e)}",
                            "citation": None,
                            "answer": None,
                        })
                        done_reasons[idx] = "error"
            
            self.logger.debug(f"[SGLang] Batch complete: {len([r for r in responses if r])} responses")
            return responses, usage_total, done_reasons
        
        # SINGLE CONVERSATION MODE
        response_text, usage, done_reason = self._single_chat(messages, **kwargs)
        return [response_text], usage, [done_reason]
    
    def schat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Synchronous chat with done_reasons (for compatibility with MinionS).
        
        Args:
            messages: Single conversation or batch of conversations
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (responses, usage, done_reasons)
        """
        return self.chat(messages, **kwargs)
    
    def check_server_health(self) -> bool:
        """
        Check if SGLang server is running and responsive.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            health_url = f"{self.base_url}/health"
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"SGLang health check failed: {e}")
            return False
