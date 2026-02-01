"""
SGLang Client for MinionS with External Logit Processor Support.

Supports external CustomLogitProcessor classes for constrained decoding (e.g., 
reducing verbosity, penalizing certain token patterns). The processor can be
provided at initialization or set afterwards.

SGLang provides RadixAttention which automatically shares KV cache across
parallel requests with common prefixes (document chunks).

Usage without logit processor (uses logit_bias fallback):
    client = SGLangClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000",
    )

Usage with external logit processor:
    from learned_logit_processor import LearnedBloatAxisProcessor
    
    client = SGLangClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000",
        logit_processor_class=LearnedBloatAxisProcessor,
    )
    
    # Optionally override token ID sets after creation:
    client.intro_token_ids = LearnedBloatAxisProcessor.EARLY_PHASE_TOKEN_IDS
    client.always_token_ids = LearnedBloatAxisProcessor.ALWAYS_TOKEN_IDS
    client.list_token_ids = LearnedBloatAxisProcessor.AFTER_NEWLINE_TOKEN_IDS
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
    Client for SGLang inference server with optional external logit processor.
    
    Supports external CustomLogitProcessor classes for constrained decoding.
    The processor can be provided via the `logit_processor_class` parameter,
    and token ID sets can be configured via `intro_token_ids`, `always_token_ids`,
    `list_token_ids`, and `newline_token_ids` attributes.
    
    If no logit processor is provided, falls back to logit_bias for EOS boosting.
    
    Generates JobOutput fields (explanation, citation, answer) sequentially to
    maximize KV cache reuse via SGLang's RadixAttention.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_workers: int = 32,
        structured_output_schema=None,
        # External logit processor class (optional)
        logit_processor_class=None,  # CustomLogitProcessor subclass from external module
        **kwargs
    ):
        """
        Initialize SGLang client with optional external logit processor.
        
        Args:
            model_name: Model name (should match what SGLang server loaded)
            base_url: SGLang server URL (default: http://localhost:8000)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum tokens to generate for full response (default: 2048)
            max_workers: Maximum parallel workers for batch requests (default: 32)
            structured_output_schema: Optional Pydantic model for structured JSON output
            logit_processor_class: Optional CustomLogitProcessor subclass for constrained decoding.
                                   If provided, must have .to_str() method and .get_default_params() classmethod.
                                   Token ID sets (intro_token_ids, etc.) can be set after initialization.
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
        
        # External logit processor class (can be set to enable custom constraint decoding)
        self.logit_processor_class = logit_processor_class
        
        # Token ID sets for constraint decoding (can be set externally after initialization)
        # These are passed to the logit processor via custom_params
        self.intro_token_ids = set()
        self.always_token_ids = set()
        self.list_token_ids = set()
        self.newline_token_ids = set()
        
        # Track whether server supports CustomLogitProcessor
        # Will be validated on first request
        self._custom_processor_supported = HAS_CUSTOM_LOGIT_PROCESSOR and logit_processor_class is not None
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
            f"[SGLang] Client initialized: {self.base_url}, model={model_name}, "
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
        Generate using external CustomLogitProcessor for constrained decoding.
        
        Uses the logit_processor_class provided at initialization (or set afterwards).
        The processor should have:
        - .to_str() method for serialization to SGLang server
        - .get_default_params() classmethod for default custom_params
        
        Token ID sets (intro_token_ids, etc.) are passed via custom_params.
        
        Requires SGLang server started with --enable-custom-logit-processor flag.
        """
        try:
            if self.logit_processor_class is None:
                return f"Error: No logit_processor_class configured", 0, 0, ""
            
            # Get default params from the processor class if available
            if hasattr(self.logit_processor_class, 'get_default_params'):
                custom_params = self.logit_processor_class.get_default_params()
            else:
                custom_params = {}
            
            # Override with instance-level settings
            custom_params["eos_token_id"] = self.stop_ids[0] if self.stop_ids else 128001
            custom_params["min_new_tokens"] = min_tokens
            
            # Use externally-set token ID sets (if any)
            if self.intro_token_ids:
                custom_params["intro_token_ids"] = list(self.intro_token_ids)
            if self.always_token_ids:
                custom_params["always_token_ids"] = list(self.always_token_ids)
            if self.list_token_ids:
                custom_params["list_token_ids"] = list(self.list_token_ids)
            if self.newline_token_ids:
                custom_params["newline_token_ids"] = list(self.newline_token_ids)
            
            extra_body = {
                "custom_logit_processor": self.logit_processor_class().to_str(),
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
            
            self.logger.debug(f"[SGLang] Using external logit processor with min_tokens={min_tokens}")
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
            avg_beta = 1.5  # Default beta for length penalty
            max_tokens = 450  # Default max tokens for all fields combined
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            
            # Use external logit processor if supported, otherwise logit_bias
            avg_min_tokens = 6  # Default minimum tokens before EOS
            
            if self._custom_processor_supported and self.logit_processor_class is not None:
                # Get default params from the processor class if available
                if hasattr(self.logit_processor_class, 'get_default_params'):
                    custom_params = self.logit_processor_class.get_default_params()
                else:
                    custom_params = {}
                
                # Override with instance-level settings
                custom_params["eos_token_id"] = self.stop_ids[0] if self.stop_ids else 128001
                custom_params["min_new_tokens"] = avg_min_tokens
                
                # Use externally-set token ID sets (if any)
                if self.intro_token_ids:
                    custom_params["intro_token_ids"] = list(self.intro_token_ids)
                if self.always_token_ids:
                    custom_params["always_token_ids"] = list(self.always_token_ids)
                if self.list_token_ids:
                    custom_params["list_token_ids"] = list(self.list_token_ids)
                if self.newline_token_ids:
                    custom_params["newline_token_ids"] = list(self.newline_token_ids)
                
                params["extra_body"] = {
                    "custom_logit_processor": self.logit_processor_class().to_str(),
                    "custom_params": custom_params,
                    "min_tokens": avg_min_tokens,
                }
                self.logger.debug(f"[SGLang] Single-shot with external logit processor, min_tokens={avg_min_tokens}")
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
        
        # Generate structured output
        if self.structured_output_schema is not None:
            json_response, usage, done_reason = self._generate_job_output_single_shot(prompt)
            # Post-process JSON output to ensure fields are strings, not dicts
            if json_response:
                json_response = _postprocess_json_output(json_response)
            return json_response, usage, done_reason
        else:
            # For non-structured output, use single generation
            text, pt, ct, _ = self._generate_with_length_penalty(
                text=prompt,
                beta=1.0,
                max_new_tokens=self.max_tokens,
                min_tokens=3,
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
