"""
SGLang Client for MinionS with WFSA Length Prior.

SGLang provides RadixAttention which automatically shares KV cache across
parallel requests with common prefixes (document chunks). This client uses
SGLang's native /generate endpoint with custom logit processors to implement
a WFSA-based length prior that encourages shorter outputs.

Usage:
    # Start SGLang server with custom logit processor enabled:
    # python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
    #   --port 8000 --enable-custom-logit-processor
    
    client = SGLangClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000",
        temperature=0.2,
        beta_explanation=1.0,
        beta_citation=2.0,
        beta_answer=1.5,
    )
"""

import json
import logging
import os
import pickle
import base64
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import openai
from transformers import AutoTokenizer

from minions.usage import Usage
from minions.clients.base import MinionsClient


class LengthPriorWFSA:
    """
    WFSA (Weighted Finite-State Acceptor) for length prior.
    
    Implements a 1-state WFSA that assigns a length prior:
      - States: one nonfinal state (s), one final state (f)
      - Transitions:
        - For any non-stop token: s -> s with weight -c
        - For stop token (EOS/EOT): s -> f with weight 0
    
    This defines log q(y) = -c * |y| (shorter strings get higher weight).
    When decoded with product-of-experts:
        argmax log p_LM(y|x) + lambda * log q(y)
    
    We just add a constant EOS boost (beta = lambda * c) at every step,
    which is equivalent to subtracting beta from all non-stop tokens.
    """
    
    def __call__(self, logits, custom_param_list):
        """
        Apply WFSA length prior by boosting stop token logits.
        
        Args:
            logits: Tensor of shape [batch_size, vocab_size]
            custom_param_list: List of dicts with 'beta' and 'stop_ids' per request
            
        Returns:
            Modified logits with EOS boost applied
        """
        assert logits.shape[0] == len(custom_param_list)
        
        for i, params in enumerate(custom_param_list):
            beta = float(params.get("beta", 2.0))
            stop_ids = params.get("stop_ids", [])
            for tid in stop_ids:
                logits[i, int(tid)] += beta
        
        return logits
    
    def to_str(self) -> str:
        """Serialize the processor for transmission to SGLang server."""
        return base64.b64encode(pickle.dumps(self)).decode()


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
        
        # Initialize tokenizer to get stop token IDs
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.eos_token_id = self.tokenizer.eos_token_id
            # Get additional stop tokens for Llama-3 instruct format
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            self.stop_ids = [self.eos_token_id]
            if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
                self.stop_ids.append(eot_id)
        except Exception as e:
            self.logger.warning(f"Could not load tokenizer: {e}. Using default stop IDs.")
            self.tokenizer = None
            self.eos_token_id = 128001  # Llama-3 default EOS
            self.stop_ids = [128001, 128009]  # EOS and EOT for Llama-3
        
        # Create WFSA processor
        self.wfsa_processor = LengthPriorWFSA()
        self.wfsa_processor_str = self.wfsa_processor.to_str()
        
        # Track if WFSA is supported - default to False since SGLang's custom
        # logit processor API requires specific JSON format that varies by version.
        # The OpenAI-compatible fallback with logit_bias works reliably.
        self._wfsa_supported = False
        
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
            f"SGLang client initialized: {self.base_url}, model={model_name}, "
            f"betas=(exp={beta_explanation}, cit={beta_citation}, ans={beta_answer})"
        )
    
    def _generate_with_wfsa(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        stop_sequences: Optional[List[str]] = None,
        rid: Optional[str] = None,
    ) -> Tuple[str, int, int, str]:
        """
        Generate text using SGLang with WFSA length prior.
        
        Tries native /generate endpoint with custom logit processor first.
        Falls back to OpenAI-compatible API with logit_bias if native fails.
        
        Args:
            text: Input text/prompt
            beta: WFSA strength (EOS boost)
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS allowed
            stop_sequences: List of stop strings
            rid: Request ID for continuation (KV cache reuse)
            
        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens, new_rid)
        """
        # Try native endpoint with WFSA if supported
        if self._wfsa_supported:
            result = self._try_native_generate(text, beta, max_new_tokens, min_new_tokens, stop_sequences, rid)
            if result is not None:
                return result
            # Mark WFSA as not supported for future calls
            self._wfsa_supported = False
            self.logger.warning("WFSA not supported, falling back to OpenAI-compatible API with logit_bias")
        
        # Fallback to OpenAI-compatible API with logit_bias
        return self._generate_with_logit_bias(text, beta, max_new_tokens, stop_sequences)
    
    def _try_native_generate(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        min_new_tokens: int,
        stop_sequences: Optional[List[str]],
        rid: Optional[str],
    ) -> Optional[Tuple[str, int, int, str]]:
        """Try native /generate endpoint with custom logit processor."""
        url = f"{self.base_url}/generate"
        
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": min_new_tokens,
                "custom_params": {
                    "beta": beta,
                    "stop_ids": self.stop_ids,
                },
            },
            "custom_logit_processor": self.wfsa_processor_str,
        }
        
        if stop_sequences:
            payload["sampling_params"]["stop"] = stop_sequences
        
        if rid:
            payload["rid"] = rid
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code != 200:
                error_detail = response.text[:500] if response.text else "No error details"
                self.logger.error(f"Native generate failed ({response.status_code}): {error_detail}")
                return None
            
            result = response.json()
            generated_text = result.get("text", "")
            meta = result.get("meta_info", {})
            prompt_tokens = meta.get("prompt_tokens", 0)
            completion_tokens = meta.get("completion_tokens", 0)
            new_rid = meta.get("id", rid or str(uuid.uuid4()))
            
            return generated_text, prompt_tokens, completion_tokens, new_rid
            
        except Exception as e:
            self.logger.error(f"Native generate exception: {e}")
            return None
    
    def _generate_with_logit_bias(
        self,
        text: str,
        beta: float,
        max_new_tokens: int,
        stop_sequences: Optional[List[str]] = None,
    ) -> Tuple[str, int, int, str]:
        """Generate using OpenAI-compatible API with logit_bias for EOS boost."""
        try:
            # Build logit_bias dict: boost EOS tokens by beta
            logit_bias = {str(tid): int(beta * 10) for tid in self.stop_ids}  # Scale beta
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": text}],
                "temperature": self.temperature,
                "max_tokens": max_new_tokens,
                "logit_bias": logit_bias,
            }
            
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
    
    def _generate_job_output_sequential(
        self,
        prompt: str,
    ) -> Tuple[str, Usage, str]:
        """
        Generate JobOutput by generating each field sequentially with WFSA.
        
        If WFSA (native /generate endpoint) is supported, generates fields one at a time
        reusing the KV cache between generations for efficiency.
        
        If WFSA is not supported, falls back to single-shot JSON generation.
        
        Args:
            prompt: The full worker prompt
            
        Returns:
            Tuple of (json_response, usage, done_reason)
        """
        # If WFSA is not supported, use single-shot generation
        if not self._wfsa_supported:
            return self._generate_job_output_single_shot(prompt)
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Start with the prompt and JSON opening
        current_text = prompt + '\n{"explanation": "'
        rid = None
        
        # Generate explanation field
        explanation, pt, ct, rid = self._generate_with_wfsa(
            text=current_text,
            beta=self.beta_explanation,
            max_new_tokens=self.max_tokens_explanation,
            min_new_tokens=self.min_tokens_explanation,
            stop_sequences=['"', '\n'],
            rid=rid,
        )
        total_prompt_tokens += pt
        total_completion_tokens += ct
        
        # If WFSA failed and we switched to fallback, use single-shot for rest
        if not self._wfsa_supported:
            return self._generate_job_output_single_shot(prompt)
        
        # Clean up explanation (remove trailing quote if present)
        explanation = explanation.rstrip('"').strip()
        
        # Continue with citation field
        current_text = current_text + explanation + '", "citation": "'
        citation, pt, ct, rid = self._generate_with_wfsa(
            text=current_text,
            beta=self.beta_citation,
            max_new_tokens=self.max_tokens_citation,
            min_new_tokens=self.min_tokens_citation,
            stop_sequences=['"', '\n'],
            rid=rid,
        )
        total_prompt_tokens += pt
        total_completion_tokens += ct
        
        # Clean up citation
        citation = citation.rstrip('"').strip()
        if citation.lower() in ['none', 'null', '']:
            citation = None
        
        # Continue with answer field
        if citation:
            current_text = current_text + citation + '", "answer": "'
        else:
            current_text = current_text + 'null, "answer": "'
        
        answer, pt, ct, rid = self._generate_with_wfsa(
            text=current_text,
            beta=self.beta_answer,
            max_new_tokens=self.max_tokens_answer,
            min_new_tokens=self.min_tokens_answer,
            stop_sequences=['"', '\n', '}'],
            rid=rid,
        )
        total_prompt_tokens += pt
        total_completion_tokens += ct
        
        # Clean up answer
        answer = answer.rstrip('"').rstrip('}').strip()
        if answer.lower() in ['none', 'null', '']:
            answer = None
        
        # Assemble final JSON
        result = {
            "explanation": explanation,
            "citation": citation,
            "answer": answer,
        }
        
        json_response = json.dumps(result)
        usage = Usage(prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens)
        
        return json_response, usage, "stop"
    
    def _generate_job_output_single_shot(
        self,
        prompt: str,
    ) -> Tuple[str, Usage, str]:
        """
        Generate JobOutput in a single API call (fallback when WFSA not available).
        
        Uses OpenAI-compatible API with response_format=json_object.
        
        Args:
            prompt: The full worker prompt
            
        Returns:
            Tuple of (json_response, usage, done_reason)
        """
        try:
            # Build logit_bias with average beta to encourage shorter outputs
            avg_beta = (self.beta_explanation + self.beta_citation + self.beta_answer) / 3
            logit_bias = {str(tid): int(avg_beta * 5) for tid in self.stop_ids}
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens_explanation + self.max_tokens_citation + self.max_tokens_answer,
                response_format={"type": "json_object"},
                logit_bias=logit_bias,
            )
            
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
        
        # Generate with WFSA using sequential field generation
        if self.structured_output_schema is not None:
            return self._generate_job_output_sequential(prompt)
        else:
            # For non-structured output, use single generation with default beta
            text, pt, ct, _ = self._generate_with_wfsa(
                text=prompt,
                beta=self.beta_answer,  # Use answer beta as default
                max_new_tokens=self.max_tokens,
                min_new_tokens=0,
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
            self.logger.info(f"[SGLang] Batch mode: Processing {len(messages)} prompts in parallel")
            
            responses = [None] * len(messages)
            usage_total = Usage()
            done_reasons = ["stop"] * len(messages)
            
            # Group messages by chunk prefix for optimal KV cache reuse
            prefix_groups = self._group_by_prefix(messages)
            self.logger.info(f"[SGLang] Grouped into {len(prefix_groups)} prefix groups for KV cache optimization")
            
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
            
            self.logger.info(f"[SGLang] Batch complete: {len([r for r in responses if r])} responses")
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
