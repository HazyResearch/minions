"""
SGLang Client for MinionS.

SGLang provides RadixAttention which automatically shares KV cache across
parallel requests with common prefixes (document chunks). This is ideal for
the MinionS protocol where multiple operations run on the same chunks.

Usage:
    # Start SGLang server:
    # python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 8000
    
    client = SGLangClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
        temperature=0.2
    )
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from minions.usage import Usage
from minions.clients.base import MinionsClient


class SGLangClient(MinionsClient):
    """
    Client for SGLang inference server with RadixAttention KV cache sharing.
    
    SGLang exposes an OpenAI-compatible API at /v1/chat/completions.
    RadixAttention automatically detects shared prefixes across parallel
    requests and reuses KV cache, significantly speeding up workloads
    where multiple operations target the same document chunks.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        api_key: Optional[str] = None,
        max_workers: int = 32,
        structured_output_schema=None,
        **kwargs
    ):
        """
        Initialize SGLang client.
        
        Args:
            model_name: Model name (should match what SGLang server loaded)
            base_url: SGLang server URL (default: http://localhost:8000/v1)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum tokens to generate (default: 2048)
            api_key: API key (optional, SGLang usually doesn't require one)
            max_workers: Maximum parallel workers for batch requests (default: 32)
            structured_output_schema: Optional Pydantic model for structured JSON output
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            local=True,  # SGLang is always a local backend
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.base_url = base_url or os.getenv("SGLANG_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("SGLANG_API_KEY", "not-needed")
        self.max_workers = max_workers
        self.structured_output_schema = structured_output_schema
        
        # Initialize OpenAI client pointing to SGLang server
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.logger.info(f"SGLang client initialized: {self.base_url}, model={model_name}")
    
    def _is_batch_of_single_prompts(self, messages) -> bool:
        """
        Check if messages is a list of individual prompts (batch mode).
        
        Batch mode: List of individual user message dicts, each to be processed separately.
        e.g., [{"role": "user", "content": "..."}, {"role": "user", "content": "..."}]
        
        This matches the format used by Minions when sending worker_chats.
        """
        if not isinstance(messages, list) or len(messages) <= 1:
            return False
        
        # Check if all messages are individual user prompts (batch mode pattern from Minions)
        # In batch mode, each message is a single {"role": "user", "content": "..."} dict
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
        
        The Minions worker prompt format is:
            "Here is a document excerpt:\n\n{context}\n\n---...---\nAnd here is your task..."
        
        Jobs on the same chunk share the same {context} prefix.
        
        Args:
            messages: List of user message dicts
            
        Returns:
            List of groups, where each group is a list of message indices sharing the same prefix
        """
        prefix_groups = {}
        separator = '-' * 30  # The separator used in worker prompt template
        
        for idx, msg in enumerate(messages):
            content = msg.get('content', '')
            # Extract the prefix (everything before the first separator, which is the chunk)
            if separator in content:
                prefix = content.split(separator)[0]
            else:
                # Fallback: use first 500 chars as prefix identifier
                prefix = content[:500]
            
            # Use hash of prefix as key to handle large chunks efficiently
            prefix_key = hash(prefix)
            if prefix_key not in prefix_groups:
                prefix_groups[prefix_key] = []
            prefix_groups[prefix_key].append(idx)
        
        return list(prefix_groups.values())
    
    def _single_chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[str, Usage, str]:
        """
        Execute a single chat request.
        
        Args:
            messages: List of message dicts for the conversation
            **kwargs: Additional arguments for the API call
            
        Returns:
            Tuple of (response_text, usage, done_reason)
        """
        try:
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # Force JSON output when structured schema is set
            if self.structured_output_schema is not None:
                params["response_format"] = {"type": "json_object"}
            
            # Add response_format if explicitly provided (overrides schema setting)
            if "response_format" in kwargs:
                params["response_format"] = kwargs["response_format"]
            
            response = self.client.chat.completions.create(**params)
            
            output_text = response.choices[0].message.content or ""
            
            # Extract finish_reason (maps to done_reason)
            done_reason = response.choices[0].finish_reason or "stop"
            
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            
            return output_text, usage, done_reason
            
        except Exception as e:
            self.logger.error(f"SGLang chat error: {e}")
            return f"Error: {str(e)}", Usage(), "error"
    
    def chat(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        **kwargs
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle chat completions, with support for batch processing.
        
        For batch mode (list of single-message conversations), requests are
        sent in parallel via ThreadPoolExecutor. SGLang's RadixAttention
        automatically detects shared prefixes and reuses KV cache.
        
        Jobs are grouped by chunk prefix to maximize KV cache hits - requests
        sharing the same document chunk are processed together so RadixAttention
        can reuse the cached KV states for the common prefix.
        
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
            # Jobs on the same chunk will be processed together, allowing
            # RadixAttention to share the KV cache for the common prefix
            prefix_groups = self._group_by_prefix(messages)
            self.logger.info(f"[SGLang] Grouped into {len(prefix_groups)} prefix groups for KV cache optimization")
            
            # Use ThreadPoolExecutor for parallel processing
            # SGLang's RadixAttention handles KV cache sharing automatically
            max_workers = min(len(messages), self.max_workers)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs - grouped by prefix but processed in parallel
                # RadixAttention will detect shared prefixes and reuse KV cache
                future_to_idx = {}
                for group in prefix_groups:
                    for idx in group:
                        msg = messages[idx]
                        # Wrap single message in list for _single_chat
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
                        responses[idx] = f"Error: {str(e)}"
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
        
        This method now simply calls chat() which returns 3 values.
        
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
            import requests
            health_url = self.base_url.replace("/v1", "/health")
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"SGLang health check failed: {e}")
            return False
