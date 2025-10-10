"""
Arch Router utility for intelligent model selection.

Wrapper around katanemo/Arch-Router-1.5B for routing queries to appropriate models.
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any
import os

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _escape_newlines_in_strings(json_str: str) -> str:
    """Escape newline characters within JSON string values."""
    return re.sub(
        r'(".*?")',
        lambda m: m.group(1).replace("\n", "\\n"),
        json_str,
        flags=re.DOTALL,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from text that may be wrapped in markdown code blocks."""
    block_matches = list(re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL))
    bracket_matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))

    if block_matches:
        json_str = block_matches[-1].group(1).strip()
    elif bracket_matches:
        json_str = bracket_matches[-1].group(0)
    else:
        json_str = text

    json_str = _escape_newlines_in_strings(json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {json_str}")
        raise


class ArchRouter:
    """
    Wrapper for Arch-Router-1.5B model for intelligent query routing.

    Routes queries to appropriate language models based on domain and task characteristics.
    """

    def __init__(
        self,
        model_name: str = "katanemo/Arch-Router-1.5B",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the Arch Router model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            cache_dir: Optional directory to cache model files
            load_in_8bit: Whether to load model in 8-bit precision
            verbose: Enable verbose logging
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for ArchRouter. "
                "Install with: pip install torch transformers"
            )

        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.logger.info(f"Loading Arch Router model: {model_name} on {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )

            if load_in_8bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
            else:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device)

            self.model.eval()
            self.logger.info(f"Successfully loaded Arch Router on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load Arch Router model: {e}")
            raise

    def route(
        self,
        query: str,
        available_clients: Dict[str, Dict[str, str]],
        return_scores: bool = False,
        temperature: float = 0.1,
        max_new_tokens: int = 256
    ) -> Dict[str, Any]:
        """
        Route a query to the most appropriate client using Arch-Router's XML format.

        Args:
            query: User query to route
            available_clients: Dict mapping client names to metadata
                Expected format: {"name": str, "description": str}
                Legacy format: {"capabilities": str, "domain": str, "cost": str} (auto-converted)
            return_scores: Whether to include confidence scores
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with routing decision: {"route": str, "reasoning": str (optional)}
        """
        if not available_clients:
            raise ValueError("available_clients cannot be empty")

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        # Convert metadata to Arch-Router expected format
        route_config = []
        for name, metadata in available_clients.items():
            # Support both new format (name/description) and legacy format (capabilities/domain/cost)
            if "description" in metadata:
                description = metadata["description"]
            else:
                # Legacy format conversion
                capabilities = metadata.get("capabilities", "General purpose model")
                domain = metadata.get("domain", "")
                cost = metadata.get("cost", "")

                desc_parts = [capabilities]
                if domain:
                    desc_parts.append(f"Domain: {domain}")
                if cost:
                    desc_parts.append(f"Cost: {cost}")
                description = ". ".join(desc_parts)

            route_config.append({
                "name": name,
                "description": description
            })

        # Format conversation in Arch-Router expected format
        conversation = [{"role": "user", "content": query}]

        # Build prompt using Arch-Router's XML-tagged format
        routes_json = json.dumps(route_config, indent=2)
        conversation_json = json.dumps(conversation, indent=2)

        TASK_INSTRUCTION = f"""You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>
{routes_json}
</routes>

<conversation>
{conversation_json}
</conversation>"""

        FORMAT_PROMPT = """Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags. Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent.
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"}"""

        prompt = f"{TASK_INSTRUCTION}\n\n{FORMAT_PROMPT}"

        self.logger.info(f"Routing query: {query[:100]}...")
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Routes: {routes_json}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract JSON from response
            json_start = response.find("{")
            if json_start != -1:
                response_json_part = response[json_start:]
            else:
                response_json_part = response

            routing_decision = _extract_json(response_json_part)

            selected_route = routing_decision.get("route", "")

            # Handle "other" route or invalid selection
            if selected_route == "other" or selected_route not in available_clients:
                if selected_route == "other":
                    self.logger.info("Arch Router returned 'other' - using fallback")
                else:
                    self.logger.warning(
                        f"Arch Router selected '{selected_route}' which is not in available_clients. "
                        f"Available: {list(available_clients.keys())}. Falling back to first available."
                    )
                routing_decision["route"] = list(available_clients.keys())[0]
                routing_decision["fallback"] = True

            self.logger.info(f"Routed to: {routing_decision.get('route')}")

            return routing_decision

        except Exception as e:
            self.logger.error(f"Error during routing: {e}")
            fallback_client = list(available_clients.keys())[0]
            self.logger.warning(f"Falling back to: {fallback_client}")
            return {
                "route": fallback_client,
                "reasoning": f"Fallback due to routing error: {str(e)}",
                "error": str(e)
            }

    def batch_route(
        self,
        queries: List[str],
        available_clients: Dict[str, Dict[str, str]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Route multiple queries in batch.

        Args:
            queries: List of queries to route
            available_clients: Same as in route()
            **kwargs: Additional arguments passed to route()

        Returns:
            List of routing decisions
        """
        return [self.route(query, available_clients, **kwargs) for query in queries]

    def __str__(self) -> str:
        return f"ArchRouter(model={self.model_name}, device={self.device})"

    def __repr__(self) -> str:
        return self.__str__()
