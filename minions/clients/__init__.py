# Core imports - should always work with base installation
from minions.clients.base import MinionsClient
from minions.clients.response import ChatResponse
from minions.clients.ollama import OllamaClient, OllamaTurboClient
from minions.clients.osaurus import OsaurusClient
from minions.clients.lemonade import LemonadeClient
from minions.clients.openai import OpenAIClient
from minions.clients.azure_openai import AzureOpenAIClient
from minions.clients.anthropic import AnthropicClient
from minions.clients.parallel import ParallelClient

# Initialize __all__ with core clients
__all__ = [
    "MinionsClient",
    "ChatResponse",
    "OllamaClient",
    "OllamaTurboClient",
    "OsaurusClient",
    "LemonadeClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "AnthropicClient",
    "ParallelClient",
]

# Optional dependencies - clients that require additional packages
_optional_clients = [
    ("minions.clients.cohere", "CohereClient"),
    ("minions.clients.perplexity", "PerplexityAIClient"),
    ("minions.clients.openrouter", "OpenRouterClient"),
    ("minions.clients.groq", "GroqClient"),
    ("minions.clients.deepseek", "DeepSeekClient"),
    ("minions.clients.qwen", "QwenClient"),
    ("minions.clients.moonshot", "MoonshotClient"),
    ("minions.clients.grok", "GrokClient"),
    ("minions.clients.llama_api", "LlamaApiClient"),
    ("minions.clients.minimax", "MiniMaxClient"),
    ("minions.clients.sarvam", "SarvamClient"),
    ("minions.clients.docker_model_runner", "DockerModelRunnerClient"),
    ("minions.clients.distributed_inference", "DistributedInferenceClient"),
    ("minions.clients.novita", "NovitaClient"),
    ("minions.clients.tencent", "TencentClient"),
    ("minions.clients.cloudflare", "CloudflareGatewayClient"),
    ("minions.clients.notdiamond", "NotDiamondAIClient"),
    ("minions.clients.vercel_gateway", "VercelGatewayClient"),
    ("minions.clients.exa", "ExaClient"),
    ("minions.clients.together", "TogetherClient"),
    ("minions.clients.sambanova", "SambanovaClient"),
    ("minions.clients.gemini", "GeminiClient"),
    ("minions.clients.mistral", "MistralClient"),
]

# Dynamically import optional clients
for module_path, class_name in _optional_clients:
    try:
        module = __import__(module_path, fromlist=[class_name])
        globals()[class_name] = getattr(module, class_name)
        __all__.append(class_name)
    except ImportError:
        pass

try:
    from minions.clients.transformers import TransformersClient

    __all__.append("TransformersClient")
except ImportError:
    # print warning that transformers is not installed
    print(
        "WARNING: Transformers is not installed. Please install it with `pip install transformers`."
    )

try:
    from .cartesia_mlx import CartesiaMLXClient

    __all__.append("CartesiaMLXClient")
except ImportError:
    # If cartesia_mlx is not installed, skip it
    print(
        "Warning: cartesia_mlx is not installed. If you want to use cartesia_mlx, please follow the instructions in the README to install it."
    )

try:
    from .huggingface_client import HuggingFaceClient

    __all__.append("HuggingFaceClient")
except ImportError:
    # print warning that huggingface is not installed
    print(
        "Warning: huggingface inference client is not installed. If you want to use huggingface inference client, please install it with `pip install huggingface-hub`"
    )

# Import all MLX clients from the consolidated file
try:
    from .mlx_clients import MLXLMClient, MLXOmniClient, MLXAudioClient, MLXParallmClient
    __all__.extend(["MLXLMClient", "MLXOmniClient", "MLXAudioClient", "MLXParallmClient"])
except ImportError:
    # Individual client imports with their specific dependencies
    try:
        from .mlx_clients import MLXLMClient
        __all__.append("MLXLMClient")
    except ImportError:
        print(
            "Warning: mlx_lm is not installed. If you want to use mlx_lm, please install it with `pip install mlx-lm`."
        )

    try:
        from .mlx_clients import MLXOmniClient
        __all__.append("MLXOmniClient")
    except ImportError:
        print(
            "Warning: mlx_omni is not installed. If you want to use mlx_omni, please install it with `pip install mlx-omni-server`"
        )

    try:
        from .mlx_clients import MLXAudioClient
        __all__.append("MLXAudioClient")
    except ImportError:
        print(
            "Warning: mlx_audio is not installed. If you want to use mlx_audio, please install it with `pip install mlx-audio`"
        )

    try:
        from .mlx_clients import MLXParallmClient
        __all__.append("MLXParallmClient")
    except ImportError:
        print(
            "Warning: mlx_parallm is not installed. If you want to use mlx_parallm, please install it with `pip install mlx-parallm`"
        )


# Duplicate import removed - TransformersClient is already imported above

try:
    from minions.clients.secure import SecureClient
    __all__.append("SecureClient")
except ImportError:
    # print warning that secure crypto utilities are not available
    print(
        "Warning: Secure crypto utilities are not available. SecureClient will not be available. "
        "Please ensure the secure module is properly installed."
    )

try:
    from minions.clients.cerebras import CerebrasClient
    __all__.append("CerebrasClient")
except ImportError:
    # print warning that cerebras-cloud-sdk is not installed
    print(
        "Warning: cerebras-cloud-sdk is not installed. If you want to use CerebrasClient, "
        "please install it with `pip install cerebras-cloud-sdk`."
    )

try:
    from minions.clients.modular import ModularClient
    __all__.append("ModularClient")
except ImportError:
    # print warning that modular is not installed
    print(
        "Warning: Modular MAX or OpenAI client is not installed. If you want to use ModularClient, "
        "please install Modular MAX (https://docs.modular.com/max/get-started) and OpenAI client (pip install openai)."
    )

try:
    from minions.clients.lmcache import LMCacheClient
    __all__.append("LMCacheClient")
except ImportError:
    # print warning that lmcache is not installed
    print(
        "Warning: LMCache or vLLM is not installed. If you want to use LMCacheClient, "
        "please install with `pip install lmcache vllm`. "
        "For detailed instructions, see: https://docs.lmcache.ai/getting_started/installation.html"
    )
