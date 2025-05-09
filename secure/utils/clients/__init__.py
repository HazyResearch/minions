from minions.utils.clients.ollama import OllamaClient
from minions.utils.clients.openai import OpenAIClient
from minions.utils.clients.azure_openai import AzureOpenAIClient
from minions.utils.clients.anthropic import AnthropicClient
from minions.utils.clients.together import TogetherClient
from minions.utils.clients.perplexity import PerplexityAIClient
from minions.utils.clients.openrouter import OpenRouterClient
from minions.utils.clients.groq import GroqClient
from minions.utils.clients.deepseek import DeepSeekClient
from minions.utils.clients.sambanova import SambanovaClient
from minions.utils.clients.gemini import GeminiClient

__all__ = [
    "OllamaClient",
    "OpenAIClient",
    "AzureOpenAIClient",
    "AnthropicClient",
    "TogetherClient",
    "PerplexityAIClient",
    "OpenRouterClient",
    "GroqClient",
    "DeepSeekClient",
    "SambanovaClient",
    "GeminiClient",
]

try:
    from minions.utils.clients.mlx_lm import MLXLMClient

    __all__.append("MLXLMClient")
except ImportError:
    # print warning that mlx_lm is not installed
    print(
        "Warning: mlx_lm is not installed. If you want to use mlx_lm, please install it with `pip install mlx-lm`."
    )

try:
    from minions.utils.clients.cartesia_mlx import CartesiaMLXClient

    __all__.append("CartesiaMLXClient")
except ImportError:
    # If cartesia_mlx is not installed, skip it
    print(
        "Warning: cartesia_mlx is not installed. If you want to use cartesia_mlx, please follow the instructions in the README to install it."
    )


try:
    from minions.utils.clients.mlx_omni import MLXOmniClient

    __all__.append("MLXOmniClient")
except ImportError:
    # print warning that mlx_omni is not installed
    print(
        "Warning: mlx_omni is not installed. If you want to use mlx_omni, please install it with `pip install mlx-omni-server`"
    )

try:
    from minions.utils.clients.huggingface_client import HuggingFaceClient

    __all__.append("HuggingFaceClient")
except ImportError:
    # print warning that huggingface is not installed
    print(
        "Warning: huggingface inference client is not installed. If you want to use huggingface inference client, please install it with `pip install huggingface-hub`"
    )

try:
    from minions.utils.clients.mlx_audio import MLXAudioClient

    __all__.append("MLXAudioClient")
except ImportError:
    # print warning that mlx_audio is not installed
    print(
        "Warning: mlx_audio is not installed. If you want to use mlx_audio, please install it with `pip install mlx-audio`"
    )

try:
    from minions.utils.clients.mlx_parallm_model import MLXParallmClient
except ImportError:
    # This allows the package to be imported even if mlx_parallm is not installed
    print(
        "Warning: mlx_parallm is not installed. If you want to use mlx_parallm, please install it with `pip install mlx-parallm`"
    )
