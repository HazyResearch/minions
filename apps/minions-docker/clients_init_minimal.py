from minions.clients.base import MinionsClient
from minions.clients.openai import OpenAIClient
from minions.clients.docker_model_runner import DockerModelRunnerClient


# Stub classes for missing clients to avoid import errors
class TogetherClient(MinionsClient):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("TogetherClient not available in minimal build")


class GeminiClient(MinionsClient):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GeminiClient not available in minimal build")


class SambanovaClient(MinionsClient):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SambanovaClient not available in minimal build")


__all__ = [
    "MinionsClient",
    "OpenAIClient",
    "DockerModelRunnerClient",
    "TogetherClient",
    "GeminiClient",
    "SambanovaClient",
]
