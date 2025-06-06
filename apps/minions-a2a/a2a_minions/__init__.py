"""
A2A-Minions Integration Package

This package provides an A2A (Agent-to-Agent) protocol server that wraps 
the Minions protocol, enabling standardized agent communication.
"""

__version__ = "0.1.0"
__author__ = "Minions A2A Team"

from .server import A2AMinionsServer
from .config import MinionsConfig, ConfigManager
from .agent_cards import get_default_agent_card, MINIONS_SKILLS

__all__ = [
    "A2AMinionsServer",
    "MinionsConfig", 
    "ConfigManager",
    "get_default_agent_card",
    "MINIONS_SKILLS",
] 