"""
Prompt Registry for MinionS.

This module provides a centralized registry for all prompts used in MinionS,
enabling runtime prompt injection for evolution experiments.

Usage:
    from minions.prompts.registry import get_prompt, register_prompts
    
    # Get a prompt by name
    prompt = get_prompt("WORKER_PROMPT_SHORT")
    
    # Override prompts at runtime
    register_prompts({"WORKER_PROMPT_SHORT": "new prompt text..."})
"""

from typing import Dict, Any, Union

# Import all prompts from the minions prompts module
from minions.prompts.minions import (
    WORKER_ICL_EXAMPLES,
    WORKER_OUTPUT_TEMPLATE,
    WORKER_PROMPT_TEMPLATE,
    WORKER_PROMPT_SHORT,
    REMOTE_ANSWER_OR_CONTINUE,
    REMOTE_ANSWER_OR_CONTINUE_SHORT,
    REMOTE_ANSWER,
    ADVICE_PROMPT,
    ADVICE_PROMPT_STEPS,
    ADVANCED_STEPS_INSTRUCTIONS,
    DECOMPOSE_TASK_PROMPT,
    DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC,
    DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC,
    DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    DECOMPOSE_TASK_PROMPT_SHORT,
    DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS,
    BM25_INSTRUCTIONS,
    EMBEDDING_INSTRUCTIONS,
    REMOTE_SYNTHESIS_COT,
    REMOTE_SYNTHESIS_JSON,
    REMOTE_SYNTHESIS_FINAL,
)


# The central prompt registry mapping names to prompt values
# This allows runtime overrides for prompt evolution experiments
PROMPT_REGISTRY: Dict[str, Any] = {
    # ==========================================================================
    # EXECUTE module (local worker prompts)
    # ==========================================================================
    "WORKER_ICL_EXAMPLES": WORKER_ICL_EXAMPLES,
    "WORKER_OUTPUT_TEMPLATE": WORKER_OUTPUT_TEMPLATE,
    "WORKER_PROMPT_TEMPLATE": WORKER_PROMPT_TEMPLATE,
    "WORKER_PROMPT_SHORT": WORKER_PROMPT_SHORT,
    
    # ==========================================================================
    # DECOMPOSE module (remote supervisor - task decomposition prompts)
    # ==========================================================================
    "ADVICE_PROMPT": ADVICE_PROMPT,
    "ADVICE_PROMPT_STEPS": ADVICE_PROMPT_STEPS,
    "ADVANCED_STEPS_INSTRUCTIONS": ADVANCED_STEPS_INSTRUCTIONS,
    "DECOMPOSE_TASK_PROMPT": DECOMPOSE_TASK_PROMPT,
    "DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC": DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC,
    "DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND": DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC": DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC,
    "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND": DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
    "DECOMPOSE_TASK_PROMPT_SHORT": DECOMPOSE_TASK_PROMPT_SHORT,
    "DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS": DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS,
    
    # Retrieval instructions
    "BM25_INSTRUCTIONS": BM25_INSTRUCTIONS,
    "EMBEDDING_INSTRUCTIONS": EMBEDDING_INSTRUCTIONS,
    
    # ==========================================================================
    # AGGREGATE module (remote supervisor - synthesis prompts)
    # ==========================================================================
    "REMOTE_ANSWER_OR_CONTINUE": REMOTE_ANSWER_OR_CONTINUE,
    "REMOTE_ANSWER_OR_CONTINUE_SHORT": REMOTE_ANSWER_OR_CONTINUE_SHORT,
    "REMOTE_ANSWER": REMOTE_ANSWER,
    "REMOTE_SYNTHESIS_COT": REMOTE_SYNTHESIS_COT,
    "REMOTE_SYNTHESIS_JSON": REMOTE_SYNTHESIS_JSON,
    "REMOTE_SYNTHESIS_FINAL": REMOTE_SYNTHESIS_FINAL,
}


def get_prompt(name: str) -> Any:
    """
    Get a prompt by name from the registry.
    
    Args:
        name: The name of the prompt to retrieve (e.g., "WORKER_PROMPT_SHORT")
        
    Returns:
        The prompt value (string or other type like list for ICL examples)
        
    Raises:
        KeyError: If the prompt name is not found in the registry
    """
    if name not in PROMPT_REGISTRY:
        available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
        raise KeyError(f"Prompt '{name}' not found in registry. Available prompts: {available}")
    return PROMPT_REGISTRY[name]


def register_prompts(overrides: Dict[str, Any]) -> None:
    """
    Override prompts in the registry at runtime.
    
    This is used by the prompt evolution system to inject evolved prompts.
    
    Args:
        overrides: A dictionary mapping prompt names to new prompt values.
                   Only prompts that exist in the registry can be overridden.
                   
    Raises:
        KeyError: If an override key is not a valid prompt name
    """
    for name, value in overrides.items():
        if name not in PROMPT_REGISTRY:
            available = ", ".join(sorted(PROMPT_REGISTRY.keys()))
            raise KeyError(f"Cannot override unknown prompt '{name}'. Available prompts: {available}")
        PROMPT_REGISTRY[name] = value


def get_all_prompts() -> Dict[str, Any]:
    """
    Get a copy of all prompts in the registry.
    
    Returns:
        A dictionary containing all prompt names and their current values.
    """
    return dict(PROMPT_REGISTRY)


def reset_prompts() -> None:
    """
    Reset all prompts to their original values.
    
    This is useful for testing or when you want to undo all overrides.
    """
    global PROMPT_REGISTRY
    PROMPT_REGISTRY.update({
        "WORKER_ICL_EXAMPLES": WORKER_ICL_EXAMPLES,
        "WORKER_OUTPUT_TEMPLATE": WORKER_OUTPUT_TEMPLATE,
        "WORKER_PROMPT_TEMPLATE": WORKER_PROMPT_TEMPLATE,
        "WORKER_PROMPT_SHORT": WORKER_PROMPT_SHORT,
        "ADVICE_PROMPT": ADVICE_PROMPT,
        "ADVICE_PROMPT_STEPS": ADVICE_PROMPT_STEPS,
        "ADVANCED_STEPS_INSTRUCTIONS": ADVANCED_STEPS_INSTRUCTIONS,
        "DECOMPOSE_TASK_PROMPT": DECOMPOSE_TASK_PROMPT,
        "DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC": DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC,
        "DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND": DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
        "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC": DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC,
        "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND": DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND,
        "DECOMPOSE_TASK_PROMPT_SHORT": DECOMPOSE_TASK_PROMPT_SHORT,
        "DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS": DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS,
        "BM25_INSTRUCTIONS": BM25_INSTRUCTIONS,
        "EMBEDDING_INSTRUCTIONS": EMBEDDING_INSTRUCTIONS,
        "REMOTE_ANSWER_OR_CONTINUE": REMOTE_ANSWER_OR_CONTINUE,
        "REMOTE_ANSWER_OR_CONTINUE_SHORT": REMOTE_ANSWER_OR_CONTINUE_SHORT,
        "REMOTE_ANSWER": REMOTE_ANSWER,
        "REMOTE_SYNTHESIS_COT": REMOTE_SYNTHESIS_COT,
        "REMOTE_SYNTHESIS_JSON": REMOTE_SYNTHESIS_JSON,
        "REMOTE_SYNTHESIS_FINAL": REMOTE_SYNTHESIS_FINAL,
    })


# Prompt-to-module mapping for reference
# This documents which module each prompt belongs to
PROMPT_MODULE_MAPPING = {
    # EXECUTE module (local worker)
    "WORKER_ICL_EXAMPLES": "EXECUTE",
    "WORKER_OUTPUT_TEMPLATE": "EXECUTE",
    "WORKER_PROMPT_TEMPLATE": "EXECUTE",
    "WORKER_PROMPT_SHORT": "EXECUTE",
    
    # DECOMPOSE module (remote supervisor)
    "ADVICE_PROMPT": "DECOMPOSE",
    "ADVICE_PROMPT_STEPS": "DECOMPOSE",
    "ADVANCED_STEPS_INSTRUCTIONS": "DECOMPOSE",
    "DECOMPOSE_TASK_PROMPT": "DECOMPOSE",
    "DECOMPOSE_TASK_PROMPT_AGGREGATION_FUNC": "DECOMPOSE",
    "DECOMPOSE_TASK_PROMPT_AGG_FUNC_LATER_ROUND": "DECOMPOSE",
    "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGGREGATION_FUNC": "DECOMPOSE",
    "DECOMPOSE_RETRIEVAL_TASK_PROMPT_AGG_FUNC_LATER_ROUND": "DECOMPOSE",
    "DECOMPOSE_TASK_PROMPT_SHORT": "DECOMPOSE",
    "DECOMPOSE_TASK_PROMPT_SHORT_JOB_OUTPUTS": "DECOMPOSE",
    "BM25_INSTRUCTIONS": "DECOMPOSE",
    "EMBEDDING_INSTRUCTIONS": "DECOMPOSE",
    
    # AGGREGATE module (remote supervisor)
    "REMOTE_ANSWER_OR_CONTINUE": "AGGREGATE",
    "REMOTE_ANSWER_OR_CONTINUE_SHORT": "AGGREGATE",
    "REMOTE_ANSWER": "AGGREGATE",
    "REMOTE_SYNTHESIS_COT": "AGGREGATE",
    "REMOTE_SYNTHESIS_JSON": "AGGREGATE",
    "REMOTE_SYNTHESIS_FINAL": "AGGREGATE",
}
