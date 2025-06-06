"""
Agent cards and skills definitions for A2A-Minions integration.
Streamlined to support only two core skills: minion_query and minions_query.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


class AgentSkill(BaseModel):
    """A2A Agent Skill definition."""
    id: str
    name: str
    description: str
    tags: List[str] = []
    examples: List[str] = []
    inputModes: List[str] = ["text"]
    outputModes: List[str] = ["text"]


class AgentCapabilities(BaseModel):
    """A2A Agent Capabilities."""
    streaming: bool = True
    pushNotifications: bool = True
    stateTransitionHistory: bool = True


class AgentCard(BaseModel):
    """A2A Agent Card definition."""
    name: str
    description: str
    url: str
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: List[str] = ["text", "file"]
    defaultOutputModes: List[str] = ["text", "data"]
    skills: List[AgentSkill]
    supportsAuthenticatedExtendedCard: bool = False


# Define core Minions skills
MINIONS_SKILLS = {
    "minion_query": AgentSkill(
        id="minion_query",
        name="Minion Query",
        description=(
            "Execute focused queries using the Minion protocol with document support. "
            "Provide your question as the first text part, and your document/context as the second part "
            "(can be text, file, or data). Ideal for single document Q&A and analysis."
        ),
        tags=["query", "document-analysis", "local-remote", "cost-efficient", "single-conversation"],
        examples=[
            "First part: 'What are the key findings?' Second part: Research paper file",
            "First part: 'Summarize the main points' Second part: Meeting transcript text",
            "First part: 'Find the error' Second part: Code file",
            "First part: 'Extract action items' Second part: Project document"
        ],
        inputModes=["text", "file", "data"],
        outputModes=["text", "data"]
    ),
    
    "minions_query": AgentSkill(
        id="minions_query", 
        name="Minions Query",
        description=(
            "Execute parallel processing queries for complex analysis using distributed Minions. "
            "Provide your question as the first text part, and your document/context as the second part "
            "(can be text, file, or data). Best for large documents requiring parallel processing."
        ),
        tags=["query", "parallel", "document-processing", "chunking", "scalable", "cost-efficient"],
        examples=[
            "First part: 'Extract all insights' Second part: Large research report PDF", 
            "First part: 'Find patterns and themes' Second part: Multiple document files",
            "First part: 'Analyze trends' Second part: Large dataset",
            "First part: 'Summarize key points' Second part: 100-page document"
        ],
        inputModes=["text", "file", "data"],
        outputModes=["text", "data"]
    )
}


def get_default_agent_card(base_url: str = "http://localhost:8000") -> AgentCard:
    """Get the default agent card for A2A-Minions server."""
    return AgentCard(
        name="Minions Protocol Agent",
        description=(
            "Cost-efficient collaboration between on-device and cloud LLMs using the Minions protocol. "
            "Supports document Q&A and analysis through two main skills: minion_query for focused analysis "
            "and minions_query for parallel processing of large documents."
        ),
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text", "file", "data"],
        defaultOutputModes=["text", "data"], 
        skills=list(MINIONS_SKILLS.values()),
        supportsAuthenticatedExtendedCard=False
    )


def get_extended_agent_card(base_url: str = "http://localhost:8000") -> AgentCard:
    """Get an extended agent card - currently same as default."""
    return get_default_agent_card(base_url)




def extract_query_and_document_from_parts(parts: List[Dict[str, Any]]) -> tuple[str, str, str]:
    """
    Extract query and document from A2A message parts.
    
    Args:
        parts: List of A2A parts (text, file, or data)
        
    Returns:
        tuple: (query, document_content, document_type)
    """
    if not parts:
        raise ValueError("No parts provided")
    
    # First part should be the query (text)
    if parts[0].get("kind") != "text":
        raise ValueError("First part must be text containing the query")
    
    query = parts[0].get("text", "")
    if not query:
        raise ValueError("Query text is empty")
    
    # Second part is optional and contains the document
    if len(parts) == 1:
        return query, "", "none"
    
    second_part = parts[1]
    part_kind = second_part.get("kind")
    
    if part_kind == "text":
        document_content = second_part.get("text", "")
        return query, document_content, "text"
    
    elif part_kind == "file":
        # Will be handled by file processing logic in converters
        return query, "", "file"
    
    elif part_kind == "data":
        # Convert data to string representation
        data = second_part.get("data", {})
        document_content = str(data)
        return query, document_content, "data"
    
    else:
        raise ValueError(f"Unsupported part kind: {part_kind}") 