"""
Agent card definitions for A2A-Minions server.
Defines the public capabilities and skills available through the A2A protocol.
"""

from typing import List, Dict, Any
from pydantic import BaseModel


class OAuthFlow(BaseModel):
    """OAuth flow configuration."""
    tokenUrl: str
    scopes: Dict[str, str]


class OAuthFlows(BaseModel):
    """OAuth flows configuration."""
    clientCredentials: OAuthFlow


class SecurityScheme(BaseModel):
    """Security scheme definition."""
    type: str
    description: str
    scheme: str = None
    bearerFormat: str = None
    flows: OAuthFlows = None
    name: str = None
    in_: str = None  # 'in' is a reserved keyword

    class Config:
        fields = {'in_': 'in'}


class AgentSkill(BaseModel):
    """Definition of a skill that an agent can perform."""
    id: str
    name: str
    description: str
    tags: List[str] = []
    examples: List[str] = []


class AgentCapabilities(BaseModel):
    """Agent capabilities definition."""
    streaming: bool = True


class AgentCard(BaseModel):
    """Agent card following A2A protocol specification."""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    defaultInputModes: List[str] = ["application/json"]
    defaultOutputModes: List[str] = ["application/json"]
    capabilities: AgentCapabilities = AgentCapabilities()
    skills: List[AgentSkill] = []
    securitySchemes: Dict[str, SecurityScheme] = {}
    security: List[Dict[str, List[str]]] = []


# Define available skills
MINIONS_SKILLS = [
    AgentSkill(
        id="minion_query",
        name="Minion Query",
        description="Focused analysis using single-conversation protocol for document Q&A and specific questions",
        tags=["document-analysis", "qa", "focused"],
        examples=[
            "What are the key findings in this research paper?",
            "Summarize the main points of this document",
            "Extract specific information from the provided text"
        ]
    ),
    AgentSkill(
        id="minions_query",
        name="Minions Query",
        description="Parallel processing for complex multi-document analysis and research tasks",
        tags=["multi-document", "parallel", "research"],
        examples=[
            "Analyze these multiple research papers for common themes",
            "Process this large dataset and identify patterns",
            "Compare and contrast information across multiple sources"
        ]
    )
]


def get_security_schemes(base_url: str) -> Dict[str, SecurityScheme]:
    """Get security schemes for the agent card."""
    return {
        "bearer_auth": SecurityScheme(
            type="http",
            scheme="bearer",
            bearerFormat="JWT",
            description="Bearer token authentication using JWT tokens"
        ),
        "api_key": SecurityScheme(
            type="apiKey",
            name="X-API-Key",
            in_="header",
            description="API key authentication for local deployments"
        ),
        "oauth2_client_credentials": SecurityScheme(
            type="oauth2",
            description="OAuth2 client credentials flow for M2M authentication",
            flows=OAuthFlows(
                clientCredentials=OAuthFlow(
                    tokenUrl=f"{base_url}/oauth/token",
                    scopes={
                        "minion:query": "Execute focused minion queries",
                        "minions:query": "Execute parallel minions queries",
                        "tasks:read": "Read task status and results",
                        "tasks:write": "Create and cancel tasks"
                    }
                )
            )
        )
    }


def get_default_agent_card(base_url: str) -> AgentCard:
    """Get the default public agent card."""
    return AgentCard(
        name="A2A-Minions Server",
        description="Agent-to-Agent server providing Minions protocol capabilities for document analysis and complex reasoning tasks",
        url=base_url,
        version="1.0.0",
        defaultInputModes=["application/json"],
        defaultOutputModes=["application/json", "text/event-stream"],
        capabilities=AgentCapabilities(streaming=True),
        skills=MINIONS_SKILLS,
        securitySchemes=get_security_schemes(base_url),
        security=[
            {
                "api_key": []  # Default to API key for local deployments
            }
        ]
    )


def get_extended_agent_card(base_url: str) -> AgentCard:
    """Get the extended agent card for authenticated users."""
    card = get_default_agent_card(base_url)
    # Extended card can include additional internal skills or capabilities
    return card


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