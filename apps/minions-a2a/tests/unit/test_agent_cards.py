#!/usr/bin/env python3
"""
Unit tests for A2A-Minions agent cards.
Tests agent card generation, skills definition, and security schemes.
"""

import unittest

from a2a_minions.agent_cards import (
    OAuthFlow, OAuthFlows, SecurityScheme, AgentSkill, AgentCapabilities,
    AgentCard, MINIONS_SKILLS, get_security_schemes, get_default_agent_card,
    get_extended_agent_card, extract_query_and_document_from_parts
)


class TestOAuthModels(unittest.TestCase):
    """Test OAuth-related models."""
    
    def test_oauth_flow(self):
        """Test OAuthFlow model."""
        flow = OAuthFlow(
            tokenUrl="https://example.com/oauth/token",
            scopes={
                "read": "Read access",
                "write": "Write access"
            }
        )
        self.assertEqual(flow.tokenUrl, "https://example.com/oauth/token")
        self.assertEqual(flow.scopes["read"], "Read access")
    
    def test_oauth_flows(self):
        """Test OAuthFlows model."""
        flow = OAuthFlow(
            tokenUrl="https://example.com/oauth/token",
            scopes={"tasks": "Task access"}
        )
        flows = OAuthFlows(clientCredentials=flow)
        self.assertEqual(flows.clientCredentials.tokenUrl, "https://example.com/oauth/token")


class TestSecurityScheme(unittest.TestCase):
    """Test SecurityScheme model."""
    
    def test_http_bearer_scheme(self):
        """Test HTTP bearer security scheme."""
        scheme = SecurityScheme(
            type="http",
            scheme="bearer",
            bearerFormat="JWT",
            description="Bearer token authentication"
        )
        self.assertEqual(scheme.type, "http")
        self.assertEqual(scheme.scheme, "bearer")
        self.assertEqual(scheme.bearerFormat, "JWT")
    
    def test_api_key_scheme(self):
        """Test API key security scheme."""
        scheme = SecurityScheme(
            type="apiKey",
            name="X-API-Key",
            in_="header",
            description="API key authentication"
        )
        self.assertEqual(scheme.type, "apiKey")
        self.assertEqual(scheme.name, "X-API-Key")
        self.assertEqual(scheme.in_, "header")
    
    def test_oauth2_scheme(self):
        """Test OAuth2 security scheme."""
        flows = OAuthFlows(
            clientCredentials=OAuthFlow(
                tokenUrl="https://example.com/token",
                scopes={"admin": "Admin access"}
            )
        )
        scheme = SecurityScheme(
            type="oauth2",
            description="OAuth2 authentication",
            flows=flows
        )
        self.assertEqual(scheme.type, "oauth2")
        self.assertIsNotNone(scheme.flows)
        self.assertEqual(scheme.flows.clientCredentials.tokenUrl, "https://example.com/token")
    
    def test_field_alias(self):
        """Test 'in' field alias handling."""
        # Test with alias
        scheme = SecurityScheme(
            type="apiKey",
            name="key",
            description="Test",
            **{"in": "query"}  # Using 'in' directly
        )
        self.assertEqual(scheme.in_, "query")


class TestAgentSkill(unittest.TestCase):
    """Test AgentSkill model."""
    
    def test_minimal_skill(self):
        """Test skill with minimal fields."""
        skill = AgentSkill(
            id="test-skill",
            name="Test Skill",
            description="A test skill"
        )
        self.assertEqual(skill.id, "test-skill")
        self.assertEqual(skill.name, "Test Skill")
        self.assertEqual(skill.description, "A test skill")
        self.assertEqual(skill.tags, [])
        self.assertEqual(skill.examples, [])
    
    def test_full_skill(self):
        """Test skill with all fields."""
        skill = AgentSkill(
            id="complex-skill",
            name="Complex Skill",
            description="A complex skill with metadata",
            tags=["analysis", "research", "qa"],
            examples=[
                "Analyze this document",
                "Research this topic",
                "Answer questions about the data"
            ]
        )
        self.assertEqual(len(skill.tags), 3)
        self.assertIn("analysis", skill.tags)
        self.assertEqual(len(skill.examples), 3)


class TestAgentCapabilities(unittest.TestCase):
    """Test AgentCapabilities model."""
    
    def test_default_capabilities(self):
        """Test default agent capabilities."""
        caps = AgentCapabilities()
        self.assertTrue(caps.streaming)
    
    def test_custom_capabilities(self):
        """Test custom agent capabilities."""
        caps = AgentCapabilities(streaming=False)
        self.assertFalse(caps.streaming)


class TestAgentCard(unittest.TestCase):
    """Test AgentCard model."""
    
    def test_minimal_card(self):
        """Test agent card with minimal fields."""
        card = AgentCard(
            name="Test Agent",
            description="A test agent",
            url="https://example.com/agent"
        )
        self.assertEqual(card.name, "Test Agent")
        self.assertEqual(card.description, "A test agent")
        self.assertEqual(card.url, "https://example.com/agent")
        self.assertEqual(card.version, "1.0.0")
        self.assertEqual(card.defaultInputModes, ["application/json"])
        self.assertEqual(card.defaultOutputModes, ["application/json"])
        self.assertTrue(card.capabilities.streaming)
        self.assertEqual(len(card.skills), 0)
        self.assertEqual(len(card.securitySchemes), 0)
        self.assertEqual(len(card.security), 0)
    
    def test_full_card(self):
        """Test agent card with all fields."""
        skill = AgentSkill(
            id="skill1",
            name="Skill 1",
            description="First skill"
        )
        
        scheme = SecurityScheme(
            type="http",
            scheme="bearer",
            description="Bearer auth"
        )
        
        card = AgentCard(
            name="Full Agent",
            description="A fully configured agent",
            url="https://example.com/agent",
            version="2.0.0",
            defaultInputModes=["application/json", "text/plain"],
            defaultOutputModes=["application/json", "text/event-stream"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[skill],
            securitySchemes={"bearer": scheme},
            security=[{"bearer": []}]
        )
        
        self.assertEqual(card.version, "2.0.0")
        self.assertEqual(len(card.defaultInputModes), 2)
        self.assertEqual(len(card.skills), 1)
        self.assertEqual(card.skills[0].id, "skill1")
        self.assertIn("bearer", card.securitySchemes)
        self.assertEqual(len(card.security), 1)


class TestMinionsSkills(unittest.TestCase):
    """Test predefined Minions skills."""
    
    def test_skills_exist(self):
        """Test that predefined skills exist."""
        self.assertEqual(len(MINIONS_SKILLS), 2)
    
    def test_minion_query_skill(self):
        """Test minion_query skill definition."""
        skill = next(s for s in MINIONS_SKILLS if s.id == "minion_query")
        self.assertEqual(skill.name, "Minion Query")
        self.assertIn("document-analysis", skill.tags)
        self.assertIn("qa", skill.tags)
        self.assertIn("focused", skill.tags)
        self.assertTrue(len(skill.examples) > 0)
    
    def test_minions_query_skill(self):
        """Test minions_query skill definition."""
        skill = next(s for s in MINIONS_SKILLS if s.id == "minions_query")
        self.assertEqual(skill.name, "Minions Query")
        self.assertIn("multi-document", skill.tags)
        self.assertIn("parallel", skill.tags)
        self.assertIn("research", skill.tags)
        self.assertTrue(len(skill.examples) > 0)


class TestSecuritySchemes(unittest.TestCase):
    """Test security scheme generation."""
    
    def test_get_security_schemes(self):
        """Test generating security schemes."""
        base_url = "https://example.com"
        schemes = get_security_schemes(base_url)
        
        # Check all expected schemes exist
        self.assertIn("bearer_auth", schemes)
        self.assertIn("api_key", schemes)
        self.assertIn("oauth2_client_credentials", schemes)
        
        # Check bearer auth
        bearer = schemes["bearer_auth"]
        self.assertEqual(bearer.type, "http")
        self.assertEqual(bearer.scheme, "bearer")
        self.assertEqual(bearer.bearerFormat, "JWT")
        
        # Check API key
        api_key = schemes["api_key"]
        self.assertEqual(api_key.type, "apiKey")
        self.assertEqual(api_key.name, "X-API-Key")
        self.assertEqual(api_key.in_, "header")
        
        # Check OAuth2
        oauth2 = schemes["oauth2_client_credentials"]
        self.assertEqual(oauth2.type, "oauth2")
        self.assertIsNotNone(oauth2.flows)
        self.assertEqual(
            oauth2.flows.clientCredentials.tokenUrl,
            f"{base_url}/oauth/token"
        )
        self.assertIn("minion:query", oauth2.flows.clientCredentials.scopes)
        self.assertIn("tasks:write", oauth2.flows.clientCredentials.scopes)


class TestAgentCardGeneration(unittest.TestCase):
    """Test agent card generation functions."""
    
    def test_get_default_agent_card(self):
        """Test default agent card generation."""
        base_url = "https://api.example.com"
        card = get_default_agent_card(base_url)
        
        self.assertEqual(card.name, "A2A-Minions Server")
        self.assertEqual(card.url, base_url)
        self.assertEqual(card.version, "1.0.0")
        self.assertTrue(card.capabilities.streaming)
        self.assertEqual(len(card.skills), 2)
        self.assertEqual(len(card.securitySchemes), 3)
        self.assertEqual(len(card.security), 1)
        self.assertEqual(card.security[0], {"api_key": []})
        
        # Check output modes include streaming
        self.assertIn("text/event-stream", card.defaultOutputModes)
    
    def test_get_extended_agent_card(self):
        """Test extended agent card generation."""
        base_url = "https://api.example.com"
        card = get_extended_agent_card(base_url)
        
        # Currently returns same as default, but structure is in place for extensions
        self.assertEqual(card.name, "A2A-Minions Server")
        self.assertEqual(card.url, base_url)


class TestExtractQueryAndDocument(unittest.TestCase):
    """Test query and document extraction from parts."""
    
    def test_empty_parts(self):
        """Test extraction with empty parts."""
        with self.assertRaises(ValueError) as context:
            extract_query_and_document_from_parts([])
        self.assertIn("No parts provided", str(context.exception))
    
    def test_non_text_first_part(self):
        """Test extraction when first part is not text."""
        parts = [{"kind": "file", "file": {"name": "test.pdf"}}]
        with self.assertRaises(ValueError) as context:
            extract_query_and_document_from_parts(parts)
        self.assertIn("First part must be text", str(context.exception))
    
    def test_empty_query_text(self):
        """Test extraction with empty query text."""
        parts = [{"kind": "text", "text": ""}]
        with self.assertRaises(ValueError) as context:
            extract_query_and_document_from_parts(parts)
        self.assertIn("Query text is empty", str(context.exception))
    
    def test_query_only(self):
        """Test extraction with query only (no document)."""
        parts = [{"kind": "text", "text": "What is the capital of France?"}]
        query, content, doc_type = extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "What is the capital of France?")
        self.assertEqual(content, "")
        self.assertEqual(doc_type, "none")
    
    def test_query_with_text_document(self):
        """Test extraction with query and text document."""
        parts = [
            {"kind": "text", "text": "Summarize this document"},
            {"kind": "text", "text": "This is the document content to summarize."}
        ]
        query, content, doc_type = extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Summarize this document")
        self.assertEqual(content, "This is the document content to summarize.")
        self.assertEqual(doc_type, "text")
    
    def test_query_with_file_document(self):
        """Test extraction with query and file document."""
        parts = [
            {"kind": "text", "text": "Analyze this PDF"},
            {"kind": "file", "file": {"name": "document.pdf", "bytes": "..."}}
        ]
        query, content, doc_type = extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Analyze this PDF")
        self.assertEqual(content, "")  # File content handled elsewhere
        self.assertEqual(doc_type, "file")
    
    def test_query_with_data_document(self):
        """Test extraction with query and data document."""
        parts = [
            {"kind": "text", "text": "Analyze this data"},
            {"kind": "data", "data": {"key": "value", "number": 42}}
        ]
        query, content, doc_type = extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Analyze this data")
        self.assertEqual(content, "{'key': 'value', 'number': 42}")
        self.assertEqual(doc_type, "data")
    
    def test_invalid_part_kind(self):
        """Test extraction with invalid part kind."""
        parts = [
            {"kind": "text", "text": "Query"},
            {"kind": "invalid", "something": "data"}
        ]
        with self.assertRaises(ValueError) as context:
            extract_query_and_document_from_parts(parts)
        self.assertIn("Unsupported part kind: invalid", str(context.exception))
    
    def test_missing_text_in_text_part(self):
        """Test extraction with missing text in text part."""
        parts = [
            {"kind": "text", "text": "Query"},
            {"kind": "text"}  # Missing text field
        ]
        query, content, doc_type = extract_query_and_document_from_parts(parts)
        
        self.assertEqual(query, "Query")
        self.assertEqual(content, "")  # Defaults to empty string
        self.assertEqual(doc_type, "text")


if __name__ == "__main__":
    unittest.main()