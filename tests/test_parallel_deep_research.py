"""Scripted model tests through the research loop and real MCP HTTP SDK.

Run with: python -m unittest discover -s tests -p test_parallel_deep_research.py
No model server, search credentials or internet connection is needed.
"""

import asyncio
import json
import os
import re
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch
from uuid import UUID

import httpx
from mcp.client.streamable_http import streamablehttp_client

from minions.minions_deep_research import DeepResearchMinions
from minions.utils import parallel_search_mcp as parallel


class ScriptedRemote:
    def __init__(self, rounds=2):
        self.rounds = rounds

    def chat(self, messages, **kwargs):
        prompt = messages[0]["content"]
        if "USER QUESTION:" in prompt:
            return [re.search(r'USER QUESTION: "(.*?)"', prompt)[1]], None
        query = re.search(r'ORIGINAL QUERY: "(.*?)"', prompt)[1]
        count = prompt.count("SEARCH QUERY:")
        return [json.dumps({"more_info_required": count < self.rounds,
                            "search_query": f"{query} followup {count}"})], None


class ScriptedLocal:
    def __init__(self):
        self.prompts = []

    def chat(self, messages):
        self.prompts.extend(message["content"] for message in messages)
        if "## Content" in messages[0]["content"]:
            return [json.dumps({"explanation": "Verified source summary. It answers the research query.",
                                "answer": "relevant"}) for _ in messages], None, ["stop"] * len(messages)
        return [messages[0]["content"]], None, ["stop"]


class MCPFixture:
    """JSON-RPC server fixture at the HTTP boundary, with captured request bytes."""
    def __init__(self):
        self.requests = []
        self.tool_calls = []
        self.failure = None
        self.text_only = False
        self.empty = False
        self.fetch_errors = False
        self.slow = False
        self.cancelled = False
        self.started = threading.Event()
        self.barrier = None
        self.http_status = None
        self.malformed = False
        self.missing_tool = False

    async def handle(self, request):
        body = json.loads(request.content) if request.content else {}
        self.requests.append((str(request.url), dict(request.headers), body))
        method = body.get("method")
        if self.http_status:
            return httpx.Response(self.http_status, text="Service error")
        if method == "initialize":
            result = {"protocolVersion": body["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in (("web_search",) if self.missing_tool
                                             else ("web_search", "web_fetch"))]}
        elif method == "tools/call":
            name = body["params"]["name"]
            args = body["params"]["arguments"]
            self.tool_calls.append((name, args))
            self.started.set()
            if self.barrier and name == "web_search" and "followup" not in args["objective"]:
                await asyncio.to_thread(self.barrier.wait, 5)
            if self.slow:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            if self.failure:
                result = {"isError": True, "content": [{"type": "text", "text": self.failure}]}
            else:
                urls = ["https://example.com/source", "https://example.com/other"]
                if name == "web_search":
                    payload = {"results": [] if self.empty else [
                        {"url": "file:///etc/passwd"}, {"url": "https://user:pass@example.com/"},
                        *[{"url": url, "excerpts": ["Search excerpt"]} for url in urls],
                        {"url": urls[0]}]}
                else:
                    payload = {"results": [] if self.fetch_errors else [
                        {"url": url, "full_content": "Verified source body. " * 12}
                        for url in reversed(args["urls"])],
                        "errors": [{"url": urls[0]}] if self.fetch_errors else []}
                result = {"content": [{"type": "text", "text": json.dumps(payload)}]}
                if not self.text_only:
                    result["structuredContent"] = payload
                if self.malformed:
                    result = {"content": [{"type": "text", "text": "not JSON"}]}
        else:
            return httpx.Response(202)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    @contextmanager
    def installed(self):
        def factory(**kwargs):
            return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)

        def transport(*args, **kwargs):
            return streamablehttp_client(*args, **kwargs, httpx_client_factory=factory)

        with patch.object(parallel, "streamablehttp_client", transport):
            yield self


class ParallelDeepResearchTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.server = MCPFixture()
        self.local = ScriptedLocal()
        self.research = DeepResearchMinions(self.local, ScriptedRemote(), max_sources_per_round=2)

    def test_fresh_default_consumes_sources_and_reuses_run_session(self):
        with self.server.installed():
            answer, urls = self.research("Minions research query")
        self.assertIn("Verified source summary", answer)
        self.assertEqual(urls, {"https://example.com/source", "https://example.com/other"})
        self.assertTrue(any("Verified source body" in p and "Source: https://" in p
                            for p in self.local.prompts))
        self.assertEqual([name for name, _ in self.server.tool_calls],
                         ["web_search", "web_fetch", "web_search", "web_fetch"])
        ids = {args["session_id"] for _, args in self.server.tool_calls}
        self.assertEqual(len(ids), 1)
        UUID(next(iter(ids)))
        for url, headers, body in self.server.requests:
            self.assertEqual(url, "https://search.parallel.ai/mcp")
            self.assertEqual(headers["user-agent"], parallel.USER_AGENT)
            self.assertNotIn("authorization", headers)
        for name, args in self.server.tool_calls:
            if name == "web_fetch":
                self.assertEqual(args["urls"], ["https://example.com/source", "https://example.com/other"])
                self.assertTrue(args["full_content"])

    def test_new_run_on_shared_instance_has_new_session(self):
        with self.server.installed():
            self.research("First research query")
            self.research("Second research query")
        ids = [args["session_id"] for name, args in self.server.tool_calls if name == "web_search"]
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[2], ids[3])
        self.assertNotEqual(ids[0], ids[2])

    def test_interleaved_runs_on_shared_instance_keep_context(self):
        self.server.barrier = threading.Barrier(2)
        with self.server.installed(), ThreadPoolExecutor(2) as executor:
            results = list(executor.map(self.research, ["Alpha research query", "Beta research query"]))
        self.assertEqual(len(results), 2)
        ids = {}
        for name, args in self.server.tool_calls:
            query = args["objective"].split()[0]
            ids.setdefault(query, set()).add(args["session_id"])
        self.assertEqual([len(values) for values in ids.values()], [1, 1])
        self.assertNotEqual(ids["Alpha"], ids["Beta"])

    def test_ambient_parallel_credentials_are_never_used(self):
        os.environ["PARALLEL_API_KEY"] = "do-not-use"
        os.environ["PARALLEL_BASE_URL"] = "https://invalid.example/"
        with self.server.installed():
            self.research("Anonymous research query")
        self.assertTrue(self.server.tool_calls)
        self.assertTrue(all("authorization" not in headers and url == parallel.MCP_URL
                            for url, headers, _ in self.server.requests))

    def test_explicit_disabled_provider_prevents_all_web_requests(self):
        with self.server.installed():
            _, urls = self.research("Offline research query", search_provider="disabled")
        self.assertFalse(urls)
        self.assertFalse(self.server.requests)
        self.assertEqual(self.research.search_provider, "auto")
        self.research.search_provider = "disabled"
        with self.server.installed():
            self.research("Saved disabled query")
        self.assertFalse(self.server.requests)

    def test_legacy_keys_and_explicit_overrides(self):
        for keys in ({"SERPAPI_API_KEY": "s"}, {"FIRECRAWL_API_KEY": "f"}):
            with patch.dict(os.environ, keys):
                with self.assertRaisesRegex(ValueError, "requires both"):
                    self.research("Partial configuration query")
        with patch.dict(os.environ, {"SERPAPI_API_KEY": "s", "FIRECRAWL_API_KEY": "f"}):
            self.assertEqual(parallel.resolve_research_provider(), "serpapi")
            with self.server.installed():
                self.research("Explicit override query", search_provider="parallel")
        with self.assertRaisesRegex(ValueError, "requires both"):
            self.research("Explicit legacy query", search_provider="serpapi")
        with self.assertRaisesRegex(ValueError, "search_provider"):
            self.research("Invalid provider query", search_provider="typo")

    def test_call_keys_reach_legacy_clients_without_mutating_environment(self):
        self.research.search_provider = "serpapi"
        with patch("minions.utils.serpapi_util.get_web_urls", return_value=["https://example.com"]) as search, \
                patch("minions.utils.firecrawl_util.scrape_url", return_value={"markdown": "Source text. " * 20}) as fetch:
            _, urls = self.research("Legacy configured query", serpapi_key="s", firecrawl_api_key="f")
        self.assertEqual(urls, {"https://example.com"})
        search.assert_called_with("Legacy configured query followup 1", num_urls=2, api_key="s")
        fetch.assert_called_with("https://example.com", api_key="f")
        self.assertNotIn("SERPAPI_API_KEY", os.environ)
        self.assertNotIn("FIRECRAWL_API_KEY", os.environ)
        self.assertFalse(self.server.requests)

    def test_invalid_inputs_fail_before_network(self):
        for query, limit, timeout in [("", 2, 1), ("query", 0, 1), ("query", 21, 1), ("query", 2, 0)]:
            with self.subTest(query=query, limit=limit, timeout=timeout):
                with self.server.installed(), self.assertRaises(ValueError):
                    parallel.research_web(query, limit, "run-id", timeout)
        self.assertFalse(self.server.requests)

    def test_text_response_empty_search_and_source_order(self):
        self.server.text_only = True
        with self.server.installed():
            urls, docs = self.research.extract_metadata("Source ordering query")
            self.assertTrue(docs[0].startswith("Source: " + urls[0]))
            self.server.empty = True
            self.assertEqual(self.research.extract_metadata("Empty search query"), ([], []))

    def test_tool_error_and_quota_propagate_without_provider_fallback(self):
        for failure in ("Upstream error", "Free anonymous quota exceeded"):
            with self.subTest(failure=failure):
                self.server.failure = failure
                with self.server.installed(), self.assertRaises(Exception) as caught:
                    self.research("Failed research query")
                self.assertIn(failure, str(caught.exception))
                self.assertEqual(self.server.tool_calls[-1][0], "web_search")

    def test_fetch_errors_are_reported(self):
        self.server.fetch_errors = True
        with self.server.installed(), self.assertRaises(Exception) as caught:
            self.research("Unreadable source query")
        self.assertIn("could not read", str(caught.exception))

    def test_http_auth_parser_and_discovery_errors_are_not_empty_successes(self):
        for attribute, value in [("http_status", 401), ("http_status", 503),
                                 ("malformed", True), ("missing_tool", True)]:
            with self.subTest(attribute=attribute, value=value):
                self.server = MCPFixture()
                setattr(self.server, attribute, value)
                with self.server.installed(), self.assertRaises(Exception):
                    self.research("Failed service query")

    def test_deadline_cancels_real_mcp_request(self):
        self.server.slow = True
        self.research.search_timeout = 0.1
        with self.server.installed(), self.assertRaises(TimeoutError):
            self.research("Slow research query")
        self.assertTrue(self.server.cancelled)

    def test_cancellation_reaches_transport(self):
        self.server.slow = True

        async def run():
            task = asyncio.create_task(parallel._research_web("Cancelled research query", 2, "run-id", 10))
            await asyncio.to_thread(self.server.started.wait, 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with self.server.installed():
            asyncio.run(run())
        self.assertTrue(self.server.cancelled)


if __name__ == "__main__":
    unittest.main()
