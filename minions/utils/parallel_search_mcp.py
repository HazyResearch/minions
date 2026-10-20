"""Anonymous web research through Parallel's Streamable HTTP MCP server."""

import asyncio
import json
import os
from datetime import timedelta
from typing import Optional
from urllib.parse import urlsplit

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "minions/0.1.0 (https://github.com/HazyResearch/minions)"


def resolve_research_provider(
    provider: str = "auto",
    serpapi_key: Optional[str] = None,
    firecrawl_key: Optional[str] = None,
) -> str:
    """Keep intentional legacy configuration; use Parallel when unconfigured."""
    if provider not in ("auto", "parallel", "serpapi", "disabled"):
        raise ValueError("search_provider must be auto, parallel, serpapi or disabled")
    serpapi_key = os.getenv("SERPAPI_API_KEY") if serpapi_key is None else serpapi_key
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY") if firecrawl_key is None else firecrawl_key
    if provider == "auto":
        provider = "serpapi" if serpapi_key or firecrawl_key else "parallel"
    if provider == "serpapi" and not (serpapi_key and firecrawl_key):
        raise ValueError("SerpAPI search requires both SERPAPI_API_KEY and FIRECRAWL_API_KEY")
    return provider


def _tool_payload(result, tool_name: str) -> dict:
    if result.isError:
        message = "\n".join(block.text for block in result.content if block.type == "text")
        raise RuntimeError(f"Parallel {tool_name} failed: {message}")
    payload = result.structuredContent
    if payload is None:
        text = "\n".join(block.text for block in result.content if block.type == "text")
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"Parallel {tool_name} returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"Parallel {tool_name} returned no results list")
    return payload


def _web_url(url) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        return parsed.scheme in ("http", "https") and bool(parsed.hostname) and not (
            parsed.username or parsed.password
        )
    except ValueError:
        return False


async def _research_web(query: str, max_sources: int, session_id: str, timeout: float):
    # No environment credentials or saved MCP configuration are loaded here.
    async with streamablehttp_client(
        MCP_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=timedelta(seconds=timeout),
        sse_read_timeout=timedelta(seconds=timeout),
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            if not {"web_search", "web_fetch"}.issubset(tool.name for tool in tools.tools):
                raise ValueError("Parallel MCP is missing web_search or web_fetch")
            search = _tool_payload(
                await session.call_tool("web_search", {
                    "objective": query,
                    "search_queries": [query],
                    "session_id": session_id,
                }),
                "web_search",
            )
            urls = []
            for result in search["results"]:
                url = result.get("url")
                if _web_url(url) and url not in urls:
                    urls.append(url)
                if len(urls) == max_sources:
                    break
            if not urls:
                return [], []
            # DeepResearch summarizes source documents, so request page markdown.
            fetched = _tool_payload(
                await session.call_tool("web_fetch", {
                    "urls": urls,
                    "objective": query[:200],
                    "search_queries": [query],
                    "full_content": True,
                    "session_id": session_id,
                }),
                "web_fetch",
            )
            documents = {}
            for result in fetched["results"]:
                url = result.get("url")
                if url not in urls:
                    continue
                content = result.get("full_content") or "\n\n".join(result.get("excerpts") or [])
                if content:
                    documents[url] = f"Source: {url}\n\n{content}"
            if not documents and fetched.get("errors"):
                raise RuntimeError("Parallel web_fetch could not read the selected sources")
            read_urls = [url for url in urls if url in documents]
            return read_urls, [documents[url] for url in read_urls]


def research_web(query: str, max_sources: int, session_id: str, timeout: float = 60.0):
    """Search and read up to 20 sources, within one cancellable deadline.

    The caller owns session_id: reuse it across rounds of a research run, and
    generate a new UUID for each independent run. Errors propagate to the caller;
    an explicit provider failure never switches to another service.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A nonempty research query is required")
    if not 1 <= max_sources <= 20:
        raise ValueError("max_sources must be between 1 and 20")
    if timeout <= 0:
        raise ValueError("search_timeout must be positive")

    async def run():
        return await asyncio.wait_for(
            _research_web(query, max_sources, session_id, timeout), timeout=timeout
        )

    try:
        return asyncio.run(run())
    except Exception as exc:
        # The MCP SDK's task groups can wrap a single actionable tool/HTTP error.
        # Expose that error to the synchronous caller and Streamlit UI.
        while len(getattr(exc, "exceptions", ())) == 1:
            exc = exc.exceptions[0]
        raise exc from None
