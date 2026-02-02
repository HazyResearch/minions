from firecrawl import FirecrawlApp
import os
from typing import Optional, List, Dict, Any


def _get_client(api_key: Optional[str] = None) -> FirecrawlApp:
    """Get a FirecrawlApp client instance."""
    if api_key is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
    
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set")
    
    return FirecrawlApp(api_key=api_key)


def scrape_url(url, api_key=None, min_age=None):
    """
    Scrape a URL using Firecrawl v2 API.
    
    Args:
        url: The URL to scrape
        api_key: Optional API key. If not provided, reads from FIRECRAWL_API_KEY env var
        min_age: (int) The minimum cached age (in milliseconds) required before re-scraping.
                 - If cache age < min_age: Returns cached data.
                 - If cache age > min_age: Triggers a fresh scrape.
                 - Default (None) uses Firecrawl's default (typically 24-48h).
        
    Returns:
        dict: A dictionary containing 'markdown' and 'html' keys with the scraped content,
              plus 'metadata' with page information
    """
    app = _get_client(api_key)
    
    # Prepare parameters
    # We map 'min_age' to the API's 'maxAge' parameter which controls cache tolerance.
    params = {}
    if min_age is not None:
        params["maxAge"] = min_age

    try:
        result = app.scrape_url(
            url, 
            params={
                "formats": ["markdown", "html"],
                **params
            }
        )
    except Exception as e:
        print(f"Scrape failed: {e}")
        raise
    
    # Return in the same format as v1 for backward compatibility
    return {
        "markdown": result.get("markdown", ""),
        "html": result.get("html", ""),
        "metadata": result.get("metadata", {})
    }


def agent(
    prompt: str,
    urls: Optional[List[str]] = None,
    schema: Optional[Dict[str, Any]] = None,
    model: str = "spark-1-mini",
    max_credits: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Firecrawl Agent to search, navigate, and gather data from the web.

    See: https://docs.firecrawl.dev/features/agent
        
    Example:
        # Simple extraction
        result = agent(prompt="Find the pricing of Firecrawl")
        print(result.data)
        
        # Complex research with Spark 1 Pro
        result = agent(
            prompt="Compare all enterprise features and pricing across Firecrawl, Apify, and ScrapingBee",
            model="spark-1-pro"
        )
    """
    app = _get_client(api_key)
    
    kwargs = {
        "prompt": prompt,
        "model": model,
    }
    
    if urls:
        kwargs["urls"] = urls
    
    if schema:
        kwargs["schema"] = schema
    
    if max_credits is not None:
        kwargs["maxCredits"] = max_credits
    
    try:
        result = app.agent(**kwargs)
        return result
    except Exception as e:
        print(f"Agent failed: {e}")
        raise
