from firecrawl import Firecrawl
import os

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
    # reads environment variable FIRECRAWL_API_KEY
    if api_key is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set")

    # Initialize Firecrawl v2 client
    firecrawl = Firecrawl(api_key=api_key)
    
    # Prepare parameters
    # We map 'min_age' to the API's 'maxAge' parameter which controls cache tolerance.
    params = {}
    if min_age is not None:
        params["maxAge"] = min_age

    try:
        # Pass params via the 'params' dictionary for v2 compatibility
        result = firecrawl.scrape(
            url, 
            formats=["markdown", "html"],
            params=params
        )
    except Exception as e:
        # Fallback logging or handling if needed
        print(f"Scrape failed: {e}")
        raise
    
    # Return in the same format as v1 for backward compatibility
    return {
        "markdown": getattr(result, "markdown", ""),
        "html": getattr(result, "html", ""),
        "metadata": getattr(result, "metadata", {})
    }