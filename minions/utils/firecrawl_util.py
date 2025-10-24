from firecrawl import Firecrawl
import os


# convert this into a function that takes a url and returns the markdown and html
def scrape_url(url, api_key=None):
    """
    Scrape a URL using Firecrawl v2 API.
    
    Args:
        url: The URL to scrape
        api_key: Optional API key. If not provided, reads from FIRECRAWL_API_KEY env var
        
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
    
    # Use the new v2 scrape method with formats as a direct parameter
    result = firecrawl.scrape(url, formats=["markdown", "html"])
    
    # The v2 API returns the data object directly when using the SDK
    # Return in the same format as v1 for backward compatibility
    return {
        "markdown": result.get("markdown", ""),
        "html": result.get("html", ""),
        "metadata": result.get("metadata", {})
    }
