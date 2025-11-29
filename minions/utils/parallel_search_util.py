try:
    from parallel import Parallel
except ImportError:
    Parallel = None

import os
from typing import List, Dict, Any, Optional


def parallel_search(
    objective: str,
    search_queries: Optional[List[str]] = None,
    max_results: int = 10,
    max_chars_per_result: int = 10000,
    api_key: Optional[str] = None
) -> Any:
    """
    Search the web using Parallel AI's Search API.
    
    Args:
        objective: Natural language objective describing what you want to find
        search_queries: Optional list of search queries. If not provided, the API will generate them.
        max_results: Maximum number of results to return (default: 10)
        max_chars_per_result: Maximum characters per result excerpt (default: 10000)
        api_key: Optional API key. If not provided, reads from PARALLEL_API_KEY env var
        
    Returns:
        Search response object with:
            - search_id: Unique identifier for the search
            - results: List of search results with url, title, publish_date, and excerpts
            - warnings: Any warnings from the API
            - usage: Usage information
    
    Raises:
        ValueError: If API key is not set or Parallel library is not available
        Exception: If the API request fails
    """
    if Parallel is None:
        raise ValueError("parallel library is required for parallel_search. Install with: pip install parallel")
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("PARALLEL_API_KEY")
    
    if not api_key:
        raise ValueError("PARALLEL_API_KEY is not set")
    
    # Initialize Parallel client
    client = Parallel(api_key=api_key)
    
    try:
        # Build kwargs for the search call
        search_kwargs = {
            "objective": objective,
            "max_results": max_results,
            "max_chars_per_result": max_chars_per_result,
            "betas": ["search-extract-2025-10-10"]
        }
        
        # Add search queries if provided
        if search_queries:
            search_kwargs["search_queries"] = search_queries
        
        # Perform the search
        search = client.beta.search(**search_kwargs)
        return search
    except Exception as e:
        print(f"[PARALLEL_SEARCH] Error performing search: {e}")
        raise


def parallel_extract(
    urls: List[str],
    objective: Optional[str] = None,
    excerpts: bool = True,
    full_content: bool = False,
    api_key: Optional[str] = None
) -> Any:
    """
    Extract clean markdown content from URLs using Parallel AI's Extract API.
    
    Converts any public URL into clean markdown, including JavaScript-heavy pages and PDFs.
    Returns focused excerpts aligned to your objective, or full page content if requested.
    
    Args:
        urls: List of URLs to extract content from
        objective: Optional natural language objective describing what you want to extract.
                   If provided, excerpts will be focused on this objective.
        excerpts: Whether to return focused excerpts (default: True)
        full_content: Whether to return full page content (default: False)
        api_key: Optional API key. If not provided, reads from PARALLEL_API_KEY env var
        
    Returns:
        Extract response object with:
            - extract_id: Unique identifier for the extraction
            - results: List of extraction results with url, title, publish_date, excerpts/full_content
            - errors: Any errors from the API
            - warnings: Any warnings from the API
            - usage: Usage information
    
    Raises:
        ValueError: If API key is not set or Parallel library is not available
        Exception: If the API request fails
        
    Reference:
        https://docs.parallel.ai/extract/extract-quickstart
    """
    if Parallel is None:
        raise ValueError("parallel library is required for parallel_extract. Install with: pip install parallel")
    
    # Get API key from parameter or environment variable
    if api_key is None:
        api_key = os.getenv("PARALLEL_API_KEY")
    
    if not api_key:
        raise ValueError("PARALLEL_API_KEY is not set")
    
    # Initialize Parallel client
    client = Parallel(api_key=api_key)
    
    try:
        # Build kwargs for the extract call
        extract_kwargs = {
            "urls": urls,
            "excerpts": excerpts,
            "full_content": full_content,
            "betas": ["search-extract-2025-10-10"]
        }
        
        # Add objective if provided
            extract_kwargs["objective"] = objective
        
        # Perform the extraction
        result = client.beta.extract(**extract_kwargs)
        return result.results
    except Exception as e:
        print(f"[PARALLEL_EXTRACT] Error performing extraction: {e}")
        raise


def get_parallel_search_urls(
    objective: str,
    search_queries: Optional[List[str]] = None,
    max_results: int = 10,
    api_key: Optional[str] = None
) -> List[str]:
    """
    Get a list of URLs from Parallel Search API.
    
    Args:
        objective: Natural language objective describing what you want to find
        search_queries: Optional list of search queries
        max_results: Maximum number of results to return
        api_key: Optional API key. If not provided, reads from PARALLEL_API_KEY env var
        
    Returns:
        list: A list of URLs from the search results
    """
    try:
        search = parallel_search(
            objective=objective,
            search_queries=search_queries,
            max_results=max_results,
            api_key=api_key
        )
        return [result.url for result in search.results]
    except Exception as e:
        print(f"[PARALLEL_SEARCH] Error getting URLs: {e}")
        return []


def extract_url_content(
    url: str,
    objective: Optional[str] = None,
    full_content: bool = True,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract content from a single URL and return as a dictionary.
    
    This is a convenience wrapper around parallel_extract for single URL extraction.
    
    Args:
        url: The URL to extract content from
        objective: Optional objective to focus the extraction
        full_content: Whether to return full content (default: True for single URL use case)
        api_key: Optional API key. If not provided, reads from PARALLEL_API_KEY env var
        
    Returns:
        dict: A dictionary containing:
            - url: The extracted URL
            - title: Page title
            - markdown: Full markdown content (if full_content=True)
            - excerpts: Focused excerpts (if full_content=False)
            - publish_date: Publication date if available
    """
    try:
        result = parallel_extract(
            urls=[url],
            objective=objective,
            excerpts=not full_content,
            full_content=full_content,
            api_key=api_key
        )
        
        if result.results and len(result.results) > 0:
            r = result.results[0]
            return {
                "url": getattr(r, "url", url),
                "title": getattr(r, "title", ""),
                "markdown": getattr(r, "full_content", "") if full_content else "",
                "excerpts": getattr(r, "excerpts", []) if not full_content else [],
                "publish_date": getattr(r, "publish_date", None)
            }
        return {"url": url, "title": "", "markdown": "", "excerpts": [], "publish_date": None}
    except Exception as e:
        print(f"[PARALLEL_EXTRACT] Error extracting URL content: {e}")
        return {"url": url, "title": "", "markdown": "", "excerpts": [], "publish_date": None, "error": str(e)}

