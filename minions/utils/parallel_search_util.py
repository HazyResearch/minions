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

