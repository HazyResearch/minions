from serpapi import GoogleSearch
import os

def get_web_urls(query: str, num_urls: int = 5) -> list[str]:
    """
    Search the web for for relevant urls using SerpAPI
    Args:
        query: The query to search the web for
    Returns:
        A list of urls
    """
    api_key = os.getenv('SERPAPI_API_KEY')
    
    # Remove quotes from query if present
    query = query.strip('"').strip("'")
    
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "gl": "us",
        "hl": "en",
        "num": num_urls,
        "google_domain": "google.com",
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Check if there are any organic results
        if 'organic_results' not in results or not results['organic_results']:
            return []
            
        return [result['link'] for result in results['organic_results']]
    except Exception as e:
        print(f"[SERPAPI] Error searching web: {e}")
        return []
