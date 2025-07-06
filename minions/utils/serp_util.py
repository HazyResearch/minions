import os
import requests

def search_serp(query: str, num_results: int = 5, api_key: str = None):
    if api_key is None:
        api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SERPAPI_KEY is not set")

    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
    }

    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    data = response.json()

    return data.get("organic_results", [])[:num_results]
