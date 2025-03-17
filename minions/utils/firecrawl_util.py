from firecrawl import FirecrawlApp
import os


# convert this into a function that takes a url and returns the markdown and html
def scrape_url(url, api_key=None):
    # reads environment variable FIRECRAWL_API_KEY
    if api_key is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")

    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY is not set")

    app = FirecrawlApp(api_key=api_key)
    return app.scrape_url(url, params={"formats": ["markdown", "html"]})
