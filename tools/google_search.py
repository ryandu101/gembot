import os
import aiohttp
from typing import Any, Dict, List

class GoogleSearchError(Exception):
    pass

async def google_search_impl(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Perform a Google Custom Search (JSON API) and return structured results.
    Requires env vars:
      - GOOGLE_CSE_API_KEY
      - GOOGLE_CSE_ID
    """
    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        raise GoogleSearchError("Missing GOOGLE_CSE_API_KEY or GOOGLE_CSE_ID environment variables.")

    num_results = max(1, min(10, int(num_results)))
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,
        "safe": "off"
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params, timeout=20) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise GoogleSearchError(f"CSE HTTP {resp.status}: {text}")
            data = await resp.json()

    items = data.get("items", []) or []
    results: List[Dict[str, Any]] = []
    for it in items:
        results.append({
            "title": it.get("title"),
            "link": it.get("link"),
            "snippet": it.get("snippet"),
            "displayLink": it.get("displayLink"),
            "source": "google_cse"
        })

    return {"query": query, "results": results}