"""SerpAPI Google Shopping integration for Live Store Recommendation."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# Load .env from the project root (parent of src/), not the process cwd.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

SERPAPI_URL = "https://serpapi.com/search.json"
MAX_PRODUCTS = 9


def get_api_key() -> str:
    """Load SerpAPI key from environment."""
    load_dotenv(_ENV_PATH, override=True)
    api_key = os.getenv("SERPAPI_API_KEY", "").strip().strip('"').strip("'")
    if not api_key or api_key == "your_serpapi_api_key_here":
        raise ValueError(
            "SERPAPI_API_KEY is missing or still set to the placeholder. "
            f"Save your real key in {_ENV_PATH}, then click Recommend Products again."
        )
    return api_key


def build_search_query(gender: str, category: str, store: str) -> str:
    """Build a shopping search query from user selections."""
    gender_prefix = {
        "Male": "Men's",
        "Female": "Women's",
        "Unisex": "Unisex",
    }.get(gender, gender)

    return f"{gender_prefix} {category} {store}"


def _extract_description(item: dict[str, Any]) -> str:
    """Pick the best available short description from a shopping result."""
    if item.get("snippet"):
        return str(item["snippet"])
    if item.get("description"):
        return str(item["description"])
    extensions = item.get("extensions")
    if isinstance(extensions, list) and extensions:
        return ", ".join(str(ext) for ext in extensions)
    return "No description available."


def _store_matches(store_name: str, preferred_store: str) -> bool:
    """Return True when a merchant name matches the selected store."""
    name = (store_name or "").lower()
    preferred = (preferred_store or "").lower()
    if not name or not preferred:
        return False
    return preferred in name or name in preferred


def _is_direct_store_url(url: str) -> bool:
    """True when the URL points at a merchant site, not Google Shopping."""
    if not url:
        return False
    host = urlparse(url).netloc.lower()
    return bool(host) and "google." not in host


def _pick_store_offer(
    stores: list[dict[str, Any]], preferred_store: str
) -> Optional[dict[str, Any]]:
    """Prefer an offer from the selected store; otherwise use the first with a link."""
    preferred_offers = [
        store
        for store in stores
        if store.get("link") and _store_matches(store.get("name", ""), preferred_store)
    ]
    if preferred_offers:
        return preferred_offers[0]

    for store in stores:
        if store.get("link") and _is_direct_store_url(store["link"]):
            return store
    return None


def _fetch_immersive_stores(api_url: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch seller/store offers for a shopping result."""
    response = requests.get(api_url, params={"api_key": api_key}, timeout=30)
    if response.status_code == 401:
        raise RuntimeError(
            "SerpAPI returned 401 Unauthorized while resolving product links. "
            "Check SERPAPI_API_KEY in .env."
        )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    product_results = data.get("product_results") or {}
    return product_results.get("stores") or []


def _resolve_product(
    item: dict[str, Any], preferred_store: str, api_key: str
) -> dict[str, str]:
    """
    Normalize a shopping result and replace Google links with the
    selected store's direct product URL when available.
    """
    price = item.get("price") or item.get("extracted_price")
    if price is not None and not isinstance(price, str):
        price = f"₹{price}"

    product = {
        "title": item.get("title") or "Untitled product",
        "price": str(price) if price else "Price not available",
        "store": item.get("source") or item.get("merchant") or "Unknown store",
        "description": _extract_description(item),
        "image": item.get("thumbnail") or item.get("serpapi_thumbnail") or "",
        "url": item.get("link") or item.get("product_link") or "",
    }

    # Prefer an already-direct merchant link when present.
    if _is_direct_store_url(product["url"]):
        return product

    immersive_api = item.get("serpapi_immersive_product_api")
    if not immersive_api:
        return product

    try:
        stores = _fetch_immersive_stores(immersive_api, api_key)
        offer = _pick_store_offer(stores, preferred_store)
    except Exception:
        return product

    if not offer:
        return product

    product["url"] = offer["link"]
    product["store"] = offer.get("name") or product["store"]
    if offer.get("price"):
        product["price"] = str(offer["price"])
    if offer.get("title"):
        product["title"] = str(offer["title"])
    return product


def search_products(
    query: str,
    preferred_store: str,
    num_results: int = MAX_PRODUCTS,
) -> list[dict[str, str]]:
    """
    Search Google Shopping via SerpAPI and return up to `num_results` products
    with Buy Now links pointing to the selected store's product page.
    """
    api_key = get_api_key()
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": api_key,
        "hl": "en",
        "gl": "in",
        "num": num_results,
    }

    response = requests.get(SERPAPI_URL, params=params, timeout=30)
    if response.status_code == 401:
        raise RuntimeError(
            "SerpAPI returned 401 Unauthorized. Check that SERPAPI_API_KEY in .env "
            "is a valid key from https://serpapi.com/manage-api-key, then fully "
            "restart Streamlit (stop the terminal process and run it again)."
        )
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(data["error"])

    shopping_results = data.get("shopping_results") or []
    preferred_lower = preferred_store.lower()
    matched = [
        item
        for item in shopping_results
        if preferred_lower in (item.get("source") or "").lower()
    ]
    others = [item for item in shopping_results if item not in matched]
    selected = (matched + others)[:num_results]

    if not selected:
        return []

    products: list[Optional[dict[str, str]]] = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=min(6, len(selected))) as executor:
        futures = {
            executor.submit(_resolve_product, item, preferred_store, api_key): index
            for index, item in enumerate(selected)
        }
        for future in as_completed(futures):
            products[futures[future]] = future.result()

    return [product for product in products if product is not None]
