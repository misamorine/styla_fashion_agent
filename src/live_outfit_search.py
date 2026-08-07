"""Build SerpAPI shopping queries for outfit slots (body shape + occasion aware)."""

from __future__ import annotations

from typing import Any, Optional

from .body_shop_queries import shoppable_queries_for_shape
from .knowledge_base import preferred_terms_for_slot
from .rules import SLOT_QUERY_HINTS

SLOT_TO_SHAPE_CATEGORIES = {
    "top": ("tops", "necklines", "sleeves"),
    "bottom": ("bottoms",),
    "layer": ("jackets",),
    "footwear": (),
    "accessory": (),
}

GENDER_LABEL = {
    "male": "Male",
    "female": "Female",
    "unisex": "Unisex",
}


def normalize_shopping_store(preferred_store: str) -> str:
    """Map UI store labels to SerpAPI-friendly store names."""
    store = (preferred_store or "").strip()
    if not store or store == "All Stores":
        return "Myntra"
    if store == "Amazon.in":
        return "Amazon"
    return store


def gender_for_serpapi(gender: str) -> str:
    return GENDER_LABEL.get((gender or "").lower(), "Unisex")


def build_slot_search_term(intent, slot: str) -> str:
    """Compose a shoppable search phrase for one outfit slot."""
    parts: list[str] = []

    if getattr(intent, "color", None):
        parts.append(str(intent.color))

    # Prefer body-shape recommendations that map to this slot.
    shape = getattr(intent, "body_shape", None)
    if shape:
        try:
            shape_queries = shoppable_queries_for_shape(shape, max_queries=8)
        except Exception:
            shape_queries = []
        wanted = SLOT_TO_SHAPE_CATEGORIES.get(slot, ())
        for row in shape_queries:
            if row.get("category") in wanted:
                parts.append(row["search_term"])
                break

    preferred = preferred_terms_for_slot(intent.occasion, slot)
    if preferred:
        parts.append(preferred[0])

    if not parts:
        parts.append(SLOT_QUERY_HINTS.get(slot, slot))

    # Include a light touch of the user's vibe/request when relevant.
    request = str(getattr(intent, "query", "") or "").strip()
    if request and slot in ("layer", "top") and len(parts) < 3:
        # Keep short — avoid dumping the whole sentence into every slot.
        words = [w for w in request.split() if len(w) > 3][:3]
        if words:
            parts.extend(words[:2])

    # Deduplicate while preserving order.
    seen = set()
    cleaned = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(part)

    return " ".join(cleaned).strip()


def serpapi_product_to_outfit_item(
    product: dict[str, Any],
    *,
    slot: str,
    intent,
    search_query: str,
    preferred_store: str,
) -> dict[str, Any]:
    """Normalize a SerpAPI product into the outfit item schema used by the UI."""
    title = product.get("title") or f"{slot.title()} recommendation"
    url = product.get("url") or ""
    item_id = f"live_{abs(hash(url or title)) % 10_000_000_000}"

    return {
        "id": item_id,
        "source": "serpapi",
        "owned": False,
        "title": title,
        "display_title": title,
        "description": product.get("description") or f"Live store suggestion for {slot}",
        "category": slot,
        "slot": slot,
        "color": "unknown",
        "gender": getattr(intent, "gender", "unisex"),
        "occasion_group": getattr(intent, "occasion", "unknown"),
        "image_path": "",
        "image": product.get("image") or "",
        "url": url,
        "store": product.get("store") or preferred_store,
        "price": product.get("price") or "Price not available",
        "rule_score": 20.0,
        "vector_score": 0.0,
        "final_score": 20.0,
        "used_where_filter": f"SerpAPI live search · store={preferred_store} · q={search_query}",
    }
