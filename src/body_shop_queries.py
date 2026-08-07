"""Derive live shopping queries from body-shape knowledge (not the catalog dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

# Categories that map cleanly to Google Shopping product searches.
SHOPPABLE_CATEGORIES = ("tops", "bottoms", "dresses", "jackets", "sleeves")

# Append a product noun when the knowledge item is a style adjective/phrase.
CATEGORY_SUFFIX = {
    "tops": "top",
    "bottoms": "pants",
    "dresses": "dress",
    "jackets": "jacket",
    "sleeves": "top",
}

# Already-concrete item phrases that should be searched as-is.
CONCRETE_TERMS = (
    "jeans",
    "trousers",
    "pants",
    "dress",
    "blazer",
    "jacket",
    "top",
    "blouse",
    "shirt",
    "skirt",
    "coat",
)


def load_shape_knowledge(shape: str) -> dict[str, Any]:
    """Load knowledge/{shape}.json for a BSTI body-shape code."""
    code = (shape or "").strip().upper()
    path = KNOWLEDGE_DIR / f"{code}.json"
    if not path.exists():
        raise FileNotFoundError(f"No body-shape knowledge file for '{code}' at {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _to_search_term(item_name: str, category: str) -> str:
    """Turn a styling recommendation into a shoppable search phrase."""
    name = (item_name or "").strip()
    if not name:
        return ""

    lower = name.lower()
    tokens = set(lower.replace("-", " ").split())
    if tokens & set(CONCRETE_TERMS):
        return name

    # Knowledge often uses plural category nouns already ("Structured Tops").
    for plural, singular in (
        ("tops", "top"),
        ("bottoms", "pants"),
        ("dresses", "dress"),
        ("jackets", "jacket"),
        ("skirts", "skirt"),
    ):
        if lower.endswith(plural):
            return f"{name[: -len(plural)].strip()} {singular}".strip()

    suffix = CATEGORY_SUFFIX.get(category, "")
    if suffix and suffix not in tokens:
        return f"{name} {suffix}".strip()
    return name


def shoppable_queries_for_shape(
    shape: str,
    max_queries: int = 4,
) -> list[dict[str, Any]]:
    """
    Build prioritized live-shopping queries from body-shape knowledge.

    Returns dicts with: category, item, reason, priority, search_term
    """
    knowledge = load_shape_knowledge(shape)
    recommendations = knowledge.get("recommendations") or {}

    candidates: list[dict[str, Any]] = []
    for category in SHOPPABLE_CATEGORIES:
        for rec in recommendations.get(category) or []:
            item = str(rec.get("item") or "").strip()
            if not item:
                continue
            search_term = _to_search_term(item, category)
            if not search_term:
                continue
            candidates.append(
                {
                    "category": category,
                    "item": item,
                    "reason": str(rec.get("reason") or ""),
                    "priority": int(rec.get("priority") or 0),
                    "search_term": search_term,
                    "shape": knowledge.get("shape", shape),
                    "shape_name": knowledge.get("name", ""),
                    "style_goal": knowledge.get("style_goal", ""),
                }
            )

    candidates.sort(key=lambda row: (-row["priority"], row["item"]))

    # Deduplicate similar search terms while keeping highest priority.
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    for row in candidates:
        key = row["search_term"].lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= max_queries:
            break

    return selected
