from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

COLORS = [
    "black", "white", "navy", "blue", "brown", "grey", "gray", "charcoal",
    "beige", "cream", "green", "red", "maroon", "pink", "purple", "yellow"
]

CATEGORY_KEYWORDS = {
    "layer": ["blazer", "jacket", "coat", "suit"],
    "top": ["shirt", "t-shirt", "tee", "polo", "kurta", "top"],
    "bottom": ["trouser", "trousers", "pants", "jeans", "chinos", "bottom"],
    "footwear": ["shoe", "shoes", "loafers", "loafter", "oxford", "sneakers", "boots"],
    "accessory": ["belt", "watch", "tie", "pocket square", "accessory", "wallet"],
}

CATEGORY_NORMALIZATION = {
    "blazer": "layer", "jacket": "layer", "coat": "layer", "suit": "layer",
    "shirt": "top", "t-shirt": "top", "tee": "top", "polo": "top", "kurta": "top",
    "trouser": "bottom", "trousers": "bottom", "pants": "bottom", "jeans": "bottom", "chinos": "bottom",
    "shoe": "footwear", "shoes": "footwear", "loafers": "footwear", "oxford": "footwear", "sneakers": "footwear", "boots": "footwear",
    "belt": "accessory", "watch": "accessory", "tie": "accessory", "pocket square": "accessory",
}


class StyleIntentPydantic(BaseModel):
    """LangChain Pydantic schema for user styling intent parsing."""
    color: Optional[str] = Field(None, description="Primary color preference mentioned in query")
    requested_slot: Optional[str] = Field(None, description="Clothing category slot: layer, top, bottom, footwear, accessory")
    requested_category_word: Optional[str] = Field(None, description="Specific clothing keyword found in query")


@dataclass
class StyleIntent:
    query: str
    gender: str
    occasion: str
    color: Optional[str] = None
    requested_slot: Optional[str] = None
    requested_category_word: Optional[str] = None
    body_shape: Optional[str] = None


def parse_intent(query: str, gender: str, occasion: str, body_shape: Optional[str] = None) -> StyleIntent:
    q = query.lower().strip()
    color = None
    for c in COLORS:
        if c in q:
            color = "grey" if c == "gray" else c
            break

    requested_slot = None
    requested_category_word = None
    for slot, words in CATEGORY_KEYWORDS.items():
        for w in words:
            if w in q:
                requested_slot = slot
                requested_category_word = w
                break
        if requested_slot:
            break

    return StyleIntent(
        query=query,
        gender=gender.lower(),
        occasion=occasion.lower(),
        color=color,
        requested_slot=requested_slot,
        requested_category_word=requested_category_word,
        body_shape=body_shape.upper() if body_shape else None,
    )


def parse_intent_with_langchain(query: str, gender: str, occasion: str, body_shape: Optional[str] = None) -> StyleIntent:
    """Parses intent using LangChain structured output with fallback to rule-based parser."""
    try:
        from rag.llm import get_stylist_llm
        llm = get_stylist_llm()
        if llm and hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(StyleIntentPydantic)
            parsed = structured_llm.invoke(f"Extract clothing slots and color from query: '{query}'")
            return StyleIntent(
                query=query,
                gender=gender.lower(),
                occasion=occasion.lower(),
                color=parsed.color,
                requested_slot=parsed.requested_slot,
                requested_category_word=parsed.requested_category_word,
                body_shape=body_shape.upper() if body_shape else None,
            )
    except Exception:
        pass

    return parse_intent(query, gender, occasion, body_shape=body_shape)


