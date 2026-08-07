from typing import Dict, List, Optional
from pathlib import Path
from .knowledge_base import (
    avoid_terms_for_occasion,
    dress_code_for_occasion,
    must_have_slots,
    preferred_colors_for_occasion,
    preferred_terms_for_slot,
    style_principles_for_occasion,
)

OCCASION_GROUPS = {
    "business dinner": ["business", "formal", "dinner", "office", "smart casual"],
    "office": ["business", "formal", "office", "smart casual"],
    "wedding": ["wedding", "formal", "party", "ethnic"],
    "college": ["casual", "college", "smart casual"],
    "casual outing": ["casual", "smart casual", "party"],
}

SLOT_QUERY_HINTS = {
    "layer": "formal blazer jacket suit",
    "top": "clean shirt formal top",
    "bottom": "formal trousers chinos pants",
    "footwear": "formal shoes loafers oxford",
    "accessory": "belt watch tie pocket square accessory",
}

COLOR_COMPATIBILITY = {
    "black": ["white", "grey", "charcoal", "navy", "beige", "brown"],
    "navy": ["white", "grey", "beige", "brown", "black"],
    "white": ["black", "navy", "grey", "blue", "brown", "beige"],
    "brown": ["white", "beige", "navy", "blue", "cream", "grey"],
    "grey": ["white", "black", "navy", "blue", "brown"],
    "charcoal": ["white", "black", "navy", "brown"],
    "beige": ["white", "brown", "navy", "black", "olive", "blue"],
    "blue": ["white", "beige", "grey", "navy", "brown"],
    "cream": ["brown", "navy", "maroon", "black", "beige"],
}


def occasion_groups(occasion: str) -> List[str]:
    return OCCASION_GROUPS.get(occasion.lower(), [occasion.lower()])


def outfit_slots(occasion: str) -> List[str]:
    return must_have_slots(occasion.lower())


def color_match_score(base_color: Optional[str], item_color: Optional[str], slot: Optional[str] = None) -> float:
    if not base_color or not item_color:
        return 0.0
    base_color = base_color.lower()
    item_color = item_color.lower()
    if base_color == item_color:
        return 12.0 if slot in ["layer", None] else 6.0
    if item_color in COLOR_COMPATIBILITY.get(base_color, []):
        return 8.0
    return -4.0


def score_item(metadata: Dict, intent, slot: Optional[str] = None) -> float:
    score = 0.0
    text = " ".join(str(metadata.get(k, "")).lower() for k in ["title", "description", "category", "slot", "color", "occasion_group", "gender"])

    item_gender = str(metadata.get("gender", "")).lower()
    if intent.gender in ["male", "female"]:
        if item_gender in [intent.gender, "unisex"]:
            score += 25
        elif item_gender in ["male", "female"] and item_gender != intent.gender:
            score -= 250.0

        if intent.gender == "female" and any(w in text for w in ["men's", "mens", "male blazer"]):
            score -= 250.0
        elif intent.gender == "male" and any(w in text for w in ["women's", "womens", "female"]):
            score -= 250.0

    if slot and metadata.get("slot") == slot:
        score += 35

    # Knowledge-base preferred terms for the occasion and slot.
    preferred_terms = preferred_terms_for_slot(intent.occasion, slot or "")
    if preferred_terms and any(term.lower() in text for term in preferred_terms):
        score += 22

    # Requested category/color from the user's query.
    if intent.color:
        score += color_match_score(intent.color, metadata.get("color"), slot)
    if intent.requested_category_word and intent.requested_category_word in text:
        score += 20

    # Occasion group and occasion-preferred colors.
    if any(g in text for g in occasion_groups(intent.occasion)):
        score += 20
    preferred_colors = preferred_colors_for_occasion(intent.occasion)
    if metadata.get("color") in preferred_colors:
        score += 8

    # Occasion-specific avoid list from the knowledge base.
    avoid_terms = avoid_terms_for_occasion(intent.occasion)
    if any(w.lower() in text for w in avoid_terms):
        score -= 45

    # Body shape rule scoring (BSTI A, H, X, Y)
    if getattr(intent, "body_shape", None):
        shape = intent.body_shape.upper()
        kb_path = Path(__file__).resolve().parents[1] / "knowledge" / f"{shape}.json"
        if kb_path.exists():
            import json
            with open(kb_path, "r", encoding="utf-8") as f:
                shape_data = json.load(f)
                avoid_list = shape_data.get("avoid", [])
                if any(av.lower() in text for av in avoid_list):
                    score -= 30
                
                recs = shape_data.get("recommendations", {})
                for key, items in recs.items():
                    if isinstance(items, list):
                        for rec in items:
                            rec_item = rec.get("item", "").lower() if isinstance(rec, dict) else str(rec).lower()
                            if rec_item and rec_item in text:
                                score += 15

    return score


def compute_look_score(look: Dict, intent) -> float:
    if not look:
        return 0.0
    slots = outfit_slots(intent.occasion)
    present_score = len([s for s in slots if s in look]) / max(len(slots), 1) * 40.0
    avg_item_score = sum(float(item.get("rule_score", 0)) for item in look.values()) / max(len(look), 1)
    preferred_color_hits = sum(1 for item in look.values() if item.get("color") in preferred_colors_for_occasion(intent.occasion))
    color_score = min(20.0, preferred_color_hits * 4.0)
    raw = present_score + 0.5 * avg_item_score + color_score
    return round(max(0.0, min(10.0, raw / 10.0)), 1)


def build_explanation(look: Dict, intent) -> str:
    lines = []
    lines.append(f"Dress code: {dress_code_for_occasion(intent.occasion)}.")
    body_str = f" · Body Shape: {intent.body_shape}" if getattr(intent, "body_shape", None) else ""
    lines.append(f"This look is composed for {intent.gender.title()} · {intent.occasion.title()}{body_str}.")
    if intent.color:
        lines.append(f"The requested {intent.color} tone is prioritized for the main item and compatible supporting items.")

    if getattr(intent, "body_shape", None):
        shape = intent.body_shape.upper()
        kb_path = Path(__file__).resolve().parents[1] / "knowledge" / f"{shape}.json"
        if kb_path.exists():
            import json
            with open(kb_path, "r", encoding="utf-8") as f:
                shape_data = json.load(f)
                lines.append(f"Body Shape Styling Goal ({shape_data.get('name', shape)}): {shape_data.get('style_goal', '')}")

    principles = style_principles_for_occasion(intent.occasion)
    if principles:
        lines.append("Styling principles used:")
        for p in principles[:4]:
            lines.append(f"• {p}")

    selected_bits = []
    for slot, item in look.items():
        title = item.get("title", "selected item")
        color = item.get("color", "unknown color")
        selected_bits.append(f"• {slot.title()}: {title} ({color})")
    if selected_bits:
        lines.append("Selected outfit structure:")
        lines.extend(selected_bits)

    lines.append("Selection method: ChromaDB metadata filters → Progressive relaxation → Body Shape (A/H/X/Y) scoring → FashionCLIP similarity.")
    return "\n\n".join(lines)

