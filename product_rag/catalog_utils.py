import re
from pathlib import Path
from typing import Dict

from PIL import Image

# from .intent_parser import CATEGORY_NORMALIZATION, COLORS
# from .rules import occasion_groups
CATEGORY_NORMALIZATION = {
    "shirt": "top",
    "tshirt": "top",
    "t-shirt": "top",
    "jeans": "bottom",
    "pants": "bottom",
    "dress": "dress",
    "skirt": "bottom",
    "blazer": "jacket",
    "jacket": "jacket",
}

COLORS = [
    "black","white","blue","red","green","pink",
    "grey","gray","brown","beige","navy","cream"
]

def safe_text(x) -> str:
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def infer_slot(text: str) -> str:
    t = text.lower()
    for word, slot in CATEGORY_NORMALIZATION.items():
        if word in t:
            return slot
    return "unknown"


def infer_color(text: str) -> str:
    t = text.lower()
    for c in COLORS:
        if c in t:
            return "grey" if c == "gray" else c
    return "unknown"


def infer_gender(text: str) -> str:
    t = text.lower()
    women_words = ["women", "woman", "female", "ladies", "girl", "girls", "dress", "skirt", "heels", "blouse"]
    men_words = ["men", "man", "male", "mens", "boys", "boy", "oxford", "loafers", "suit", "blazer"]
    if any(w in t for w in women_words):
        return "female"
    if any(w in t for w in men_words):
        return "male"
    return "unisex"


def infer_occasion_group(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["blazer", "suit", "formal", "office", "business", "oxford", "loafers", "trouser"]):
        return "business"
    if any(w in t for w in ["wedding", "party", "gown", "dress"]):
        return "wedding"
    if any(w in t for w in ["jeans", "tee", "t-shirt", "sneaker", "casual"]):
        return "casual"
    return "smart casual"


def metadata_from_item(item: Dict, idx: int, image_path: str) -> Dict:
    possible_text_fields = ["title", "name", "description", "category", "semantic_category", "fine_category"]
    combined = " ".join(safe_text(item.get(f, "")) for f in possible_text_fields)
    if not combined.strip():
        combined = f"fashion item {idx}"

    slot = infer_slot(combined)
    color = infer_color(combined)
    gender = infer_gender(combined)
    occasion_group = infer_occasion_group(combined)

    return {
        "title": safe_text(item.get("title") or item.get("name") or f"Fashion Item {idx}"),
        "description": safe_text(item.get("description") or combined),
        "category": safe_text(item.get("category") or item.get("semantic_category") or slot),
        "slot": slot,
        "color": color,
        "gender": gender,
        "occasion_group": occasion_group,
        "image_path": image_path,
    }


def save_dataset_image(image, out_dir: Path, idx: int) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"item_{idx}.jpg"
    if isinstance(image, Image.Image):
        img = image.convert("RGB")
        img.save(path, quality=90)
        return str(path)
    raise ValueError("Dataset item does not contain a PIL image in expected field")
