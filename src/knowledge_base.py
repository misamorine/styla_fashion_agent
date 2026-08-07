import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any

DEFAULT_KB_PATH = Path(__file__).resolve().parents[1] / "data" / "fashion_knowledge_base.json"


@lru_cache(maxsize=1)
def load_fashion_kb(path: str = str(DEFAULT_KB_PATH)) -> Dict[str, Any]:
    kb_path = Path(path)
    if not kb_path.exists():
        raise FileNotFoundError(f"Fashion knowledge base not found: {kb_path}")
    with kb_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_occasion_rules(occasion: str) -> Dict[str, Any]:
    kb = load_fashion_kb()
    return kb.get(occasion.lower(), {})


def preferred_terms_for_slot(occasion: str, slot: str) -> List[str]:
    rules = get_occasion_rules(occasion)
    return rules.get("preferred_slots", {}).get(slot, [])


def avoid_terms_for_occasion(occasion: str) -> List[str]:
    rules = get_occasion_rules(occasion)
    return rules.get("avoid_keywords", [])


def preferred_colors_for_occasion(occasion: str) -> List[str]:
    rules = get_occasion_rules(occasion)
    return rules.get("preferred_colors", [])


def style_principles_for_occasion(occasion: str) -> List[str]:
    rules = get_occasion_rules(occasion)
    return rules.get("style_principles", [])


def dress_code_for_occasion(occasion: str) -> str:
    rules = get_occasion_rules(occasion)
    return rules.get("dress_code", occasion.title())


def must_have_slots(occasion: str) -> List[str]:
    rules = get_occasion_rules(occasion)
    return rules.get("must_have_slots", ["top", "bottom", "footwear"])
