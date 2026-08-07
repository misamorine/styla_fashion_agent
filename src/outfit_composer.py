from typing import Dict, List, Optional

from .live_outfit_search import (
    build_slot_search_term,
    gender_for_serpapi,
    normalize_shopping_store,
    serpapi_product_to_outfit_item,
)
from .rules import outfit_slots, build_explanation, compute_look_score
from .serpapi_service import build_search_query, search_products


class OutfitComposer:
    def __init__(self, retriever=None, wardrobe_manager=None):
        self.retriever = retriever
        self.wardrobe_manager = wardrobe_manager

    def _live_candidates_for_slot(
        self,
        intent,
        slot: str,
        preferred_store: str,
        n_results: int = 5,
    ) -> List[Dict]:
        store = normalize_shopping_store(preferred_store)
        search_term = build_slot_search_term(intent, slot)
        query = build_search_query(gender_for_serpapi(intent.gender), search_term, store)
        try:
            products = search_products(query, preferred_store=store, num_results=n_results)
        except Exception:
            return []

        return [
            serpapi_product_to_outfit_item(
                product,
                slot=slot,
                intent=intent,
                search_query=query,
                preferred_store=store,
            )
            for product in products
        ]

    def compose(
        self,
        intent,
        use_wardrobe_first: bool = True,
        preferred_store: str = "Myntra",
        use_live_api: bool = True,
        catalog_fallback: bool = True,
    ) -> Dict:
        slots = outfit_slots(intent.occasion)
        # If user explicitly asks for a category, make sure it appears first.
        if intent.requested_slot and intent.requested_slot in slots:
            slots = [intent.requested_slot] + [s for s in slots if s != intent.requested_slot]

        selected = {}
        alternatives = {}
        owned_selected = {}
        missing_slots = []
        used_ids = set()
        sources_used = set()

        for slot in slots:
            wardrobe_candidates: List[Dict] = []
            if use_wardrobe_first and self.wardrobe_manager is not None:
                wardrobe_candidates = self.wardrobe_manager.candidates_for_slot(intent, slot=slot)

            live_candidates: List[Dict] = []
            if use_live_api:
                live_candidates = self._live_candidates_for_slot(
                    intent, slot, preferred_store=preferred_store, n_results=5
                )
                if live_candidates:
                    sources_used.add("serpapi")

            catalog_candidates: List[Dict] = []
            if catalog_fallback and self.retriever is not None and (
                not wardrobe_candidates and not live_candidates
            ):
                catalog_candidates = self.retriever.retrieve_for_slot(
                    intent, slot=slot, n_results=10
                )
                if catalog_candidates:
                    sources_used.add("catalog")

            # Prefer wardrobe when enabled; otherwise prefer live API over dataset.
            if use_wardrobe_first and wardrobe_candidates:
                candidates = wardrobe_candidates + live_candidates + catalog_candidates
                sources_used.add("wardrobe")
            else:
                candidates = live_candidates + wardrobe_candidates + catalog_candidates

            candidates = [c for c in candidates if c.get("id") not in used_ids]
            alternatives[slot] = candidates

            if candidates:
                selected[slot] = candidates[0]
                used_ids.add(candidates[0]["id"])
                if candidates[0].get("owned"):
                    owned_selected[slot] = candidates[0]
                    sources_used.add("wardrobe")
                else:
                    missing_slots.append(slot)
            else:
                missing_slots.append(slot)

        explanation = build_explanation(selected, intent)
        if "serpapi" in sources_used:
            explanation += (
                "\n\nSelection method: Wardrobe-first (optional) → SerpAPI live store "
                "search (body shape + occasion) → catalog fallback only if needed."
            )

        return {
            "intent": intent,
            "selected": selected,
            "alternatives": alternatives,
            "look_score": compute_look_score(selected, intent),
            "owned_selected": owned_selected,
            "missing_slots": missing_slots,
            "shopping_needed": {
                slot: item for slot, item in selected.items() if not item.get("owned")
            },
            "sources_used": sorted(sources_used),
            "explanation": explanation,
        }
