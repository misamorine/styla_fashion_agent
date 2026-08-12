from typing import Dict, List, Optional

from .rules import outfit_slots, build_explanation, compute_look_score


class OutfitComposer:
    def __init__(self, retriever=None, wardrobe_manager=None):
        self.retriever = retriever
        self.wardrobe_manager = wardrobe_manager

    def compose(
        self,
        intent,
        use_wardrobe_first: bool = True,
        preferred_store: str = "All Stores",
        use_live_api: bool = False,
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

            catalog_candidates: List[Dict] = []
            if self.retriever is not None:
                catalog_candidates = self.retriever.retrieve_for_slot(
                    intent, slot=slot, n_results=10
                )
                if catalog_candidates:
                    sources_used.add("catalog")

            # Prioritize wardrobe first (if enabled), then Marqo Polyvore dataset catalog items.
            if use_wardrobe_first and wardrobe_candidates:
                candidates = wardrobe_candidates + catalog_candidates
                sources_used.add("wardrobe")
            else:
                candidates = catalog_candidates + wardrobe_candidates

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
