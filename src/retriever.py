from typing import Dict, List, Optional

from .rules import occasion_groups, score_item, SLOT_QUERY_HINTS
from .vector_store import and_filter, FashionVectorStore


def build_filters(intent, slot: Optional[str] = None, include_color: bool = False, include_occasion: bool = True) -> Dict:
    filters = []
    if intent.gender in ["male", "female"]:
        filters.append({"gender": {"$in": [intent.gender, "unisex"]}})
    if slot:
        filters.append({"slot": slot})
    if include_color and intent.color:
        filters.append({"color": intent.color})
    if include_occasion:
        filters.append({"occasion_group": {"$in": occasion_groups(intent.occasion)}})
    return and_filter(filters)


class FashionRetriever:
    def __init__(self, store: FashionVectorStore, embedder):
        self.store = store
        self.embedder = embedder

    def retrieve_for_slot(self, intent, slot: str, n_results: int = 12) -> List[Dict]:
        # Progressive relaxation: strict -> relaxed. Each call uses ChromaDB `where` metadata filtering.
        query_text = f"{intent.query}. {intent.gender} {intent.occasion} {SLOT_QUERY_HINTS.get(slot, slot)}"
        if intent.color and (slot == intent.requested_slot or slot == "layer"):
            query_text += f" {intent.color}"
        q_emb = self.embedder.text_embedding(query_text)

        gender_only_filter = build_filters(intent, slot=None, include_color=False, include_occasion=False)
        filter_attempts = [
            build_filters(intent, slot=slot, include_color=True, include_occasion=True),
            build_filters(intent, slot=slot, include_color=False, include_occasion=True),
            build_filters(intent, slot=slot, include_color=False, include_occasion=False),
            gender_only_filter,
        ]
        if not (intent.gender in ["male", "female"]):
            filter_attempts.append(None)

        seen = set()
        combined = []
        for where in filter_attempts:
            try:
                results = self.store.query(q_emb, where=where, n_results=n_results)
            except Exception:
                continue
            rows = self.store.flatten_results(results)
            for r in rows:
                if r["id"] not in seen:
                    r["used_where_filter"] = str(where)
                    r["rule_score"] = score_item(r, intent, slot)
                    # Chroma distance is smaller-is-better. Convert to a simple similarity component.
                    dist = r.get("distance")
                    r["vector_score"] = 0.0 if dist is None else max(0.0, 100.0 - float(dist) * 100.0)
                    r["final_score"] = r["rule_score"] + 0.25 * r["vector_score"]
                    combined.append(r)
                    seen.add(r["id"])
            # Stop early when we have enough strong slot-matching candidates.
            strong = [x for x in combined if x.get("slot") == slot and x.get("rule_score", 0) > 20]
            if len(strong) >= 3:
                break

        combined = [x for x in combined if x.get("rule_score", 0) >= 0]
        combined.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return combined[:n_results]
