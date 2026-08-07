from typing import Dict, List, Optional, Tuple
import chromadb


def and_filter(filters: List[Dict]) -> Optional[Dict]:
    filters = [f for f in filters if f]
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


class FashionVectorStore:
    def __init__(self, db_path: str = "./fashion_vector_db_v2", collection_name: str = "fashion_items_v2"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, item_id: str, embedding: List[float], document: str, metadata: Dict):
        self.collection.add(ids=[item_id], embeddings=[embedding], documents=[document], metadatas=[metadata])

    def count(self) -> int:
        return self.collection.count()

    def query(self, query_embedding: List[float], where: Optional[Dict], n_results: int = 30):
        kwargs = {"query_embeddings": [query_embedding], "n_results": n_results}
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    @staticmethod
    def flatten_results(results) -> List[Dict]:
        rows = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)
        for i, item_id in enumerate(ids):
            row = dict(metas[i] or {})
            row["id"] = item_id
            row["document"] = docs[i] if i < len(docs) else ""
            row["distance"] = dists[i] if i < len(dists) else None
            rows.append(row)
        return rows
