from .embedder import FashionEmbedder
from .vector_store import FashionVectorStore


class ProductRetriever:

    def __init__(
        self,
        db_path="./fashion_vector_db_v2"
    ):

        self.embedder = FashionEmbedder()

        self.store = FashionVectorStore(
            db_path=db_path
        )

    def retrieve(
        self,
        query,
        n_results=4
    ):

        embedding = self.embedder.text_embedding(query)

        results = self.store.query(
            query_embedding=embedding,
            where=None,
            n_results=n_results
        )

        return self.store.flatten_results(results)