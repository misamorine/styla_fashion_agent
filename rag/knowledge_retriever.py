from rag.text_embedder import TextEmbedder
from rag.vector_store import FashionVectorStore
from rag.llm import get_stylist_llm
from rag.prompt_builder import BODY_SHAPE_STYLING_PROMPT
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class KnowledgeRetriever:

    def __init__(
        self,
        db_path="./knowledge_vector_db"
    ):
        self.embedder = TextEmbedder()
        self.store = FashionVectorStore(
            db_path=db_path,
            collection_name="body_shape_knowledge"
        )

    def retrieve(self, shape, n_results=1):
        embedding = self.embedder.embed(shape)
        results = self.store.query(
            query_embedding=embedding,
            where={"shape": shape},
            n_results=n_results
        )
        return self.store.flatten_results(results)

    def get_advice_with_langchain(self, shape: str) -> str:
        """Executes RAG pipeline using LangChain Expression Language (LCEL)."""
        docs = self.retrieve(shape)
        knowledge_text = docs[0]["document"] if docs else ""
        
        llm = get_stylist_llm()
        if not llm:
            from rag.llm import generate
            from rag.prompt_builder import build_prompt
            return generate(build_prompt(shape, knowledge_text))

        chain = (
            BODY_SHAPE_STYLING_PROMPT
            | llm
            | StrOutputParser()
        )
        return chain.invoke({"body_shape": shape, "knowledge": knowledge_text})