import json
from pathlib import Path
from rag.text_embedder import TextEmbedder
from rag.vector_store import FashionVectorStore
from rag.llm import get_stylist_llm
from rag.prompt_builder import BODY_SHAPE_STYLING_PROMPT
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
        self.knowledge_dir = Path(__file__).resolve().parent.parent / "knowledge"

    def _fallback_load_json(self, shape_code: str) -> str:
        code = shape_code.strip()
        path = self.knowledge_dir / f"{code}.json"
        if not path.exists():
            return ""
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return f"""
Body Shape Code: {data.get('shape', code)}
Name: {data.get('name', '')}
Summary: {data.get('summary', '')}
Style Goal: {data.get('style_goal', '')}
Characteristics: {' '.join(data.get('characteristics', []))}
Recommendations: {json.dumps(data.get('recommendations', {}))}
Avoid: {json.dumps(data.get('avoid', []))}
Reasoning: {json.dumps(data.get('reasoning', {}))}
Styling Tips: {' '.join(data.get('styling_tips', []))}
"""

    def retrieve(self, shape: str, n_results=1):
        shape_code = (shape or "").strip()
        if "-" in shape_code:
            parts = [p.strip() for p in shape_code.split("-") if p.strip()]
        else:
            parts = [shape_code]

        all_docs = []
        for part in parts:
            embedding = self.embedder.embed(part)
            results = self.store.query(
                query_embedding=embedding,
                where={"shape": part},
                n_results=n_results
            )
            flattened = self.store.flatten_results(results)
            if flattened:
                all_docs.extend(flattened)
            else:
                fallback_text = self._fallback_load_json(part)
                if fallback_text:
                    all_docs.append({
                        "shape": part,
                        "document": fallback_text
                    })

        if not all_docs:
            return [{"shape": shape_code, "document": f"Body Shape styling rules for {shape_code}"}]

        return all_docs

    def get_advice_with_langchain(self, shape: str) -> str:
        """Executes RAG pipeline using LangChain Expression Language (LCEL)."""
        docs = self.retrieve(shape)
        knowledge_texts = [d.get("document", "") for d in docs if d.get("document")]
        knowledge_text = "\n\n========================================\n\n".join(knowledge_texts)

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
