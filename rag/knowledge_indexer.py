import json
from pathlib import Path

from rag.text_embedder import TextEmbedder
from rag.vector_store import FashionVectorStore


class KnowledgeIndexer:

    def __init__(
        self,
        knowledge_dir="knowledge",
        db_path="./knowledge_vector_db"
    ):

        self.knowledge_dir = Path(knowledge_dir)

        self.embedder = TextEmbedder()

        self.store = FashionVectorStore(
            db_path=db_path,
            collection_name="body_shape_knowledge"
        )

    def json_to_document(self, data):

        text = f"""
Body Shape: {data['shape']}
Name: {data['name']}

Summary:
{data['summary']}

Style Goal:
{data['style_goal']}

Characteristics:
{' '.join(data['characteristics'])}

Recommendations:
{json.dumps(data['recommendations'])}

Avoid:
{json.dumps(data['avoid'])}

Reasoning:
{json.dumps(data['reasoning'])}

Styling Tips:
{' '.join(data['styling_tips'])}
"""

        return text

    def index(self):

        json_files = list(
            self.knowledge_dir.glob("*.json")
        )

        for file in json_files:

            with open(file, "r") as f:
                data = json.load(f)

            document = self.json_to_document(data)

            embedding = self.embedder.embed(document)

            metadata = {
                "shape": data["shape"],
                "name": data["name"]
            }

            self.store.add(
                item_id=data["shape"],
                embedding=embedding,
                document=document,
                metadata=metadata
            )

        print(
            f"Indexed {len(json_files)} body shapes."
        )