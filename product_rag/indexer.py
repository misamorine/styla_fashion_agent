from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from .catalog_utils import metadata_from_item, save_dataset_image
from .embedder import FashionEmbedder
from .vector_store import FashionVectorStore


def get_image_field(item):
    for key in ["image", "images", "img"]:
        if key in item and item[key] is not None:
            value = item[key]
            if isinstance(value, list) and value:
                return value[0]
            return value
    return None


def index_dataset(limit: int = 2000, dataset_name: str = "Marqo/polyvore", split: str = "data", db_path: str = "./fashion_vector_db_v2", image_dir: str = "./fashion_item_images"):
    ds = load_dataset(dataset_name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    embedder = FashionEmbedder()
    store = FashionVectorStore(db_path=db_path)
    img_dir = Path(image_dir)

    added = 0
    for idx, item in enumerate(tqdm(ds, desc="Indexing fashion items")):
        try:
            image = get_image_field(item)
            if image is None:
                continue
            image_path = save_dataset_image(image, img_dir, idx)
            metadata = metadata_from_item(item, idx, image_path)
            doc = f"{metadata['title']} {metadata['description']} category:{metadata['category']} slot:{metadata['slot']} color:{metadata['color']} gender:{metadata['gender']} occasion:{metadata['occasion_group']}"
            emb = embedder.image_embedding(image)
            store.add(str(idx), emb, doc, metadata)
            added += 1
        except Exception as e:
            # Skip malformed rows, common in research datasets.
            continue
    return added
