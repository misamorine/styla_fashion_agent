"""LangChain agent tools wrapping core fashion stylist functionalities."""

from langchain_core.tools import tool
from src.wardrobe_manager import WardrobeManager
from src.serpapi_service import search_products
from rag.knowledge_retriever import KnowledgeRetriever
from src.intent_parser import parse_intent
from src.outfit_composer import OutfitComposer
from src.vector_store import FashionVectorStore
from src.embedder import FashionEmbedder
from src.retriever import FashionRetriever


@tool
def lookup_wardrobe_items(category: str = "") -> str:
    """Retrieves saved fashion items from the user's personal wardrobe.
    Argument category can be: 'top', 'bottom', 'layer', 'footwear', 'accessory', or empty for all.
    """
    try:
        wm = WardrobeManager()
        if category:
            items = wm.get_items_by_category(category.lower())
        else:
            items = wm.list_items()
        
        if not items:
            return "No items found in wardrobe for this category."
        
        summary = []
        for idx, item in enumerate(items, 1):
            title = item.get("title") or item.get("category", "Item")
            color = item.get("color", "")
            cat = item.get("category", "")
            summary.append(f"{idx}. {title} (Category: {cat}, Color: {color})")
        return "\n".join(summary)
    except Exception as e:
        return f"Error retrieving wardrobe items: {e}"


@tool
def search_live_products(query: str, store: str = "All Stores") -> str:
    """Searches live fashion e-commerce platforms (Myntra, Ajio, Amazon, Nykaa) for items matching the search query."""
    try:
        results = search_products(query, store=store, limit=4)
        if not results:
            return f"No live products found for query '{query}' on {store}."
        
        formatted = []
        for p in results[:4]:
            title = p.get("title", "Fashion Item")
            price = p.get("price", "N/A")
            source = p.get("source_name") or p.get("source", "Online Store")
            url = p.get("url") or p.get("link", "#")
            formatted.append(f"- **{title}** | Price: {price} | Store: {source} | Link: {url}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Error searching live products: {e}"


@tool
def get_body_shape_styling_tips(body_shape: str) -> str:
    """Retrieves body-shape specific fashion recommendations and rules.
    Accepted body_shape values: Single shapes ('A', 'H', 'X', 'Y', 'B', 'P', etc.) or 2-view combined codes ('A-B', 'X-S', 'H-b', etc.).
    """
    try:
        kr = KnowledgeRetriever()
        return kr.get_advice_with_langchain(body_shape.strip())
    except Exception as e:
        return f"Error retrieving styling tips for body shape {body_shape}: {e}"



@tool
def compose_outfit_recommendation(query: str, gender: str = "female", occasion: str = "casual outing") -> str:
    """Composes a complete multi-piece outfit (layer, top, bottom, footwear, accessory) matching an occasion and query."""
    try:
        store = FashionVectorStore(db_path="./fashion_vector_db_v2")
        embedder = FashionEmbedder()
        retriever = None
        if store.count() > 0:
            retriever = FashionRetriever(store, embedder)
        
        wardrobe = WardrobeManager()
        composer = OutfitComposer(retriever, wardrobe_manager=wardrobe)
        intent = parse_intent(query, gender, occasion)
        
        look = composer.compose(
            intent,
            use_wardrobe_first=True,
            use_live_api=True,
            catalog_fallback=store.count() > 0,
        )
        
        selected = look.get("selected", {})
        if not selected:
            return "Could not compose an outfit matching the criteria."
        
        lines = [f"### Recommended Outfit for {occasion.title()} (Score: {look.get('look_score', 0)}/10):"]
        for slot, item in selected.items():
            title = item.get("display_title") or item.get("title", slot)
            source = "OWNED" if item.get("owned") else "LIVE PICK"
            lines.append(f"- **{slot.title()}** ({source}): {title}")
        
        if look.get("explanation"):
            lines.append(f"\n*Styling Note:* {look['explanation']}")
            
        return "\n".join(lines)
    except Exception as e:
        return f"Error composing outfit: {e}"


def get_stylist_tools():
    """Returns list of LangChain tools for the AI Fashion Stylist agent."""
    return [
        lookup_wardrobe_items,
        search_live_products,
        get_body_shape_styling_tips,
        compose_outfit_recommendation,
    ]
