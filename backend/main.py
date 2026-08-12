"""
FastAPI backend for the Personal Stylist app.

This file does NOT modify any existing logic in src/, rag/, product_rag/,
or knowledge/. It only imports and calls those modules the same way
app.py (the original Streamlit app) does, and exposes the results as
JSON over HTTP so a static HTML/CSS/JS frontend can talk to them.

IMPORTANT: run this from the project ROOT directory (the folder that
contains app.py), because several of the original modules use relative
paths (e.g. "./fashion_vector_db_v2", "pose_landmarker.task") that only
resolve correctly when the process's working directory is the project
root - exactly like `streamlit run app.py` did.

    cd styla_fashion_agent
    uvicorn backend.main:app --reload --port 8000
"""

import re
import sys
import tempfile
import traceback
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Make sure the project root (parent of this backend/ folder) is importable,
# so "from src.xxx import yyy" / "from rag.xxx import yyy" work exactly as
# they do in app.py.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.embedder import FashionEmbedder
from src.intent_parser import parse_intent_with_langchain
from src.vector_store import FashionVectorStore
from src.retriever import FashionRetriever
from src.outfit_composer import OutfitComposer
from src.wardrobe_manager import WardrobeManager
from src.serpapi_service import build_search_query, search_products
from src.body_shop_queries import load_shape_knowledge, shoppable_queries_for_shape

from rag.knowledge_retriever import KnowledgeRetriever
from rag.prompt_builder import build_prompt
from rag.llm import generate, get_stylist_llm

from src.agent_tools import get_stylist_tools

# ---------------------------------------------------------------------------
# Same constants app.py uses.
# ---------------------------------------------------------------------------
DB_PATH = str(ROOT_DIR / "fashion_vector_db_v2")
IMAGE_DIR = str(ROOT_DIR / "fashion_item_images")
WARDROBE_DB = str(ROOT_DIR / "wardrobe_db.json")
WARDROBE_IMAGE_DIR = str(ROOT_DIR / "wardrobe_images")

SHAPE_NAMES = {
    # Front Profile Shapes
    "A": "Pear Shape (Triangle - Hips wider than bust)",
    "H": "Rectangle Shape (Straight - Similar bust, waist, hip)",
    "X": "Hourglass Shape (Curvy - Balanced bust & hip, narrow waist)",
    "Y": "Inverted Triangle Shape (Broad shoulders / bust wider than hip)",
    # Side Profile Shapes
    "I": "Balanced Side Profile (Neutral side depth)",
    "P": "Prominent Chest Side Profile",
    "b": "Prominent Belly Side Profile",
    "B": "Prominent Chest & Belly Side Profile",
    "S": "Prominent Chest & Butt Side Profile (S-Line)",
    "d": "Prominent Butt Side Profile",
    "db": "Prominent Belly & Butt Side Profile",
    "dB": "Full-Curve Side Profile (Chest, Belly & Butt)",
}


def get_shape_display_name(code: str) -> str:
    if not code or code == "None":
        return "Not Specified"
    if "-" in code:
        parts = code.split("-")
        front_name = SHAPE_NAMES.get(parts[0], f"Front {parts[0]}")
        side_name = SHAPE_NAMES.get(parts[1], f"Side {parts[1]}")
        return f"{front_name} + {side_name} ({code})"
    return SHAPE_NAMES.get(code, f"Shape {code}")


SLOT_NAME_TEMPLATES = {
    "layer": ["Tailored Double-Breasted Blazer", "Structured Wool-Blend Coat", "Casual Overshirt Jacket", "Slim-Fit Suit Jacket"],
    "top": ["Classic Oxford Cotton Shirt", "Relaxed Fit Silk-Blend Blouse", "Ribbed Knit Crewneck Top", "Structured Square-Neck Top"],
    "bottom": ["High-Waisted Wide-Leg Trousers", "Slim-Fit Chino Pants", "Tailored Ankle-Length Pants", "Classic Straight-Leg Denim Jeans"],
    "footwear": ["Leather Derby Shoes", "Minimalist Leather Loafers", "Classic Suede Oxfords", "Clean Leather Sneakers"],
    "accessory": ["Classic Leather Belt", "Minimalist Analog Watch", "Silk Pocket Square", "Leather Crossbody Bag"],
}

LIVE_GENDERS = ["Male", "Female", "Unisex"]
LIVE_STORES = ["Myntra", "Ajio", "Amazon", "Flipkart", "H&M", "Zara", "Nykaa Fashion"]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Personal Stylist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve locally stored images (wardrobe photos, indexed catalog thumbnails,
# sample imgs) so the frontend can display them via plain <img src="...">.
for mount_path, folder in [
    ("/media/wardrobe_images", WARDROBE_IMAGE_DIR),
    ("/media/fashion_item_images", IMAGE_DIR),
    ("/media/imgs", str(ROOT_DIR / "imgs")),
]:
    Path(folder).mkdir(parents=True, exist_ok=True)
    app.mount(mount_path, StaticFiles(directory=folder), name=mount_path.strip("/").replace("/", "_"))

# Serve the static frontend itself, so the user can just open
# http://127.0.0.1:8000/ instead of dealing with file:// + CORS.
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Lazily-created singletons (these load heavy ML models - CLIP, BGE, etc. -
# so we only pay that cost once, and only when actually needed, exactly
# like the Streamlit app did inside button handlers / on tab render).
# ---------------------------------------------------------------------------
_store: Optional[FashionVectorStore] = None
_wardrobe: Optional[WardrobeManager] = None
_embedder: Optional[FashionEmbedder] = None
_knowledge_retriever: Optional[KnowledgeRetriever] = None


def get_store() -> FashionVectorStore:
    global _store
    if _store is None:
        _store = FashionVectorStore(db_path=DB_PATH)
    return _store


def get_wardrobe() -> WardrobeManager:
    global _wardrobe
    if _wardrobe is None:
        _wardrobe = WardrobeManager(db_path=WARDROBE_DB, image_dir=WARDROBE_IMAGE_DIR)
    return _wardrobe


def get_embedder() -> FashionEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FashionEmbedder()
    return _embedder


def get_knowledge_retriever() -> KnowledgeRetriever:
    global _knowledge_retriever
    if _knowledge_retriever is None:
        _knowledge_retriever = KnowledgeRetriever()
    return _knowledge_retriever


# ---------------------------------------------------------------------------
# Helpers ported from app.py's display logic (no Streamlit involved, so
# they are rewritten here to build JSON instead of calling st.* widgets).
# The underlying data/behavior is identical to app.py.
# ---------------------------------------------------------------------------
def enrich_product_display(item: dict) -> dict:
    item = dict(item)
    raw_title = item.get("title", "")
    slot = item.get("slot", "top")
    color = item.get("color", "neutral").title()
    item_id = str(item.get("id", "0"))
    hash_val = abs(hash(item_id))

    if item.get("display_title"):
        pass
    elif not raw_title or raw_title.lower().startswith("fashion item") or raw_title.lower().startswith("item"):
        templates = SLOT_NAME_TEMPLATES.get(slot, ["Fashion Styling Piece"])
        template = templates[hash_val % len(templates)]
        item["display_title"] = f"{color} {template}"
    else:
        item["display_title"] = raw_title.title() if item.get("source") != "serpapi" else raw_title

    if "price" not in item or not item.get("price"):
        price_base = {"layer": 3999, "top": 1799, "bottom": 2499, "footwear": 3499, "accessory": 1299}
        base = price_base.get(slot, 1999)
        variation = (hash_val % 10) * 100
        item["price"] = f"₹{base + variation:,}"

    return item


def build_similar_links(title: str, gender: str = "") -> Dict[str, str]:
    clean_title = re.sub(
        r"^(Zara|H&M|Mango|Uniqlo|ASOS|Massimo Dutti|Nike|Levi's|Ralph Lauren|Forever New|Amazon\.in|Myntra|Nykaa|Ajio)\s*(Recommended)?\s*",
        "",
        title or "",
        flags=re.IGNORECASE,
    ).strip()
    search_q = f"{gender} {clean_title}".strip()
    enc_q = urllib.parse.quote_plus(search_q)
    return {
        "Myntra": f"https://www.myntra.com/{enc_q}",
        "Amazon.in": f"https://www.amazon.in/s?k={enc_q}",
        "Nykaa Fashion": f"https://www.nykaafashion.com/catalogsearch/result/?q={enc_q}",
        "Ajio": f"https://www.ajio.com/search/?text={enc_q}",
    }


def to_media_url(image_path: Optional[str]) -> Optional[str]:
    """Convert a local file path used by the original app into a URL this
    FastAPI process can serve."""
    if not image_path:
        return None
    p = Path(image_path)
    if not p.is_absolute():
        p = (ROOT_DIR / image_path).resolve()
    else:
        p = p.resolve()

    if not p.exists():
        filename = p.name
        img_dir_path = Path(IMAGE_DIR).resolve() / filename
        if img_dir_path.exists():
            return f"/media/fashion_item_images/{filename}"
        return None

    for folder, mount in [
        (Path(WARDROBE_IMAGE_DIR).resolve(), "/media/wardrobe_images"),
        (Path(IMAGE_DIR).resolve(), "/media/fashion_item_images"),
        ((ROOT_DIR / "imgs").resolve(), "/media/imgs"),
    ]:
        try:
            rel = p.relative_to(folder)
            return f"{mount}/{rel.as_posix()}"
        except ValueError:
            continue

    return None


def serialize_item(item: dict, gender_hint: str = "") -> dict:
    """Enrich + make an outfit/catalog/wardrobe item JSON-safe & browser-ready."""
    enriched = enrich_product_display(item)
    out = dict(enriched)
    out["image_url"] = to_media_url(enriched.get("image_path")) or enriched.get("image") or None
    out["owned"] = bool(enriched.get("owned"))
    out["is_live_store"] = enriched.get("source") == "serpapi"
    out["final_score"] = round(float(enriched.get("final_score", 0) or 0), 2)
    out["rule_score"] = round(float(enriched.get("rule_score", 0) or 0), 2)
    out["vector_score"] = round(float(enriched.get("vector_score", 0) or 0), 2)
    out["similar_links"] = {}
    return out


def serialize_selected(selected: Dict[str, dict], gender_hint: str = "") -> Dict[str, dict]:
    return {slot: serialize_item(item, gender_hint) for slot, item in selected.items()}


def serialize_alternatives(alternatives: Dict[str, List[dict]], gender_hint: str = "") -> Dict[str, List[dict]]:
    return {slot: [serialize_item(item, gender_hint) for item in items] for slot, items in alternatives.items()}


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class CreateOutfitRequest(BaseModel):
    query: str = "blazer outfit for business dinner"
    gender: str = "female"
    occasion: str = "business dinner"
    body_shape: Optional[str] = None
    use_wardrobe_first: bool = True



class ShopForYouRequest(BaseModel):
    shape: str
    gender: str = "Female"
    preferred_store: str = "Myntra"
    products_per_query: int = 3
    max_queries: int = 4


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return {
        "shape_names": SHAPE_NAMES,
        "live_genders": LIVE_GENDERS,
        "live_stores": LIVE_STORES,
        "occasions": ["business dinner", "office", "wedding", "college", "casual outing"],
        "genders": ["female", "male", "unisex"],
        "stores": ["All Stores", "Myntra", "Amazon.in", "Nykaa Fashion", "Ajio", "Flipkart", "H&M", "Zara"],
        "catalog_count": get_store().count(),
        "wardrobe_count": get_wardrobe().count(),
        "chat_enabled": get_stylist_llm() is not None,
    }


# ---------------------------------------------------------------------------
# Create Outfit  (mirrors the "Create Outfit" tab in app.py)
# ---------------------------------------------------------------------------
@app.post("/api/create-outfit")
def create_outfit(req: CreateOutfitRequest):
    try:
        store = get_store()
        wardrobe = get_wardrobe()
        count = store.count()

        retriever = None
        if count > 0:
            retriever = FashionRetriever(store, get_embedder())

        composer = OutfitComposer(retriever, wardrobe_manager=wardrobe)
        intent = parse_intent_with_langchain(
            req.query, req.gender, req.occasion, body_shape=req.body_shape
        )

        look = composer.compose(
            intent,
            use_wardrobe_first=req.use_wardrobe_first,
            use_live_api=False,
            catalog_fallback=True,
        )


        advice_text = ""
        if intent.body_shape:
            try:
                kr = get_knowledge_retriever()
                advice_text = kr.get_advice_with_langchain(intent.body_shape)
            except Exception:
                advice_text = f"Body Shape {intent.body_shape} rule applied."

        gender_hint = intent.gender

        return {
            "intent": {
                "query": intent.query,
                "gender": intent.gender,
                "occasion": intent.occasion,
                "color": intent.color,
                "requested_slot": intent.requested_slot,
                "body_shape": intent.body_shape,
                "body_shape_label": SHAPE_NAMES.get(intent.body_shape, intent.body_shape or "Not Specified"),
            },
            "look_score": look.get("look_score", 0),
            "sources_used": look.get("sources_used") or [],
            "selected": serialize_selected(look.get("selected", {}), gender_hint),
            "owned_selected": serialize_selected(look.get("owned_selected", {}), gender_hint),
            "shopping_needed": serialize_selected(look.get("shopping_needed", {}), gender_hint),
            "missing_slots": look.get("missing_slots", []),
            "alternatives": serialize_alternatives(look.get("alternatives", {}), gender_hint),
            "explanation": look.get("explanation", ""),
            "advice_text": advice_text,
            "wardrobe_count": wardrobe.count(),
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# My Wardrobe  (mirrors the "My Wardrobe" tab in app.py)
# ---------------------------------------------------------------------------
@app.get("/api/wardrobe")
def list_wardrobe():
    wardrobe = get_wardrobe()
    items = wardrobe.list_items()
    out = []
    for item in items:
        row = dict(item)
        row["image_url"] = to_media_url(row.get("image_path"))
        out.append(row)
    return {"count": len(out), "items": out}


@app.post("/api/wardrobe")
async def add_wardrobe_item(
    title: str = Form(...),
    description: str = Form(""),
    gender: str = Form("auto"),
    slot: str = Form("auto"),
    color: str = Form("auto"),
    occasion_group: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
):
    if not title.strip():
        raise HTTPException(status_code=400, detail="Please enter an item name.")
    wardrobe = get_wardrobe()
    try:
        image_file = image.file if image is not None and image.filename else None
        row = wardrobe.add_item(image_file, title, description, gender, slot, color, occasion_group)
        row = dict(row)
        row["image_url"] = to_media_url(row.get("image_path"))
        return row
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/wardrobe/{item_id}")
def delete_wardrobe_item(item_id: str):
    wardrobe = get_wardrobe()
    deleted = wardrobe.delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"deleted": True, "id": item_id}


@app.delete("/api/wardrobe")
def clear_wardrobe():
    wardrobe = get_wardrobe()
    wardrobe.clear()
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Body Shape  (mirrors the "Body Shape" tab in app.py)
# ---------------------------------------------------------------------------
@app.post("/api/body-shape")
async def analyze_body_shape(
    front_photo: Optional[UploadFile] = File(None),
    side_photo: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),  # Legacy fallback for single photo
):
    from src.vision.segment2 import detect_dual_body_shape

    target_front = front_photo or photo
    if not target_front or not target_front.filename:
        raise HTTPException(status_code=400, detail="Please upload a front-view full-body photo.")

    suffix_f = Path(target_front.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_f) as tmp_f:
        tmp_f.write(await target_front.read())
        front_path = tmp_f.name

    side_path = None
    if side_photo and side_photo.filename:
        suffix_s = Path(side_photo.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_s) as tmp_s:
            tmp_s.write(await side_photo.read())
            side_path = tmp_s.name

    try:
        front_shape, side_shape, combined_shape = detect_dual_body_shape(front_path, side_path)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error analyzing body shape: {exc}")

    try:
        kr = get_knowledge_retriever()
        advice = kr.get_advice_with_langchain(combined_shape)
    except Exception as exc:
        traceback.print_exc()
        advice = f"Could not generate styling advice: {exc}"

    return {
        "shape": combined_shape,
        "front_shape": front_shape,
        "side_shape": side_shape,
        "shape_label": get_shape_display_name(combined_shape),
        "advice": advice,
    }



# ---------------------------------------------------------------------------
# Shop For You  (mirrors the "Shop For You" tab in app.py)
# ---------------------------------------------------------------------------
@app.post("/api/shop-for-you")
def shop_for_you(req: ShopForYouRequest):
    try:
        knowledge = load_shape_knowledge(req.shape)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    style_queries = shoppable_queries_for_shape(req.shape, max_queries=req.max_queries)
    grouped_results = []
    for style_query in style_queries:
        search_term = style_query["search_term"]
        query = build_search_query(req.gender, search_term, req.preferred_store)
        try:
            products = search_products(query, preferred_store=req.preferred_store, num_results=req.products_per_query)
            error = ""
        except Exception as exc:
            products = []
            error = str(exc)

        grouped_results.append({**style_query, "search_query": query, "products": products, "error": error})

    return {
        "shape": req.shape,
        "shape_label": SHAPE_NAMES.get(req.shape, req.shape),
        "style_goal": knowledge.get("style_goal", ""),
        "groups": grouped_results,
    }


# ---------------------------------------------------------------------------
# Stylist Chatbot  (mirrors the "💬 Stylist Chatbot" tab in app.py)
# ---------------------------------------------------------------------------
@app.post("/api/chat")
def chat(req: ChatRequest):
    llm = get_stylist_llm()
    if not llm:
        return {
            "reply": (
                "### 💡 AI Stylist (Offline Mode)\n\n"
                "Set your `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env` to activate "
                "full interactive chatbot capabilities with tool calling!"
            )
        }

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    history_messages = []
    for m in req.history:
        if m.role == "user":
            history_messages.append(HumanMessage(content=m.content))
        else:
            history_messages.append(AIMessage(content=m.content))

    tools = get_stylist_tools()
    try:
        llm_with_tools = llm.bind_tools(tools)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert AI fashion stylist assistant. Use tools when needed to look up "
                    "wardrobe items, search live stores (Myntra/Amazon/Ajio), get body shape rules, or "
                    "compose outfits. Be warm, helpful, and concise.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt_template | llm_with_tools
        response = chain.invoke({"input": req.message, "history": history_messages})

        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_outputs = []
            for tc in response.tool_calls:
                t_name = tc["name"]
                t_args = tc["args"]
                t_obj = next((t for t in tools if t.name == t_name), None)
                if t_obj:
                    t_res = t_obj.invoke(t_args)
                    tool_outputs.append(f"Result from tool `{t_name}`:\n{t_res}")

            synthesis_input = f"User query: '{req.message}'\n\nTool Results:\n" + "\n\n".join(tool_outputs)
            synthesis_res = llm.invoke(
                [
                    ("system", "Synthesize tool execution results into a friendly, helpful, beautifully formatted response."),
                    ("human", synthesis_input),
                ]
            )
            reply = synthesis_res.content
        else:
            reply = response.content

        return {"reply": reply}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chatbot encountered an error: {exc}")
