import argparse
import re
import tempfile
import urllib.parse
from pathlib import Path

import streamlit as st

from src.embedder import FashionEmbedder
from src.indexer import index_dataset
from src.intent_parser import parse_intent, parse_intent_with_langchain
from src.vector_store import FashionVectorStore
from src.retriever import FashionRetriever
from src.outfit_composer import OutfitComposer
from src.wardrobe_manager import WardrobeManager
from src.serpapi_service import build_search_query, search_products
from src.body_shop_queries import load_shape_knowledge, shoppable_queries_for_shape

# Vision & RAG imports (from fashion agent new integration)
from src.vision.segment2 import detect_body_shape
from rag.knowledge_retriever import KnowledgeRetriever
from rag.prompt_builder import build_prompt
from rag.llm import generate, get_stylist_llm
from product_rag.recommendation_parser import extract_recommendations

# LangChain Chatbot imports
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agent_tools import get_stylist_tools

LIVE_GENDERS = ["Male", "Female", "Unisex"]
LIVE_STORES = [
    "Myntra",
    "Ajio",
    "Amazon",
    "Flipkart",
    "H&M",
    "Zara",
    "Nykaa Fashion",
]


def render_view_similar_links(title: str, gender: str = "", preferred_store: str = "All Stores"):
    clean_title = re.sub(r'^(Zara|H&M|Mango|Uniqlo|ASOS|Massimo Dutti|Nike|Levi\'s|Ralph Lauren|Forever New|Amazon\.in|Myntra|Nykaa|Ajio)\s*(Recommended)?\s*', '', title, flags=re.IGNORECASE).strip()
    search_q = f"{gender} {clean_title}".strip()
    enc_q = urllib.parse.quote_plus(search_q)

    store_urls = {
        "Myntra": f"https://www.myntra.com/{enc_q}",
        "Amazon.in": f"https://www.amazon.in/s?k={enc_q}",
        "Nykaa Fashion": f"https://www.nykaafashion.com/catalogsearch/result/?q={enc_q}",
        "Ajio": f"https://www.ajio.com/search/?text={enc_q}"
    }

    if preferred_store in store_urls:
        url = store_urls[preferred_store]
        st.markdown(f"[🔍 View Similar on {preferred_store}]({url})")
    else:
        st.markdown(f"[🛍️ Myntra]({store_urls['Myntra']}) · [📦 Amazon]({store_urls['Amazon.in']}) · [✨ Nykaa]({store_urls['Nykaa Fashion']}) · [👗 Ajio]({store_urls['Ajio']})")

DB_PATH = "./fashion_vector_db_v2"
IMAGE_DIR = "./fashion_item_images"
WARDROBE_DB = "./wardrobe_db.json"
WARDROBE_IMAGE_DIR = "./wardrobe_images"

SHAPE_NAMES = {
    "A": "Pear Shape (Triangle - Hips wider than bust)",
    "H": "Rectangle Shape (Straight - Similar bust, waist, hip)",
    "X": "Hourglass Shape (Curvy - Balanced bust & hip, narrow waist)",
    "Y": "Inverted Triangle Shape (Broad shoulders / bust wider than hip)",
}

SLOT_NAME_TEMPLATES = {
    "layer": ["Tailored Double-Breasted Blazer", "Structured Wool-Blend Coat", "Casual Overshirt Jacket", "Slim-Fit Suit Jacket"],
    "top": ["Classic Oxford Cotton Shirt", "Relaxed Fit Silk-Blend Blouse", "Ribbed Knit Crewneck Top", "Structured Square-Neck Top"],
    "bottom": ["High-Waisted Wide-Leg Trousers", "Slim-Fit Chino Pants", "Tailored Ankle-Length Pants", "Classic Straight-Leg Denim Jeans"],
    "footwear": ["Leather Derby Shoes", "Minimalist Leather Loafers", "Classic Suede Oxfords", "Clean Leather Sneakers"],
    "accessory": ["Classic Leather Belt", "Minimalist Analog Watch", "Silk Pocket Square", "Leather Crossbody Bag"],
}


def cli():
    parser = argparse.ArgumentParser(description="AI Fashion Stylist Agent V2")
    parser.add_argument("--index", action="store_true", help="Index Hugging Face fashion dataset into ChromaDB")
    parser.add_argument("--limit", type=int, default=2000, help="Number of dataset rows to index")
    parser.add_argument("--dataset", type=str, default="Marqo/polyvore", help="Hugging Face dataset name")
    args, _ = parser.parse_known_args()
    if args.index:
        added = index_dataset(limit=args.limit, dataset_name=args.dataset, db_path=DB_PATH, image_dir=IMAGE_DIR)
        print(f"Indexed {added} fashion items into {DB_PATH}")
        raise SystemExit(0)


def enrich_product_display(item: dict) -> dict:
    """Enriches generic dataset items with realistic titles and pricing metadata without dummy brand names."""
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


def _render_item_image(item: dict) -> None:
    """Show wardrobe local image or live SerpAPI thumbnail."""
    img_path = item.get("image_path")
    if img_path and Path(img_path).exists():
        st.image(img_path, use_container_width=True)
        return
    if item.get("image"):
        st.image(item["image"], use_container_width=True)


def _render_buy_or_similar(item: dict) -> None:
    """Buy Now for live products; similar-store links for catalog/wardrobe."""
    if item.get("url") and not item.get("owned"):
        st.link_button("Buy Now", item["url"], use_container_width=True)
        store = item.get("store")
        if store:
            st.caption(f"Store: **{store}**")
        return
    render_view_similar_links(
        item.get("display_title") or item.get("title", ""),
        item.get("gender", ""),
    )


def inject_card_css():
    st.markdown(
        """
        <style>
        .fashion-card {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 18px;
            padding: 14px;
            margin-bottom: 14px;
            background: rgba(255,255,255,0.03);
            min-height: 170px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        .fashion-slot {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.72;
            margin-bottom: 6px;
        }
        .fashion-title {
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 4px;
        }
        .fashion-brand {
            font-size: 0.85rem;
            font-weight: 600;
            color: #ff4b4b;
            margin-bottom: 6px;
        }
        .fashion-meta {
            font-size: 0.82rem;
            opacity: 0.84;
            margin-bottom: 8px;
        }
        .owned-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 9px;
            border: 1px solid rgba(128,128,128,0.28);
            font-size: 0.75rem;
            margin-bottom: 8px;
            font-weight: 700;
        }
        .fashion-score {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 9px;
            border: 1px solid rgba(128,128,128,0.28);
            font-size: 0.78rem;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_item_card(item, slot_name: str, show_debug: bool = True):
    item = enrich_product_display(item)
    st.markdown(f"### {slot_name.title()}")
    _render_item_image(item)
    st.markdown(f"**{item.get('display_title')}**")
    st.caption(f"💰 {item.get('price')}")
    _render_buy_or_similar(item)
    st.caption(item.get("description", "")[:220])
    c1, c2, c3 = st.columns(3)
    c1.metric("Gender", item.get("gender", "unknown"))
    c2.metric("Slot", item.get("slot", "unknown"))
    c3.metric("Color", item.get("color", "unknown"))
    source = item.get("source", "catalog")
    st.caption(f"Occasion group: {item.get('occasion_group', 'unknown')} · Source: {source}")
    if show_debug:
        with st.expander("Why this was picked"):
            st.write("Overall match:", round(item.get("final_score", 0), 2))
            st.write("Style fit:", round(item.get("rule_score", 0), 2))
            st.write("Similarity match:", round(item.get("vector_score", 0), 2))
            st.code(item.get("used_where_filter", "No filter recorded"))


def render_visual_card(item, slot_name: str, show_debug: bool = False):
    item = enrich_product_display(item)
    _render_item_image(item)
    title = item.get("display_title")
    price = item.get("price")
    desc = item.get("description", "")[:120]
    if item.get("owned"):
        owned_label = "OWNED"
    elif item.get("source") == "serpapi":
        owned_label = "LIVE STORE"
    else:
        owned_label = "TO BUY"
    st.markdown(
        f"""
        <div class="fashion-card">
            <div class="fashion-slot">{slot_name.title()}</div>
            <span class="owned-badge">{owned_label}</span>
            <div class="fashion-title">{title}</div>
            <div class="fashion-brand">{price}</div>
            <div class="fashion-meta">{item.get('gender', 'unknown')} · {item.get('slot', 'unknown')} · {item.get('color', 'unknown')} · {item.get('occasion_group', 'unknown')}</div>
            <div class="fashion-meta">{desc}</div>
            <span class="fashion-score">Score: {round(item.get('final_score', 0), 2)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_buy_or_similar(item)
    if show_debug:
        with st.expander("Match details"):
            st.write("Style fit:", round(item.get("rule_score", 0), 2))
            st.write("Similarity match:", round(item.get("vector_score", 0), 2))
            st.code(item.get("used_where_filter", "No filter recorded"))


def render_card_grid(items_by_slot, columns: int = 3, show_debug: bool = False):
    pairs = list(items_by_slot.items())
    rows = [pairs[i:i + columns] for i in range(0, len(pairs), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, (slot, item) in zip(cols, row):
            with col:
                render_visual_card(item, slot, show_debug=show_debug)


def render_alternative_cards(alternatives, columns: int = 4, show_debug: bool = False):
    for slot, candidates in alternatives.items():
        with st.expander(f"{slot.title()} alternatives ({len(candidates)})", expanded=False):
            cards = candidates[:8]
            for i in range(0, len(cards), columns):
                cols = st.columns(columns)
                for col, item in zip(cols, cards[i:i + columns]):
                    with col:
                        render_visual_card(item, slot, show_debug=show_debug)


def render_wardrobe_item(item, show_delete: bool = False, manager=None):
    img = item.get("image_path")
    if img and Path(img).exists():
        st.image(img, use_container_width=True)
    st.markdown(f"**{item.get('title', 'Wardrobe item')}**")
    st.caption(item.get("description", "")[:160])
    st.caption(f"{item.get('gender')} · {item.get('slot')} · {item.get('color')} · {item.get('occasion_group')}")
    if show_delete and manager is not None:
        if st.button("Delete", key=f"delete_{item.get('id')}"):
            manager.delete_item(item.get("id"))
            st.rerun()


def render_wardrobe_page(manager: WardrobeManager):
    st.subheader("My Wardrobe")
    st.write("Upload clothes you already own. The stylist will use these first and suggest only missing items to buy.")

    with st.expander("Add wardrobe item", expanded=True):
        uploaded = st.file_uploader("Upload item photo", type=["jpg", "jpeg", "png"])
        title = st.text_input("Item name", placeholder="Example: white oxford shirt, black blazer, brown loafers")
        description = st.text_area("Description / notes", placeholder="Example: formal cotton shirt, slim fit, good for office")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gender = st.selectbox("Gender", ["auto", "male", "female", "unisex"], key="wardrobe_gender")
        with c2:
            slot = st.selectbox("Slot", ["auto", "layer", "top", "bottom", "footwear", "accessory"], key="wardrobe_slot")
        with c3:
            color = st.selectbox("Color", ["auto", "black", "navy", "white", "brown", "grey", "charcoal", "beige", "blue", "cream", "olive"], key="wardrobe_color")
        with c4:
            occasion_group = st.selectbox("Occasion group", ["auto", "business", "formal", "smart casual", "casual", "wedding", "party", "ethnic"], key="wardrobe_occasion")

        if st.button("Save to wardrobe", type="primary"):
            if not title.strip():
                st.error("Please enter an item name. The name is used to infer category and color.")
            else:
                manager.add_item(uploaded, title, description, gender, slot, color, occasion_group)
                st.success("Saved to wardrobe.")
                st.rerun()

    items = manager.list_items()
    st.metric("Wardrobe items", len(items))
    if not items:
        st.info("No wardrobe items yet. Add 3–5 items first: a shirt, blazer, trouser, shoes, and belt.")
        return

    cols = st.columns(4)
    for i, item in enumerate(items):
        with cols[i % 4]:
            render_wardrobe_item(item, show_delete=True, manager=manager)

    with st.expander("Danger zone"):
        if st.button("Clear complete wardrobe"):
            manager.clear()
            st.rerun()


def render_body_shape_page():
    st.subheader("Body Shape Analysis")
    st.caption("Upload a full-body photo and we'll work out your body shape and put together styling advice just for you.")

    uploaded_file = st.file_uploader("Upload full-body photo", type=["jpg", "jpeg", "png"], key="body_photo_upload_main")

    if uploaded_file is not None:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(uploaded_file.read())
        temp.close()
        image_path = temp.name

        st.image(image_path, width=320, caption="Your photo")

        if st.button("Analyze My Body Shape", type="primary"):
            progress = st.progress(0, text="Analyzing your photo...")
            try:
                shape = detect_body_shape(image_path)
            except Exception as e:
                progress.empty()
                st.error(f"Error analyzing body shape: {e}")
                return
            progress.progress(40, text="Analyzing your photo...")

            st.session_state["detected_body_shape"] = shape

            progress.progress(70, text="Finding styling guidance for you...")
            try:
                kr = KnowledgeRetriever()
                k_res = kr.retrieve(shape)
                knowledge_doc = k_res[0]["document"] if k_res else f"Knowledge for shape {shape}"
            except Exception as e:
                knowledge_doc = f"Body shape {shape} rules"

            progress.progress(90, text="Putting together your styling advice...")
            prompt = build_prompt(shape, knowledge_doc)
            advice = generate(prompt)
            st.session_state["body_shape_advice"] = advice
            progress.progress(100, text="Done!")
            progress.empty()

    if "detected_body_shape" in st.session_state:
        shape = st.session_state["detected_body_shape"]
        full_name = SHAPE_NAMES.get(shape, f"Shape {shape}")

        st.success(f"Your Body Shape: **{shape}** ({full_name})")

        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            st.metric("Body Shape", shape)
            st.info(f"Classified as: **{full_name}**")
            st.write(
                "This shape is used to personalize your shopping picks "
                "and how outfits are scored for you."
            )
            st.caption("Open the **Shop For You** tab to see picks for this shape.")

        with c2:
            st.subheader("Your Personalized Styling Advice")
            advice = st.session_state.get("body_shape_advice", "")
            st.markdown(
                f"""
                <div style="height:420px; overflow-y:auto; padding:16px; border:1px solid rgba(128,128,128,0.3); border-radius:12px; background:rgba(255,255,255,0.03);">
                {advice}
                </div>
                """,
                unsafe_allow_html=True
            )


def render_owned_vs_missing(look: dict):
    owned = look.get("owned_selected", {})
    shopping = look.get("shopping_needed", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Owned items used", len(owned))
    c2.metric("Missing / to buy", len(shopping))
    c3.metric("Total look items", len(look.get("selected", {})))

    left, right = st.columns(2)
    with left:
        st.markdown("### Already owned")
        if owned:
            for slot, item in owned.items():
                st.write(f"✅ **{slot.title()}**: {item.get('title')}")
        else:
            st.caption("No owned wardrobe item was suitable for this look yet.")
    with right:
        st.markdown("### Suggested to buy")
        if shopping:
            for slot, item in shopping.items():
                enriched = enrich_product_display(item)
                st.write(f"🛒 **{slot.title()}**: {enriched.get('display_title')} ({enriched.get('price')})")
        else:
            st.success("Great — this complete look can be built from your existing wardrobe.")


def render_live_product_card(product: dict) -> None:
    """Render one SerpAPI product card."""
    with st.container(border=True):
        if product.get("image"):
            st.image(product["image"], use_container_width=True)
        else:
            st.info("No image available")

        st.markdown(f"**{product.get('title', 'Untitled product')[:70]}**")
        st.write(f"Price: {product.get('price', 'Price not available')}")
        st.caption(f"Store: **{product.get('store', 'Unknown store')}**")
        st.write(product.get("description") or "No description available.")
        if product.get("url"):
            st.link_button("Buy Now", product["url"], use_container_width=True)
        else:
            st.write("Product link not available.")


def fetch_body_shape_live_products(
    shape: str,
    gender: str,
    preferred_store: str,
    products_per_query: int = 3,
    max_queries: int = 4,
) -> list[dict]:
    """
    Recommend live store products from body-shape knowledge queries.
    Does not use the Polyvore/catalog dataset.
    """
    style_queries = shoppable_queries_for_shape(shape, max_queries=max_queries)
    grouped_results: list[dict] = []

    for style_query in style_queries:
        search_term = style_query["search_term"]
        query = build_search_query(gender, search_term, preferred_store)
        try:
            products = search_products(
                query,
                preferred_store=preferred_store,
                num_results=products_per_query,
            )
        except Exception as exc:
            products = []
            error = str(exc)
        else:
            error = ""

        grouped_results.append(
            {
                **style_query,
                "search_query": query,
                "products": products,
                "error": error,
            }
        )

    return grouped_results


def render_live_recommendations_page():
    st.subheader("Shop For You")
    st.caption(
        "Live store picks matched to your body shape."
    )

    detected_shape = st.session_state.get("detected_body_shape")
    if not detected_shape:
        st.warning(
            "We don't have your body shape yet. Go to **Body Shape** tab, upload a "
            "full-body photo, and come back here once it's done."
        )
        return

    shape_name = SHAPE_NAMES.get(detected_shape, detected_shape)
    try:
        knowledge = load_shape_knowledge(detected_shape)
        style_goal = knowledge.get("style_goal", "")
    except Exception:
        knowledge = {}
        style_goal = ""

    st.success(f"Using body shape **{detected_shape}** — {shape_name}")
    if style_goal:
        st.info(f"Style goal: {style_goal}")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", LIVE_GENDERS, key="live_tab_gender")
    with c2:
        preferred_store = st.selectbox(
            "Preferred Store",
            LIVE_STORES,
            key="live_tab_store",
        )

    if st.button("Find Picks For My Body Shape", type="primary", key="live_tab_btn"):
        progress = st.progress(20, text="Finding pieces for your shape...")
        try:
            progress.progress(50, text="Checking live store listings...")
            grouped = fetch_body_shape_live_products(
                detected_shape, gender, preferred_store
            )
        except ValueError as exc:
            progress.empty()
            st.error(str(exc))
            return
        except Exception as exc:
            progress.empty()
            st.error(f"Failed to fetch products: {exc}")
            return

        progress.progress(100, text="Done!")
        progress.empty()

        st.session_state["live_tab_body_results"] = grouped
        st.session_state["live_tab_body_shape_done"] = detected_shape
        st.session_state["live_tab_preferred_store"] = preferred_store
        st.session_state["live_tab_gender_done"] = gender

    grouped_results = st.session_state.get("live_tab_body_results", [])
    if not grouped_results:
        return

    any_products = any(group.get("products") for group in grouped_results)
    if not any_products:
        st.warning("No live products found for your body-shape recommendations. Try another store.")
        return

    for group in grouped_results:
        st.divider()
        st.subheader(f"{group['item']}")
        if group.get('reason'):
            st.caption(group.get('reason', ''))
        if group.get("error"):
            st.error(group["error"])
            continue
        products = group.get("products") or []
        if not products:
            st.info("No products found for this style recommendation.")
            continue
        cols = st.columns(min(3, len(products)))
        for idx, product in enumerate(products):
            with cols[idx % len(cols)]:
                render_live_product_card(product)


def render_chatbot_page():
    st.subheader("💬 AI Stylist Chatbot")
    st.caption("Chat with your personal AI Stylist powered by LangChain memory and tools.")

    msgs = StreamlitChatMessageHistory(key="langchain_chat_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Hi there! I am your AI Fashion Stylist. Ask me anything about outfits, body shape styling, live store items, or your saved wardrobe!")

    for msg in msgs.messages:
        avatar = "👤" if msg.type == "human" else "👗"
        st.chat_message(msg.type, avatar=avatar).write(msg.content)

    if prompt_input := st.chat_input("Ask your AI Stylist (e.g., 'What tops suit an Hourglass shape?', 'Search for navy blazers on Myntra')"):
        st.chat_message("human", avatar="👤").write(prompt_input)

        llm = get_stylist_llm()
        if not llm:
            fallback_text = "### 💡 AI Stylist (Offline Mode)\n\nSet your `OPENROUTER_API_KEY` or `OPENAI_API_KEY` in `.env` to activate full interactive chatbot capabilities with tool calling!"
            st.chat_message("ai", avatar="👗").write(fallback_text)
            msgs.add_user_message(prompt_input)
            msgs.add_ai_message(fallback_text)
            return

        tools = get_stylist_tools()
        try:
            with st.spinner("AI Stylist is thinking & fetching recommendations..."):
                llm_with_tools = llm.bind_tools(tools)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert AI fashion stylist assistant. Use tools when needed to look up wardrobe items, search live stores (Myntra/Amazon/Ajio), get body shape rules, or compose outfits. Be warm, helpful, and concise."),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ])
                
                chain = prompt_template | llm_with_tools
                response = chain.invoke({"input": prompt_input, "history": msgs.messages})

                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_outputs = []
                    for tc in response.tool_calls:
                        t_name = tc["name"]
                        t_args = tc["args"]
                        t_obj = next((t for t in tools if t.name == t_name), None)
                        if t_obj:
                            t_res = t_obj.invoke(t_args)
                            tool_outputs.append(f"Result from tool `{t_name}`:\n{t_res}")
                    
                    synthesis_input = f"User query: '{prompt_input}'\n\nTool Results:\n" + "\n\n".join(tool_outputs)
                    synthesis_res = llm.invoke([
                        ("system", "Synthesize tool execution results into a friendly, helpful, beautifully formatted response."),
                        ("human", synthesis_input)
                    ])
                    reply = synthesis_res.content
                else:
                    reply = response.content

            st.chat_message("ai", avatar="👗").write(reply)
            msgs.add_user_message(prompt_input)
            msgs.add_ai_message(reply)
        except Exception as exc:
            st.error(f"Chatbot encountered an error: {exc}")


def run_app():
    st.set_page_config(page_title="Personal Stylist", layout="wide")
    st.title("👗 Personal Stylist")
    st.caption("Outfits built from your wardrobe and live store picks, matched to your body shape.")
    inject_card_css()

    store = FashionVectorStore(db_path=DB_PATH)
    wardrobe = WardrobeManager(db_path=WARDROBE_DB, image_dir=WARDROBE_IMAGE_DIR)

    tabs = st.tabs([
        "Create Outfit",
        "Shop For You",
        "Body Shape",
        "My Wardrobe",
        "💬 Stylist Chatbot"
    ])

    with tabs[4]:
        render_chatbot_page()

    with tabs[3]:
        render_wardrobe_page(wardrobe)

    with tabs[2]:
        render_body_shape_page()

    with tabs[1]:
        render_live_recommendations_page()

    with tabs[0]:
        count = store.count()

        with st.sidebar:
            st.header("Styling Inputs")
            gender = st.selectbox("Gender", ["female", "male", "unisex"], index=0)
            occasion = st.selectbox("Occasion", ["business dinner", "office", "wedding", "college", "casual outing"], index=0)
            query = st.text_input("Request / Vibe", "blazer outfit for business dinner")

            detected_shape = st.session_state.get("detected_body_shape", None)
            shape_options = ["None", "A", "H", "X", "Y"]
            default_index = shape_options.index(detected_shape) if detected_shape in shape_options else 0
            body_shape_choice = st.selectbox(
                "Body Shape",
                shape_options,
                index=default_index,
                help="A: Pear | H: Rectangle | X: Hourglass | Y: Inverted Triangle"
            )
            body_shape_arg = None if body_shape_choice == "None" else body_shape_choice

            use_wardrobe_first = st.checkbox(
                "Use my wardrobe first",
                value=True,
                help="If enabled, owned wardrobe items are preferred. Missing slots are filled with live store picks.",
            )
            preferred_store = st.selectbox(
                "Shopping Platform",
                ["All Stores", "Myntra", "Amazon.in", "Nykaa Fashion", "Ajio", "Flipkart", "H&M", "Zara"],
                index=1,
                help="Buy Now links prefer this store when available",
            )
            view_mode = st.radio(
                "Visualization",
                ["Visual cards", "Outfit board", "Compact list"],
                index=0,
            )
            card_columns = st.slider("Cards per row", min_value=2, max_value=5, value=3)
            show_alternatives = st.checkbox("Show alternatives", value=True)
            show_debug = st.checkbox("Show match details", value=False)

        st.info(f"Wardrobe items saved: {wardrobe.count()}")
        st.caption(
            "Create Outfit uses your wardrobe when enabled, then fills in missing pieces with "
            "live store picks based on occasion and body shape."
        )

        if st.button("Create Outfit", type="primary"):
            progress = st.progress(15, text="Checking your wardrobe...")

            retriever = None
            if count > 0:
                embedder = FashionEmbedder()
                retriever = FashionRetriever(store, embedder)
            composer = OutfitComposer(retriever, wardrobe_manager=wardrobe)
            intent = parse_intent_with_langchain(query, gender, occasion, body_shape=body_shape_arg)

            progress.progress(45, text="Finding pieces for your look...")
            look = composer.compose(
                intent,
                use_wardrobe_first=use_wardrobe_first,
                preferred_store=preferred_store,
                use_live_api=True,
                catalog_fallback=count > 0,
            )

            advice_text = ""
            if intent.body_shape:
                progress.progress(80, text="Adding styling notes for your shape...")
                try:
                    kr = KnowledgeRetriever()
                    advice_text = kr.get_advice_with_langchain(intent.body_shape)
                except Exception as e:
                    advice_text = f"Body Shape {intent.body_shape} rule applied."

            progress.progress(100, text="Done!")
            progress.empty()

            st.subheader("Styling Summary")
            shape_label = SHAPE_NAMES.get(intent.body_shape, intent.body_shape or 'Not Specified')
            c_info1, c_info2, c_info3, c_info4 = st.columns(4)
            c_info1.metric("Occasion", intent.occasion.title())
            c_info2.metric("Gender", intent.gender.title())
            c_info3.metric("Body Shape", intent.body_shape or "Auto/None", help=shape_label)
            c_info4.metric("Sources", ", ".join(look.get("sources_used") or ["none"]))

            st.divider()
            st.subheader("Your Recommended Outfit")
            st.metric("Overall Style Score", f"{look.get('look_score', 0)}/10")
            render_owned_vs_missing(look)

            selected = look["selected"]
            if not selected:
                st.warning(
                    "No suitable outfit items found. Try adding a few wardrobe items "
                    "or a different occasion/store."
                )
            else:
                if view_mode == "Visual cards":
                    render_card_grid(selected, columns=card_columns, show_debug=show_debug)
                elif view_mode == "Outfit board":
                    cols = st.columns(min(5, len(selected)))
                    for idx, (slot, item) in enumerate(selected.items()):
                        with cols[idx % len(cols)]:
                            render_item_card(item, slot, show_debug=show_debug)
                else:
                    for slot, item in selected.items():
                        item_e = enrich_product_display(item)
                        if item.get("owned"):
                            owned_tag = "OWNED"
                        elif item.get("source") == "serpapi":
                            owned_tag = "LIVE STORE"
                        else:
                            owned_tag = "TO BUY"
                        st.write(
                            f"**{slot.title()}** ({owned_tag}): {item_e.get('display_title')} "
                            f"({item_e.get('price')}) — score {round(item.get('final_score', 0), 2)}"
                        )
                        if item.get("url") and not item.get("owned"):
                            st.link_button(
                                f"Buy {slot.title()}",
                                item["url"],
                                key=f"buy_{slot}_{item.get('id')}",
                            )

            st.divider()
            st.subheader("Styling Notes")
            st.markdown(look["explanation"])
            if advice_text:
                with st.expander("More styling details for your shape"):
                    st.markdown(advice_text)

            if show_alternatives:
                st.divider()
                st.subheader("More Options")
                render_alternative_cards(look["alternatives"], columns=card_columns, show_debug=show_debug)


if __name__ == "__main__":
    cli()
    run_app()