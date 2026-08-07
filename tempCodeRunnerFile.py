            <div class="fashion-meta">{item.get('gender', 'unknown')} · {item.get('slot', 'unknown')} · {item.get('color', 'unknown')} · {item.get('occasion_group', 'unknown')}</div>
            <div class="fashion-meta">{desc}</div>
            <span class="fashion-score">Score: {round(item.get('final_score', 0), 2)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if show_debug:
        with st.expander("Debug filter"):
            st.write("Rule score:", round(item.get("rule_score", 0), 2))
            st.write("Vector score:", round(item.get("vector_score", 0), 2))
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
    st.subheader("🧍 Body Shape Analysis & AI Styling Advice")
    st.caption("Upload a full-body photo to automatically detect your BSTI body shape code (A, H, X, Y) and receive custom AI styling advice.")

    uploaded_file = st.file_uploader("Upload full-body photo", type=["jpg", "jpeg", "png"], key="body_photo_upload")

    if uploaded_file is not None:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp.write(uploaded_file.read())
        temp.close()
        image_path = temp.name

        st.image(image_path, width=300, caption="Uploaded Full-Body Photo")

        if st.button("✨ Detect Body Shape & Retrieve RAG Styling Knowledge", type="primary"):
            with st.spinner("🧍 Running OpenCV & MediaPipe pose segmentation..."):
                try:
                    shape = detect_body_shape(image_path)
                except Exception as e:
                    st.error(f"Error analyzing body shape: {e}")
                    return

            st.session_state["detected_body_shape"] = shape

            with st.spinner("📚 Retrieving knowledge for shape code..."):
                try:
                    kr = KnowledgeRetriever()
                    k_res = kr.retrieve(shape)
                    knowledge_doc = k_res[0]["document"] if k_res else f"Knowledge for shape {shape}"
                except Exception as e:
                    knowledge_doc = f"Body shape {shape} rules"

            with st.spinner("✨ Generating AI styling recommendations..."):
                prompt = build_prompt(shape, knowledge_doc)
                advice = generate(prompt)
                st.session_state["body_shape_advice"] = advice

    if "detected_body_shape" in st.session_state:
        shape = st.session_state["detected_body_shape"]
        full_name = SHAPE_NAMES.get(shape, f"Shape {shape}")

        st.success(f"Detected Body Shape: **{shape}** ({full_name})")

        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            st.metric("Body Shape Code", shape)
            st.info(f"Classified as: **{full_name}**")
            st.write("This shape code will be automatically used to guide outfit composition in the Outfit Generator.")

        with c2:
            st.subheader("✨ Personalized Styling Advice")
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
                st.write(f"🛒 **{slot.title()}**: {item.get('title')}")
        else:
            st.success("Great — this complete look can be built from your existing wardrobe.")


def run_app():
    st.set_page_config(page_title="AI Integrated Personal Stylist Agent", layout="wide")
    st.title("👗 AI Integrated Personal Stylist Agent")
    st.write("Integrated System: MediaPipe/OpenCV Body Analysis (A, H, X, Y) + RAG Knowledge Base + ChromaDB Catalog + Wardrobe Agent.")
    inject_card_css()

    store = FashionVectorStore(db_path=DB_PATH)
    wardrobe = WardrobeManager(db_path=WARDROBE_DB, image_dir=WARDROBE_IMAGE_DIR)

    tabs = st.tabs(["✨ Create Outfit", "🧍 Body Shape Analysis", "🧺 My Wardrobe"])

    with tabs[2]:
        render_wardrobe_page(wardrobe)

