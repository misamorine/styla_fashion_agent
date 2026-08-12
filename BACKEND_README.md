# Running the FastAPI + static frontend version

This adds two things to your existing project, without changing anything
inside `app.py`, `src/`, `rag/`, `product_rag/`, or `knowledge/`:

- `backend/main.py` - a FastAPI server that imports and calls your existing
  modules exactly the way `app.py` did, and returns JSON instead of
  rendering Streamlit widgets.
- `frontend/` - a static `index.html` + `style.css` + `script.js` that
  recreates the five tabs (Create Outfit, Shop For You, Body Shape,
  My Wardrobe, Stylist Chatbot) in your espresso/satin theme, and talks to
  the backend over HTTP.

Your original Streamlit app (`streamlit run app.py`) still works exactly
as before - none of that code was touched.

## 1. Install the extra backend dependency

You already have `uvicorn` in `requirements.txt`. You just need `fastapi`
and `python-multipart` (for file uploads):

```bash
pip install -r backend/requirements-backend.txt
```

## 2. Run the backend

**Run this from the project ROOT** (the folder with `app.py` in it) -
not from inside `backend/`. Several existing modules use relative paths
(`./fashion_vector_db_v2`, `./wardrobe_db.json`, `pose_landmarker.task`,
etc.) that only resolve correctly when the working directory is the
project root, exactly like `streamlit run app.py` required.

```bash
uvicorn backend.main:app --reload --port 8000
```

The first request that needs the CLIP model, the wardrobe embedder, or
the body-shape knowledge embedder will take a little while to load those
models into memory, same as the first click on a Streamlit tab did.

## 3. Open the frontend

Two ways, pick whichever you like:

- **Simplest:** open `http://127.0.0.1:8000/app/` in your browser. The
  backend also serves the frontend folder directly, so there's nothing
  else to run and there are no CORS issues.
- **Or:** just double-click `frontend/index.html` to open it as a local
  file. The frontend detects it's running from `file://` and talks to
  `http://127.0.0.1:8000` automatically (CORS is already enabled on the
  backend for this).

## API surface

| Endpoint | Method | Mirrors |
|---|---|---|
| `/api/meta` | GET | sidebar counts / dropdown options |
| `/api/create-outfit` | POST | "Create Outfit" tab |
| `/api/wardrobe` | GET / POST / DELETE | "My Wardrobe" tab |
| `/api/wardrobe/{id}` | DELETE | delete one wardrobe item |
| `/api/body-shape` | POST (multipart photo) | "Body Shape" tab |
| `/api/shop-for-you` | POST | "Shop For You" tab |
| `/api/chat` | POST | "💬 Stylist Chatbot" tab |

`.env` values (`OPENROUTER_API_KEY` / `OPENAI_API_KEY`, `SERPAPI_API_KEY`)
are read the same way they always were, from the project root.
