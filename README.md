# AI Personal Stylist Agent V2.2

A local AI-powered fashion stylist built with Streamlit, ChromaDB, LangChain, and product search integrations.

This project combines:
- wardrobe management for owned clothing items
- fashion outfit composition using a vector catalog
- body shape analysis from photos
- live store recommendations for missing pieces
- an AI stylist chatbot with tool-enabled search and wardrobe access

## Key Features

- **Create Outfit**: build looks from wardrobe items and fill missing pieces using indexed fashion catalog and live store results
- **My Wardrobe**: upload owned clothing images, save metadata locally, and reuse these items in outfit composition
- **Body Shape Analysis**: upload a full-body photo and get personalized styling advice with body-shape-aware recommendations
- **Shop For You**: search live fashion listings from stores like Myntra, Amazon.in, Ajio, and more
- **Stylist Chatbot**: chat with a LangChain-based stylist assistant that can use tools for wardrobe lookup, search, and outfit advice
- **Local data persistence**: wardrobe saved in `wardrobe_db.json`, images in `wardrobe_images/`, and indexed catalog in `fashion_vector_db_v2/`

## Repository Structure

- `app.py` - Streamlit app entrypoint and main UI logic
- `environment.yml` - Conda environment specification
- `requirements.txt` - Python dependency list
- `src/` - core application modules
  - `embedder.py`, `indexer.py`, `retriever.py`, `outfit_composer.py`, `wardrobe_manager.py`, `vision/` etc.
- `rag/` - RAG prompt and LLM utilities for body shape and stylist advice
- `product_rag/` - catalog/recommender tooling and vector store utilities
- `knowledge/` - shape guidance and fashion knowledge base files
- `fashion_item_images/` - downloaded image assets for the catalog
- `fashion_vector_db_v2/` - local ChromaDB index storage
- `wardrobe_images/` - user wardrobe photos stored locally

## Requirements

Recommended Python environment is defined in `environment.yml`.

The project also depends on optional environment variables for AI and search integrations:
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for the chatbot and RAG-based recommendations
- `SERPAPI_API_KEY` if live SerpAPI product search is enabled

## Install

Using conda:

```bash
conda env create -f environment.yml
conda activate fashion-stylist-agent-v2
```

Or install with pip into an existing environment:

```bash
python -m pip install -r requirements.txt
```

## Index the Fashion Catalog

Before using catalog-powered outfit composition, index the dataset:

```bash
python app.py --index --limit 2000
```

This downloads the fashion dataset, generates embeddings, and stores the index in `fashion_vector_db_v2/`.

## Run the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Usage Guide

1. **My Wardrobe**
   - Upload owned clothing photos and enter item metadata
   - Save items to `wardrobe_db.json`
   - The stylist will prefer owned items when creating outfits

2. **Body Shape**
   - Upload a full-body image
   - Detect your body shape and generate personalized styling advice
   - The body shape is used in outfit scoring and live shop recommendations

3. **Create Outfit**
   - Choose gender, occasion, body shape, and request text
   - Enable `Use my wardrobe first` to prefer owned items
   - Create outfit recommendations with owned pieces and missing item suggestions

4. **Shop For You**
   - Use live store picks tailored to your detected body shape
   - Search products from Myntra, Amazon, Nykaa Fashion, Ajio, and other supported stores

5. **Stylist Chatbot**
   - Chat with the AI stylist
   - Ask outfit, body shape, wardrobe, and shopping questions
   - The chatbot can call tools where configured

## Notes

- The wardrobe item uploader currently uses text metadata plus simple inference for color/slot labels.
- Images and wardrobe metadata are stored locally; this project is designed as a self-hosted stylist prototype.
- For full AI chatbot functionality, configure the relevant API keys in a `.env` file or environment.

## Troubleshooting

- If the app cannot find `fashion_vector_db_v2`, run the index command first.
- If the chat assistant falls back to offline mode, verify that `OPENAI_API_KEY` or `OPENROUTER_API_KEY` is set.
- If image uploads fail, ensure the uploaded files are valid `jpg`, `jpeg`, or `png` files.

## Quick Commands

```bash
conda env create -f environment.yml
conda activate fashion-stylist-agent-v2
python app.py --index --limit 2000
streamlit run app.py
```

## License

No license is specified in this repository. Add one if you plan to share or publish the project.
