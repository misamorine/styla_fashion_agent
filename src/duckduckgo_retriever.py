import json
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from ddgs import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False


STORE_INFO = {
    "Myntra": {
        "badge": "🛍️ Myntra",
        "domain": "myntra.com",
        "url_template": "https://www.myntra.com/{query}",
    },
    "Amazon.in": {
        "badge": "📦 Amazon.in",
        "domain": "amazon.in",
        "url_template": "https://www.amazon.in/s?k={query}",
    },
    "Nykaa Fashion": {
        "badge": "✨ Nykaa",
        "domain": "nykaafashion.com",
        "url_template": "https://www.nykaafashion.com/catalogsearch/result/?q={query}",
    },
    "Ajio": {
        "badge": "👗 Ajio",
        "domain": "ajio.com",
        "url_template": "https://www.ajio.com/search/?text={query}",
    },
}

_PDP_PATTERNS: Dict[str, List[re.Pattern]] = {
    "Amazon.in": [
        re.compile(r"amazon\.in/.*/dp/[A-Z0-9]{10}", re.I),
        re.compile(r"amazon\.in/dp/[A-Z0-9]{10}", re.I),
        re.compile(r"amazon\.in/gp/product/[A-Z0-9]{10}", re.I),
    ],
    "Myntra": [
        re.compile(r"myntra\.com/.+/\d{5,}/buy/?", re.I),
        re.compile(r"myntra\.com/[^/]+/[^/]+/[^/]+/\d{5,}(?:/|$|\?)", re.I),
    ],
    "Ajio": [
        re.compile(r"ajio\.com/p/", re.I),
        re.compile(r"ajio\.com/.+/p/[0-9a-z_]+", re.I),
        re.compile(r"ajio\.com/.+/\d{6,}(?:_|$|\?|/)", re.I),
    ],
    "Nykaa Fashion": [
        re.compile(r"nykaafashion\.com/.+/p/\d+", re.I),
        re.compile(r"nykaafashion\.com/.+/\d{6,}(?:/|$|\?)", re.I),
    ],
}

_GENDER_WORDS = {
    "female": ("female", "women", "woman", "ladies", "lady"),
    "male": ("male", "men", "man", "gents"),
    "unisex": ("unisex",),
}

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Short in-process cache so Streamlit reruns don't re-hit stores.
_RESULT_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
_CACHE_TTL_SEC = 180.0


def _gender_term(gender: str) -> str:
    g = (gender or "").lower()
    if g == "female":
        return "women"
    if g == "male":
        return "men"
    return ""


def _clean_query(query: str, gender: str) -> str:
    q = (query or "").strip()
    words = _GENDER_WORDS.get((gender or "").lower(), ()) + (
        "unisex", "female", "male", "women", "men", "woman", "man",
    )
    changed = True
    while changed and q:
        changed = False
        lower = q.lower()
        for w in words:
            prefix = w + " "
            if lower.startswith(prefix):
                q = q[len(prefix):].strip()
                changed = True
                break
    return q or (query or "").strip()


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.rstrip("/").lower()
        return f"{netloc}{path}"
    except Exception:
        return url.strip().lower()


def _normalize_image_url(url: str) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        netloc = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.rstrip("/").lower()
        return f"{netloc}{path}"
    except Exception:
        return url.strip().lower()


def store_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    host = urlparse(url).netloc.lower().replace("www.", "")
    for name, meta in STORE_INFO.items():
        if meta["domain"] in host:
            return name
    return None


def is_product_page(url: str, store_name: Optional[str] = None) -> bool:
    if not url:
        return False
    lower = url.lower()
    if any(h in lower for h in ("/s?", "/search", "catalogsearch", "/shop/", "/live/")):
        if not (("amazon.in" in lower) and ("/dp/" in lower or "/gp/product/" in lower)):
            return False
    stores = [store_name] if store_name in _PDP_PATTERNS else list(_PDP_PATTERNS.keys())
    for name in stores:
        for pattern in _PDP_PATTERNS[name]:
            if pattern.search(url):
                return True
    return False


def canonicalize_product_url(url: str, store_name: Optional[str] = None) -> str:
    if not url:
        return url
    store = store_name or store_from_url(url)
    try:
        if store == "Amazon.in":
            m = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", url, re.I)
            if m:
                return f"https://www.amazon.in/dp/{m.group(1).upper()}"
        parsed = urlparse(url)
        return urlunparse(parsed._replace(query="", fragment=""))
    except Exception:
        return url


def search_url_for_store(store_name: str, query: str, gender: str) -> str:
    clean = _clean_query(query, gender)
    g = _gender_term(gender)
    enc_q = urllib.parse.quote_plus(f"{g} {clean}".strip())
    return STORE_INFO[store_name]["url_template"].format(query=enc_q)


def _build_result(
    *,
    title: str,
    image_url: str,
    source_url: str,
    store_name: str,
    query: str,
    is_pdp: bool,
) -> Dict:
    meta = STORE_INFO[store_name]
    return {
        "title": (title or query.title())[:65],
        "image_url": image_url,
        "thumbnail": image_url,
        "source_url": source_url,
        "store_name": store_name,
        "store_badge": meta["badge"],
        "is_pdp": is_pdp,
        "query": query,
    }


def _should_keep(
    image_url: str,
    source_url: str,
    seen_images: Set[str],
    seen_urls: Set[str],
) -> bool:
    img_key = _normalize_image_url(image_url)
    url_key = _normalize_url(source_url)
    if not img_key:
        return False
    if img_key in seen_images:
        return False
    if url_key and url_key in seen_urls:
        return False
    return True


def _mark_seen(image_url: str, source_url: str, seen_images: Set[str], seen_urls: Set[str]) -> None:
    img_key = _normalize_image_url(image_url)
    url_key = _normalize_url(source_url)
    if img_key:
        seen_images.add(img_key)
    if url_key:
        seen_urls.add(url_key)


def _cache_key(query: str, max_results: int, preferred_store: str, gender: str) -> str:
    return f"{preferred_store}|{gender}|{query.strip().lower()}|{max_results}"


def _fetch_html(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _amazon_search_url(clean_q: str, gender_term: str) -> str:
    q = f"{gender_term} {clean_q}".strip() if gender_term else clean_q
    return f"https://www.amazon.in/s?k={urllib.parse.quote_plus(q)}"


def _myntra_search_urls(clean_q: str, gender_term: str) -> List[str]:
    """Build likely Myntra listing URLs for a fashion query."""
    slug = re.sub(r"\s+", "-", clean_q.strip().lower())
    slug = re.sub(r"[^a-z0-9\-]+", "", slug)
    urls: List[str] = []
    if gender_term and slug:
        urls.append(f"https://www.myntra.com/{gender_term}-{slug}")
        # plural-ish category guesses
        if not slug.endswith("s"):
            urls.append(f"https://www.myntra.com/{gender_term}-{slug}s")
    if slug:
        gender_filter = ""
        if gender_term == "men":
            gender_filter = "&f=Gender%3Amen%2Cmen%20women"
        elif gender_term == "women":
            gender_filter = "&f=Gender%3Awomen%2Cmen%20women"
        urls.append(f"https://www.myntra.com/{slug}?rawQuery={urllib.parse.quote(clean_q)}{gender_filter}")
    # de-dupe
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _scrape_amazon_products(clean_q: str, gender_term: str, max_keep: int) -> List[Dict]:
    """Scrape Amazon.in search results for unique PDP + image pairs."""
    out: List[Dict] = []
    try:
        html = _fetch_html(_amazon_search_url(clean_q, gender_term))
    except Exception as e:
        print(f"Amazon scrape notice: {e}")
        return out

    blocks = re.split(r'data-asin="', html)[1:]
    seen_asins: Set[str] = set()
    for block in blocks:
        if len(out) >= max_keep:
            break
        asin = block[:10]
        if not re.match(r"^[A-Z0-9]{10}$", asin) or asin in seen_asins:
            continue

        chunk = block[:6000]
        img = None
        for pat in (
            r'src="(https://m\.media-amazon\.com/images/I/[^"]+\.jpg)"',
            r'data-src="(https://m\.media-amazon\.com/images/I/[^"]+\.jpg)"',
            r'srcset="(https://m\.media-amazon\.com/images/I/[^"\s]+\.jpg)',
        ):
            m = re.search(pat, chunk)
            if m:
                img = m.group(1)
                break
        if not img:
            continue

        title = None
        for pat in (
            r'alt="([^"]{8,180})"',
            r'a-size-base-plus[^"]*"[^>]*>([^<]{8,140})</span>',
            r'a-size-medium[^"]*a-color-base[^"]*"[^>]*>([^<]{8,140})</span>',
            r'a-text-normal"[^>]*>\s*<span[^>]*>([^<]{8,140})</span>',
        ):
            m = re.search(pat, chunk)
            if m:
                cand = re.sub(r"^Sponsored Ad\s*-\s*", "", m.group(1).strip(), flags=re.I)
                if cand and cand.lower() not in {"shop", "sponsored", "amazon"}:
                    title = cand
                    break
        if not title:
            title = f"{clean_q.title()} ({asin})"

        seen_asins.add(asin)
        source_url = f"https://www.amazon.in/dp/{asin}"
        # Prefer larger image variant when possible.
        img = re.sub(r"\._AC_[^.]+_\.", "._AC_UL480_.", img)
        out.append(
            _build_result(
                title=title,
                image_url=img,
                source_url=source_url,
                store_name="Amazon.in",
                query=clean_q,
                is_pdp=True,
            )
        )
    return out


def _iter_myntra_products(obj) -> Iterable[Dict]:
    if isinstance(obj, dict):
        if "productId" in obj and ("searchImage" in obj or "landingPageUrl" in obj):
            yield obj
        for v in obj.values():
            yield from _iter_myntra_products(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_myntra_products(v)


def _scrape_myntra_products(clean_q: str, gender_term: str, max_keep: int) -> List[Dict]:
    """Scrape Myntra listing pages for unique PDP + image pairs."""
    out: List[Dict] = []
    seen_ids: Set[str] = set()

    for url in _myntra_search_urls(clean_q, gender_term):
        if len(out) >= max_keep:
            break
        try:
            html = _fetch_html(url)
        except Exception as e:
            print(f"Myntra scrape notice ({url}): {e}")
            continue

        products: List[Dict] = []
        m = re.search(r"<script>\s*window\.__myx\s*=\s*(\{.*?\})\s*</script>", html, re.S)
        if m:
            try:
                payload = json.loads(m.group(1))
                products = list(_iter_myntra_products(payload))
            except Exception as e:
                print(f"Myntra JSON notice: {e}")

        if not products:
            # Regex fallback from embedded JSON strings.
            ids = re.findall(r'"productId"\s*:\s*(\d+)', html)
            names = re.findall(r'"productName"\s*:\s*"([^"]+)"', html)
            imgs = re.findall(r'"searchImage"\s*:\s*"([^"]+)"', html)
            lands = re.findall(r'"landingPageUrl"\s*:\s*"([^"]+)"', html)
            for i, pid in enumerate(ids):
                products.append(
                    {
                        "productId": pid,
                        "productName": names[i] if i < len(names) else clean_q.title(),
                        "searchImage": imgs[i].encode().decode("unicode_escape") if i < len(imgs) else "",
                        "landingPageUrl": lands[i].encode().decode("unicode_escape") if i < len(lands) else "",
                    }
                )

        for p in products:
            if len(out) >= max_keep:
                break
            pid = str(p.get("productId") or "")
            if not pid or pid in seen_ids:
                continue
            land = (p.get("landingPageUrl") or "").lstrip("/")
            if not land:
                continue
            if land.startswith("http"):
                source_url = land
            else:
                source_url = "https://www.myntra.com/" + land.replace("\\u002F", "/")
            source_url = urllib.parse.unquote(source_url)
            if not is_product_page(source_url, "Myntra"):
                # Force buy URL shape when we have an id.
                source_url = re.sub(r"/+$", "", source_url)
                if not source_url.endswith("/buy"):
                    source_url = source_url + "/buy" if re.search(r"/\d{5,}$", source_url) else source_url
                if not is_product_page(source_url, "Myntra"):
                    continue

            img = p.get("searchImage") or ""
            if isinstance(img, str):
                img = img.replace("\\u002F", "/").replace("http://", "https://")
            if not img and isinstance(p.get("images"), list) and p["images"]:
                img = (p["images"][0] or {}).get("src") or ""
                img = str(img).replace("http://", "https://")
            if not img:
                continue

            title = p.get("productName") or p.get("product") or clean_q.title()
            brand = p.get("brand") or ""
            if brand and brand.lower() not in str(title).lower():
                title = f"{brand} {title}"

            seen_ids.add(pid)
            out.append(
                _build_result(
                    title=str(title),
                    image_url=img,
                    source_url=canonicalize_product_url(source_url, "Myntra"),
                    store_name="Myntra",
                    query=clean_q,
                    is_pdp=True,
                )
            )

        if out:
            break
    return out


def _safe_images(ddgs: "DDGS", term: str, max_results: int) -> List[Dict]:
    try:
        return list(ddgs.images(term, max_results=max_results, region="in-en"))
    except Exception as e:
        print(f"Image search notice: {e}")
        return []


def _ingest_image_hits(
    hits: Iterable[Dict],
    *,
    allowed_stores: Set[str],
    query: str,
    seen_images: Set[str],
    seen_urls: Set[str],
    max_keep: int,
) -> List[Dict]:
    out: List[Dict] = []
    for item in hits:
        if len(out) >= max_keep:
            break
        img_url = item.get("image") or item.get("thumbnail") or ""
        page_url = item.get("url") or ""
        title = item.get("title") or f"{query.title()}"
        store = store_from_url(page_url)
        if store not in allowed_stores:
            continue
        if not img_url or not page_url or not is_product_page(page_url, store):
            continue
        source_url = canonicalize_product_url(page_url, store)
        if not _should_keep(img_url, source_url, seen_images, seen_urls):
            continue
        _mark_seen(img_url, source_url, seen_images, seen_urls)
        out.append(
            _build_result(
                title=title,
                image_url=img_url,
                source_url=source_url,
                store_name=store,
                query=query,
                is_pdp=True,
            )
        )
    return out


def search_trendy_fashion_products(
    query: str, max_results: int = 8, preferred_store: str = "All Stores", gender: str = "female"
) -> List[Dict]:
    """
    Live store product search across Indian e-commerce sites.

    Primary: scrape Amazon.in / Myntra listing pages (reliable product URL + image pairs).
    Optional: DuckDuckGo images when available (not required).
    """
    if preferred_store in STORE_INFO:
        selected_stores = [preferred_store]
    else:
        # Prefer Amazon + Myntra for reliability.
        selected_stores = ["Amazon.in", "Myntra", "Ajio", "Nykaa Fashion"]

    cache_key = _cache_key(query, max_results, preferred_store, gender)
    cached = _RESULT_CACHE.get(cache_key)
    if cached:
        ts, payload = cached
        if time.time() - ts < _CACHE_TTL_SEC and payload:
            return [dict(item) for item in payload[:max_results]]

    clean_q = _clean_query(query, gender)
    gender_term = _gender_term(gender)

    results: List[Dict] = []
    seen_images: Set[str] = set()
    seen_urls: Set[str] = set()

    def _extend(batch: List[Dict]) -> None:
        for item in batch:
            if len(results) >= max_results:
                return
            if not _should_keep(item["image_url"], item["source_url"], seen_images, seen_urls):
                continue
            _mark_seen(item["image_url"], item["source_url"], seen_images, seen_urls)
            results.append(item)

    per_store = max(3, (max_results + 1) // max(1, min(2, len(selected_stores))))

    # 1) Direct store scrapes — independent of DuckDuckGo rate limits.
    if "Amazon.in" in selected_stores and len(results) < max_results:
        _extend(_scrape_amazon_products(clean_q, gender_term, per_store))
    if "Myntra" in selected_stores and len(results) < max_results:
        need = max_results - len(results) if preferred_store == "Myntra" else min(per_store, max_results - len(results))
        _extend(_scrape_myntra_products(clean_q, gender_term, need))

    # 2) Optional DDG enrichment when still short (ignore failures).
    if HAS_DDG and len(results) < max_results:
        try:
            with DDGS() as ddgs:
                for store in selected_stores:
                    if len(results) >= max_results:
                        break
                    if store not in ("Amazon.in", "Myntra"):
                        continue
                    domain = STORE_INFO[store]["domain"]
                    phrase = f"{gender_term} {clean_q}".strip()
                    hits = _safe_images(ddgs, f"site:{domain} {phrase}", max(max_results * 2, 8))
                    batch = _ingest_image_hits(
                        hits,
                        allowed_stores={store},
                        query=clean_q,
                        seen_images=seen_images,
                        seen_urls=seen_urls,
                        max_keep=max_results - len(results),
                    )
                    results.extend(batch)
        except Exception as e:
            print(f"Web search notice: {e}")

    results = [r for r in results if r.get("is_pdp") and r.get("source_url") and r.get("image_url")]
    results = results[:max_results]
    if results:
        _RESULT_CACHE[cache_key] = (time.time(), [dict(r) for r in results])
    return results
