const API_BASE = (location.protocol === 'file:' || location.port === '5500')
  ? 'http://127.0.0.1:8000'
  : '';

const SHAPE_LABELS = {
  A: 'Pear Shape (Triangle - Hips wider than bust)',
  H: 'Rectangle Shape (Straight - Similar bust, waist, hip)',
  X: 'Hourglass Shape (Curvy - Balanced bust & hip, narrow waist)',
  Y: 'Inverted Triangle Shape (Broad shoulders / bust wider than hip)',
};

// ---------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'shop-for-you') refreshShopForYouVisibility();
    if (btn.dataset.tab === 'wardrobe') loadWardrobe();
  });
});

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (e) {}
    throw new Error(detail);
  }
  return res.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function titleCase(s) {
  if (!s) return '';
  return s.replace(/\b\w/g, (c) => c.toUpperCase());
}

function badgeInfo(item) {
  if (item.owned) return { text: 'OWNED', cls: 'owned' };
  if (item.is_live_store) return { text: 'LIVE STORE', cls: 'live' };
  return { text: 'TO BUY', cls: 'buy' };
}

function renderItemCard(item, slotName, showDebug) {
  const tpl = document.getElementById('tpl-item-card');
  const node = tpl.content.firstElementChild.cloneNode(true);

  const img = node.querySelector('.item-image');
  const imageWrap = node.querySelector('.item-image-wrap');
  if (item.image_url) {
    img.src = item.image_url;
    img.alt = item.display_title || item.title || '';
  } else {
    img.remove();
    const fallback = el('div', 'no-image-fallback', 'No image');
    imageWrap.appendChild(fallback);
  }

  const badge = badgeInfo(item);
  const badgeEl = node.querySelector('.owned-badge');
  badgeEl.textContent = badge.text;
  badgeEl.classList.add(badge.cls);

  node.querySelector('.item-slot').textContent = titleCase(slotName || item.slot || '');
  node.querySelector('.item-title').textContent = item.display_title || item.title || 'Item';
  node.querySelector('.item-price').textContent = item.price || '';
  node.querySelector('.item-meta').textContent =
    `${titleCase(item.gender || 'unknown')} \u00b7 ${titleCase(item.color || 'unknown')} \u00b7 ${titleCase(item.occasion_group || 'unknown')}`;

  const actions = node.querySelector('.item-actions');
  if (item.store) {
    const storeTag = el('span', 'item-meta', `Source: ${item.store}`);
    actions.appendChild(storeTag);
  }

  const debugEl = node.querySelector('.item-debug');
  if (showDebug) {
    debugEl.classList.remove('hidden');
    debugEl.innerHTML =
      `Overall match: ${item.final_score ?? 0}<br>` +
      `Style fit: ${item.rule_score ?? 0}<br>` +
      `Similarity match: ${item.vector_score ?? 0}`;
  }

  return node;
}

function renderCardGrid(itemsBySlot, showDebug) {
  const grid = el('div', 'card-grid');
  Object.entries(itemsBySlot).forEach(([slot, item]) => {
    grid.appendChild(renderItemCard(item, slot, showDebug));
  });
  return grid;
}

// ---------------------------------------------------------------------
// Meta / wardrobe count pill
// ---------------------------------------------------------------------
async function refreshMeta() {
  try {
    const meta = await apiFetch('/api/meta');
    document.getElementById('wardrobeCountPill').textContent = `Wardrobe: ${meta.wardrobe_count} items`;
  } catch (e) {
    document.getElementById('wardrobeCountPill').textContent = 'Backend not connected';
  }
}
refreshMeta();

// ---------------------------------------------------------------------
// CREATE OUTFIT
// ---------------------------------------------------------------------
const coSubmit = document.getElementById('co-submit');
const coEmpty = document.getElementById('co-empty');
const coLoading = document.getElementById('co-loading');
const coLoadingText = document.getElementById('co-loading-text');
const coResults = document.getElementById('co-results');

coSubmit.addEventListener('click', async () => {
  coEmpty.classList.add('hidden');
  coResults.classList.add('hidden');
  coLoading.classList.remove('hidden');
  coLoadingText.textContent = 'Checking your wardrobe & Marqo Polyvore catalog...';
  coSubmit.disabled = true;

  const payload = {
    query: document.getElementById('co-query').value || 'outfit',
    gender: document.getElementById('co-gender').value,
    occasion: document.getElementById('co-occasion').value,
    body_shape: document.getElementById('co-body-shape').value || null,
    use_wardrobe_first: document.getElementById('co-use-wardrobe').checked,
  };

  const timer = setTimeout(() => { coLoadingText.textContent = 'Finding pieces for your look...'; }, 900);

  try {
    const look = await apiFetch('/api/create-outfit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    clearTimeout(timer);
    renderCreateOutfitResults(look, document.getElementById('co-show-debug').checked);
    document.getElementById('wardrobeCountPill').textContent = `Wardrobe: ${look.wardrobe_count} items`;
  } catch (err) {
    clearTimeout(timer);
    coResults.innerHTML = '';
    coResults.appendChild(el('div', 'notice-card notice-error', `Could not create an outfit: ${err.message}`));
    coResults.classList.remove('hidden');
  } finally {
    coLoading.classList.add('hidden');
    coSubmit.disabled = false;
  }
});

function renderCreateOutfitResults(look, showDebug) {
  coResults.innerHTML = '';

  const summary = el('div', 'summary-strip');
  const metrics = [
    ['Occasion', titleCase(look.intent.occasion)],
    ['Gender', titleCase(look.intent.gender)],
    ['Body shape', look.intent.body_shape || 'Auto / none'],
    ['Sources', (look.sources_used && look.sources_used.length) ? look.sources_used.join(', ') : 'none'],
  ];
  metrics.forEach(([label, value]) => {
    const box = el('div', 'metric-box');
    box.appendChild(el('div', 'metric-label', label));
    box.appendChild(el('div', 'metric-value', value));
    summary.appendChild(box);
  });
  coResults.appendChild(summary);

  const scoreRow = el('div', 'score-banner');
  scoreRow.appendChild(el('span', 'score-num', `${look.look_score}/10`));
  scoreRow.appendChild(el('span', 'score-label', 'Overall style score'));
  coResults.appendChild(scoreRow);

  // owned vs missing
  const ownedCount = Object.keys(look.owned_selected || {}).length;
  const missingCount = Object.keys(look.shopping_needed || {}).length;
  const omGrid = el('div', 'owned-missing-grid');

  const ownedCard = el('div', 'card');
  ownedCard.appendChild(el('h4', '', `Already owned (${ownedCount})`));
  if (ownedCount) {
    const ul = el('ul');
    Object.entries(look.owned_selected).forEach(([slot, item]) => {
      ul.appendChild(el('li', '', `${titleCase(slot)}: ${item.title || item.display_title}`));
    });
    ownedCard.appendChild(ul);
  } else {
    ownedCard.appendChild(el('p', '', 'No owned wardrobe item was suitable for this look yet.'));
  }
  omGrid.appendChild(ownedCard);

  const missingCard = el('div', 'card');
  missingCard.appendChild(el('h4', '', `Suggested to buy (${missingCount})`));
  if (missingCount) {
    const ul = el('ul');
    Object.entries(look.shopping_needed).forEach(([slot, item]) => {
      ul.appendChild(el('li', '', `${titleCase(slot)}: ${item.display_title || item.title} (${item.price || ''})`));
    });
    missingCard.appendChild(ul);
  } else {
    missingCard.appendChild(el('p', '', 'This complete look can be built from your existing wardrobe.'));
  }
  omGrid.appendChild(missingCard);
  coResults.appendChild(omGrid);

  // selected items
  coResults.appendChild(el('div', 'subsection-title', 'Your recommended outfit'));
  const selectedKeys = Object.keys(look.selected || {});
  if (!selectedKeys.length) {
    coResults.appendChild(el('div', 'notice-card', 'No suitable outfit items found. Try adding a few wardrobe items or a different occasion/store.'));
  } else {
    coResults.appendChild(renderCardGrid(look.selected, showDebug));
  }

  // styling notes
  coResults.appendChild(el('div', 'subsection-title', 'Styling notes'));
  coResults.appendChild(el('div', 'explanation-box', look.explanation || ''));

  if (look.advice_text) {
    const details = document.createElement('details');
    details.className = 'card';
    details.style.marginTop = '14px';
    const summaryEl = el('summary', '', 'More styling details for your shape');
    summaryEl.style.cursor = 'pointer';
    summaryEl.style.fontWeight = '700';
    details.appendChild(summaryEl);
    const adviceDiv = el('div', 'advice-scroll', look.advice_text);
    adviceDiv.style.marginTop = '12px';
    details.appendChild(adviceDiv);
    coResults.appendChild(details);
  }

  // alternatives
  const altEntries = Object.entries(look.alternatives || {}).filter(([, items]) => items.length);
  if (altEntries.length) {
    coResults.appendChild(el('div', 'subsection-title', 'More options'));
    altEntries.forEach(([slot, items]) => {
      const details = document.createElement('details');
      details.className = 'alt-group';
      details.appendChild(el('summary', '', `${titleCase(slot)} alternatives (${items.length})`));
      const grid = el('div', 'card-grid');
      items.slice(0, 8).forEach((item) => grid.appendChild(renderItemCard(item, slot, showDebug)));
      details.appendChild(grid);
      coResults.appendChild(details);
    });
  }

  coResults.classList.remove('hidden');
}

// ---------------------------------------------------------------------
// BODY SHAPE
// ---------------------------------------------------------------------
const bsFileFront = document.getElementById('bs-file-front');
const bsPreviewFront = document.getElementById('bs-preview-front');
const bsFileSide = document.getElementById('bs-file-side');
const bsPreviewSide = document.getElementById('bs-preview-side');
const bsSubmit = document.getElementById('bs-submit');
const bsLoading = document.getElementById('bs-loading');
const bsResults = document.getElementById('bs-results');

if (bsFileFront) {
  bsFileFront.addEventListener('change', () => {
    const file = bsFileFront.files[0];
    if (!file) {
      bsPreviewFront.classList.add('hidden');
      bsSubmit.disabled = true;
      return;
    }
    bsPreviewFront.src = URL.createObjectURL(file);
    bsPreviewFront.classList.remove('hidden');
    bsSubmit.disabled = false;
  });
}

if (bsFileSide) {
  bsFileSide.addEventListener('change', () => {
    const file = bsFileSide.files[0];
    if (!file) {
      bsPreviewSide.classList.add('hidden');
      return;
    }
    bsPreviewSide.src = URL.createObjectURL(file);
    bsPreviewSide.classList.remove('hidden');
  });
}

if (bsSubmit) {
  bsSubmit.addEventListener('click', async () => {
    const frontFile = bsFileFront ? bsFileFront.files[0] : null;
    if (!frontFile) return;

    bsResults.classList.add('hidden');
    bsLoading.classList.remove('hidden');
    bsSubmit.disabled = true;

    const formData = new FormData();
    formData.append('front_photo', frontFile);
    formData.append('photo', frontFile);

    const sideFile = bsFileSide ? bsFileSide.files[0] : null;
    if (sideFile) {
      formData.append('side_photo', sideFile);
    }

    try {
      const data = await apiFetch('/api/body-shape', { method: 'POST', body: formData });
      localStorage.setItem('stylist_body_shape', data.shape);
      localStorage.setItem('stylist_body_shape_label', data.shape_label);

      document.getElementById('bs-shape-banner').textContent = `Your body shape profile: ${data.shape} (${data.shape_label})`;
      document.getElementById('bs-advice').textContent = data.advice;
      bsResults.classList.remove('hidden');

      const coShapeSelect = document.getElementById('co-body-shape');
      if (coShapeSelect) {
        if (![...coShapeSelect.options].some((o) => o.value === data.shape)) {
          const opt = document.createElement('option');
          opt.value = data.shape;
          opt.textContent = `${data.shape} (${data.shape_label})`;
          coShapeSelect.appendChild(opt);
        }
        coShapeSelect.value = data.shape;
      }
      refreshShopForYouVisibility();
    } catch (err) {
      bsResults.innerHTML = '';
      bsResults.appendChild(el('div', 'notice-card notice-error', `Error analyzing body shape: ${err.message}`));
      bsResults.classList.remove('hidden');
    } finally {
      bsLoading.classList.add('hidden');
      bsSubmit.disabled = false;
    }
  });
}


// ---------------------------------------------------------------------
// SHOP FOR YOU
// ---------------------------------------------------------------------
function refreshShopForYouVisibility() {
  const shape = localStorage.getItem('stylist_body_shape');
  const noShape = document.getElementById('sfy-no-shape');
  const body = document.getElementById('sfy-body');
  if (!shape) {
    noShape.classList.remove('hidden');
    body.classList.add('hidden');
    return;
  }
  noShape.classList.add('hidden');
  body.classList.remove('hidden');
  const label = localStorage.getItem('stylist_body_shape_label') || shape;
  document.getElementById('sfy-shape-banner').textContent = `Using body shape ${shape} \u2014 ${label}`;
}
refreshShopForYouVisibility();

document.getElementById('sfy-submit').addEventListener('click', async () => {
  const shape = localStorage.getItem('stylist_body_shape');
  if (!shape) return;

  const loading = document.getElementById('sfy-loading');
  const resultsEl = document.getElementById('sfy-results');
  resultsEl.innerHTML = '';
  loading.classList.remove('hidden');

  const payload = {
    shape,
    gender: document.getElementById('sfy-gender').value,
    preferred_store: document.getElementById('sfy-store').value,
  };

  try {
    const data = await apiFetch('/api/shop-for-you', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderShopForYouResults(data);
  } catch (err) {
    resultsEl.appendChild(el('div', 'notice-card notice-error', `Failed to fetch products: ${err.message}`));
  } finally {
    loading.classList.add('hidden');
  }
});

function renderShopForYouResults(data) {
  const resultsEl = document.getElementById('sfy-results');
  resultsEl.innerHTML = '';

  if (data.style_goal) {
    resultsEl.appendChild(el('div', 'notice-card notice-success', `Style goal: ${data.style_goal}`));
  }

  const anyProducts = (data.groups || []).some((g) => (g.products || []).length);
  if (!anyProducts) {
    resultsEl.appendChild(el('div', 'notice-card', 'No live products found for your body-shape recommendations. Try another store.'));
    return;
  }

  data.groups.forEach((group) => {
    const wrap = el('div', 'shop-group');
    wrap.appendChild(el('h3', '', group.item));
    if (group.reason) wrap.appendChild(el('p', 'reason', group.reason));

    if (group.error) {
      wrap.appendChild(el('div', 'notice-card notice-error', group.error));
      resultsEl.appendChild(wrap);
      return;
    }
    if (!group.products || !group.products.length) {
      wrap.appendChild(el('div', 'notice-card', 'No products found for this style recommendation.'));
      resultsEl.appendChild(wrap);
      return;
    }

    const grid = el('div', 'card-grid');
    group.products.forEach((product) => grid.appendChild(renderLiveProductCard(product)));
    wrap.appendChild(grid);
    resultsEl.appendChild(wrap);
  });
}

function renderLiveProductCard(product) {
  const card = el('div', 'item-card');
  const imgWrap = el('div', 'item-image-wrap');
  if (product.image) {
    const img = el('img', 'item-image');
    img.src = product.image;
    img.alt = product.title || '';
    imgWrap.appendChild(img);
  } else {
    imgWrap.appendChild(el('div', 'no-image-fallback', 'No image available'));
  }
  card.appendChild(imgWrap);

  const body = el('div', 'item-body');
  body.appendChild(el('div', 'item-title', (product.title || 'Untitled product').slice(0, 70)));
  body.appendChild(el('div', 'item-price', product.price || 'Price not available'));
  body.appendChild(el('div', 'item-meta', `Store: ${product.store || 'Unknown store'}`));
  if (product.description) body.appendChild(el('div', 'item-meta', product.description));

  const actions = el('div', 'item-actions');
  if (product.url) {
    const a = el('a', '', 'Buy now');
    a.href = product.url;
    a.target = '_blank';
    a.rel = 'noopener';
    actions.appendChild(a);
  } else {
    actions.appendChild(el('span', 'item-meta', 'Product link not available.'));
  }
  body.appendChild(actions);
  card.appendChild(body);
  return card;
}

// ---------------------------------------------------------------------
// MY WARDROBE
// ---------------------------------------------------------------------
async function loadWardrobe() {
  const grid = document.getElementById('wd-grid');
  const emptyEl = document.getElementById('wd-empty');
  const countEl = document.getElementById('wd-count');
  try {
    const data = await apiFetch('/api/wardrobe');
    countEl.textContent = `${data.count} items`;
    document.getElementById('wardrobeCountPill').textContent = `Wardrobe: ${data.count} items`;
    grid.innerHTML = '';
    if (!data.count) {
      emptyEl.classList.remove('hidden');
      return;
    }
    emptyEl.classList.add('hidden');
    data.items.forEach((item) => grid.appendChild(renderWardrobeCard(item)));
  } catch (err) {
    grid.innerHTML = '';
    grid.appendChild(el('div', 'notice-card notice-error', `Could not load wardrobe: ${err.message}`));
  }
}

function renderWardrobeCard(item) {
  const card = el('div', 'item-card wardrobe-card');

  const imgWrap = el('div', 'item-image-wrap');
  if (item.image_url) {
    const img = el('img', 'item-image');
    img.src = item.image_url;
    img.alt = item.title || '';
    imgWrap.appendChild(img);
  } else {
    imgWrap.appendChild(el('div', 'no-image-fallback', 'No image'));
  }
  card.appendChild(imgWrap);

  const del = el('button', 'delete-btn', '\u2715');
  del.title = 'Delete';
  del.addEventListener('click', async () => {
    try {
      await apiFetch(`/api/wardrobe/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
      loadWardrobe();
    } catch (err) {
      alert(`Could not delete item: ${err.message}`);
    }
  });
  card.appendChild(del);

  const body = el('div', 'item-body');
  body.appendChild(el('div', 'item-title', item.title || 'Wardrobe item'));
  if (item.description) body.appendChild(el('div', 'item-meta', item.description.slice(0, 160)));
  body.appendChild(el('div', 'item-meta', `${titleCase(item.gender)} \u00b7 ${titleCase(item.slot)} \u00b7 ${titleCase(item.color)} \u00b7 ${titleCase(item.occasion_group)}`));
  card.appendChild(body);

  return card;
}

document.getElementById('wd-submit').addEventListener('click', async () => {
  const title = document.getElementById('wd-title').value.trim();
  if (!title) {
    alert('Please enter an item name. The name is used to infer category and color.');
    return;
  }

  const formData = new FormData();
  formData.append('title', title);
  formData.append('description', document.getElementById('wd-description').value);
  formData.append('gender', document.getElementById('wd-gender').value);
  formData.append('slot', document.getElementById('wd-slot').value);
  formData.append('color', document.getElementById('wd-color').value);
  formData.append('occasion_group', document.getElementById('wd-occasion').value);
  const file = document.getElementById('wd-file').files[0];
  if (file) formData.append('image', file);

  const btn = document.getElementById('wd-submit');
  btn.disabled = true;
  btn.textContent = 'Saving...';
  try {
    await apiFetch('/api/wardrobe', { method: 'POST', body: formData });
    document.getElementById('wd-title').value = '';
    document.getElementById('wd-description').value = '';
    document.getElementById('wd-file').value = '';
    loadWardrobe();
  } catch (err) {
    alert(`Could not save item: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save to wardrobe';
  }
});

document.getElementById('wd-clear').addEventListener('click', async () => {
  if (!confirm('Clear your complete wardrobe? This cannot be undone.')) return;
  try {
    await apiFetch('/api/wardrobe', { method: 'DELETE' });
    loadWardrobe();
  } catch (err) {
    alert(`Could not clear wardrobe: ${err.message}`);
  }
});

// ---------------------------------------------------------------------
// CHATBOT
// ---------------------------------------------------------------------
const chatLog = document.getElementById('chat-log');
const chatHistory = [];

function appendChatMessage(role, content, thinking = false) {
  const wrap = el('div', `chat-msg ${role}${thinking ? ' thinking' : ''}`);
  const avatar = el('div', 'chat-avatar', role === 'user' ? '\ud83d\udc64' : '\ud83d\udc57');
  const bubble = el('div', 'chat-bubble', content);
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return bubble;
}

appendChatMessage('assistant', "Hi there! I am your AI Fashion Stylist. Ask me anything about outfits, body shape styling, live store items, or your saved wardrobe!");

document.getElementById('chat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  if (!message) return;

  appendChatMessage('user', message);
  chatHistory.push({ role: 'user', content: message });
  input.value = '';

  const thinkingBubble = appendChatMessage('assistant', 'AI Stylist is thinking & fetching recommendations...', true);

  try {
    const data = await apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, history: chatHistory }),
    });
    thinkingBubble.parentElement.classList.remove('thinking');
    thinkingBubble.textContent = data.reply;
    chatHistory.push({ role: 'assistant', content: data.reply });
  } catch (err) {
    thinkingBubble.parentElement.classList.remove('thinking');
    thinkingBubble.textContent = `Chatbot encountered an error: ${err.message}`;
  }
});

// initial load
loadWardrobe();
