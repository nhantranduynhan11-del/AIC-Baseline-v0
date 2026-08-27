/* UI giai doan 1 - grid ket qua, panel chi tiet, gio nop bai.
 *
 * Khong framework, khong build step: sua file la F5 thay ngay. Doi lai phai tu
 * quan ly trang thai, nen moi thu goi vao mot object `S` duy nhat, va moi thay
 * doi deu di qua render() thay vi vá DOM rai rac.
 *
 * Ba trang thai can phan biet:
 *   S.hits    - ket qua search hien tai (theo thu tu rank)
 *   S.cursor  - vi tri con tro trong grid (dieu khien bang phim mui ten)
 *   S.picked  - gio nop bai, giu THU TU CHON (quan trong cho TRAKE)
 */

const MAX_PICK = 100;

const S = {
  hits: [],
  cursor: -1,
  picked: [],          // [{idx, video_id, frame_idx, pts_time}]
  detailIdx: null,     // keyframe dang xem - co the KHONG nam trong S.hits
  neighbors: [],       // dai lan can cua S.neighborsFor
  neighborsFor: null,  // dai tren thuoc ve keyframe nao
  busy: false,
  odFilters: [],       // [{label: "dog", count: 1}]
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };

/* ---------- API ---------- */

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* khong phai JSON */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

const getJSON = async (path) => (await api(path)).json();

const postJSON = async (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ---------- trang thai server ---------- */

async function checkHealth() {
  const s = $("status");
  try {
    const h = await getJSON("/health");
    if (!h.ready) { s.textContent = `chưa sẵn sàng — ${h.error}`; s.className = "err"; return; }
    s.className = "";
    s.textContent = `${h.n_keyframes.toLocaleString()} keyframe · RRF k=${h.rrf_k} · OCR ${h.ocr ? "có" : "chưa có"}`;
    if (!h.ocr) $("ocr").placeholder = "chưa có DB OCR";
    $("ocr").disabled = !h.ocr;
  } catch (e) {
    s.textContent = `không gọi được API — ${e.message}`;
    s.className = "err";
  }
}

/* ---------- search ---------- */

async function doSearch(imageIdx = null) {
  if (S.busy) return;
  const query = $("query").value.trim();
  if (imageIdx === null && !query) { $("query").focus(); return; }

  const body = {
    top_k_per_model: Number($("k").value) || undefined,
    top_n: Number($("topn").value) || 100,
  };
  if (imageIdx !== null) body.image_idx = imageIdx; else body.query = query;

  const ocr = $("ocr").value.trim();
  if (ocr) {
    body.ocr = ocr;
    body.ocr_phrase = $("ocr-phrase").checked;
    body.ocr_min_confidence = Number($("ocr-conf").value);
  }

  if (S.odFilters && S.odFilters.length > 0) {
    body.od_filters = S.odFilters;
  }

  S.busy = true;
  const s = $("status");
  const before = s.textContent;
  s.textContent = "đang tìm…";
  const t0 = performance.now();

  try {
    const data = await (await postJSON("/search", body)).json();
    S.hits = data.hits;
    S.cursor = data.hits.length ? 0 : -1;
    const ms = Math.round(performance.now() - t0);
    let note = `${data.n_hits} kết quả · ${ms}ms`;
    if (data.translated_query && data.translated_query !== query) {
      note += ` · Dịch: "${data.translated_query}"`;
    }
    if (data.ocr_filter) {
      note += ` · OCR khớp ${data.ocr_filter.n_frames} keyframe`;
      if (data.ocr_filter.n_unknown) note += ` (${data.ocr_filter.n_unknown} lệch manifest)`;
    }
    s.textContent = note;
    s.className = "";
    renderGrid();
    if (S.cursor >= 0) showDetail(S.hits[0].idx);
  } catch (e) {
    s.textContent = `lỗi: ${e.message}`;
    s.className = "err";
    if (before) setTimeout(() => { if (s.className === "err") checkHealth(); }, 4000);
  } finally {
    S.busy = false;
  }
}

/* ---------- grid ---------- */

function renderGrid() {
  const box = $("results");
  box.textContent = "";
  if (!S.hits.length) {
    const e = el("div", "empty");
    e.textContent = "Không có kết quả nào.";
    box.append(e);
    return;
  }

  const grid = el("div");
  grid.id = "grid";
  const pickedIdx = new Set(S.picked.map((p) => p.idx));

  S.hits.forEach((hit, i) => {
    const card = el("div", "card");
    card.dataset.pos = i;
    if (i === S.cursor) card.classList.add("cursor");
    if (pickedIdx.has(hit.idx)) card.classList.add("picked");

    let mediaElement;
    if (hit.sequence_idxs && hit.sequence_idxs.length > 1) {
      card.classList.add("seq-card");
      // Cố gắng mở rộng card chiếm nhiều cột lưới, tối đa 3 cột để không phá grid
      const spanCols = Math.min(hit.sequence_idxs.length, 3);
      card.style.gridColumn = `span ${spanCols}`;

      mediaElement = el("div", "seq-images");
      hit.sequence_idxs.forEach((seqIdx) => {
        const simg = el("img");
        simg.src = `/thumb/${seqIdx}`;
        simg.loading = "lazy";
        simg.alt = `frame ${seqIdx}`;
        simg.title = `frame ${seqIdx} (nhấp để xem chi tiết frame này)`;
        if (seqIdx === S.detailIdx) {
          simg.classList.add("active-seq-frame");
        }
        simg.addEventListener("click", (ev) => {
          ev.stopPropagation();
          S.cursor = i;
          // Update active sub-frame highlight without full grid re-render to avoid scrollbar jumps
          document.querySelectorAll(".seq-images img.active-seq-frame").forEach((el) => el.classList.remove("active-seq-frame"));
          simg.classList.add("active-seq-frame");
          document.querySelectorAll(".card.cursor").forEach((c) => c.classList.remove("cursor"));
          card.classList.add("cursor");
          showDetail(seqIdx);
        });
        mediaElement.append(simg);
      });
    } else {
      mediaElement = el("img");
      mediaElement.src = `/thumb/${hit.idx}`;
      mediaElement.loading = "lazy";
      mediaElement.alt = `${hit.video_id} frame ${hit.frame_idx}`;
    }

    const rank = el("div", "rank");
    rank.textContent = `#${hit.rank}`;
    const tick = el("div", "tick");
    tick.textContent = "✓";

    const meta = el("div", "meta");
    const left = el("b");
    left.textContent = hit.video_id;
    const right = el("span");
    right.textContent = hit.frame_idx;
    meta.append(left, right);

    card.append(mediaElement, rank, tick, meta);
    card.addEventListener("click", () => { S.cursor = i; renderGrid(); showDetail(hit.idx); });
    card.addEventListener("dblclick", () => togglePick(hit));
    grid.append(card);
  });

  box.append(grid);
  scrollCursorIntoView();
}

function scrollCursorIntoView() {
  const card = document.querySelector(".card.cursor");
  if (card) card.scrollIntoView({ block: "nearest" });
}

function columnCount() {
  const grid = $("grid");
  if (!grid) return 1;
  return getComputedStyle(grid).gridTemplateColumns.split(" ").length;
}

function moveCursor(delta) {
  if (!S.hits.length) return;
  const next = Math.max(0, Math.min(S.hits.length - 1, S.cursor + delta));
  if (next === S.cursor) return;
  S.cursor = next;
  renderGrid();
  showDetail(S.hits[next].idx);
}

/* ---------- panel chi tiet ---------- */

async function showDetail(idx) {
  S.detailIdx = idx;
  S.neighbors = [];
  S.neighborsFor = null;
  const box = $("detail");
  box.textContent = "";

  const hit = S.hits.find((h) => h.idx === idx);
  const parentHit = hit || S.hits.find((h) => h.sequence_idxs && h.sequence_idxs.includes(idx));

  const img = el("img", "big");
  img.src = `/keyframe/${idx}`;
  box.append(img);

  const title = el("h3");
  title.textContent = hit ? `${hit.video_id} • frame ${hit.frame_idx}` : `Đang tải chi tiết...`;
  box.append(title);

  let currentItem = hit;
  if (!currentItem) {
    try {
      const entry = await fetchEntry(idx);
      if (S.detailIdx !== idx) return; // stale request
      currentItem = {
        idx: entry.idx,
        video_id: entry.video_id,
        frame_idx: entry.frame_idx,
        pts_time: entry.pts_time,
        score: parentHit ? parentHit.score : 0,
        rank: parentHit ? parentHit.rank : 0,
        ranks: parentHit ? parentHit.ranks : {},
      };
      title.textContent = `${entry.video_id} • frame ${entry.frame_idx}`;
    } catch (e) {
      title.textContent = `idx ${idx}`;
    }
  }

  if (currentItem) {
    const kv = el("div", "kv");
    const ranks = (parentHit && parentHit.ranks) ? Object.entries(parentHit.ranks).map(([m, r]) => `${m}#${r}`).join("  ") : "";
    const scoreStr = parentHit ? ` • RRF ${(parentHit.score || 0).toFixed(6)}` : "";
    const rankStr = ranks ? ` • ${ranks}` : "";
    kv.textContent = `t=${(currentItem.pts_time || 0).toFixed(2)}s${scoreStr}${rankStr}`;
    box.append(kv);
  }

  const actions = el("div", "row");
  actions.style.marginTop = "8px";

  const pickBtn = el("button");
  const inBasket = S.picked.some((p) => p.idx === idx);
  pickBtn.textContent = inBasket ? "Bỏ khỏi giỏ" : "Thêm vào giỏ";
  pickBtn.addEventListener("click", async () => {
    const itemToPick = currentItem ?? (await fetchEntry(idx));
    togglePick(itemToPick);
  });

  const simBtn = el("button");
  simBtn.textContent = "Tìm ảnh giống";
  simBtn.title = "Dùng chính keyframe này làm query (Video KIS)";
  simBtn.addEventListener("click", () => doSearch(idx));

  actions.append(pickBtn, simBtn);
  box.append(actions);

  box.append(el("div", "sep"));
  const nb = el("h3");
  nb.textContent = "Keyframe lân cận";
  const nbHint = el("div", "kv");
  nbHint.textContent = "cùng video, theo thời gian — , và . để trượt";
  const strip = el("div");
  strip.id = "neighbors";
  box.append(nb, nbHint, strip);

  box.append(el("div", "sep"));
  const oh = el("h3");
  oh.textContent = "Chữ trên màn hình (OCR)";
  const obox = el("div");
  obox.id = "ocrbox";
  box.append(oh, obox);

  loadNeighbors(idx);
  loadOcr(idx);
}

async function fetchEntry(idx) {
  /* Keyframe lan can khong nam trong ket qua search nen thieu metadata.
     /neighbors tra ve du video_id + frame_idx de bo vao gio nop bai. */
  const data = await getJSON(`/neighbors/${idx}?w=0`);
  const item = data.items.find((i) => i.idx === idx) ?? data.items[0];
  return { idx, video_id: data.video_id, frame_idx: item.frame_idx, pts_time: item.pts_time };
}

async function loadNeighbors(idx) {
  const strip = $("neighbors");
  try {
    const data = await getJSON(`/neighbors/${idx}?w=7`);
    // Bam mui ten nhanh se ban nhieu request; phan hoi den khong dung thu tu.
    // Bo qua phan hoi cua keyframe khong con dang xem.
    if (idx !== S.detailIdx) return;
    S.neighbors = data.items;
    S.neighborsFor = idx;
    strip.textContent = "";
    for (const item of data.items) {
      const t = el("img");
      t.src = `/thumb/${item.idx}`;
      t.loading = "lazy";
      t.title = `frame ${item.frame_idx} · t=${item.pts_time.toFixed(2)}s`;
      if (item.is_current) t.classList.add("here");
      t.addEventListener("click", () => showDetail(item.idx));
      strip.append(t);
    }
  } catch (e) {
    strip.textContent = e.message;
  }
}

function stepNeighbor(delta) {
  // Dai lan can phai dung la cua keyframe dang xem: bam phim nhanh co the lam
  // dai con la cua keyframe truoc do, va truot tiep se nhay sang VIDEO KHAC.
  if (S.neighborsFor !== S.detailIdx || !S.neighbors.length) return;
  const at = S.neighbors.findIndex((i) => i.idx === S.detailIdx);
  if (at < 0) return;        // findIndex tra -1; S.neighbors[-1 + 1] la phan tu DAU dai
  const next = S.neighbors[at + delta];
  if (next) showDetail(next.idx);
}

async function loadOcr(idx) {
  const box = $("ocrbox");
  try {
    const data = await getJSON(`/ocr/${idx}`);
    if (idx !== S.detailIdx) return;
    box.textContent = "";
    if (!data.available) { box.className = "kv"; box.textContent = "chưa chạy OCR"; return; }
    if (!data.texts.length) { box.className = "kv"; box.textContent = "không đọc được chữ nào"; return; }
    box.className = "";
    for (const t of data.texts) {
      const line = el("div", "ocrline");
      const conf = el("span");
      conf.textContent = ` ${t.confidence.toFixed(2)}`;
      line.append(document.createTextNode(t.text), conf);
      box.append(line);
    }
  } catch (e) {
    box.className = "kv";
    box.textContent = e.message;
  }
}

/* ---------- gio nop bai ---------- */

function togglePick(item) {
  if (!item) return;
  const at = S.picked.findIndex((p) => p.idx === item.idx);
  if (at >= 0) {
    S.picked.splice(at, 1);
  } else {
    if (S.picked.length >= MAX_PICK) {
      $("export-note").textContent = `Đã đủ ${MAX_PICK} — giới hạn nộp bài.`;
      return;
    }
    S.picked.push({
      idx: item.idx, video_id: item.video_id,
      frame_idx: item.frame_idx, pts_time: item.pts_time,
    });
  }
  renderBasket();
  renderGrid();
  if (S.detailIdx !== null) showDetail(S.detailIdx);
}

function renderBasket() {
  const box = $("basket");
  box.textContent = "";
  for (const p of S.picked) {
    const chip = el("div", "chip");
    chip.textContent = `${p.video_id},${p.frame_idx}`;
    chip.title = "Bấm để bỏ khỏi giỏ";
    chip.addEventListener("click", () => togglePick(p));
    box.append(chip);
  }
  const count = $("count");
  count.textContent = `${S.picked.length} / ${MAX_PICK}`;
  count.classList.toggle("full", S.picked.length >= MAX_PICK);
  $("btn-export").disabled = S.picked.length === 0;
}

/* ---------- export ---------- */

function buildExportBody() {
  const task = $("task").value;
  if (task === "trake") {
    /* Mot dong = mot video. Gom theo video theo THU TU CHON, frame sap tang dan
       de dung yeu cau "theo thu tu thoi gian cua cac su kien". */
    const groups = new Map();
    for (const p of S.picked) {
      if (!groups.has(p.video_id)) groups.set(p.video_id, []);
      groups.get(p.video_id).push(p.frame_idx);
    }
    return {
      task,
      items: [...groups].map(([video_id, frames]) => ({
        video_id, frames: frames.slice().sort((a, b) => a - b),
      })),
    };
  }
  const body = {
    task,
    items: S.picked.map((p) => ({ video_id: p.video_id, frame_idx: p.frame_idx })),
  };
  if (task === "qa") body.answer = $("answer").value.trim();
  return body;
}

async function doExport() {
  const note = $("export-note");
  const body = buildExportBody();
  if (body.task === "qa" && !body.answer) { note.textContent = "Q&A cần nhập answer."; return; }

  try {
    const res = await postJSON("/export", body);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = el("a");
    a.href = url;
    a.download = `submission_${body.task}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    const rows = body.task === "trake" ? body.items.length : S.picked.length;
    note.textContent = `đã xuất ${rows} dòng (${body.task})`;
  } catch (e) {
    note.textContent = `lỗi: ${e.message}`;
  }
}

/* ---------- phim tat ---------- */

function typing(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement;
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { document.activeElement?.blur(); return; }

  if (ev.ctrlKey && ev.key.toLowerCase() === "e") { ev.preventDefault(); doExport(); return; }

  if (typing(ev.target)) {
    if (ev.key === "Enter" && !ev.shiftKey) { 
      ev.preventDefault(); 
      ev.target.blur(); 
      doSearch(); 
    }
    return;
  }

  switch (ev.key) {
    case "/": ev.preventDefault(); $("query").focus(); $("query").select(); break;
    case "ArrowRight": ev.preventDefault(); moveCursor(1); break;
    case "ArrowLeft": ev.preventDefault(); moveCursor(-1); break;
    case "ArrowDown": ev.preventDefault(); moveCursor(columnCount()); break;
    case "ArrowUp": ev.preventDefault(); moveCursor(-columnCount()); break;
    case " ":
      ev.preventDefault();
      if (S.detailIdx !== null) {
        const hit = S.hits.find((h) => h.idx === S.detailIdx);
        hit ? togglePick(hit) : fetchEntry(S.detailIdx).then(togglePick);
      }
      break;
    case ",": ev.preventDefault(); stepNeighbor(-1); break;
    case ".": ev.preventDefault(); stepNeighbor(1); break;
    case "Enter": if (S.detailIdx !== null) doSearch(S.detailIdx); break;
    default: break;
  }
});

/* ---------- khoi tao ---------- */

$("btn-search").addEventListener("click", () => doSearch());
$("btn-export").addEventListener("click", doExport);
$("btn-clear").addEventListener("click", () => {
  S.picked = [];
  $("export-note").textContent = "";
  renderBasket();
  renderGrid();
});
$("task").addEventListener("change", (ev) => {
  $("answer").style.display = ev.target.value === "qa" ? "" : "none";
  $("export-note").textContent = ev.target.value === "trake"
    ? "TRAKE: mỗi video thành một dòng, frame sắp theo thời gian."
    : "";
});

/* ---------- OD filter ---------- */
function renderODTags() {
  const box = $("od-tags");
  box.textContent = "";
  S.odFilters.forEach((f, i) => {
    const chip = el("div", "chip");
    const colorTxt = (f.color && f.color !== "none") ? `[${f.color}] ` : "";
    chip.textContent = `${colorTxt}${f.label}: ${f.count}`;
    chip.title = "Bấm để xóa";
    chip.addEventListener("click", () => {
      S.odFilters.splice(i, 1);
      renderODTags();
    });
    box.append(chip);
  });
}

$("btn-add-od").addEventListener("click", () => {
  const lbl = $("od-label").value.trim();
  const color = $("od-color").value;
  const cnt = parseInt($("od-count").value, 10);
  if (lbl && cnt > 0) {
    S.odFilters.push({ label: lbl, count: cnt, color: color });
    $("od-label").value = "";
    $("od-color").value = "none";
    renderODTags();
  }
});

renderBasket();
checkHealth();

/* ---------- Resizable Splitter & Auto-expand Search ---------- */

function initSplitter() {
  const splitter = $("splitter");
  const detail = $("detail");
  if (!splitter || !detail) return;

  const savedWidth = localStorage.getItem("aic_detail_width");
  if (savedWidth) {
    const w = parseInt(savedWidth, 10);
    if (w >= 240 && w <= window.innerWidth * 0.75) {
      detail.style.width = w + "px";
    }
  }

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  splitter.addEventListener("mousedown", (e) => {
    isDragging = true;
    startX = e.clientX;
    startWidth = detail.getBoundingClientRect().width;
    splitter.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const deltaX = startX - e.clientX;
    const newWidth = Math.max(240, Math.min(Math.round(startWidth + deltaX), Math.round(window.innerWidth * 0.75)));
    detail.style.width = newWidth + "px";
  });

  window.addEventListener("mouseup", () => {
    if (!isDragging) return;
    isDragging = false;
    splitter.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    const finalWidth = parseInt(detail.style.width, 10);
    if (finalWidth) {
      localStorage.setItem("aic_detail_width", finalWidth);
    }
  });
}

function initAutoExpandQuery() {
  const queryInput = $("query");
  if (!queryInput) return;
  function autoExpand() {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.max(60, Math.min(queryInput.scrollHeight, 240)) + "px";
  }
  queryInput.addEventListener("input", autoExpand);
  autoExpand();
}

initSplitter();
initAutoExpandQuery();

/* ---------- Header Collapse / Expand ---------- */

function initHeaderToggle() {
  const btn = $("btn-toggle-header");
  const header = document.querySelector("header");
  if (!btn || !header) return;

  function setCollapsed(collapsed) {
    header.classList.toggle("collapsed", collapsed);
    btn.textContent = collapsed ? "▼ Mở rộng" : "▲ Thu gọn";
    localStorage.setItem("aic_header_collapsed", collapsed ? "1" : "0");
  }

  // Restore saved state
  if (localStorage.getItem("aic_header_collapsed") === "1") {
    setCollapsed(true);
  }

  btn.addEventListener("click", () => {
    const isNow = !header.classList.contains("collapsed");
    setCollapsed(isNow);
  });

  // Hotkey Ctrl+H to toggle header
  document.addEventListener("keydown", (ev) => {
    if (ev.ctrlKey && ev.key.toLowerCase() === "h") {
      ev.preventDefault();
      const isNow = !header.classList.contains("collapsed");
      setCollapsed(isNow);
    }
  });
}

initHeaderToggle();
