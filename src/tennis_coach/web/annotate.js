// Stroke / contact-frame annotation for the evaluation set. Labels are saved to data/labels/annotations/.
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const STROKE_LABEL = { forehand: "正手", backhand_1h: "单手反手", backhand_2h: "双手反手", serve: "发球", smash: "高压", volley: "截击", other: "其他" };
const KEYS = { f: "forehand", b: "backhand_1h", d: "backhand_2h", s: "serve", m: "smash", v: "volley", o: "other" };
const KEY_OF = Object.fromEntries(Object.entries(KEYS).map(([k, v]) => [v, k.toUpperCase()]));
const ST = { key: null, fps: 30, nFrames: 0, events: [], sel: -1, dirty: false, timer: null };
const video = $("#video");

async function api(url, opt) {
  const r = await fetch(url, opt);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}
function annotator() {
  const a = $("#annotator").value.trim();
  try { localStorage.setItem("annotator", a); } catch (_) { /* storage may be unavailable */ }
  return a;
}
try { $("#annotator").value = localStorage.getItem("annotator") || ""; } catch (_) { /* ignore */ }
$("#annotator").onchange = () => { annotator(); refreshList(); if (ST.key) openVideo(ST.key); };

async function refreshList() {
  const vids = await api("/api/annot/videos"), me = annotator();
  $("#vlist").innerHTML = vids.map((v) => {
    const mine = v.annotators[me];
    const others = Object.entries(v.annotators).filter(([a]) => a !== me).map(([a, b]) => `${esc(a)} ${b.n_events}${b.status === "done" ? "✓" : ""}`).join("，");
    return `<li data-key="${esc(v.key)}" class="${v.key === ST.key ? "active" : ""}"><div class="nm">${esc(v.key)}</div>
      <div class="st">${mine ? `我：${mine.n_events} 条 · ${mine.status === "done" ? "已完成" : "标注中"}` : "我：未标注"}${others ? ` · 其他：${others}` : ""}</div></li>`;
  }).join("") || `<li class="empty">没有视频</li>`;
  document.querySelectorAll("#vlist li[data-key]").forEach((li) => li.onclick = () => openVideo(li.dataset.key));
}

async function openVideo(key) {
  if (ST.dirty) await save();
  const me = annotator();
  if (!me) { alert("请先在左上角填写标注人（字母/数字）"); $("#annotator").focus(); return; }
  const d = await api(`/api/annot/video?key=${encodeURIComponent(key)}`);
  ST.key = key; ST.fps = d.probe.fps || 30; ST.nFrames = d.probe.n_frames || Math.round(d.probe.duration * ST.fps);
  const mine = d.labels.annotations?.[me] || {};
  ST.events = (mine.events || []).map((e) => ({ ...e })); ST.sel = -1; ST.dirty = false;
  $("#m-view").value = mine.view || ""; $("#m-hand").value = mine.hand || ""; $("#m-level").value = mine.level || "";
  $("#m-status").value = mine.status || "in_progress"; $("#m-notes").value = mine.notes || "";
  $("#vtitle").textContent = key;
  const s = d.source || {};
  $("#vsource").innerHTML = s.url ? `来源：<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.title || s.url)}</a> · ${esc(s.uploader || "")} · ${esc(s.license || "")} · ${ST.fps.toFixed(2)} fps · ${ST.nFrames} 帧`
    : `${ST.fps.toFixed(2)} fps · ${ST.nFrames} 帧`;
  const st0 = d.probe.start_time || 0;
  if (d.probe.frame_exact === false) $("#vsource").innerHTML += `<br><b style="color:#c0392b">这个视频${d.probe.cfr ? "" : "不是恒定帧率"}${d.probe.cfr || Math.abs(st0) < 1e-3 ? "" : "、"}${Math.abs(st0) < 1e-3 ? "" : `首帧不在 0 秒（${st0.toFixed(3)} s）`}，帧号会对不准。请先运行 <code>uv run python scripts/prep_annotate.py &lt;视频&gt;</code> 转换后再标注。</b>`;
  video.src = `/api/annot/file?key=${encodeURIComponent(key)}`;
  $("#work").style.display = "";
  $("#strokes").innerHTML = Object.entries(STROKE_LABEL).map(([k, l]) => `<button class="s-${k}" data-s="${k}">${l}<kbd>${KEY_OF[k]}</kbd></button>`).join("");
  document.querySelectorAll("#strokes button").forEach((b) => b.onclick = () => mark(b.dataset.s));
  $("#savest").textContent = mine.saved ? `上次保存 ${mine.saved}` : "";
  renderEvents(); refreshList();
}

// ---- frames: frame n spans [n/fps, (n+1)/fps); seek to its middle so the decoder shows exactly that frame
const curFrame = () => Math.min(ST.nFrames - 1, Math.max(0, Math.floor(video.currentTime * ST.fps + 1e-3)));
function seekFrame(n) { video.pause(); video.currentTime = (Math.max(0, Math.min(ST.nFrames - 1, n)) + 0.5) / ST.fps; }
function updateDisp() {
  const f = curFrame();
  const near = ST.events.findIndex((e) => Math.abs(e.frame - f) <= 0);
  $("#fdisp").textContent = `帧 ${f} / ${ST.nFrames - 1} · ${video.currentTime.toFixed(3)} s${near >= 0 ? ` · ★ ${STROKE_LABEL[ST.events[near].stroke]}` : ""}`;
  $("#cur").style.left = `${(f / Math.max(1, ST.nFrames - 1)) * 100}%`;
}
function onFrame() { updateDisp(); if (!video.paused && "requestVideoFrameCallback" in video) video.requestVideoFrameCallback(onFrame); }
video.addEventListener("seeked", updateDisp);
video.addEventListener("loadedmetadata", () => { video.playbackRate = +$("#rate").value; updateDisp(); });
video.addEventListener("play", () => { $("#play").textContent = "⏸ 暂停"; if ("requestVideoFrameCallback" in video) video.requestVideoFrameCallback(onFrame); });
video.addEventListener("pause", () => { $("#play").textContent = "▶ 播放"; updateDisp(); });
video.addEventListener("timeupdate", () => { if (!("requestVideoFrameCallback" in video)) updateDisp(); });
$("#play").onclick = () => (video.paused ? video.play() : video.pause());
$("#rate").onchange = () => { video.playbackRate = +$("#rate").value; };
document.querySelectorAll("[data-step]").forEach((b) => b.onclick = () => seekFrame(curFrame() + +b.dataset.step));
$("#tl").onclick = (e) => {
  const r = $("#tl").getBoundingClientRect();
  seekFrame(Math.round(((e.clientX - r.left) / r.width) * (ST.nFrames - 1)));
};

// ---- events
function mark(stroke) {
  const f = curFrame();
  const i = ST.events.findIndex((e) => Math.abs(e.frame - f) <= 3);
  if (i >= 0) { ST.events[i].stroke = stroke; ST.sel = i; }
  else {
    ST.events.push({ frame: f, t: +(f / ST.fps).toFixed(4), stroke, unsure: false, note: "" });
    ST.events.sort((a, b) => a.frame - b.frame);
    ST.sel = ST.events.findIndex((e) => e.frame === f);
  }
  changed();
}
function renderEvents() {
  $("#evbody").innerHTML = ST.events.map((e, i) => `<tr data-i="${i}" class="${i === ST.sel ? "sel" : ""}">
    <td>${i + 1}</td><td><a href="#" class="go">${e.frame}</a></td><td>${(e.frame / ST.fps).toFixed(2)}s</td>
    <td><select class="st">${Object.entries(STROKE_LABEL).map(([k, l]) => `<option value="${k}" ${k === e.stroke ? "selected" : ""}>${l}</option>`).join("")}</select></td>
    <td><input type="checkbox" class="un" ${e.unsure ? "checked" : ""}></td>
    <td><input type="text" class="nt" value="${esc(e.note)}"></td>
    <td><button class="del" title="删除">✕</button></td></tr>`).join("") || `<tr><td colspan="7" class="hint">还没有标注。找到击球帧后按快捷键（F/B/D/S/M/V/O）。</td></tr>`;
  document.querySelectorAll("#evbody tr[data-i]").forEach((tr) => {
    const i = +tr.dataset.i;
    tr.querySelector(".go").onclick = (ev) => { ev.preventDefault(); ST.sel = i; seekFrame(ST.events[i].frame); renderEvents(); };
    tr.querySelector(".st").onchange = (ev) => { ST.events[i].stroke = ev.target.value; changed(); };
    tr.querySelector(".un").onchange = (ev) => { ST.events[i].unsure = ev.target.checked; changed(); };
    tr.querySelector(".nt").onchange = (ev) => { ST.events[i].note = ev.target.value; changed(); };
    tr.querySelector(".del").onclick = () => { ST.events.splice(i, 1); ST.sel = -1; changed(); };
  });
  const tl = $("#tl"); tl.querySelectorAll(".ev").forEach((n) => n.remove());
  const colors = { forehand: "#e4572e", backhand_1h: "#2e86de", backhand_2h: "#8e44ad", serve: "#16a085", smash: "#d4a017", volley: "#7f8c8d", other: "#555" };
  ST.events.forEach((e) => {
    const d = document.createElement("div"); d.className = "ev"; d.title = `${e.frame} ${STROKE_LABEL[e.stroke]}`;
    d.style.left = `${(e.frame / Math.max(1, ST.nFrames - 1)) * 100}%`; d.style.background = colors[e.stroke];
    tl.appendChild(d);
  });
  updateDisp();
}
function changed() { ST.dirty = true; renderEvents(); clearTimeout(ST.timer); ST.timer = setTimeout(save, 800); $("#savest").textContent = "未保存…"; }
["#m-view", "#m-hand", "#m-level", "#m-status", "#m-notes"].forEach((s) => $(s).onchange = changed);

async function save() {
  if (!ST.key) return;
  clearTimeout(ST.timer);
  const body = { annotator: annotator(), status: $("#m-status").value, view: $("#m-view").value || null, hand: $("#m-hand").value || null,
    level: $("#m-level").value || null, notes: $("#m-notes").value, events: ST.events };
  try {
    const r = await api(`/api/annot/labels?key=${encodeURIComponent(ST.key)}`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    ST.dirty = false; $("#savest").textContent = `已保存 ${new Date().toLocaleTimeString()}（${r.n_events} 条）`; refreshList();
  } catch (e) { $("#savest").textContent = `保存失败：${e.message}`; }
}
window.addEventListener("beforeunload", (e) => { if (ST.dirty) { save(); e.preventDefault(); } });

document.addEventListener("keydown", (e) => {
  if (!ST.key || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const k = e.key.toLowerCase();
  if (k === "arrowleft" || k === "arrowright") { e.preventDefault(); seekFrame(curFrame() + (k === "arrowleft" ? -1 : 1) * (e.shiftKey ? 10 : 1)); }
  else if (k === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  else if (KEYS[k] && !e.ctrlKey && !e.metaKey) { e.preventDefault(); mark(KEYS[k]); }
  else if ((k === "delete" || k === "backspace") && ST.sel >= 0) { e.preventDefault(); ST.events.splice(ST.sel, 1); ST.sel = -1; changed(); }
  else if (k === "u" && ST.sel >= 0) { ST.events[ST.sel].unsure = !ST.events[ST.sel].unsure; changed(); }
  else if (k === "s" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); save(); }
});
refreshList();
