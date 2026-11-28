"use strict";
// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (_) {} throw new Error(m); }
  return r.json();
}
const jpatch = (path, body) => api(path, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
let META = null;
const ACTION = (a) => (META?.actions?.[a]) || a;
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const PHASE_COLORS = { ready: "#8aa0b4", backswing: "#3b7dd8", forward: "#d9822b", contact: "#c0392b", follow: "#2f7d4a", toss: "#7d5bd8", drop: "#b0478f", finish: "#555" };
const EDGES = [[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];
const LEFT = new Set([1,3,5,7,9,11,13,15]);

// ------------------------------------------------------------------ tabs
$$("nav button").forEach((b) => b.onclick = () => {
  $$("nav button").forEach((x) => x.classList.toggle("active", x === b));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "library") refreshLibrary();
});

// ------------------------------------------------------------------ uploads
function formMeta(form) {
  const fd = new FormData(form); const meta = {};
  for (const [k, v] of fd.entries()) if (k !== "file" && v !== "") meta[k] = v;
  if (meta.time_scale && meta.time_scale !== "unknown") meta.time_scale = parseFloat(meta.time_scale);
  return meta;
}
async function upload(form, kind) {
  const fd = new FormData(); const file = form.file.files[0];
  if (!file) return;
  fd.append("file", file); fd.append("kind", kind); fd.append("meta", JSON.stringify(formMeta(form)));
  const btn = $("button[type=submit]", form); btn.disabled = true; btn.textContent = "上传中…";
  try { const r = await api("/api/videos", { method: "POST", body: fd }); form.reset(); return r.id; }
  catch (e) { alert("上传失败：" + e.message); }
  finally { btn.disabled = false; btn.textContent = kind === "query" ? "上传并分析" : "上传入库"; }
}
$("#form-query").onsubmit = async (e) => { e.preventDefault(); const id = await upload(e.target, "query"); if (id) { await refreshQueries(); openAnalysis(id); } };
$("#form-ref").onsubmit = async (e) => { e.preventDefault(); const id = await upload(e.target, "reference"); if (id) refreshLibrary(); };

// ------------------------------------------------------------------ video lists + polling
function statusBadge(v) {
  if (v.status === "done") { const q = v.summary?.quality; return `<span class="badge ${q === "bad" ? "bad" : q === "warn" ? "warn" : "good"}">完成</span>`; }
  if (v.status === "failed") return `<span class="badge bad">失败</span>`;
  return `<span class="badge info">${esc(v.stage || v.status)} ${pct(v.progress)}</span>`;
}
function videoLi(v, sel) {
  const s = v.summary || {};
  const acts = (s.actions || []).filter((a) => a !== "unknown").map(ACTION);
  return `<li data-id="${v.id}" class="${sel ? "sel" : ""}">${statusBadge(v)} ${esc(v.filename)}
    <div class="meta">${new Date(v.created_at * 1000).toLocaleString()} · ${s.n_swings ?? "?"} 次挥拍 ${acts.length ? "· " + acts.join("/") : ""} ${s.view ? "· " + esc(s.view) : ""}</div>
    ${v.status === "failed" ? `<div class="meta" style="color:var(--bad)">${esc(v.error)}</div>` : ""}
    ${["queued", "running"].includes(v.status) ? `<div class="progress"><div style="width:${pct(v.progress)}"></div></div>` : ""}</li>`;
}
let currentId = null, pollTimer = null;
async function refreshQueries() {
  const vs = await api("/api/videos?kind=query");
  $("#query-list").innerHTML = vs.map((v) => videoLi(v, v.id === currentId)).join("") || `<li class="empty">暂无</li>`;
  $$("#query-list li[data-id]").forEach((li) => li.onclick = async () => {
    await openAnalysis(li.dataset.id);
    // single-column layout: the result is below the upload form, bring it into view
    if (matchMedia("(max-width: 860px)").matches) $("#analysis-view").scrollIntoView({ behavior: "smooth" });
  });
  return vs;
}
async function poll() {
  clearTimeout(pollTimer);
  try {
    const [q, r] = await Promise.all([refreshQueries(), api("/api/videos?kind=reference")]);
    renderRefVideos(r);
    const busy = [...q, ...r].some((v) => ["queued", "running"].includes(v.status));
    if (currentId) {
      const cur = q.find((v) => v.id === currentId);
      if (cur && cur.status === "done" && AN?.video?.status !== "done") openAnalysis(currentId);
      else if (cur && ["queued", "running", "failed"].includes(cur.status)) $(".status-line") && ($(".status-line").innerHTML = statusBadge(cur) + (cur.error ? esc(cur.error) : ""));
    }
    if ($("#tab-library").classList.contains("active") && busy) refreshRefTable();
    pollTimer = setTimeout(poll, busy ? 1500 : 6000);
  } catch (e) { pollTimer = setTimeout(poll, 5000); }
}

// ------------------------------------------------------------------ analysis view
let AN = null;   // state of the open analysis
async function openAnalysis(id) {
  currentId = id;
  $$("#query-list li").forEach((li) => li.classList.toggle("sel", li.dataset.id === id));
  const v = await api(`/api/videos/${id}`);
  const root = $("#analysis-view");
  root.innerHTML = ""; root.appendChild($("#tpl-analysis").content.cloneNode(true));
  $(".an-title", root).textContent = v.filename;
  AN = { video: v, root };
  $(".btn-delete", root).onclick = async () => {
    if (!confirm("永久删除该视频、转码文件、姿态缓存和全部分析结果？此操作不可撤销。")) return;
    await api(`/api/videos/${id}`, { method: "DELETE" }); currentId = null; root.innerHTML = `<p class="empty">已删除。</p>`; refreshQueries();
  };
  $(".btn-reanalyse", root).onclick = async () => { await api(`/api/videos/${id}/reanalyse`, { method: "POST" }); poll(); };
  $(".btn-annotate", root).onclick = (e) => annotateVideo(id, e.currentTarget);
  if (v.status !== "done" || !v.analysis) {
    $(".status-line", root).innerHTML = statusBadge(v) + (v.error ? " " + esc(v.error) : " 处理中，完成后自动显示…");
    $(".an-body", root).style.display = "none"; poll(); return;
  }
  const a = v.analysis;
  $(".status-line", root).innerHTML = `${a.n_swings} 次挥拍 · 姿态模型 ${esc(a.pose_model)}（${a.pose_cache_hit ? "命中缓存" : `${esc(a.pose_meta?.device || "")} ${fmt(a.pose_meta?.ms_per_frame, 0)} ms/帧`}）`
    + ` · 分析版本 ${esc(a.analysis_version)} · 参考库 ${a.n_refs_available} 条 · 时间尺度 ${a.time_known ? fmt(a.time_scale, 3) : "未知（慢动作？）"}`;
  AN.pose = await api(`/api/videos/${id}/pose`).catch(() => null);
  const vu = $(".v-user", root); vu.src = `/api/videos/${id}/file`;
  AN.vu = vu; AN.vrEl = $(".v-ref", root); AN.vr = AN.vrEl; AN.cu = $(".c-user", root); AN.cr = $(".c-ref", root);
  const tabs = $(".swing-tabs", root);
  if (!a.swings.length) { tabs.innerHTML = `<span class="empty">未检测到挥拍。</span>`; renderQuality(a, null); return; }
  tabs.innerHTML = a.swings.map((s) => `<button data-i="${s.index}">#${s.index + 1} ${esc(ACTION(s.action))} @${fmt(s.window.tv_peak, 1)}s</button>`).join("");
  $$("button", tabs).forEach((b) => b.onclick = () => selectSwing(+b.dataset.i));
  setupPlayback();
  selectSwing(a.primary_swing ?? a.swings[0].index);
}

// annotation copy (constant frame rate, first frame at 0 s) of this video, then open it on the annotation page,
// where the system's swings are shown as hints on the timeline
async function annotateVideo(id, btn) {
  const old = btn.textContent; btn.disabled = true;
  try {
    let j = await api(`/api/annot/from_video/${id}`, { method: "POST" });
    while (j.state === "running") {
      btn.textContent = `准备标注副本：${j.stage}…`;
      await new Promise((r) => setTimeout(r, 800));
      j = await api(`/api/annot/jobs/${j.id}`);
    }
    if (j.state !== "done") throw new Error(j.error || "失败");
    location.href = `annotate.html?key=${encodeURIComponent(j.key)}`;
  } catch (e) { alert(`无法打开标注：${e.message}`); btn.textContent = old; btn.disabled = false; }
}

async function selectSwing(i) {
  const a = AN.video.analysis; const sw = a.swings.find((s) => s.index === i);
  AN.sw = sw; AN.ref = null;
  $$(".swing-tabs button", AN.root).forEach((b) => b.classList.toggle("active", +b.dataset.i === i));
  // the right-hand player shows the most similar professional swing when there is one
  const pn = sw.pro?.nearest;
  const cand = pn ? { ref_id: pn.ref_id, cost: pn.cost, path: pn.path, pro: true } : sw.match?.candidates?.[0];
  if (cand) {
    const rd = await api(`/api/references/${cand.ref_id}`);
    AN.ref = { cand, detail: rd, sw: rd.swing, ts: rd.time_scale || 1, pose: await api(`/api/videos/${rd.video.id}/pose`).catch(() => null) };
    useRefSource(rd, AN.ref.pose);
    $(".ref-title", AN.root).innerHTML = `参考：${esc(rd.video.meta?.player || rd.video.filename)} · 挥拍 #${rd.ref.swing_index + 1} · `
      + (cand.pro ? `<span class="badge info">职业示范 · 最接近你的职业挥拍</span> ` : "")
      + (rd.video.has_video === false ? `<span class="badge info">参考包：只有骨架，没有原视频</span> ` : "")
      + `<span class="badge ${rd.ref.review_status === "coach_reviewed" ? "good" : "warn"}">${rd.ref.review_status === "coach_reviewed" ? "教练审核" : "未经教练审核"}</span> DTW 距离 ${fmt(cand.cost)}`
      + `<br>库内标注：<b>${esc(ACTION(rd.ref.action))}</b> <a href="#" class="ref-wrong">标注不对？排除这条并重新分析</a>`;
    $(".ref-wrong", AN.root).onclick = async (e) => {
      e.preventDefault();
      if (!confirm(`把这条参考（${rd.video.meta?.player || ""} #${rd.ref.swing_index + 1}，库内标为「${ACTION(rd.ref.action)}」）从参考库停用？\n之后的匹配不会再用它，可在“参考库”页重新启用。`)) return;
      await jpatch(`/api/references/${rd.ref.id}`, { enabled: false });
      await api(`/api/videos/${AN.video.id}/reanalyse`, { method: "POST" }); poll();
    };
  } else {
    useRefSource(null, null);
    $(".ref-title", AN.root).textContent = "参考：无（" + (sw.match?.message || "未匹配") + "）";
  }
  AN.vu.currentTime = sw.analysed_range_v ? Math.max(0, sw.analysed_range_v[0] - 0.3) : sw.window.tv0;
  buildMapping(); renderTimeline(); renderPro(); renderRecog(); renderCorrect(); renderFeedback(); renderMetrics(); renderCurves(); renderQuality(a, sw);
  drawFrame();
}

// Reference-pack entries have keypoints but no video: the right-hand player is then a "virtual video"
// (an object with currentTime / videoWidth ...) and the skeleton is drawn on an empty background, zoomed to
// the player's area during this swing.
function useRefSource(rd, pose) {
  const el = AN.vrEl;
  el.classList.remove("novideo");
  if (!rd) { el.removeAttribute("src"); el.load(); AN.vr = el; return; }
  if (rd.video.has_video !== false) { AN.vr = el; el.src = `/api/videos/${rd.video.id}/file`; return; }
  el.removeAttribute("src"); el.load(); el.classList.add("novideo");
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  if (pose) {
    const w = rd.swing?.window || {}, t0 = (w.tv0 ?? 0) - 0.5, t1 = (w.tv1 ?? 1e9) + 0.5;
    pose.t.forEach((t, i) => {
      if (t < t0 || t > t1) return;
      pose.kpts[i].forEach((k, j) => { if (pose.scores[i][j] >= 0.3 && k[0] >= 0) { x0 = Math.min(x0, k[0]); x1 = Math.max(x1, k[0]); y0 = Math.min(y0, k[1]); y1 = Math.max(y1, k[1]); } });
    });
  }
  if (!(x1 > x0)) { x0 = 0; y0 = 0; x1 = pose?.width || 1280; y1 = pose?.height || 720; }
  const m = 0.12 * Math.max(x1 - x0, y1 - y0);
  const crop = { x0: x0 - m, y0: y0 - m, w: x1 - x0 + 2 * m, h: y1 - y0 + 2 * m };
  AN.vr = { virtual: true, currentTime: 0, readyState: 4, videoWidth: crop.w, videoHeight: crop.h, crop };
}

// ---- time mapping user video time -> reference video time
function buildMapping() {
  AN.map = null;
  if (!AN.ref) return;
  const su = AN.sw.curves, sr = AN.ref.sw.curves;
  const path = AN.ref.cand.path || [];
  const pts = path.map(([i, j]) => [su.tv[Math.min(i, su.tv.length - 1)], sr.tv[Math.min(j, sr.tv.length - 1)]]).filter((p) => p[0] !== undefined && p[1] !== undefined);
  // collapse duplicates of user time (DTW may map one user frame to several ref frames)
  const m = new Map(); pts.forEach(([u, r]) => { if (!m.has(u)) m.set(u, []); m.get(u).push(r); });
  AN.map = [...m.entries()].map(([u, rs]) => [u, rs.reduce((x, y) => x + y) / rs.length]).sort((x, y) => x[0] - y[0]);
}
function refTimeFor(tu) {
  if (!AN.ref) return null;
  const cu = AN.sw.phases?.events?.contact, cr = AN.ref.sw.phases?.events?.contact;
  const tsU = AN.video.analysis.time_scale || 1, tsR = AN.ref.ts || 1;
  const byContact = () => cr.tv + (tu - cu.tv) * tsU / tsR;   // real-time offset from the contact estimate
  if ($(".sel-sync", AN.root).value === "contact" || !AN.map?.length) return byContact();
  const M = AN.map;
  if (tu <= M[0][0]) return M[0][1] + (tu - M[0][0]) * tsU / tsR;
  if (tu >= M[M.length - 1][0]) return M[M.length - 1][1] + (tu - M[M.length - 1][0]) * tsU / tsR;
  let lo = 0, hi = M.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (M[mid][0] <= tu) lo = mid; else hi = mid; }
  const f = (tu - M[lo][0]) / (M[hi][0] - M[lo][0] || 1);
  return M[lo][1] + f * (M[hi][1] - M[lo][1]);
}

// ---- playback + skeleton overlay
function setupPlayback() {
  const vu = AN.vu, vr = AN.vr, root = AN.root;
  const rate = () => parseFloat($(".sel-rate", root).value);
  $(".btn-play", root).onclick = () => {
    if (vu.paused) {
      const [a0, a1] = AN.sw.analysed_range_v || [AN.sw.window.tv0, AN.sw.window.tv1];
      if (vu.currentTime >= a1 || vu.currentTime < a0 - 0.6) vu.currentTime = Math.max(0, a0 - 0.3);
      vu.playbackRate = rate(); vu.play();
    } else vu.pause();
  };
  vu.onplay = () => { $(".btn-play", root).textContent = "⏸ 暂停"; loop(); };
  vu.onpause = () => { $(".btn-play", root).textContent = "▶ 同步播放"; };
  vu.onseeked = drawFrame; vr.onseeked = () => drawPose(AN.cr, AN.vr, AN.ref?.pose);
  vu.onloadeddata = drawFrame;
  $(".chk-skel", root).onchange = drawFrame;
  $(".chk-ghost", root).onchange = drawFrame;
  $(".sel-sync", root).onchange = drawFrame;
  function loop() {
    if (vu.paused) return;
    const [, a1] = AN.sw.analysed_range_v || [0, 1e9];
    if (vu.currentTime > a1 + 0.4) { vu.pause(); }
    drawFrame(); requestAnimationFrame(loop);
  }
  new ResizeObserver(drawFrame).observe(vu);
}
function drawFrame() {
  if (!AN?.vu) return;
  const tu = AN.vu.currentTime;
  drawPose(AN.cu, AN.vu, AN.pose);
  if (AN.ref && AN.vr.readyState >= 1) {
    const tr = refTimeFor(tu);
    if (tr !== null && Math.abs(AN.vr.currentTime - tr) > 0.02) AN.vr.currentTime = Math.max(0, tr);
    drawPose(AN.cr, AN.vr, AN.ref.pose);
    if (tr !== null) drawGhost(tu, tr);
  }
  const ev = AN.sw.phases?.segments?.find((s) => tu >= s.tv0 && tu <= s.tv1);
  $(".tdisp", AN.root).textContent = `你 ${fmt(tu, 2)}s${AN.ref ? ` ↔ 参考 ${fmt(AN.vr.currentTime, 2)}s` : ""}${ev ? " · " + ev.label : ""}`;
  updateCursor(tu);
}
// reference skeleton drawn over the user's video: hip-centred, scaled by torso length, mirrored when the
// two players face opposite screen directions (canonical x sign differs)
function bodyFrame(k, s) {
  const ok = (j) => s[j] >= 0.3 && k[j][0] >= 0;
  if (![5, 6, 11, 12].every(ok)) return null;
  const hc = [(k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2];
  const sc = [(k[5][0] + k[6][0]) / 2, (k[5][1] + k[6][1]) / 2];
  const torso = Math.hypot(sc[0] - hc[0], sc[1] - hc[1]);
  return torso > 1 ? { hc, torso } : null;
}
function drawGhost(tu, tr) {
  if (!$(".chk-ghost", AN.root)?.checked || !AN.pose || !AN.ref?.pose || !AN.vu.videoWidth) return;
  const pu = AN.pose, pr = AN.ref.pose;
  const iu = nearestIdx(pu.t, tu), ir = nearestIdx(pr.t, tr);
  const fu = bodyFrame(pu.kpts[iu], pu.scores[iu]), fr = bodyFrame(pr.kpts[ir], pr.scores[ir]);
  if (!fu || !fr) return;
  const flip = Math.sign(AN.sw.x_sign || 1) !== Math.sign(AN.ref.sw?.x_sign || 1) ? -1 : 1;
  const kr = pr.kpts[ir], sr = pr.scores[ir];
  const g = (j) => [fu.hc[0] + flip * (kr[j][0] - fr.hc[0]) * fu.torso / fr.torso, fu.hc[1] + (kr[j][1] - fr.hc[1]) * fu.torso / fr.torso];
  const canvas = AN.cu, r = canvas.getBoundingClientRect(), video = AN.vu, ctx = canvas.getContext("2d");
  const sc = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
  const ox = (r.width - video.videoWidth * sc) / 2, oy = (r.height - video.videoHeight * sc) / 2;
  const sx = sc * video.videoWidth / pu.width, sy = sc * video.videoHeight / pu.height;
  const P = (j) => { const [x, y] = g(j); return [ox + x * sx, oy + y * sy]; };
  ctx.save(); ctx.globalAlpha = 0.75; ctx.strokeStyle = "#39d98a"; ctx.lineWidth = 4; ctx.setLineDash([7, 5]);
  for (const [a, b] of EDGES) {
    if (sr[a] < 0.3 || sr[b] < 0.3) continue;
    ctx.beginPath(); ctx.moveTo(...P(a)); ctx.lineTo(...P(b)); ctx.stroke();
  }
  ctx.restore();
}
function nearestIdx(t, x) { let lo = 0, hi = t.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= x) lo = m; else hi = m; } return (x - t[lo] < t[hi] - x) ? lo : hi; }
function drawPose(canvas, video, pose) {
  const r = canvas.getBoundingClientRect(); canvas.width = r.width; canvas.height = r.height;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!pose || !(video.virtual || $(".chk-skel", AN.root).checked) || !video.videoWidth) return;
  const i = nearestIdx(pose.t, video.currentTime);
  const k = pose.kpts[i], s = pose.scores[i];
  const sc = Math.min(r.width / video.videoWidth, r.height / video.videoHeight);
  const ox = (r.width - video.videoWidth * sc) / 2, oy = (r.height - video.videoHeight * sc) / 2;
  const c = video.crop;   // virtual reference: pose pixels map straight onto the zoomed crop
  const sx = c ? sc : sc * video.videoWidth / pose.width, sy = c ? sc : sc * video.videoHeight / pose.height;
  const P = (j) => [ox + (k[j][0] - (c ? c.x0 : 0)) * sx, oy + (k[j][1] - (c ? c.y0 : 0)) * sy];
  ctx.lineWidth = c ? 4 : 3;
  for (const [a, b] of EDGES) {
    if (s[a] < 0.3 || s[b] < 0.3 || k[a][0] < 0 || k[b][0] < 0) continue;
    ctx.strokeStyle = LEFT.has(a) && LEFT.has(b) ? "#29a3ff" : (!LEFT.has(a) && !LEFT.has(b) ? "#ff8a1f" : "#ddd");
    ctx.beginPath(); ctx.moveTo(...P(a)); ctx.lineTo(...P(b)); ctx.stroke();
  }
  for (let j = 0; j < 17; j++) {
    if (s[j] < 0.3 || k[j][0] < 0) continue;
    ctx.fillStyle = s[j] > 0.6 ? (LEFT.has(j) ? "#29a3ff" : "#ff8a1f") : "#ccc";
    ctx.beginPath(); ctx.arc(...P(j), 3.5, 0, 6.3); ctx.fill();
  }
}

// ---- phase timeline
function tlRow(label, segs, t0, t1, cls, seekFn, events) {
  const w = (x) => `${((x - t0) / (t1 - t0)) * 100}%`;
  const segsHtml = (segs || []).map((s) => `<div class="tl-seg" title="${esc(s.label)} ${fmt(s.tv0)}–${fmt(s.tv1)}s" style="left:${w(s.tv0)};width:calc(${w(s.tv1)} - ${w(s.tv0)});background:${PHASE_COLORS[s.name] || "#888"}">${esc(s.label)}</div>`).join("");
  const evHtml = events ? `<div class="tl-ev" style="left:${w(events.tv)};background:#c0392b;opacity:.9" title="估计击球"></div>` : "";
  return `<div class="tl-row"><span class="lab">${label}</span><div class="tl-bar ${cls}" data-t0="${t0}" data-t1="${t1}">${segsHtml}${evHtml}<div class="tl-cursor"></div></div></div>`;
}
function renderTimeline() {
  const el = $(".timeline", AN.root); const sw = AN.sw;
  if (!sw.phases) { el.innerHTML = `<p class="hint">未切分阶段（动作未识别为目标动作）。</p>`; return; }
  const [u0, u1] = [sw.window.tv0, sw.window.tv1];
  let html = tlRow("你", sw.phases.segments, u0, u1, "tl-user", null, sw.phases.events.contact);
  if (AN.ref?.sw?.phases) html += tlRow("参考", AN.ref.sw.phases.segments, AN.ref.sw.window.tv0, AN.ref.sw.window.tv1, "tl-ref", null, AN.ref.sw.phases.events.contact);
  html += `<p class="hint">阶段边界与击球时刻均为基于手腕运动的<b>估计</b>（未检测球和球拍；腕部轨迹≠拍头轨迹）。点击时间条可跳转。</p>`;
  el.innerHTML = html;
  $$(".tl-bar", el).forEach((bar) => bar.onclick = (e) => {
    const r = bar.getBoundingClientRect(); const f = (e.clientX - r.left) / r.width;
    const t0 = +bar.dataset.t0, t1 = +bar.dataset.t1;
    if (bar.classList.contains("tl-user")) AN.vu.currentTime = t0 + f * (t1 - t0);
    else { // seek user video to the time that maps to this ref time
      const tr = t0 + f * (t1 - t0); let best = AN.vu.currentTime, bd = 1e9;
      for (let x = AN.sw.window.tv0; x <= AN.sw.window.tv1; x += 0.01) { const d = Math.abs(refTimeFor(x) - tr); if (d < bd) { bd = d; best = x; } }
      AN.vu.currentTime = best;
    }
  });
}
function updateCursor(tu) {
  const bar = $(".tl-user", AN.root); if (!bar) return;
  const t0 = +bar.dataset.t0, t1 = +bar.dataset.t1;
  $(".tl-cursor", bar).style.left = `${Math.min(100, Math.max(0, (tu - t0) / (t1 - t0) * 100))}%`;
  const rb = $(".tl-ref", AN.root);
  if (rb && AN.ref) { const r0 = +rb.dataset.t0, r1 = +rb.dataset.t1; $(".tl-cursor", rb).style.left = `${Math.min(100, Math.max(0, (AN.vr.currentTime - r0) / (r1 - r0) * 100))}%`; }
}

// ---- recognition / corrections
function evidList(list) { return `<ul class="evid">${(list || []).map((e) => `<li>${esc(e)}</li>`).join("")}</ul>`; }
function renderRecog() {
  const a = AN.video.analysis, sw = AN.sw, c = sw.classification, v = sw.view, h = a.hand;
  const confB = (x) => `<span class="badge ${x >= 0.7 ? "good" : x >= 0.4 ? "warn" : "bad"}">置信度 ${pct(x)}</span>`;
  const auto = sw.auto_action && sw.auto_action !== sw.action ? `（自动识别为 ${esc(ACTION(sw.auto_action))}）` : "";
  $(".recog", AN.root).innerHTML = `
    <div><b>动作：${esc(ACTION(sw.action))}</b> ${confB(c.confidence)} ${sw.action === "serve" && (c.scores || {}).smash_like ? `<span class="badge warn">可能是高压</span>` : ""} ${auto}</div>${evidList(c.evidence)}
    ${c.knn ? `<div class="hint">最近参考：${c.knn.neighbours.map((n) => `${esc(ACTION(n.action))}(${fmt(n.cost)})`).join("，")}</div>` : ""}
    <div><b>机位：${esc(sw.view_key_label)}</b> ${confB(v.confidence)} <span class="hint">整段：${esc(a.view_overall.label)}</span></div>${evidList(v.evidence)}
    <div><b>持拍手：${h.hand === "left" ? "左手" : "右手"}</b> ${h.source === "user" ? `<span class="badge info">用户指定</span>` : confB(h.confidence)}
      ${h.source === "default_right_unverified" ? `<span class="badge bad">无法判断，暂按右手处理，请纠正</span>` : ""}</div>${evidList([h.evidence])}
    ${sw.match ? `<div><b>匹配依据</b></div><div class="evid">${esc(sw.match.message)}</div>` : ""}`;
}
function renderCorrect() {
  const a = AN.video.analysis, sw = AN.sw, m = AN.video.meta || {};
  const opts = (pairs, cur) => pairs.map(([v, l]) => `<option value="${v}" ${String(cur ?? "") === String(v) ? "selected" : ""}>${l}</option>`).join("");
  const vo = m.view_override || {};
  $(".correct", AN.root).innerHTML = `
    <label>持拍手 <select class="c-hand">${opts([["", "自动"], ["right", "右手"], ["left", "左手"]], m.hand || "")}</select></label>
    <label>本次动作 <select class="c-action">${opts([["", "自动"], ...Object.entries(META.actions)], (m.action_overrides || {})[sw.index] || "")}</select></label>
    <label>机位 <select class="c-view">${opts([["", "自动"], ["side", "侧面"], ["front", "正面"], ["back", "背面"], ["oblique_front", "斜前方"], ["oblique_back", "斜后方"]], vo.view || "")}</select></label>
    <label>侧面朝向 <select class="c-facing">${opts([["screen_right", "面向画面右侧"], ["screen_left", "面向画面左侧"]], vo.facing || "screen_right")}</select></label>
    <label>时间尺度 <select class="c-ts">${opts([["1", "实时"], ["0.5", "2×慢放"], ["0.25", "4×慢放"], ["0.125", "8×慢放"], ["unknown", "未知"]], m.time_scale ?? "1")}</select></label>
    <div><button class="c-save">保存纠正并重新分析</button> <span class="hint">纠正会记录在纠错日志中；姿态已缓存，重新分析只需几秒。</span></div>`;
  $(".c-save", AN.root).onclick = async () => {
    const R = AN.root, body = {};
    body.hand = $(".c-hand", R).value;
    const act = $(".c-action", R).value; const ao = { ...(m.action_overrides || {}) };
    if (act) ao[sw.index] = act; else delete ao[sw.index];
    body.action_overrides = ao;
    const vv = $(".c-view", R).value; body.view_override = vv ? { view: vv, facing: $(".c-facing", R).value } : {};
    const ts = $(".c-ts", R).value; body.time_scale = ts === "unknown" ? "unknown" : parseFloat(ts);
    await jpatch(`/api/videos/${AN.video.id}`, body);
    AN.video.status = "queued"; poll();
    $(".status-line", R).innerHTML = `<span class="badge info">已提交纠正，重新分析中…</span>`;
  };
}

// ---- professional comparison
const EVENT_LABEL = { backswing_end: "引拍终点", contact: "估计击球", end: "随挥结束", trophy: "奖杯姿势", drop: "持拍手下落", start: "开始" };
function proRange(c) {
  const scope = { same_view: "同机位职业", same_angle: "相近角度职业", pooled: "各机位职业" }[c.pro_scope] || "职业";
  return `${scope}中位 ${fmt(c.pro_median)}，常见范围 ${fmt(c.pro_p10)}–${fmt(c.pro_p90)}（10%–90%，n=${c.n_pro}）`;
}
function renderPro() {
  const el = $(".pro", AN.root), sw = AN.sw, p = sw.pro;
  if (!p) { el.innerHTML = `<p class="hint">无职业对比（参考视频或旧版分析结果）。</p>`; return; }
  if (p.status !== "ok") { el.innerHTML = `<p class="hint">${esc(p.message)}</p>`; return; }
  const ts = AN.video.analysis.time_scale || 1;
  const img = (c) => `/api/videos/${AN.video.id}/swings/${sw.index}/compare.jpg?cp=${encodeURIComponent(c.key)}`;
  const issue = (c, n) => `
    <div class="pro-issue">
      <div class="pi-head"><span class="pi-n">${n}</span><b>${esc(c.title)}</b>
        <span class="badge ${c.deviation >= 1.5 ? "bad" : "warn"}">偏离 ${fmt(c.deviation, 1)} 个职业四分位距</span></div>
      <div class="pi-body">
        <figure><img loading="lazy" src="${img(c)}" alt="${esc(c.label)} 对比"><figcaption>左：你　右：最接近的职业挥拍（${esc(p.nearest?.player || "无同机位职业参考")}）· ${esc(EVENT_LABEL[c.event] || c.event)}，黄色为相关关节</figcaption></figure>
        <div class="pi-text">
          <div>${esc(c.explanation)}</div>
          <div class="pi-num"><b>${esc(c.label)}</b>：你 ${fmt(c.value)} ${esc(c.unit)}；${proRange(c)}</div>
          <div class="hint">${esc(c.desc)}</div>
          <div class="p">${esc(c.practice)}</div>
          <div class="ev" data-t="${c.t / ts}">▶ 在视频中查看（${fmt(c.t / ts)}s）</div>
        </div>
      </div>
    </div>`;
  const ST = { ok: ["在职业范围内", "good"], borderline: ["接近边界（容差内）", "warn"], issue: ["偏离", "bad"], outside_not_fault: ["超出范围（非问题方向）", "info"],
    insufficient_reference: ["参考不足", "info"], not_applicable: ["机位不适用", ""], not_measured: ["未测到", ""] };
  const row = (c) => { const [t, cls] = ST[c.status] || [c.status, ""];
    return `<tr class="${["not_applicable", "not_measured", "insufficient_reference"].includes(c.status) ? "unrel" : ""}"><td>${esc(c.label)}</td>
      <td>${c.value !== undefined ? `${fmt(c.value)} ${esc(c.unit)}` : "–"}</td>
      <td>${c.pro_median !== undefined ? `${fmt(c.pro_median)}（${fmt(c.pro_p10)}–${fmt(c.pro_p90)}，n=${c.n_pro}）` : "–"}</td>
      <td><span class="badge ${cls}">${t}</span></td>
      <td>${c.validation ? `${c.validated ? "✓ 已验证" : "未通过"}（职业误报 ${pct(c.validation.pro_false_alarm)}，n=${c.validation.pro_measured}）` : "未评估"}</td>
      <td class="hint">${esc(c.note || c.desc)}</td></tr>`; };
  el.innerHTML = `
    <p>${esc(p.message)}</p>
    ${p.issues.length ? `<h4>主要问题（按偏离程度 × 重要性排序）</h4>${p.issues.map((c, k) => issue(c, k + 1)).join("")}`
      : `<p class="ok-note">在可测量的检查点上，没有发现明显超出职业范围的地方。这只说明 2D 骨架上找不到可靠差异，不代表动作没有问题。</p>`}
    ${p.more_issues?.length ? `<p class="hint">另有 ${p.more_issues.length} 项较小偏离：${p.more_issues.map((c) => esc(c.title)).join("；")}</p>` : ""}
    ${p.tentative?.length ? `<h4>参考性提示 <small class="hint">（这些检查点在职业球员之间的留一验证中误报偏多或样本不足，尚未通过验证，不计入主要问题）</small></h4>
      <ul>${p.tentative.map((c) => `<li><b>${esc(c.title)}</b>：你 ${fmt(c.value)} ${esc(c.unit)}，${proRange(c)}
        <span class="hint">（职业误报率 ${c.validation ? pct(c.validation.pro_false_alarm) : "未评估"}）</span>
        <span class="ev" data-t="${c.t / ts}">▶ ${fmt(c.t / ts)}s</span></li>`).join("")}</ul>` : ""}
    ${p.strengths?.length ? `<h4>和职业接近的地方</h4><ul>${p.strengths.map((c) => `<li>${esc(c.label)}：你 ${fmt(c.value)} ${esc(c.unit)}，${proRange(c)}</li>`).join("")}</ul>` : ""}
    <details><summary>全部检查点（${p.checks.length}）</summary><div class="tbl"><table><thead><tr><th>检查点</th><th>你</th><th>职业中位（范围，n）</th><th>判断</th><th>验证</th><th>说明</th></tr></thead>
      <tbody>${p.checks.map(row).join("")}</tbody></table></div></details>
    <p class="hint">${esc(p.limits)} 职业参考球员：${esc((p.pro_players || []).join("、"))}。</p>`;
  $$(".ev", el).forEach((e) => e.onclick = () => { AN.vu.currentTime = Math.max(0, +e.dataset.t - 0.15); AN.vu.pause(); AN.vu.scrollIntoView({ block: "center" }); });
}

// ---- feedback / metrics / quality
function renderFeedback() {
  const items = AN.sw.feedback || [];
  const ts = AN.video.analysis.time_scale || 1;
  $(".feedback", AN.root).innerHTML = items.map((f) => `
    <div class="fb-item ${f.kind}"><div class="t">${esc(f.title)} ${f.confidence ? `<span class="badge warn">证据强度：${esc(f.confidence)}</span>` : ""}</div>
      <div>${esc(f.observation)}</div>
      ${f.evidence ? `<div class="ev" data-t="${f.evidence.t0 / ts}">▶ 查看证据：${fmt(f.evidence.t0 / ts)}s – ${fmt(f.evidence.t1 / ts)}s（视频时间）</div>` : ""}
      ${f.practice ? `<div class="p">练习建议：${esc(f.practice)}</div>` : ""}
      ${f.confidence_note ? `<div class="hint">${esc(f.confidence_note)}</div>` : ""}</div>`).join("") || `<p class="hint">无</p>`;
  $$(".feedback .ev", AN.root).forEach((e) => e.onclick = () => { AN.vu.currentTime = Math.max(0, +e.dataset.t - 0.15); AN.vu.pause(); AN.vu.scrollIntoView({ block: "center" }); });
}
function renderMetrics() {
  const comp = AN.sw.match?.comparison || {};
  const own = AN.sw.metrics || {};
  const keys = Object.keys(own);
  if (!keys.length) { $(".metrics", AN.root).innerHTML = `<p class="hint">无可用指标。</p>`; return; }
  const ST = { within: "范围内", above: "高于参考", below: "低于参考" };
  $(".metrics", AN.root).innerHTML = `<div class="tbl"><table><thead><tr><th>指标</th><th>你</th><th>参考组（均值 / 范围 / n）</th><th>判断</th><th>说明</th></tr></thead><tbody>${
    keys.map((k) => { const m = own[k], c = comp[k];
      return `<tr class="${m.reliable ? "" : "unrel"}"><td>${esc(m.label)}</td><td>${fmt(m.value)} ${esc(m.unit)}</td>
        <td>${c ? `${fmt(c.ref_mean)} / ${fmt(c.ref_min)}–${fmt(c.ref_max)} / ${c.n_ref}` : "–"}</td>
        <td>${!m.reliable ? "停用" : c ? `<span class="st-${c.status}">${ST[c.status]}</span>` : "–"}</td>
        <td class="hint">${esc(m.reliable ? m.desc : m.why_unreliable)}</td></tr>`; }).join("")}</tbody></table></div>
    <p class="hint">所有角度、位置均为 2D 投影，单位“躯干长”= 肩中心到髋中心距离；速度为相对强度，不代表真实球拍速度、力或功率。灰色行在当前机位/条件下不可靠，不参与反馈。</p>`;
}
const CURVES = { wrist_speed: "手腕相对速度", elbow_angle: "肘角(投影)", knee_angle: "膝角(投影)", wrist_x: "手腕水平位置", wrist_y: "手腕高度" };
function renderCurves() {
  const ctl = $(".curve-ctrl", AN.root);
  ctl.innerHTML = Object.entries(CURVES).map(([k, l], i) => `<label><input type="radio" name="cv" value="${k}" ${i === 0 ? "checked" : ""}> ${l}</label>`).join("");
  $$("input", ctl).forEach((r) => r.onchange = plotCurve);
  const box = $(".panel-curves", AN.root);
  if (box) box.ontoggle = () => box.open && plotCurve();   // canvas has no width while collapsed
  plotCurve();
}
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function plotCurve() {
  const key = $("input[name=cv]:checked", AN.root)?.value || "wrist_speed";
  const cv = $(".curves", AN.root); const W = cv.clientWidth; if (!W) return;
  const dpr = window.devicePixelRatio || 1, H = 220;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + "px";
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const C = { grid: cssVar("--line"), text: cssVar("--muted"), you: cssVar("--accent"), ref: cssVar("--orange") };
  const su = AN.sw.curves; const xs = su.t; const yu = su[key];
  let yr = null;
  if (AN.ref && AN.ref.cand.path) {   // ref values mapped onto the user's time axis by the DTW path
    yr = new Array(xs.length).fill(null); const cnt = new Array(xs.length).fill(0);
    for (const [i, j] of AN.ref.cand.path) { if (i < xs.length && AN.ref.sw.curves[key][j] !== undefined) { yr[i] = (yr[i] || 0) + AN.ref.sw.curves[key][j]; cnt[i]++; } }
    yr = yr.map((v, i) => cnt[i] ? v / cnt[i] : null);
  }
  const all = [...yu, ...(yr || []).filter((v) => v !== null)];
  const ymin = Math.min(...all), ymax = Math.max(...all), pad = 28;
  const X = (t) => pad + (t - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - pad - 8);
  const Y = (v) => H - 18 - (v - ymin) / ((ymax - ymin) || 1) * (H - 30);
  ctx.strokeStyle = C.grid; ctx.fillStyle = C.text; ctx.font = "11px sans-serif"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const v = ymin + (ymax - ymin) * g / 4; ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), 0, Y(v) + 4); }
  (AN.sw.phases?.segments || []).forEach((s) => { ctx.fillStyle = (PHASE_COLORS[s.name] || "#888") + "22"; ctx.fillRect(X(s.t0), 0, X(s.t1) - X(s.t0), H - 18); });
  const line = (ys, color, dash) => { ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = 2; ctx.beginPath(); let started = false;
    ys.forEach((v, i) => { if (v === null) { started = false; return; } if (!started) { ctx.moveTo(X(xs[i]), Y(v)); started = true; } else ctx.lineTo(X(xs[i]), Y(v)); }); ctx.stroke(); ctx.setLineDash([]); };
  if (yr) line(yr, C.ref, [5, 4]);
  line(yu, C.you, []);
  ctx.fillStyle = C.text; ctx.fillText(`${CURVES[key]} · 横轴：你的动作时间（${AN.video.analysis.time_known ? "秒" : "时间尺度未知"}）`, pad, H - 4);
}
function renderQuality(a, sw) {
  const q = sw ? sw.quality : a.quality; const all = a.quality;
  const qb = $(".q-badge", AN.root);
  if (qb) qb.innerHTML = `<span class="badge ${q.level === "good" ? "good" : q.level}">${q.level === "good" ? "良好" : q.level === "warn" ? "有提醒" : "不合格"}</span>`;
  const li = (i) => `<li><span class="badge ${i.level === "bad" ? "bad" : "warn"}">${i.level === "bad" ? "需重拍" : "注意"}</span>${esc(i.message)}</li>`;
  $(".quality", AN.root).innerHTML = `<div>本次挥拍：<span class="badge ${q.level === "good" ? "good" : q.level}">${q.level === "good" ? "良好" : q.level === "warn" ? "有提醒" : "不合格"}</span></div>
    <ul>${q.issues.map(li).join("") || "<li>未发现问题</li>"}</ul>
    <div>整段视频：</div><ul>${all.issues.map(li).join("") || "<li>未发现问题</li>"}</ul>`;
}

// ------------------------------------------------------------------ library
function renderRefVideos(vs) {
  $("#ref-video-list").innerHTML = vs.map((v) => videoLi(v, false)).join("") || `<li class="empty">暂无</li>`;
  $$("#ref-video-list li[data-id]").forEach((li) => li.onclick = async () => {
    const v = vs.find((x) => x.id === li.dataset.id);
    if (v.status === "failed") return alert(v.error);
    if (confirm(`删除示范视频「${v.filename}」及其全部参考条目？\n（取消 = 不删除）`)) { await api(`/api/videos/${v.id}`, { method: "DELETE" }); refreshLibrary(); }
  });
}
async function refreshLibrary() { renderRefVideos(await api("/api/videos?kind=reference")); refreshRefTable(); }
async function refreshRefTable() {
  const refs = await api("/api/references");
  $("#ref-count").textContent = `共 ${refs.length} 条，启用 ${refs.filter((r) => r.enabled && r.review_status !== "rejected").length} 条`;
  const opt = (obj, cur) => Object.entries(obj).map(([k, l]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${esc(l)}</option>`).join("");
  $("#ref-table tbody").innerHTML = refs.map((r) => `<tr data-id="${r.id}">
    <td>${r.id}</td><td><a href="#" class="prev">${esc(r.filename)} #${r.swing_index + 1}</a></td><td>${esc(r.meta?.player || "")}</td>
    <td><select data-f="action">${opt(META.actions, r.action)}</select></td>
    <td>${esc(META.view_keys[r.view_key] || r.view_key)}</td><td>${r.hand === "left" ? "左" : "右"}</td>
    <td>${r.label_source === "user" ? "人工" : "自动"}</td>
    <td><select data-f="review_status">${opt(META.review_status, r.review_status)}</select></td>
    <td><input data-f="style" value="${esc(r.style || "")}" placeholder="如：大引拍"></td>
    <td><input type="checkbox" data-f="enabled" ${r.enabled ? "checked" : ""}></td>
    <td><button class="danger del">移除</button></td></tr>`).join("");
  $$("#ref-table tbody tr").forEach((tr) => {
    const id = tr.dataset.id;
    $$("[data-f]", tr).forEach((el) => el.onchange = async () => {
      const f = el.dataset.f; let val = el.type === "checkbox" ? el.checked : el.value; const body = { [f]: val };
      if (f === "review_status" && val === "coach_reviewed") { const who = prompt("审核教练姓名（必填）："); if (!who) { refreshRefTable(); return; } body.reviewer = who; }
      try { await jpatch(`/api/references/${id}`, body); } catch (e) { alert(e.message); refreshRefTable(); }
    });
    $(".del", tr).onclick = async () => { if (confirm("从参考库移除该条目？（源视频保留）")) { await api(`/api/references/${id}`, { method: "DELETE" }); refreshRefTable(); } };
    $(".prev", tr).onclick = (e) => { e.preventDefault(); previewRef(id); };
  });
}
async function previewRef(id) {
  const d = await api(`/api/references/${id}`); const el = $("#ref-preview");
  const sw = d.swing;
  el.innerHTML = `<h3>参考 #${id} 预览：${esc(ACTION(d.ref.action))} · ${esc(META.view_keys[d.ref.view_key] || "")}</h3>
    ${d.video.has_video === false
      ? `<p class="hint">这条来自职业参考包：只有关键点数据，没有原视频（视频不随项目分发）。对比时以骨架显示；原视频见下方来源链接。</p>`
      : `<div class="vwrap" style="max-width:520px"><video controls muted playsinline src="/api/videos/${d.video.id}/file#t=${Math.max(0, sw.window.tv0)},${sw.window.tv1}"></video></div>`}
    <div class="hint">来源：${esc(d.video.meta?.source || "未填写")} · 许可：${esc(d.video.meta?.license || "未说明")} · 商用：${esc(d.video.meta?.license_commercial || "未说明")} · 再分发：${esc(d.video.meta?.redistribution || "未说明")}</div>
    <div class="hint">阶段：${(sw.phases?.segments || []).map((s) => `${esc(s.label)} ${fmt(s.tv0)}–${fmt(s.tv1)}s`).join(" | ")}</div>`;
}

// ------------------------------------------------------------------ about
function renderAbout() {
  $("#about").innerHTML = `<h2>工作原理</h2><ol>
  <li>上传 → 格式校验 → 按旋转信息转正并转码（保留真实时间戳）→ 姿态估计（${esc(META.pose_model)}，本机 OpenVINO 推理，结果按文件内容缓存）。</li>
  <li>关键点清洗：低置信度插值、平滑、左右肢体标签跳变修正；以髋部中心为原点、躯干长为单位做尺度和平移归一化，<b>保留真实时长</b>。</li>
  <li>挥拍检测：手腕相对髋部速度的峰值；持拍手：峰值速度和轨迹长度两条证据一致才判断；机位：用整段准备/收拍姿态的髋宽、左右顺序、面部可见性统计，而非单帧肩膀朝向。</li>
  <li>左手持拍：骨架镜像并<b>同步交换左右关节标签</b>，等价为右手后与参考比较（单元测试覆盖）。不同机位不做 2D 旋转“校正”，机位不符时拒绝打分。</li>
  <li>动作识别：可解释规则（发球/双反/正反手）+ 参考库最近邻 DTW 投票；不一致或距离过大判为“未知/非目标”。未训练深度分类器。</li>
  <li>阶段：准备 → 引拍/蓄力 → 加速 → 击球附近（<b>估计</b>，手腕速度峰值）→ 随挥。发球为 抛球/奖杯 → 持拍手下落 → 加速 → 击球 → 随挥。</li>
  <li>匹配：同动作 + 相近机位筛选参考 → 以阶段事件为锚点逐段 DTW → 与参考组分布比较各指标 → 超出范围且可靠的差异生成反馈。</li></ol>
  <h2>局限</h2><ul>
  <li>只有 2D：髋肩真实旋转、深度方向运动、地面反作用力等无法从单目视频可靠获得，相关指标已停用或标注为投影观察。</li>
  <li>未检测球和球拍：击球时刻为估计；手腕轨迹不等于拍头轨迹。</li>
  <li>背面/斜后方机位下持拍手和正反手判断明显不可靠；首版以侧面、单人、固定机位、全身入镜为主。</li>
  <li>参考默认是“参考示范”，不是专业标准；反馈不涉及伤病诊断。</li></ul>`;
}

// ------------------------------------------------------------------ init
(async function init() {
  META = await api("/api/meta");
  for (const sel of $$("select[name=action]")) for (const [k, l] of Object.entries(META.actions)) if (k !== "unknown") sel.insertAdjacentHTML("beforeend", `<option value="${k}">${esc(l)}</option>`);
  renderAbout();
  await refreshQueries();
  const deep = new URLSearchParams(location.search).get("video");   // annotate.html links back with ?video=<id>
  if (deep) openAnalysis(deep).catch(() => {});
  poll();
})();
