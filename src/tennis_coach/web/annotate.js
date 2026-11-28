// Stroke / contact-frame annotation for the evaluation set. Labels are saved to data/labels/annotations/.
const $ = (s, r = document) => r.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const STROKE_LABEL = { forehand: "正手", backhand_1h: "单手反手", backhand_2h: "双手反手", serve: "发球", smash: "高压", volley: "截击", other: "其他" };
const KEYS = { f: "forehand", b: "backhand_1h", d: "backhand_2h", s: "serve", m: "smash", v: "volley", o: "other" };
const KEY_OF = Object.fromEntries(Object.entries(KEYS).map(([k, v]) => [v, k.toUpperCase()]));
const ST = { key: null, fps: 30, nFrames: 0, events: [], sel: -1, dirty: false, timer: null, sugg: [], suggTimer: null, linked: null };
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
    const tags = (v.root === "eval_real" ? `<span class="tag">评测集</span>` : `<span class="tag">导入</span>`) + (v.linked_video ? `<span class="tag">已分析</span>` : "");
    return `<li data-key="${esc(v.key)}" class="${v.key === ST.key ? "active" : ""}"><div class="nm">${esc(v.key)}${tags}</div>
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
  ST.linked = d.linked_video || null;
  $("#open-an").style.display = ST.linked ? "" : "none"; $("#open-an").href = ST.linked ? `/?video=${encodeURIComponent(ST.linked)}` : "#";
  $("#del-video").style.display = d.deletable ? "" : "none";
  let show = d.suggest_default;
  try { const m = localStorage.getItem("sysShow:" + key); if (m !== null) show = m === "1"; } catch (_) { /* ignore */ }
  $("#sys-show").checked = !!show;
  ST.sugg = []; loadSuggestions();
  try { history.replaceState(null, "", `?key=${encodeURIComponent(key)}`); } catch (_) { /* ignore */ }
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
  else if ((k === "[" || k === "]") && $("#sys-show").checked && ST.sugg.length) {
    e.preventDefault();
    const f = curFrame(), fs = ST.sugg.map((g) => g.frame);
    const n = k === "]" ? fs.find((x) => x > f) : [...fs].reverse().find((x) => x < f);
    if (n !== undefined) seekFrame(n);
  }
});

// ---- system hints: swings found by the analysis of the linked app video. Display only, never saved as labels.
async function loadSuggestions() {
  clearTimeout(ST.suggTimer);
  const key = ST.key;
  if (!key) return;
  let d;
  try { d = await api(`/api/annot/suggestions?key=${encodeURIComponent(key)}`); } catch (e) { d = { status: "error", swings: [] }; }
  if (key !== ST.key) return;
  ST.sugg = d.swings || [];
  const st = $("#sys-st");
  if (d.status === "none") st.textContent = "（这个视频没有系统分析：导入时勾选“同时做系统分析”，或在分析页上传后点“标注这个视频”）";
  else if (d.status === "missing") st.textContent = "（对应的分析已被删除）";
  else if (["queued", "running"].includes(d.status)) {
    st.textContent = `系统分析中：${d.stage || ""} ${Math.round((d.progress || 0) * 100)}%`;
    ST.suggTimer = setTimeout(loadSuggestions, 3000);
  } else if (d.status === "failed") st.textContent = "（系统分析失败，可在分析页查看原因）";
  else if (d.status === "done") st.textContent = `系统检出 ${ST.sugg.length} 次挥拍`;
  else st.textContent = "";
  renderSys();
}
const SYS_COLORS = { forehand: "#e4572e", backhand_1h: "#2e86de", backhand_2h: "#8e44ad", serve: "#16a085", unknown: "#888" };
function renderSys() {
  const on = $("#sys-show").checked && ST.sugg.length > 0;
  const tl = $("#tl-sys"), chips = $("#sgchips");
  tl.style.display = on ? "" : "none"; chips.style.display = on ? "" : "none";
  tl.innerHTML = ""; chips.innerHTML = "";
  if (!on) return;
  ST.sugg.forEach((g) => {
    const d = document.createElement("div");
    d.className = "sg" + (g.action === "unknown" ? " unk" : "");
    d.style.left = `${(g.frame / Math.max(1, ST.nFrames - 1)) * 100}%`; d.style.borderColor = SYS_COLORS[g.action] || "#888";
    d.title = `系统：帧 ${g.frame} ${g.label}（置信 ${g.confidence}）${g.contact_estimated ? "" : "，未估计击球时刻（速度峰值）"}`;
    d.onclick = (e) => { e.stopPropagation(); seekFrame(g.frame); };
    tl.appendChild(d);
  });
  chips.innerHTML = ST.sugg.map((g, i) => `<button data-i="${i}" title="${esc(g.non_stroke ? "系统判为非挥拍动作" : "")}">${g.frame} · ${esc(g.label)}${g.action === "unknown" ? "" : ` ${g.confidence}`}</button>`).join("");
  chips.querySelectorAll("button").forEach((b) => b.onclick = () => seekFrame(ST.sugg[+b.dataset.i].frame));
}
$("#sys-show").onchange = () => { try { localStorage.setItem("sysShow:" + ST.key, $("#sys-show").checked ? "1" : "0"); } catch (_) { /* ignore */ } renderSys(); };
$("#tl-sys").onclick = (e) => {
  const r = $("#tl-sys").getBoundingClientRect();
  seekFrame(Math.round(((e.clientX - r.left) / r.width) * (ST.nFrames - 1)));
};

// ---- delete an imported video (and everything derived from it)
$("#del-video").onclick = async () => {
  if (!ST.key) return;
  if (!confirm(`永久删除视频「${ST.key}」及转换产生的文件？此操作不可撤销。`)) return;
  const labels = confirm("同时删除这个视频的标注 JSON（所有标注人）？\n确定 = 删除，取消 = 保留标注");
  const appv = ST.linked ? confirm("同时删除对应的系统分析（分析页里的这条视频、姿态缓存和分析结果）？\n确定 = 删除，取消 = 保留") : false;
  try {
    await api(`/api/annot/video?key=${encodeURIComponent(ST.key)}&labels=${labels}&app_video=${appv}`, { method: "DELETE" });
    ST.key = null; ST.dirty = false; $("#work").style.display = "none";
    try { history.replaceState(null, "", location.pathname); } catch (_) { /* ignore */ }
    refreshList();
  } catch (e) { alert(`删除失败：${e.message}`); }
};

// ---- import: local upload or one YouTube link -> CFR copy in data/raw/annotate/<name>/ (+ optional analysis)
async function followJob(j) {
  const el = $("#jobst"); el.classList.remove("err");
  while (j.state === "running") {
    el.textContent = `${j.name || ""}：${j.stage}…`;
    await new Promise((r) => setTimeout(r, 1000));
    j = await api(`/api/annot/jobs/${j.id}`);
  }
  if (j.state !== "done") { el.classList.add("err"); el.textContent = `导入失败：${j.error || ""}`; return; }
  el.textContent = `已导入 ${j.key}${j.video_id ? "，系统分析已排队（可以先开始标注）" : ""}`;
  await refreshList();
  openVideo(j.key);
}
function importBusy(b) { $("#imp-upload").disabled = b; $("#imp-yt").disabled = b; }
$("#imp-upload").onclick = async () => {
  const f = $("#imp-file").files[0];
  if (!f) { alert("先选择一个视频文件"); return; }
  const fd = new FormData(); fd.append("file", f); fd.append("analyse", $("#imp-analyse").checked);
  importBusy(true); $("#jobst").classList.remove("err"); $("#jobst").textContent = `上传 ${f.name}…`;
  try { await followJob(await api("/api/annot/import/upload", { method: "POST", body: fd })); }
  catch (e) { $("#jobst").classList.add("err"); $("#jobst").textContent = `导入失败：${e.message}`; }
  finally { importBusy(false); }
};
$("#imp-yt").onclick = async () => {
  const url = $("#imp-url").value.trim();
  if (!url) { alert("先粘贴 YouTube 链接"); return; }
  const num = (s) => ($(s).value === "" ? null : +$(s).value);
  importBusy(true); $("#jobst").classList.remove("err"); $("#jobst").textContent = "提交下载…";
  try {
    await followJob(await api("/api/annot/import/youtube", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, start: num("#imp-start"), end: num("#imp-end"), analyse: $("#imp-analyse").checked }) }));
  } catch (e) { $("#jobst").classList.add("err"); $("#jobst").textContent = `导入失败：${e.message}`; }
  finally { importBusy(false); }
};

refreshList().then(() => {
  const k = new URLSearchParams(location.search).get("key");
  if (k) openVideo(k);
});
