/* Deal Desk Helper — demo replay + live SSE, one renderer for both.
 *
 * The recorded runs come from /api/demo; a live run streams from POST /api/ask.
 * Both are the same event schema, so handleEvent() renders either source. The
 * only difference is transport: the demo is replayed on a timer, the live path
 * is read frame-by-frame off the SSE body.
 */
(function () {
  "use strict";

  var PACE = 750; // ms between trace (stage) rows
  var PLACEHOLDER = "Pick one of the questions above…";

  var runsById = {};
  var state = { runId: null, running: false, timer: null, abort: null, doneStages: 0 };

  var $question = document.getElementById("dd-question");
  var $ask = document.getElementById("dd-ask");
  var $picker = document.getElementById("dd-picker");
  var $run = document.getElementById("dd-run");
  var $akey = document.getElementById("dd-akey");
  var $ckey = document.getElementById("dd-ckey");

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function hasKeys() { return $akey.value.trim() && $ckey.value.trim(); }
  function currentQuestion() {
    var t = ($question.textContent || "").trim();
    return t === PLACEHOLDER ? "" : t;
  }

  // ---- layout: build the run scaffold once per run --------------------------
  function newRunDom() {
    $run.innerHTML =
      '<div style="display:grid; gap:var(--space-6);">' +
      '  <div id="dd-trace" style="display:grid; gap:var(--space-2); border-left:1px solid var(--color-divider); padding-left:var(--space-6);"></div>' +
      '  <div id="dd-spinner"></div>' +
      '  <div id="dd-route"></div>' +
      '  <div id="dd-rewrite"></div>' +
      '  <div id="dd-result"></div>' +
      "</div>";
  }

  function setSpinner(on, label) {
    var slot = document.getElementById("dd-spinner");
    if (!slot) return;
    slot.innerHTML = on
      ? '<div style="display:flex; align-items:center; gap:var(--space-3); font-size:14px; color:var(--color-neutral-700);">' +
        '<span style="width:13px;height:13px;border:1.5px solid var(--color-accent-300);border-top-color:var(--color-accent);border-radius:50%;animation:ddSpin .8s linear infinite;"></span>' +
        "<span>" + esc(label) + "</span></div>"
      : "";
  }

  // ---- per-event renderers --------------------------------------------------
  function renderStage(d) {
    if (d.state === "running") {
      setSpinner(true, state.doneStages >= 3 ? "Grounding the answer…" : "Working through the pipeline…");
      return;
    }
    state.doneStages += 1;
    var trace = document.getElementById("dd-trace");
    if (!trace) return;
    var row = document.createElement("div");
    row.className = "ddrow dd-trace-row";
    row.style.cssText = "display:grid; grid-template-columns:150px 1fr auto; gap:var(--space-4); align-items:baseline; padding:var(--space-2) 0;";
    row.innerHTML =
      '<span style="font-family:var(--font-heading); font-size:14px; letter-spacing:0.06em; text-transform:uppercase; color:var(--color-accent-700);">' + esc(d.label) + "</span>" +
      '<span class="dd-detail" style="font-size:14px; color:var(--color-neutral-800);">' + esc(d.detail) + "</span>" +
      '<span class="text-muted" style="font-size:11px; font-family:ui-monospace, monospace;">' + esc(d.ms) + "</span>";
    trace.appendChild(row);
  }

  function renderRoute(d) {
    var slot = document.getElementById("dd-route");
    if (!slot) return;
    slot.innerHTML =
      '<div class="ddrow" style="display:grid; gap:var(--space-1); padding:var(--space-4) var(--space-6); border-left:2px solid var(--color-accent);">' +
      '<span style="font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:var(--color-accent-700);">Why it picked this skill</span>' +
      '<span style="font-size:15px; color:var(--color-neutral-900);">' + esc(d.reason) + "</span></div>";
  }

  function renderRewrite(d) {
    var slot = document.getElementById("dd-rewrite");
    if (!slot || !d.changed) return;
    slot.innerHTML =
      '<div class="ddrow dd-rewrite" style="display:grid; grid-template-columns:1fr 1fr; gap:var(--space-6); padding:var(--space-4) var(--space-6); background:var(--color-accent-100);">' +
      '<div style="display:grid; gap:var(--space-1);"><span class="text-muted" style="font-size:11px; letter-spacing:0.08em; text-transform:uppercase;">You typed</span>' +
      '<span style="font-size:14px;">' + esc(d.original) + "</span></div>" +
      '<div style="display:grid; gap:var(--space-1);"><span style="font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color:var(--color-accent-700);">Searched as</span>' +
      '<span style="font-size:14px; color:var(--color-accent-900);">' + esc(d.rewritten) + "</span></div></div>";
  }

  function renderAnswer(d) {
    var slot = document.getElementById("dd-result");
    if (!slot) return;
    if (d.kind === "find") return renderFind(slot, d);
    // draft or single: a heading, optional subheading, cited body, sources.
    var html = '<div class="ddrow" style="display:grid; gap:var(--space-4);">';
    html += '<h3 style="font-size:22px; margin:0;">' + esc(d.heading || "Answer") + "</h3>";
    if (d.subheading) html += '<span style="font-family:var(--font-heading); font-size:16px;">' + esc(d.subheading) + "</span>";
    html += '<p style="font-size:15px; line-height:1.65; margin:0; color:var(--color-neutral-900);">' + (d.body_html || esc(d.body_plain)) + "</p>";
    if (d.note) html += '<div style="border-left:2px solid var(--color-accent); padding-left:var(--space-4); font-size:14px; color:var(--color-neutral-800);"><strong>Deal desk note:</strong> ' + esc(d.note) + "</div>";
    if (d.disclaimer) html += '<p class="text-muted" style="font-size:13px; margin:0;">' + esc(d.disclaimer) + "</p>";
    var sources = d.sources || (d.citations || []).map(function (c) { return { doc_id: c.doc_id, score: c.score }; });
    if (sources.length) {
      html += '<div style="display:grid; gap:var(--space-2); margin-top:var(--space-2);"><span class="text-muted" style="font-size:11px; letter-spacing:0.08em; text-transform:uppercase;">Sources</span>';
      sources.forEach(function (s) {
        html += '<div style="display:flex; justify-content:space-between; gap:var(--space-4); padding:var(--space-2) 0; border-bottom:1px solid var(--color-divider); font-size:13px;">' +
          '<span style="font-family:ui-monospace, monospace;">' + esc(s.doc_id) + "</span>" +
          '<span class="text-muted" style="font-size:12px;">relevance ' + esc(s.score) + "</span></div>";
      });
      html += "</div>";
    }
    html += "</div>";
    slot.innerHTML = html;
  }

  function renderFind(slot, d) {
    var html = '<div class="ddrow" style="display:grid; gap:var(--space-4);">';
    html += '<h3 style="font-size:22px; margin:0;">' + esc(d.summary) + "</h3>";
    (d.findings || []).forEach(function (f) {
      html += '<div style="display:grid; gap:var(--space-1); padding:var(--space-3) 0; border-bottom:1px solid var(--color-divider);">' +
        '<div style="display:flex; align-items:center; gap:var(--space-3);">' +
        '<span style="font-family:ui-monospace, monospace; font-size:13px;">' + esc(f.doc_id || f.doc) + "</span>" +
        (f.tag ? '<span class="tag tag-accent">' + esc(f.tag) + "</span>" : "") + "</div>" +
        '<span style="font-size:14px; color:var(--color-neutral-800);">' + esc(f.reason) + "</span></div>";
    });
    if (d.footer) html += '<p class="text-muted" style="font-size:13px; margin:0;">' + esc(d.footer) + "</p>";
    html += "</div>";
    slot.innerHTML = html;
  }

  function renderDecline(d) {
    var slot = document.getElementById("dd-result");
    if (!slot) return;
    slot.innerHTML =
      '<div class="ddrow" style="display:grid; gap:var(--space-3);">' +
      '<div style="padding:var(--space-4) var(--space-6); background:var(--color-neutral-200); border-left:2px solid var(--color-neutral-600);">' +
      '<span style="font-family:var(--font-heading); font-size:18px;">' + esc(d.reason) + "</span></div>" +
      '<p style="font-size:15px; color:var(--color-neutral-800); margin:0; max-width:46em;">' + esc(d.why) + "</p></div>";
  }

  function renderError(d) {
    if (d.code === "busy" && state.runId) { replay(state.runId); return; } // fall back to the recorded run
    var slot = document.getElementById("dd-result");
    if (slot) slot.innerHTML = '<div style="padding:var(--space-4) var(--space-6); background:var(--color-neutral-200); border-left:2px solid var(--color-neutral-600); font-size:14px;">' + esc(d.message) + "</div>";
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "stage": renderStage(ev.data); break;
      case "route": renderRoute(ev.data); break;
      case "rewrite": renderRewrite(ev.data); break;
      case "answer": renderAnswer(ev.data); break;
      case "decline": renderDecline(ev.data); break;
      case "error": renderError(ev.data); break;
      case "retrieval": case "judge": break; // informational; the stage/answer rows cover these
      case "done": finish(); break;
    }
  }

  function finish() {
    state.running = false;
    setSpinner(false, "");
    $ask.textContent = "Run again";
    $ask.disabled = false;
  }

  // ---- demo replay ----------------------------------------------------------
  function replay(runId) {
    stop();
    var run = runsById[runId];
    if (!run) return;
    state.runId = runId;
    state.running = true;
    state.doneStages = 0;
    newRunDom();
    setSpinner(true, "Working through the pipeline…");
    $ask.textContent = "Running…";
    $ask.disabled = true;
    var evs = run.events, i = 0, first = true;
    function tick() {
      if (i >= evs.length) return;
      var ev = evs[i++];
      handleEvent(ev);
      if (i < evs.length) {
        var delay = ev.type === "stage" ? (first ? PACE * 0.66 : PACE) : 0;
        first = false;
        state.timer = setTimeout(tick, delay);
      }
    }
    tick();
  }

  // ---- live SSE -------------------------------------------------------------
  async function live(question) {
    stop();
    state.runId = null;
    state.running = true;
    state.doneStages = 0;
    newRunDom();
    setSpinner(true, "Working through the pipeline…");
    $ask.textContent = "Running…";
    $ask.disabled = true;
    var controller = new AbortController();
    state.abort = controller;
    try {
      var resp = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Anthropic-Key": $akey.value.trim(),
          "X-Cohere-Key": $ckey.value.trim(),
        },
        body: JSON.stringify({ question: question, mode: "auto", doc_id: null }),
        signal: controller.signal,
      });
      if (!resp.ok || !resp.body) { handleEvent({ type: "error", data: { code: "http", message: "Request failed (" + resp.status + ")." } }); finish(); return; }
      var reader = resp.body.getReader(), dec = new TextDecoder(), buf = "";
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        var idx;
        while ((idx = buf.indexOf("\n\n")) >= 0) {
          var frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (!frame || frame[0] === ":") continue; // heartbeat/comment
          var type = null, data = "";
          frame.split("\n").forEach(function (line) {
            if (line.indexOf("event:") === 0) type = line.slice(6).trim();
            else if (line.indexOf("data:") === 0) data += line.slice(5).trim();
          });
          if (type) { try { handleEvent({ type: type, data: JSON.parse(data || "{}") }); } catch (e) {} }
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") handleEvent({ type: "error", data: { code: "internal", message: "Stream error." } });
    } finally {
      if (state.running) finish();
    }
  }

  function stop() {
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    if (state.abort) { try { state.abort.abort(); } catch (e) {} state.abort = null; }
    state.running = false;
  }

  // ---- wiring ---------------------------------------------------------------
  function pick(runId) {
    stop();
    var run = runsById[runId];
    if (!run) return;
    state.runId = runId;
    $question.textContent = run.question;
    $run.innerHTML = "";
    $ask.textContent = "Ask";
    $ask.disabled = false;
    Array.prototype.forEach.call($picker.children, function (b) {
      b.setAttribute("aria-current", b.dataset.run === runId ? "true" : "false");
    });
  }

  function onAsk() {
    var q = currentQuestion();
    if (hasKeys() && q) { live(q); return; }
    if (state.runId) { replay(state.runId); return; }
  }

  function refreshEditable() {
    var on = !!hasKeys();
    $question.setAttribute("contenteditable", on ? "true" : "false");
    $question.style.caretColor = "var(--color-accent)";
    $ask.disabled = !(state.runId || (on && currentQuestion()));
  }

  function init(runs) {
    runs.forEach(function (r) { runsById[r.id] = r; });
    $picker.innerHTML = "";
    runs.forEach(function (r) {
      var b = document.createElement("button");
      b.className = "btn btn-secondary";
      b.style.textAlign = "left";
      b.textContent = r.question;
      b.dataset.run = r.id;
      b.addEventListener("click", function () { pick(r.id); });
      $picker.appendChild(b);
    });
    $ask.addEventListener("click", onAsk);
    [$akey, $ckey].forEach(function (i) { i.addEventListener("input", refreshEditable); });
    $question.addEventListener("input", refreshEditable);
    $question.addEventListener("focus", function () { if (currentQuestion() === "" && hasKeys()) $question.textContent = ""; });
  }

  fetch("/api/demo")
    .then(function (r) { return r.json(); })
    .then(function (j) { init(j.runs || []); })
    .catch(function () { $picker.innerHTML = '<span class="text-muted" style="font-size:13px;">Demo unavailable — the API isn\'t reachable.</span>'; });
})();
