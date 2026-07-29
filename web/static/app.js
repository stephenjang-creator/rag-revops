/* Deal Desk Helper — demo replay + live SSE, one renderer for both (v2).
 *
 * The recorded runs come from /api/demo; a live run streams from POST /api/ask.
 * Both speak the same event schema, so handleEvent() renders either source —
 * the only difference is transport (the demo replays on a timer, the live path
 * reads frames off the SSE body).
 *
 * v2 copy rule: no scores, no chunk ids, no millisecond timings, no raw CUAD
 * filenames. The API still returns the raw values; we humanize them here —
 * prettyDoc() turns "041__NICELTD_2003-EX-4.5-OUTSOURCING_AGREEMENT" into
 * "Niceltd — Outsourcing Agreement", and stripMarkers() drops inline [n] tags.
 */
(function () {
  "use strict";

  var PACE = 1050; // ms between trace (stage) rows
  var PICK_PLACEHOLDER = "Pick one of the questions above…";
  var EDIT_PLACEHOLDER = "Type your own question…";

  var runsById = {};
  var state = {
    runId: null, running: false, timer: null, abort: null,
    doneStages: 0, route: null, rewrite: null,
  };

  var $question = document.getElementById("dd-question");
  var $ask = document.getElementById("dd-ask");
  var $picker = document.getElementById("dd-picker");
  var $run = document.getElementById("dd-run");
  var $akey = document.getElementById("dd-akey");
  var $ckey = document.getElementById("dd-ckey");
  var $keysToggle = document.getElementById("dd-keys-toggle");
  var $keyFields = document.getElementById("dd-key-fields");
  var $keyAck = document.getElementById("dd-key-ack");
  var $keyChange = document.getElementById("dd-key-change");
  var $keyClear = document.getElementById("dd-key-clear");

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function hasKeys() { return $akey.value.trim() && $ckey.value.trim(); }
  function currentQuestion() {
    if ($question.classList.contains("placeholder")) return "";
    return ($question.textContent || "").trim();
  }
  function setPlaceholder(text) {
    $question.classList.add("placeholder");
    $question.textContent = text;
  }

  // ---- BYO-key panel state machine ------------------------------------------
  // collapsed (a quiet link) -> fields (the two inputs) -> ack (a compact
  // confirmation once both keys are set). Once acknowledged the fields are
  // hidden; "Change" reopens them and "Clear" wipes both and collapses.
  function setKeyState(s) {
    $keysToggle.hidden = s !== "collapsed";
    $keyFields.hidden = s !== "fields";
    $keyAck.hidden = s !== "ack";
  }

  // ---- humanizers (the v2 copy rule) ----------------------------------------
  function humanize(tok) {
    return String(tok || "").split(/[_\-]+/).filter(Boolean)
      .map(function (w) { return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase(); })
      .join(" ");
  }
  function prettyDoc(id) {
    id = String(id == null ? "" : id).trim();
    if (!/^\d+__/.test(id)) return id; // already a readable name (demo data)
    var body = id.replace(/^\d+__/, "");
    var ym = body.match(/(19|20)\d{2}/);
    var party, type = "";
    if (ym) {
      party = body.slice(0, ym.index);
      var tail = body.slice(ym.index + ym[0].length).replace(/^[_\-]+/, "");
      tail = tail.replace(/^EX[-_]?[\d.]+[-_]?/i, "");
      type = humanize(tail);
    } else {
      party = body;
    }
    party = party.replace(/[_\-]+$/, "").replace(/(?:[_\-]\d{1,2}){1,2}$/, "");
    var name = humanize(party);
    return type ? name + " — " + type : name;
  }
  function stripMarkers(s) {
    return String(s == null ? "" : s).replace(/\s*(?:\[\d+(?:\s*,\s*\d+)*\])+/g, "");
  }

  // ---- layout: build the run scaffold once per run --------------------------
  function newRunDom() {
    state.doneStages = 0;
    state.route = null;
    state.rewrite = null;
    $run.innerHTML =
      '<div style="display:grid; gap:32px;">' +
      '  <div id="dd-trace" style="display:grid; gap:14px;"></div>' +
      '  <div id="dd-spinner"></div>' +
      '  <div id="dd-reason"></div>' +
      '  <div id="dd-result"></div>' +
      "</div>";
  }

  function setSpinner(on, label) {
    var slot = document.getElementById("dd-spinner");
    if (!slot) return;
    slot.innerHTML = on
      ? '<div style="display:flex; align-items:center; gap:12px; font-size:15.5px; color:var(--ink-faint);">' +
        '<span class="spinner"></span><span>' + esc(label) + "</span></div>"
      : "";
  }

  // ---- per-event renderers --------------------------------------------------
  function renderStage(d) {
    if (d.state === "running") {
      setSpinner(true, state.doneStages >= 4 ? "Putting the answer together…" : "Working through it…");
      return;
    }
    state.doneStages += 1;
    var trace = document.getElementById("dd-trace");
    if (!trace) return;
    var row = document.createElement("div");
    row.className = "sfin dd-trace-row";
    row.style.cssText = "display:grid; grid-template-columns:132px 1fr; gap:24px; align-items:baseline;";
    row.innerHTML =
      '<span class="label label-accent">' + esc(d.label) + "</span>" +
      '<span style="font-size:16px; color:var(--ink-soft);">' + esc(d.detail) + "</span>";
    trace.appendChild(row);
  }

  // The reasoning card merges the router decision and the query rewrite — they
  // arrive as two events (in either order across demo/live), one card.
  function renderReason() {
    var slot = document.getElementById("dd-reason");
    if (!slot) return;
    var r = state.route, rw = state.rewrite;
    if (!r && !rw) return;
    var typed = (rw && rw.original) || currentQuestion() || (state.runId && runsById[state.runId] && runsById[state.runId].question) || "";
    var html = '<div class="sfin raised-card" style="display:grid; gap:22px;">';
    if (r && r.reason) {
      html +=
        '<div style="display:grid; gap:6px;">' +
        '<span class="label">What it decided you were asking</span>' +
        '<span style="font-size:17px; color:var(--ink);">' + esc(r.reason) + "</span></div>";
    }
    if (rw && rw.rewritten) {
      html += '<div style="height:1px; background:var(--line);"></div>';
      html +=
        '<div class="dd-reason-2col" style="display:grid; grid-template-columns:1fr 1fr; gap:32px;">' +
        '<div style="display:grid; gap:6px;"><span class="label">You typed</span>' +
        '<span style="font-size:16px; color:var(--ink-soft);">' + esc(typed) + "</span></div>" +
        '<div style="display:grid; gap:6px;"><span class="label label-accent">Looked for</span>' +
        '<span style="font-size:16px; color:var(--ink);">' + esc(rw.rewritten) + "</span></div></div>";
    }
    html += "</div>";
    slot.innerHTML = html;
  }

  function renderRoute(d) { state.route = d; renderReason(); }
  function renderRewrite(d) { state.rewrite = d; renderReason(); }

  function sourceLines(items, getName) {
    // Unique, readable "Drawn from" names — no scores, one per line.
    var seen = {}, out = "";
    (items || []).forEach(function (it) {
      var name = prettyDoc(getName(it));
      if (!name || seen[name]) return;
      seen[name] = true;
      out += '<span style="font-size:16px; color:var(--ink-soft);">' + esc(name) + "</span>";
    });
    return out;
  }

  function renderAnswer(d) {
    var slot = document.getElementById("dd-result");
    if (!slot) return;
    if (d.kind === "find") return renderFind(slot, d);

    var body = stripMarkers(d.body_plain != null ? d.body_plain : "");
    var html = '<div class="sfin" style="display:grid; gap:22px;">';
    html += '<h3 style="font-size:30px;">' + esc(d.heading || "Answer") + "</h3>";

    html += '<div class="raised-card" style="display:grid; gap:12px; padding:32px 34px;">';
    if (d.subheading) html += '<span style="font-family:var(--serif); font-size:20px;">' + esc(d.subheading) + "</span>";
    html += '<p style="font-size:17px; line-height:1.7; color:var(--ink);">' + esc(body) + "</p></div>";

    if (d.note) html += '<p style="font-size:16px; color:var(--ink-soft); max-width:42em;"><span style="color:var(--accent-deep);">A note from the tool —</span> ' + esc(d.note) + "</p>";

    // "Drawn from": draft carries `sources`, single carries `citations`.
    var src = d.sources || d.citations || [];
    if (src.length) {
      html += '<div style="display:grid; gap:12px; padding-top:8px;"><span class="label">Drawn from</span>';
      html += sourceLines(src, function (s) { return s.doc_id; });
      if (d.disclaimer) html += '<p style="font-size:14.5px; color:var(--ink-faint); max-width:40em; padding-top:6px;">' + esc(d.disclaimer) + "</p>";
      html += "</div>";
    } else if (d.disclaimer) {
      html += '<p style="font-size:14.5px; color:var(--ink-faint); max-width:40em;">' + esc(d.disclaimer) + "</p>";
    }
    html += "</div>";
    slot.innerHTML = html;
  }

  function renderFind(slot, d) {
    var html = '<div class="sfin" style="display:grid; gap:22px;">';
    html += '<h3 style="font-size:30px;">' + esc(d.summary || "Matching contracts") + "</h3>";
    html += '<div style="display:grid; gap:18px;">';
    (d.findings || []).forEach(function (f) {
      var name = prettyDoc(f.doc_id || f.doc);
      html += '<div class="raised-card" style="display:grid; gap:8px; padding:24px 28px;">' +
        '<div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">' +
        '<span style="font-family:var(--serif); font-size:20px;">' + esc(name) + "</span>" +
        (f.tag ? '<span class="chip">' + esc(f.tag) + "</span>" : "") + "</div>" +
        '<span style="font-size:16px; color:var(--ink-soft);">' + esc(f.reason) + "</span></div>";
    });
    html += "</div>";
    if (d.footer) html += '<p style="font-size:16px; color:var(--ink-soft); max-width:40em;">' + esc(d.footer) + "</p>";
    html += "</div>";
    slot.innerHTML = html;
  }

  function renderDecline(d) {
    var slot = document.getElementById("dd-result");
    if (!slot) return;
    slot.innerHTML =
      '<div class="sfin" style="display:grid; gap:18px;">' +
      '<div class="raised-card" style="padding:30px 34px;">' +
      '<span style="font-family:var(--serif); font-size:25px;">' + esc(d.reason) + "</span></div>" +
      '<p style="font-size:17px; color:var(--ink-soft); max-width:40em;">' + esc(d.why) + "</p></div>";
  }

  function renderError(d) {
    if (d.code === "busy" && state.runId) { replay(state.runId); return; } // fall back to the recorded run
    var slot = document.getElementById("dd-result");
    if (slot) slot.innerHTML = '<div class="raised-card" style="padding:24px 28px; font-size:16px; color:var(--ink-soft);">' + esc(d.message) + "</div>";
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "stage": renderStage(ev.data); break;
      case "route": renderRoute(ev.data); break;
      case "rewrite": renderRewrite(ev.data); break;
      case "answer": renderAnswer(ev.data); break;
      case "decline": renderDecline(ev.data); break;
      case "error": renderError(ev.data); break;
      case "retrieval": case "judge": break; // informational; stage/answer rows cover these
      case "done": finish(); break;
    }
  }

  function finish() {
    state.running = false;
    setSpinner(false, "");
    $ask.textContent = "Run it again";
    $ask.disabled = false;
  }

  // ---- demo replay ----------------------------------------------------------
  function replay(runId) {
    stop();
    var run = runsById[runId];
    if (!run) return;
    state.runId = runId;
    state.running = true;
    newRunDom();
    setSpinner(true, "Working through it…");
    $ask.textContent = "Working…";
    $ask.disabled = true;
    var evs = run.events, i = 0, first = true;
    function tick() {
      if (i >= evs.length) return;
      var ev = evs[i++];
      handleEvent(ev);
      if (i < evs.length) {
        var delay = ev.type === "stage" ? (first ? PACE * 0.7 : PACE) : 0;
        if (ev.type === "stage") first = false;
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
    newRunDom();
    setSpinner(true, "Working through it…");
    $ask.textContent = "Working…";
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
    $question.classList.remove("placeholder");
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
    // Swap the placeholder copy to match what's possible right now.
    if ($question.classList.contains("placeholder")) {
      setPlaceholder(on ? EDIT_PLACEHOLDER : PICK_PLACEHOLDER);
    }
    $ask.disabled = !(state.runId || (on && currentQuestion()));
  }

  function onKeyInput() {
    refreshEditable();
    if (!$keyFields.hidden && hasKeys()) setKeyState("ack"); // both entered → collapse
  }

  function init(runs) {
    runs.forEach(function (r) { runsById[r.id] = r; });
    $picker.innerHTML = "";
    runs.forEach(function (r) {
      var b = document.createElement("button");
      b.className = "qpick";
      b.textContent = r.question;
      b.dataset.run = r.id;
      b.setAttribute("aria-current", "false");
      b.addEventListener("click", function () { pick(r.id); });
      $picker.appendChild(b);
    });
    $ask.addEventListener("click", onAsk);

    // BYO-key panel wiring.
    $keysToggle.addEventListener("click", function () { setKeyState("fields"); $akey.focus(); });
    $keyChange.addEventListener("click", function () { setKeyState("fields"); $akey.focus(); });
    $keyClear.addEventListener("click", function () {
      $akey.value = ""; $ckey.value = "";
      refreshEditable();
      setKeyState("collapsed");
    });
    [$akey, $ckey].forEach(function (i) { i.addEventListener("input", onKeyInput); });
    setKeyState(hasKeys() ? "ack" : "collapsed"); // in case the browser autofilled

    $question.addEventListener("input", refreshEditable);
    $question.addEventListener("focus", function () {
      if (hasKeys() && $question.classList.contains("placeholder")) {
        $question.classList.remove("placeholder");
        $question.textContent = "";
      }
    });
    $question.addEventListener("blur", function () {
      if (hasKeys() && ($question.textContent || "").trim() === "") {
        setPlaceholder(EDIT_PLACEHOLDER);
        refreshEditable();
      }
    });
  }

  fetch("/api/demo")
    .then(function (r) { return r.json(); })
    .then(function (j) { init(j.runs || []); })
    .catch(function () { $picker.innerHTML = '<span style="font-size:15px; color:var(--ink-faint);">Demo unavailable — the API isn\'t reachable.</span>'; });
})();
