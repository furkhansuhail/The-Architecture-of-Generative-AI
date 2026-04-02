"""
Self-contained HTML for the RAG Design Patterns interactive visual.
Covers all 7 patterns: Naive RAG, Retrieve-and-Rerank, Multimodal RAG,
Graph RAG, Hybrid RAG, Agentic RAG (Router), and Multi-Agent RAG.
Embed in Streamlit via st.components.v1.html(RAG_VISUAL_HTML, height=RAG_VISUAL_HEIGHT).
"""

RAG_VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; overflow-x: hidden; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes flow { 0%{stroke-dashoffset:24} 100%{stroke-dashoffset:0} }
  .fade-in { animation: fadeIn 0.3s ease forwards; }
  #loading {
    position: fixed; inset: 0;
    display: flex; align-items: center; justify-content: center;
    background: #0a0a0f; color: #71717a;
    font-family: monospace; font-size: 13px; z-index: 999;
  }
  #error-banner {
    display: none; padding: 20px; color: #ef4444;
    background: #12121a; border: 1px solid #ef4444;
    font-family: monospace; font-size: 12px; margin: 20px; border-radius: 8px;
  }
</style>
</head>
<body>
<div id="loading">\u23f3 Loading RAG Design Patterns…</div>
<div id="error-banner"></div>
<div id="root"></div>
<script>
window.onerror = function(msg, src) {
  document.getElementById('loading').style.display = 'none';
  var b = document.getElementById('error-banner');
  b.style.display = 'block';
  b.innerText = '\u274c Script load error: ' + msg + (src ? ' (' + src + ')' : '');
};
</script>
<script>
var useState = React.useState;
var useEffect = React.useEffect;
var C = {
  bg: "#0a0a0f",
  card: "#12121a",
  border: "#1e1e2e",
  accent: "#ff6b35",
  blue: "#4ecdc4",
  purple: "#a78bfa",
  yellow: "#fbbf24",
  text: "#e4e4e7",
  muted: "#71717a",
  dim: "#3f3f46",
  red: "#ef4444",
  green: "#4ade80",
  cyan: "#38bdf8",
  pink: "#f472b6",
  orange: "#fb923c",
  teal: "#2dd4bf",
  indigo: "#818cf8",
  lime: "#a3e635"
};
var ARR = "\\\\u2192";
var DASH = "\\\\u2014";
var CHK = "\\\\u2713";
var BULB = "\\\\uD83D\\\\uDCA1";
var WARN = "\\\\u26A0";
var DOC = "\\\\uD83D\\\\uDCCB";
var DB = "\\\\uD83D\\\\uDDC4";
var ROBOT = "\\\\uD83E\\\\uDD16";
var SEARCH = "\\\\uD83D\\\\uDD0D";
var GRAPH = "\\\\uD83D\\\\uDD78";
var MULTI = "\\\\uD83E\\\\uDDE9";

/* ── Shared components ── */
function TabBar(props) {
  var tabs = props.tabs,
    active = props.active,
    onChange = props.onChange;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 0,
      borderBottom: "2px solid " + C.border,
      marginBottom: 20,
      overflowX: "auto",
      flexWrap: "nowrap"
    }
  }, tabs.map(function (t, i) {
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      onClick: function () {
        onChange(i);
      },
      style: {
        padding: "10px 14px",
        background: "none",
        border: "none",
        borderBottom: active === i ? "2px solid " + C.accent : "2px solid transparent",
        color: active === i ? C.accent : C.muted,
        cursor: "pointer",
        fontSize: 10,
        fontWeight: 700,
        fontFamily: "'JetBrains Mono',monospace",
        transition: "all 0.2s",
        whiteSpace: "nowrap",
        marginBottom: -2
      }
    }, t);
  }));
}
function Card(props) {
  return /*#__PURE__*/React.createElement("div", {
    style: Object.assign({
      background: C.card,
      borderRadius: 10,
      padding: "16px 20px",
      border: "1px solid " + (props.color || C.border),
      transition: "border 0.3s"
    }, props.style || {})
  }, props.children);
}
function Insight(props) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 820,
      margin: "14px auto 0",
      padding: "12px 18px",
      background: "rgba(255,107,53,0.06)",
      borderRadius: 10,
      border: "1px solid rgba(255,107,53,0.2)"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.accent,
      marginBottom: 4
    }
  }, BULB + " " + (props.title || "Key Insight")), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, props.children));
}
function Badge(props) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: 4,
      margin: "2px 3px 2px 0",
      background: (props.color || C.accent) + "22",
      border: "1px solid " + (props.color || C.accent) + "50",
      color: props.color || C.accent,
      fontSize: 9,
      fontWeight: 700,
      fontFamily: "monospace"
    }
  }, props.children);
}

/* ── Node helper for pipeline diagrams ── */
function PNode(props) {
  var x = props.x,
    y = props.y,
    w = props.w || 100,
    h = props.h || 40;
  var color = props.color || C.blue,
    label = props.label,
    sub = props.sub;
  var icon = props.icon || "";
  return /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("rect", {
    x: x,
    y: y,
    width: w,
    height: h,
    rx: 7,
    fill: color + "20",
    stroke: color,
    strokeWidth: 1.5
  }), /*#__PURE__*/React.createElement("text", {
    x: x + w / 2,
    y: y + (sub ? h / 2 - 5 : h / 2 + 1),
    textAnchor: "middle",
    fill: color,
    fontSize: 10,
    fontWeight: 700,
    fontFamily: "monospace"
  }, icon && /*#__PURE__*/React.createElement("tspan", null, icon + " "), label), sub && /*#__PURE__*/React.createElement("text", {
    x: x + w / 2,
    y: y + h / 2 + 9,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 8,
    fontFamily: "monospace"
  }, sub));
}
function Arrow(props) {
  var x1 = props.x1,
    y1 = props.y1,
    x2 = props.x2,
    y2 = props.y2;
  var color = props.color || C.dim,
    dashed = props.dashed;
  var mid = props.mid;
  var d;
  if (mid) {
    d = "M" + x1 + "," + y1 + " L" + mid[0] + "," + mid[1] + " L" + x2 + "," + y2;
  } else {
    d = "M" + x1 + "," + y1 + " L" + x2 + "," + y2;
  }
  return /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("marker", {
    id: "arr-" + props.id,
    viewBox: "0 0 10 10",
    refX: "8",
    refY: "5",
    markerWidth: "5",
    markerHeight: "5",
    orient: "auto-start-reverse"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M2 2L8 5L2 8",
    fill: "none",
    stroke: color,
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))), /*#__PURE__*/React.createElement("path", {
    d: d,
    fill: "none",
    stroke: color,
    strokeWidth: 1.5,
    strokeDasharray: dashed ? "6,4" : "none",
    markerEnd: "url(#arr-" + props.id + ")",
    style: props.animated ? {
      animation: "flow 0.8s linear infinite"
    } : {}
  }), props.label && /*#__PURE__*/React.createElement("text", {
    x: (x1 + x2) / 2,
    y: (y1 + y2) / 2 - 5,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 8,
    fontFamily: "monospace"
  }, props.label));
}

/* ================================================================
   TAB 1: NAIVE RAG
   ================================================================ */
function TabNaive() {
  var _s = useState(0);
  var step = _s[0];
  var setStep = _s[1];
  var steps = [{
    t: "Documents chunked",
    d: "Raw documents are split into fixed-size overlapping chunks (e.g. 512 tokens, 50-token overlap). Each chunk is embedded into a dense vector and stored in the vector database. This happens offline — once, before any queries arrive."
  }, {
    t: "Query arrives",
    d: "A user query is received. The same embedding model used at index time converts the query text into a dense vector. This embedding captures the semantic meaning of the query."
  }, {
    t: "Similarity search",
    d: "The query vector is compared against all chunk vectors using cosine similarity (or dot product). The top-K most similar chunks are retrieved. K is typically 3–10."
  }, {
    t: "Prompt assembly",
    d: "The retrieved chunks are injected into a prompt template alongside the original query. The template instructs the LLM to answer only using the provided context."
  }, {
    t: "LLM generates response",
    d: "The assembled prompt is sent to a generative LLM. The model produces a grounded response citing facts from the retrieved context, reducing hallucination."
  }];
  var animated = step >= 1;
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.accent
    }
  }, "Naive RAG \\u2014 The Baseline"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Single-stage vector similarity retrieval ", DASH, " fast, simple, and the right starting point")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "820",
    height: "200",
    viewBox: "0 0 820 200",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement("text", {
    x: 200,
    y: 20,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "OFFLINE INDEXING"), /*#__PURE__*/React.createElement(PNode, {
    x: 20,
    y: 30,
    w: 90,
    h: 40,
    color: C.blue,
    label: "Documents",
    icon: DOC
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n1",
    x1: 112,
    y1: 50,
    x2: 135,
    y2: 50,
    color: C.dim,
    dashed: true
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 137,
    y: 30,
    w: 90,
    h: 40,
    color: C.cyan,
    label: "Chunks",
    sub: "512 tok"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n2",
    x1: 229,
    y1: 50,
    x2: 252,
    y2: 50,
    color: step >= 0 ? C.cyan : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 254,
    y: 30,
    w: 100,
    h: 40,
    color: C.purple,
    label: "Embed Model",
    sub: "bi-encoder"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n3",
    x1: 356,
    y1: 50,
    x2: 379,
    y2: 50,
    color: step >= 0 ? C.purple : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 381,
    y: 30,
    w: 100,
    h: 40,
    color: C.green,
    label: "Vector DB",
    icon: DB
  }), /*#__PURE__*/React.createElement("text", {
    x: 200,
    y: 115,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "ONLINE RETRIEVAL"), /*#__PURE__*/React.createElement(PNode, {
    x: 20,
    y: 125,
    w: 90,
    h: 40,
    color: C.yellow,
    label: "Query"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n4",
    x1: 112,
    y1: 145,
    x2: 252,
    y2: 145,
    color: step >= 1 ? C.yellow : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 254,
    y: 125,
    w: 100,
    h: 40,
    color: C.purple,
    label: "Embed Model",
    sub: "same model"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n5",
    x1: 356,
    y1: 145,
    x2: 379,
    y2: 145,
    color: step >= 2 ? C.purple : C.dim
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n6",
    x1: 431,
    y1: 72,
    x2: 431,
    y2: 123,
    color: step >= 2 ? C.green : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 381,
    y: 125,
    w: 100,
    h: 40,
    color: C.green,
    label: "Top-K Context",
    sub: "cosine sim"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n7",
    x1: 483,
    y1: 145,
    x2: 506,
    y2: 145,
    color: step >= 3 ? C.green : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 508,
    y: 125,
    w: 110,
    h: 40,
    color: C.orange,
    label: "Prompt Template"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n8",
    x1: 620,
    y1: 145,
    x2: 643,
    y2: 145,
    color: step >= 4 ? C.orange : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 645,
    y: 125,
    w: 90,
    h: 40,
    color: C.pink,
    label: "LLM",
    icon: ROBOT
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "n9",
    x1: 737,
    y1: 145,
    x2: 760,
    y2: 145,
    color: step >= 4 ? C.pink : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 762,
    y: 125,
    w: 50,
    h: 40,
    color: C.accent,
    label: "Ans"
  }), [{
    x: 20,
    y: 30,
    w: 90,
    h: 40
  }, {
    x: 20,
    y: 125,
    w: 90,
    h: 40
  }, {
    x: 381,
    y: 30,
    w: 100,
    h: 40
  }, {
    x: 508,
    y: 125,
    w: 110,
    h: 40
  }, {
    x: 645,
    y: 125,
    w: 90,
    h: 40
  }].map(function (b, i) {
    if (i !== step) return null;
    return /*#__PURE__*/React.createElement("rect", {
      key: i,
      x: b.x - 2,
      y: b.y - 2,
      width: b.w + 4,
      height: b.h + 4,
      rx: 9,
      fill: "none",
      stroke: C.accent,
      strokeWidth: 2.5,
      style: {
        animation: "pulse 1.2s infinite"
      }
    });
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 6,
      justifyContent: "center",
      marginBottom: 14
    }
  }, steps.map(function (_, i) {
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      onClick: function () {
        setStep(i);
      },
      style: {
        width: 32,
        height: 32,
        borderRadius: 6,
        border: "1.5px solid " + (step === i ? C.accent : C.border),
        background: step === i ? C.accent + "25" : C.card,
        color: step === i ? C.accent : C.muted,
        cursor: "pointer",
        fontSize: 10,
        fontWeight: 700,
        fontFamily: "monospace"
      }
    }, i + 1);
  })), /*#__PURE__*/React.createElement(Card, {
    color: C.accent,
    style: {
      maxWidth: 820,
      margin: "0 auto 14px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 800,
      color: C.accent,
      marginBottom: 4
    }
  }, step + 1 + ". " + steps[step].t), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, steps[step].d)), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 10,
      justifyContent: "center",
      flexWrap: "wrap",
      marginBottom: 4
    }
  }, [{
    l: "Latency",
    v: "80–200ms",
    c: C.green
  }, {
    l: "Complexity",
    v: "Low",
    c: C.green
  }, {
    l: "Cost/1K queries",
    v: "$0.02",
    c: C.green
  }, {
    l: "Precision",
    v: "65%",
    c: C.yellow
  }, {
    l: "Best for",
    v: "Prototypes",
    c: C.blue
  }].map(function (s, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        textAlign: "center",
        background: C.card,
        borderRadius: 8,
        padding: "8px 14px",
        border: "1px solid " + C.border
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 9,
        color: C.muted,
        marginBottom: 3
      }
    }, s.l), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 13,
        fontWeight: 700,
        color: s.c
      }
    }, s.v));
  })), /*#__PURE__*/React.createElement(Insight, {
    title: "When to start here"
  }, "Always begin with Naive RAG. It costs 1 day to build and sets your ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.accent,
      fontWeight: 700
    }
  }, "performance baseline"), ". Measure precision, recall, and latency. Only add complexity (reranking, graphs, agents) where measurements show clear failure."));
}

/* ================================================================
   TAB 2: RETRIEVE AND RERANK
   ================================================================ */
function TabRerank() {
  var _s = useState(0);
  var stage = _s[0];
  var setStage = _s[1];
  var stages = [{
    t: "Stage 1 — Broad Recall (N=50)",
    desc: "Fast ANN vector search retrieves the top-50 candidate chunks using a bi-encoder. Optimised for RECALL — cast a wide net, don't worry about false positives yet.",
    color: C.blue
  }, {
    t: "Stage 2 — Cross-Encoder Reranking",
    desc: "A cross-encoder model scores each (query, chunk) pair jointly. Unlike bi-encoders, it sees BOTH query and document simultaneously through full bidirectional attention — far more accurate. Returns top-5.",
    color: C.purple
  }, {
    t: "Generation with High-Quality Context",
    desc: "Only the top-5 reranked chunks enter the prompt. Higher signal-to-noise ratio means the LLM generates more accurate, grounded responses with fewer irrelevant facts.",
    color: C.green
  }];
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.purple
    }
  }, "Retrieve-and-Rerank \\u2014 Two-Stage Precision"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Bi-encoder for speed, cross-encoder for accuracy ", DASH, " the best precision pipeline for text")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 10,
      justifyContent: "center",
      marginBottom: 16
    }
  }, stages.map(function (s, i) {
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      onClick: function () {
        setStage(i);
      },
      style: {
        padding: "8px 16px",
        borderRadius: 8,
        border: "1.5px solid " + (stage === i ? s.color : C.border),
        background: stage === i ? s.color + "20" : C.card,
        color: stage === i ? s.color : C.muted,
        cursor: "pointer",
        fontSize: 10,
        fontWeight: 700,
        fontFamily: "monospace"
      }
    }, "Stage " + (i + 1));
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "center",
      marginBottom: 14
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "820",
    height: "160",
    viewBox: "0 0 820 160",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 20,
    w: 90,
    h: 40,
    color: C.blue,
    label: "Documents",
    icon: DOC
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r0",
    x1: 102,
    y1: 40,
    x2: 120,
    y2: 40,
    color: C.dim,
    dashed: true
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 122,
    y: 20,
    w: 80,
    h: 40,
    color: C.cyan,
    label: "Chunks"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r1",
    x1: 204,
    y1: 40,
    x2: 222,
    y2: 40,
    color: C.dim,
    dashed: true
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 224,
    y: 20,
    w: 100,
    h: 40,
    color: C.purple,
    label: "Vector DB",
    icon: DB
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 100,
    w: 90,
    h: 40,
    color: C.yellow,
    label: "Query"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r2",
    x1: 102,
    y1: 120,
    x2: 222,
    y2: 120,
    color: stage >= 0 ? C.yellow : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 224,
    y: 100,
    w: 100,
    h: 40,
    color: C.blue,
    label: "Embed Model"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r3",
    x1: 274,
    y1: 62,
    x2: 274,
    y2: 98,
    color: stage >= 0 ? C.blue : C.dim
  }), /*#__PURE__*/React.createElement("text", {
    x: 282,
    y: 84,
    fill: C.muted,
    fontSize: 8,
    fontFamily: "monospace"
  }, "top-50"), /*#__PURE__*/React.createElement(Arrow, {
    id: "r4",
    x1: 326,
    y1: 120,
    x2: 350,
    y2: 120,
    color: stage >= 0 ? C.blue : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 352,
    y: 100,
    w: 110,
    h: 40,
    color: C.blue,
    label: "Retrieved",
    sub: "50 candidates"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r5",
    x1: 464,
    y1: 120,
    x2: 488,
    y2: 120,
    color: stage >= 1 ? C.blue : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 490,
    y: 100,
    w: 110,
    h: 40,
    color: C.purple,
    label: "Reranker",
    sub: "cross-encoder"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r6",
    x1: 602,
    y1: 120,
    x2: 626,
    y2: 120,
    color: stage >= 1 ? C.purple : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 628,
    y: 100,
    w: 90,
    h: 40,
    color: C.green,
    label: "Top-5",
    sub: "re-ranked"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "r7",
    x1: 720,
    y1: 120,
    x2: 744,
    y2: 120,
    color: stage >= 2 ? C.green : C.dim
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 746,
    y: 100,
    w: 60,
    h: 40,
    color: C.accent,
    label: "LLM",
    icon: ROBOT
  }), stage === 0 && /*#__PURE__*/React.createElement("rect", {
    x: 350,
    y: 98,
    width: 116,
    height: 44,
    rx: 9,
    fill: "none",
    stroke: C.blue,
    strokeWidth: 2.5,
    style: {
      animation: "pulse 1.2s infinite"
    }
  }), stage === 1 && /*#__PURE__*/React.createElement("rect", {
    x: 488,
    y: 98,
    width: 116,
    height: 44,
    rx: 9,
    fill: "none",
    stroke: C.purple,
    strokeWidth: 2.5,
    style: {
      animation: "pulse 1.2s infinite"
    }
  }), stage === 2 && /*#__PURE__*/React.createElement("rect", {
    x: 626,
    y: 98,
    width: 96,
    height: 44,
    rx: 9,
    fill: "none",
    stroke: C.green,
    strokeWidth: 2.5,
    style: {
      animation: "pulse 1.2s infinite"
    }
  }))), /*#__PURE__*/React.createElement(Card, {
    color: stages[stage].color,
    style: {
      maxWidth: 820,
      margin: "0 auto 14px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 800,
      color: stages[stage].color,
      marginBottom: 4
    }
  }, stages[stage].t), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, stages[stage].desc)), /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 820,
      margin: "0 auto",
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Card, {
    color: C.blue
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.blue,
      marginBottom: 6
    }
  }, "Bi-Encoder (Stage 1)"), ["Query embeds once — O(1) per query", "Doc embeddings pre-computed offline", "Cosine similarity — extremely fast", "Lower accuracy (separate embeddings)", "Optimise for RECALL — wide net"].map(function (t, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 10,
        color: C.muted,
        marginBottom: 3
      }
    }, "+ " + t);
  })), /*#__PURE__*/React.createElement(Card, {
    color: C.purple
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.purple,
      marginBottom: 6
    }
  }, "Cross-Encoder (Stage 2)"), ["Sees query+doc together — rich interaction", "Cannot be pre-indexed (O(N) per query)", "Full bidirectional attention across both", "Much higher accuracy than bi-encoder", "Optimise for PRECISION — tight filter"].map(function (t, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 10,
        color: C.muted,
        marginBottom: 3
      }
    }, "+ " + t);
  }))), /*#__PURE__*/React.createElement(Insight, {
    title: "Rule of thumb"
  }, "Retrieve ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.blue,
      fontWeight: 700
    }
  }, "N=50"), " with bi-encoder (fast, cheap), rerank to ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.purple,
      fontWeight: 700
    }
  }, "K=5"), " with cross-encoder (slow, accurate). This two-stage design is the ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.accent,
      fontWeight: 700
    }
  }, "highest ROI upgrade"), " from Naive RAG \\u2014 adds ~200ms but 20-30% precision improvement."));
}

/* ================================================================
   TAB 3: MULTIMODAL RAG
   ================================================================ */
function TabMultimodal() {
  var modalities = [{
    name: "Text",
    icon: "\\\\uD83D\\\\uDCDD",
    color: C.blue,
    embed: "Text Embedder",
    example: "Product descriptions, manuals"
  }, {
    name: "Images",
    icon: "\\\\uD83D\\\\uDDBC",
    color: C.purple,
    embed: "CLIP/ViT",
    example: "Product photos, diagrams, charts"
  }, {
    name: "Audio",
    icon: "\\\\uD83C\\\\uDFA4",
    color: C.orange,
    embed: "Wav2Vec/Whisper",
    example: "Podcasts, call recordings"
  }, {
    name: "Video",
    icon: "\\\\uD83C\\\\uDFA5",
    color: C.pink,
    embed: "Video CLIP",
    example: "Tutorial videos, recordings"
  }];
  var _s = useState(0);
  var sel = _s[0];
  var setSel = _s[1];
  var mod = modalities[sel];
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.pink
    }
  }, "Multimodal RAG \\u2014 Beyond Text"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Cross-modal embedding (CLIP, ImageBind) maps all modalities into one shared vector space")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 8,
      justifyContent: "center",
      marginBottom: 16
    }
  }, modalities.map(function (m, i) {
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      onClick: function () {
        setSel(i);
      },
      style: {
        padding: "8px 14px",
        borderRadius: 8,
        border: "1.5px solid " + (sel === i ? m.color : C.border),
        background: sel === i ? m.color + "20" : C.card,
        color: sel === i ? m.color : C.muted,
        cursor: "pointer",
        fontSize: 11,
        fontFamily: "monospace",
        fontWeight: 700
      }
    }, m.icon + " " + m.name);
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "center",
      marginBottom: 14
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "820",
    height: "220",
    viewBox: "0 0 820 220",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement("text", {
    x: 80,
    y: 18,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "MULTIMODAL DOCS"), modalities.map(function (m, i) {
    return /*#__PURE__*/React.createElement("g", {
      key: i
    }, /*#__PURE__*/React.createElement(PNode, {
      x: 10,
      y: 28 + i * 44,
      w: 140,
      h: 36,
      color: m.color,
      label: m.icon + " " + m.name + " content"
    }), /*#__PURE__*/React.createElement(Arrow, {
      id: "mm" + i,
      x1: 152,
      y1: 46 + i * 44,
      x2: 230,
      y2: 100,
      color: m.color
    }));
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 233,
    y: 70,
    w: 140,
    h: 60,
    color: C.yellow,
    label: "Multimodal",
    sub: "Embedding Model"
  }), /*#__PURE__*/React.createElement("text", {
    x: 303,
    y: 145,
    textAnchor: "middle",
    fill: C.yellow,
    fontSize: 8,
    fontFamily: "monospace"
  }, "CLIP / ImageBind"), /*#__PURE__*/React.createElement(Arrow, {
    id: "mm-vec",
    x1: 375,
    y1: 100,
    x2: 430,
    y2: 100,
    color: C.yellow
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 433,
    y: 70,
    w: 130,
    h: 60,
    color: C.green,
    label: "Shared Vector",
    sub: "Space"
  }), /*#__PURE__*/React.createElement("text", {
    x: 498,
    y: 148,
    textAnchor: "middle",
    fill: C.green,
    fontSize: 8,
    fontFamily: "monospace"
  }, "All modalities together"), /*#__PURE__*/React.createElement(Arrow, {
    id: "mm-db",
    x1: 565,
    y1: 100,
    x2: 610,
    y2: 100,
    color: C.green
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 613,
    y: 70,
    w: 100,
    h: 60,
    color: C.cyan,
    label: "Vector DB",
    icon: DB
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "mm-media",
    x1: 715,
    y1: 100,
    x2: 750,
    y2: 100,
    color: C.cyan
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 753,
    y: 70,
    w: 60,
    h: 60,
    color: C.pink,
    label: "Media",
    sub: "store"
  }), /*#__PURE__*/React.createElement("text", {
    x: 80,
    y: 190,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "CROSS-MODAL QUERY"), /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 197,
    w: 140,
    h: 36,
    color: C.yellow,
    label: "\\\\\\\\uD83D\\\\\\\\uDCDD Text query finds \\\\\\\\uD83D\\\\\\\\uDDBC images"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "mm-q",
    x1: 152,
    y1: 215,
    x2: 433,
    y2: 130,
    color: C.yellow,
    dashed: true
  }))), /*#__PURE__*/React.createElement(Card, {
    color: mod.color,
    style: {
      maxWidth: 820,
      margin: "0 auto 12px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 800,
      color: mod.color,
      marginBottom: 6
    }
  }, mod.icon + " " + mod.name + " in Multimodal RAG"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 4
    }
  }, "EMBEDDING MODEL"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      fontWeight: 700,
      color: mod.color,
      fontFamily: "monospace"
    }
  }, mod.embed)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 4
    }
  }, "EXAMPLE USES"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.text
    }
  }, mod.example)))), /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 820,
      margin: "0 auto",
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10,
      marginBottom: 4
    }
  }, /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.pink,
      marginBottom: 6
    }
  }, "Key Models"), [["CLIP", "Text + Images (OpenAI)"], ["ImageBind", "6 modalities (Meta)"], ["Voyage Multimodal", "PDFs + diagrams"], ["E5-V", "Text + visual reasoning"]].map(function (r, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        display: "flex",
        justifyContent: "space-between",
        marginBottom: 4
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.cyan,
        fontFamily: "monospace"
      }
    }, r[0]), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.muted
      }
    }, r[1]));
  })), /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.pink,
      marginBottom: 6
    }
  }, "When to Use"), ["PDFs with charts and diagrams", "Product catalogs with images", "Video Q&A (frame + transcript)", "Medical imaging + clinical notes", "Technical manuals with figures"].map(function (t, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 10,
        color: C.muted,
        marginBottom: 3
      }
    }, CHK + " " + t);
  }))), /*#__PURE__*/React.createElement(Insight, {
    title: "Caption generation trick"
  }, "At index time, run a vision LLM on every image to generate a text caption. Store both the image embedding ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.purple,
      fontWeight: 700
    }
  }, "and"), " the caption embedding. Text queries now match images via caption \\u2014 ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.accent,
      fontWeight: 700
    }
  }, "dramatically improving recall"), " without requiring users to submit image queries."));
}

/* ================================================================
   TAB 4: GRAPH RAG
   ================================================================ */
function TabGraph() {
  var _s = useState(null);
  var sel = _s[0];
  var setSel = _s[1];
  var nodes = [{
    id: "RAG",
    x: 300,
    y: 80,
    color: C.accent
  }, {
    id: "Vector DB",
    x: 160,
    y: 160,
    color: C.blue
  }, {
    id: "Knowledge Graph",
    x: 440,
    y: 160,
    color: C.purple
  }, {
    id: "LLM",
    x: 100,
    y: 260,
    color: C.pink
  }, {
    id: "Entities",
    x: 370,
    y: 260,
    color: C.orange
  }, {
    id: "Relationships",
    x: 520,
    y: 260,
    color: C.yellow
  }, {
    id: "GraphRAG",
    x: 300,
    y: 350,
    color: C.green
  }, {
    id: "Multi-hop Q",
    x: 440,
    y: 430,
    color: C.cyan
  }];
  var edges = [["RAG", "Vector DB", "uses", C.blue], ["RAG", "Knowledge Graph", "uses", C.purple], ["Knowledge Graph", "Entities", "contains", C.orange], ["Knowledge Graph", "Relationships", "contains", C.yellow], ["GraphRAG", "Multi-hop Q", "improves", C.cyan], ["GraphRAG", "Knowledge Graph", "uses", C.green], ["LLM", "Knowledge Graph", "extracts", C.pink], ["LLM", "RAG", "powers", C.pink]];
  function getNode(id) {
    return nodes.find(function (n) {
      return n.id === id;
    });
  }
  var info = {
    "RAG": {
      desc: "Retrieval-Augmented Generation — the parent framework. Graph RAG extends it with structured knowledge."
    },
    "Vector DB": {
      desc: "Stores chunk embeddings for semantic similarity search. Handles the 'what does this text mean' queries."
    },
    "Knowledge Graph": {
      desc: "Stores entities (nodes) and relationships (edges) extracted by an LLM from all documents. Handles 'how are things related' queries."
    },
    "LLM": {
      desc: "Used both to EXTRACT the knowledge graph from raw text (indexing) and to GENERATE the final response (query time). Two distinct roles."
    },
    "Entities": {
      desc: "Named things in the knowledge graph: people, organisations, products, drugs, places. Extracted by an LLM from document text."
    },
    "Relationships": {
      desc: "Typed edges connecting entities: (Apple, FOUNDED_BY, Steve Jobs), (Drug X, INHIBITS, Enzyme Y). The structural knowledge."
    },
    "GraphRAG": {
      desc: "Microsoft's 2024 system. Builds a hierarchical community graph. 3-6x better than Naive RAG on multi-hop questions."
    },
    "Multi-hop Q": {
      desc: "Questions requiring traversal across multiple entities: 'Who leads the team that built the product acquired by X?' Vector search alone cannot answer this."
    }
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.purple
    }
  }, "Graph RAG ", DASH, " Structured Knowledge"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Knowledge graph + vector search ", DASH, " click any node to explore. Answers multi-hop relational questions.")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 16,
      alignItems: "flex-start",
      flexWrap: "wrap"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      flex: "0 0 auto"
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: 560,
    height: 480,
    viewBox: "0 0 560 480",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, edges.map(function (e, i) {
    var n1 = getNode(e[0]);
    var n2 = getNode(e[1]);
    if (!n1 || !n2) return null;
    return /*#__PURE__*/React.createElement("g", {
      key: i
    }, /*#__PURE__*/React.createElement("line", {
      x1: n1.x,
      y1: n1.y,
      x2: n2.x,
      y2: n2.y,
      stroke: e[3],
      strokeWidth: 1.5,
      strokeOpacity: 0.5,
      strokeDasharray: "5,3"
    }), /*#__PURE__*/React.createElement("text", {
      x: (n1.x + n2.x) / 2,
      y: (n1.y + n2.y) / 2 - 5,
      textAnchor: "middle",
      fill: e[3],
      fontSize: 8,
      fontFamily: "monospace",
      opacity: 0.8
    }, e[2]));
  }), nodes.map(function (n) {
    var active = sel === n.id;
    return /*#__PURE__*/React.createElement("g", {
      key: n.id,
      onClick: function () {
        setSel(active ? null : n.id);
      },
      style: {
        cursor: "pointer"
      }
    }, /*#__PURE__*/React.createElement("circle", {
      cx: n.x,
      cy: n.y,
      r: active ? 30 : 22,
      fill: active ? n.color + "40" : "#0d0d14",
      stroke: n.color,
      strokeWidth: active ? 3 : 1.5,
      style: {
        transition: "all 0.2s",
        filter: active ? "drop-shadow(0 0 10px " + n.color + "80)" : "none"
      }
    }), /*#__PURE__*/React.createElement("text", {
      x: n.x,
      y: n.y + 4,
      textAnchor: "middle",
      fill: n.color,
      fontSize: 9,
      fontWeight: 700,
      fontFamily: "monospace"
    }, n.id));
  }), /*#__PURE__*/React.createElement("text", {
    x: 280,
    y: 465,
    textAnchor: "middle",
    fill: C.dim,
    fontSize: 8,
    fontFamily: "monospace"
  }, "click a node to learn more"))), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      minWidth: 220
    }
  }, sel ? /*#__PURE__*/React.createElement(Card, {
    color: getNode(sel).color,
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 14,
      fontWeight: 800,
      color: getNode(sel).color,
      marginBottom: 8
    }
  }, sel), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, info[sel] ? info[sel].desc : "")) : /*#__PURE__*/React.createElement(Card, {
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      fontWeight: 700,
      color: C.text,
      marginBottom: 8
    }
  }, "Graph RAG Architecture"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, "Graph RAG maintains ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.purple
    }
  }, "two indexes"), " in parallel: a vector database for semantic similarity, and a knowledge graph for structural relationships. Click any node to explore how it fits.")), /*#__PURE__*/React.createElement(Card, {
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.purple,
      marginBottom: 8
    }
  }, "LLM Graph Extraction"), /*#__PURE__*/React.createElement("div", {
    style: {
      background: "#08080d",
      borderRadius: 6,
      padding: "10px 12px",
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 4
    }
  }, "INPUT TEXT"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.text,
      lineHeight: 1.6
    }
  }, "\\"Apple was founded by Steve Jobs and Steve Wozniak in 1976.\\""), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.yellow,
      margin: "6px 0 2px"
    }
  }, "LLM EXTRACTS TRIPLES"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      fontFamily: "monospace",
      color: C.orange,
      lineHeight: 1.8
    }
  }, "(Apple, FOUNDED_BY, Steve Jobs)", "\\n", "(Apple, FOUNDED_BY, S. Wozniak)", "\\n", "(Apple, FOUNDED_IN, 1976)"))), /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.green,
      marginBottom: 8
    }
  }, "Use When"), ["Multi-hop questions across entities", "Org chart / hierarchy queries", "Drug-disease interaction graphs", "Legal entity relationships", "Scientific citation networks"].map(function (t, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 10,
        color: C.muted,
        marginBottom: 3
      }
    }, CHK + " " + t);
  })))));
}

/* ================================================================
   TAB 5: HYBRID RAG
   ================================================================ */
function TabHybrid() {
  var _s = useState(0);
  var step = _s[0];
  var setStep = _s[1];
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.teal
    }
  }, "Hybrid RAG ", DASH, " Vector + Graph + RRF"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Two parallel indexes, fused with Reciprocal Rank Fusion ", DASH, " highest accuracy for enterprise knowledge bases")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "center",
      marginBottom: 14
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "820",
    height: "260",
    viewBox: "0 0 820 260",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 110,
    w: 100,
    h: 40,
    color: C.blue,
    label: "Documents",
    icon: DOC
  }), /*#__PURE__*/React.createElement("text", {
    x: 220,
    y: 18,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "DUAL INDEXING (parallel)"), /*#__PURE__*/React.createElement(Arrow, {
    id: "h-top",
    x1: 112,
    y1: 120,
    x2: 145,
    y2: 60,
    color: C.blue
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 147,
    y: 30,
    w: 110,
    h: 40,
    color: C.blue,
    label: "Chunk+Embed"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "h1",
    x1: 259,
    y1: 50,
    x2: 282,
    y2: 50,
    color: C.blue
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 284,
    y: 30,
    w: 100,
    h: 40,
    color: C.blue,
    label: "Vector DB",
    icon: DB
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "h-bot",
    x1: 112,
    y1: 130,
    x2: 145,
    y2: 180,
    color: C.purple
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 147,
    y: 160,
    w: 110,
    h: 40,
    color: C.purple,
    label: "LLM KG Extract"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "h2",
    x1: 259,
    y1: 180,
    x2: 282,
    y2: 180,
    color: C.purple
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 284,
    y: 160,
    w: 100,
    h: 40,
    color: C.purple,
    label: "Graph DB",
    icon: GRAPH
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 210,
    w: 100,
    h: 40,
    color: C.yellow,
    label: "Query"
  }), /*#__PURE__*/React.createElement("text", {
    x: 480,
    y: 18,
    textAnchor: "middle",
    fill: C.muted,
    fontSize: 9,
    fontWeight: 700,
    fontFamily: "monospace"
  }, "DUAL RETRIEVAL"), /*#__PURE__*/React.createElement(Arrow, {
    id: "h-qv",
    x1: 112,
    y1: 220,
    x2: 398,
    y2: 60,
    color: C.yellow,
    dashed: true
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "hv2",
    x1: 386,
    y1: 50,
    x2: 428,
    y2: 50,
    color: C.blue
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 430,
    y: 30,
    w: 120,
    h: 40,
    color: C.blue,
    label: "Vector Results",
    sub: "semantic"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "h-qg",
    x1: 112,
    y1: 225,
    x2: 398,
    y2: 185,
    color: C.yellow,
    dashed: true
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "hg2",
    x1: 386,
    y1: 180,
    x2: 428,
    y2: 180,
    color: C.purple
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 430,
    y: 160,
    w: 120,
    h: 40,
    color: C.purple,
    label: "Graph Results",
    sub: "structural"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "hrrf1",
    x1: 552,
    y1: 50,
    x2: 596,
    y2: 110,
    color: C.blue
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "hrrf2",
    x1: 552,
    y1: 180,
    x2: 596,
    y2: 130,
    color: C.purple
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 598,
    y: 90,
    w: 110,
    h: 60,
    color: C.teal,
    label: "RRF Fusion",
    sub: "rank merging"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "hrrf3",
    x1: 710,
    y1: 120,
    x2: 735,
    y2: 120,
    color: C.teal
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 737,
    y: 100,
    w: 70,
    h: 40,
    color: C.accent,
    label: "LLM",
    icon: ROBOT
  }))), /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 820,
      margin: "0 auto 12px",
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Card, {
    color: C.teal
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.teal,
      marginBottom: 8
    }
  }, "Reciprocal Rank Fusion (RRF)"), /*#__PURE__*/React.createElement("div", {
    style: {
      background: "#08080d",
      borderRadius: 6,
      padding: "10px 12px",
      border: "1px solid " + C.teal + "40",
      fontFamily: "monospace",
      fontSize: 11,
      color: C.teal,
      marginBottom: 8
    }
  }, "score(d) = ", "\\\\u03A3", " 1 / (rank_d + k)"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 10,
      color: C.muted,
      lineHeight: 1.7
    }
  }, "k=60 (default). Sum the reciprocal rank from each retriever. No score normalisation needed \\u2014 ranks from any retriever are directly comparable.")), /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.text,
      marginBottom: 8
    }
  }, "Fusion Strategies"), [{
    n: "RRF",
    d: "Rank-based, no normalisation needed. Default choice.",
    c: C.teal
  }, {
    n: "Score fusion",
    d: "Normalise scores to [0,1] then weighted sum. Needs tuning.",
    c: C.blue
  }, {
    n: "LLM rerank",
    d: "Ask LLM to pick best from combined. Highest quality, highest cost.",
    c: C.purple
  }].map(function (s, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        marginBottom: 6
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        fontWeight: 700,
        color: s.c,
        fontFamily: "monospace"
      }
    }, s.n, ": "), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.muted
      }
    }, s.d));
  }))), /*#__PURE__*/React.createElement(Insight, {
    title: "When you need Hybrid RAG"
  }, "Enterprise knowledge bases where documents describe ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.blue,
      fontWeight: 700
    }
  }, "both facts (text)"), " and ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.purple,
      fontWeight: 700
    }
  }, "relationships (structure)"), ". Example: financial research (news + company graph), healthcare (clinical notes + drug interactions), customer support (FAQ + product knowledge graph). Requires maintaining ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.teal,
      fontWeight: 700
    }
  }, "both a vector DB and graph DB"), " in sync."));
}

/* ================================================================
   TAB 6: AGENTIC RAG (ROUTER)
   ================================================================ */
function TabAgentic() {
  var _s = useState(null);
  var sel = _s[0];
  var setSel = _s[1];
  var tools = [{
    name: "Vector Search",
    icon: "\\\\uD83D\\\\uDD0D",
    color: C.blue,
    when: "Factual questions, document Q&A, semantic search over internal knowledge base",
    example: "What is our refund policy?"
  }, {
    name: "SQL Database",
    icon: "\\\\uD83D\\\\uDDC4",
    color: C.green,
    when: "Structured data: revenue, metrics, comparisons, trends, numbers, KPIs",
    example: "What was Q3 vs Q4 revenue growth?"
  }, {
    name: "Web Search",
    icon: "\\\\uD83C\\\\uDF10",
    color: C.orange,
    when: "Real-time info: news, current events, live data, competitor prices today",
    example: "What happened in AI news today?"
  }, {
    name: "Doc Analysis",
    icon: "\\\\uD83D\\\\uDCCB",
    color: C.purple,
    when: "User has uploaded a specific file that needs direct processing and analysis",
    example: "Summarise this uploaded contract."
  }, {
    name: "Code Execution",
    icon: "\\\\uD83D\\\\uDCBB",
    color: C.yellow,
    when: "Calculations, data transformations, chart generation, programmatic tasks",
    example: "Calculate the CAGR from 2020–2024."
  }];
  var t = sel !== null ? tools[sel] : null;
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.orange
    }
  }, "Agentic RAG ", DASH, " AI Router"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "An AI agent decides WHICH retrieval tool to call per query ", DASH, " one pipeline for all query types")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "center",
      marginBottom: 14
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: "820",
    height: "200",
    viewBox: "0 0 820 200",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement(PNode, {
    x: 10,
    y: 80,
    w: 80,
    h: 40,
    color: C.yellow,
    label: "Query"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "ag0",
    x1: 92,
    y1: 100,
    x2: 115,
    y2: 100,
    color: C.yellow
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 117,
    y: 60,
    w: 110,
    h: 80,
    color: C.orange,
    label: "AI Agent",
    sub: "function calling"
  }), tools.map(function (tool, i) {
    var ty = 20 + i * 36;
    return /*#__PURE__*/React.createElement("g", {
      key: i
    }, /*#__PURE__*/React.createElement(Arrow, {
      id: "ag" + i,
      x1: 229,
      y1: 100,
      x2: 310,
      y2: ty + 18,
      color: C.dim,
      dashed: true
    }), /*#__PURE__*/React.createElement(PNode, {
      x: 312,
      y: ty,
      w: 120,
      h: 36,
      color: tool.color,
      label: tool.icon + " " + tool.name
    }));
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "ag-ctx",
    x1: 434,
    y1: 100,
    x2: 488,
    y2: 100,
    color: C.cyan
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 490,
    y: 80,
    w: 110,
    h: 40,
    color: C.cyan,
    label: "Context",
    sub: "selected tool result"
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "ag-llm",
    x1: 602,
    y1: 100,
    x2: 626,
    y2: 100,
    color: C.cyan
  }), /*#__PURE__*/React.createElement(PNode, {
    x: 628,
    y: 70,
    w: 130,
    h: 60,
    color: C.pink,
    label: "Multimodal LLM",
    icon: ROBOT
  }), /*#__PURE__*/React.createElement(Arrow, {
    id: "ag-r",
    x1: 760,
    y1: 100,
    x2: 795,
    y2: 100,
    color: C.pink
  }), /*#__PURE__*/React.createElement("text", {
    x: 800,
    y: 104,
    fill: C.accent,
    fontSize: 9,
    fontFamily: "monospace"
  }, "Ans"))), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 8,
      justifyContent: "center",
      marginBottom: 12,
      flexWrap: "wrap"
    }
  }, tools.map(function (tool, i) {
    return /*#__PURE__*/React.createElement("button", {
      key: i,
      onClick: function () {
        setSel(sel === i ? null : i);
      },
      style: {
        padding: "7px 13px",
        borderRadius: 8,
        border: "1.5px solid " + (sel === i ? tool.color : C.border),
        background: sel === i ? tool.color + "20" : C.card,
        color: sel === i ? tool.color : C.muted,
        cursor: "pointer",
        fontSize: 10,
        fontWeight: 700,
        fontFamily: "monospace"
      }
    }, tool.icon + " " + tool.name);
  })), t ? /*#__PURE__*/React.createElement(Card, {
    color: t.color,
    style: {
      maxWidth: 820,
      margin: "0 auto 12px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 800,
      color: t.color,
      marginBottom: 6
    }
  }, t.icon + " " + t.name), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 3
    }
  }, "ROUTE WHEN"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.text,
      lineHeight: 1.7
    }
  }, t.when)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 3
    }
  }, "EXAMPLE QUERY"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontStyle: "italic",
      color: t.color
    }
  }, "\\"", t.example, "\\"")))) : /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: 820,
      margin: "0 auto 12px",
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 10
    }
  }, /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.orange,
      marginBottom: 6
    }
  }, "Agent Routing Logic"), ["Keyword pattern matching (fast, deterministic)", "LLM function-calling (flexible, accurate)", "Query classification model (ML-based routing)", "Confidence threshold with fallback to vector"].map(function (t, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 10,
        color: C.muted,
        marginBottom: 3
      }
    }, CHK + " " + t);
  })), /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.orange,
      marginBottom: 6
    }
  }, "Tradeoffs"), [{
    k: "Flexibility",
    v: "Very High",
    c: C.green
  }, {
    k: "Determinism",
    v: "Medium",
    c: C.yellow
  }, {
    k: "Added latency",
    v: "+200–500ms",
    c: C.orange
  }, {
    k: "Debug complexity",
    v: "Higher",
    c: C.red
  }].map(function (r, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        display: "flex",
        justifyContent: "space-between",
        marginBottom: 5
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.muted
      }
    }, r.k), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        fontWeight: 700,
        color: r.c
      }
    }, r.v));
  }))), /*#__PURE__*/React.createElement(Insight, {
    title: "The key advantage"
  }, "Agentic RAG can ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.orange,
      fontWeight: 700
    }
  }, "skip retrieval entirely"), " for computational queries (just run code), use web search for real-time questions (skip the stale vector DB), and use SQL for structured data. A single static pipeline would use the wrong tool for all of these."));
}

/* ================================================================
   TAB 7: MULTI-AGENT RAG
   ================================================================ */
function TabMultiAgent() {
  var _s = useState(null);
  var sel = _s[0];
  var setSel = _s[1];
  var agents = [{
    id: "orchestrator",
    x: 100,
    y: 130,
    r: 35,
    color: C.orange,
    label: "Orchestrator",
    sub: "decomposes + routes"
  }, {
    id: "agA",
    x: 280,
    y: 50,
    r: 28,
    color: C.blue,
    label: "Agent A",
    sub: "Vector A"
  }, {
    id: "agB",
    x: 280,
    y: 140,
    r: 28,
    color: C.cyan,
    label: "Agent B",
    sub: "Vector B"
  }, {
    id: "agC",
    x: 280,
    y: 230,
    r: 28,
    color: C.purple,
    label: "Agent C",
    sub: "Web search"
  }, {
    id: "agD",
    x: 280,
    y: 320,
    r: 28,
    color: C.pink,
    label: "Agent D",
    sub: "Slack/Gmail"
  }, {
    id: "vecDB",
    x: 450,
    y: 95,
    r: 28,
    color: C.blue,
    label: "Vector DB",
    sub: "source A+B"
  }, {
    id: "webS",
    x: 450,
    y: 230,
    r: 28,
    color: C.purple,
    label: "Web Index",
    sub: "live results"
  }, {
    id: "tools",
    x: 450,
    y: 320,
    r: 28,
    color: C.pink,
    label: "Tools",
    sub: "Slack/Gmail"
  }, {
    id: "synth",
    x: 610,
    y: 185,
    r: 35,
    color: C.green,
    label: "Synthesis",
    sub: "orchestrator"
  }, {
    id: "genLLM",
    x: 750,
    y: 185,
    r: 30,
    color: C.accent,
    label: "Gen LLM",
    sub: "final ans"
  }];
  var edges = [["orchestrator", "agA", C.orange], ["orchestrator", "agB", C.orange], ["orchestrator", "agC", C.orange], ["orchestrator", "agD", C.orange], ["agA", "vecDB", C.blue], ["agB", "vecDB", C.cyan], ["agC", "webS", C.purple], ["agD", "tools", C.pink], ["vecDB", "synth", C.blue], ["webS", "synth", C.purple], ["tools", "synth", C.pink], ["synth", "genLLM", C.green]];
  function getNode(id) {
    return agents.find(function (n) {
      return n.id === id;
    });
  }
  var info = {
    "orchestrator": "Receives the user query, decomposes it into sub-tasks, assigns each to a specialised agent, waits for all results, and synthesises the final response. The brain of the system.",
    "agA": "Specialised for domain A's vector database. Optimised prompts, retrieval parameters, and schema knowledge specific to this data source.",
    "agB": "Second vector specialist — different knowledge base, different retrieval strategy. Can run concurrently with Agent A.",
    "agC": "Real-time web search agent. Fetches live news, competitor data, pricing, and any information not in the internal knowledge base.",
    "agD": "Communication tools agent. Searches Slack conversations, Gmail threads, calendar, and internal tool integrations.",
    "vecDB": "Primary vector database serving Agents A and B. May hold millions of document embeddings for semantic retrieval.",
    "webS": "Live web index. Agent C calls search APIs (Google, Bing, Perplexity) to get real-time information unavailable in static indexes.",
    "tools": "Enterprise tool integrations: Slack, Gmail, Jira, Salesforce. Agent D queries these for communication and operational context.",
    "synth": "Orchestrator synthesises all agent results: deduplicates, resolves conflicts, orders by relevance, and assembles the final context for the LLM.",
    "genLLM": "Generative LLM receives the synthesised multi-source context and generates the final comprehensive response."
  };
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.green
    }
  }, "Multi-Agent RAG ", DASH, " Parallel Specialists"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Orchestrator + N parallel specialised agents ", DASH, " each owns a domain, all run simultaneously")), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 16,
      alignItems: "flex-start",
      flexWrap: "wrap"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      flex: "0 0 auto"
    }
  }, /*#__PURE__*/React.createElement("svg", {
    width: 530,
    height: 400,
    viewBox: "0 0 530 400",
    style: {
      background: "#08080d",
      borderRadius: 10,
      border: "1px solid " + C.border
    }
  }, /*#__PURE__*/React.createElement("text", {
    x: 30,
    y: 30,
    fill: C.yellow,
    fontSize: 9,
    fontFamily: "monospace"
  }, "Query"), /*#__PURE__*/React.createElement("line", {
    x1: 55,
    y1: 35,
    x2: 62,
    y2: 128,
    stroke: C.yellow,
    strokeWidth: 1.5,
    strokeDasharray: "4,3"
  }), /*#__PURE__*/React.createElement("text", {
    x: 755,
    y: 190,
    fill: C.accent,
    fontSize: 9,
    fontFamily: "monospace"
  }, "Response"), edges.map(function (e, i) {
    var n1 = getNode(e[0]);
    var n2 = getNode(e[1]);
    if (!n1 || !n2) return null;
    return /*#__PURE__*/React.createElement("line", {
      key: i,
      x1: n1.x,
      y1: n1.y,
      x2: n2.x,
      y2: n2.y,
      stroke: e[2],
      strokeWidth: 1.5,
      strokeOpacity: 0.6,
      strokeDasharray: "5,3"
    });
  }), agents.map(function (n) {
    var active = sel === n.id;
    return /*#__PURE__*/React.createElement("g", {
      key: n.id,
      onClick: function () {
        setSel(active ? null : n.id);
      },
      style: {
        cursor: "pointer"
      }
    }, /*#__PURE__*/React.createElement("circle", {
      cx: n.x,
      cy: n.y,
      r: active ? n.r + 4 : n.r,
      fill: active ? n.color + "35" : "#0d0d14",
      stroke: n.color,
      strokeWidth: active ? 2.5 : 1.5,
      style: {
        transition: "all 0.2s"
      }
    }), /*#__PURE__*/React.createElement("text", {
      x: n.x,
      y: n.y - 2,
      textAnchor: "middle",
      fill: n.color,
      fontSize: 9,
      fontWeight: 700,
      fontFamily: "monospace"
    }, n.label), /*#__PURE__*/React.createElement("text", {
      x: n.x,
      y: n.y + 10,
      textAnchor: "middle",
      fill: C.muted,
      fontSize: 7,
      fontFamily: "monospace"
    }, n.sub));
  }), /*#__PURE__*/React.createElement("rect", {
    x: 248,
    y: 18,
    width: 168,
    height: 318,
    rx: 6,
    fill: "none",
    stroke: C.dim,
    strokeWidth: 1,
    strokeDasharray: "4,4"
  }), /*#__PURE__*/React.createElement("text", {
    x: 332,
    y: 14,
    textAnchor: "middle",
    fill: C.dim,
    fontSize: 8,
    fontFamily: "monospace"
  }, "parallel execution"), /*#__PURE__*/React.createElement("text", {
    x: 265,
    y: 390,
    textAnchor: "middle",
    fill: C.dim,
    fontSize: 8,
    fontFamily: "monospace"
  }, "click any node to learn its role"))), /*#__PURE__*/React.createElement("div", {
    style: {
      flex: 1,
      minWidth: 220
    }
  }, sel && info[sel] ? /*#__PURE__*/React.createElement(Card, {
    color: getNode(sel).color,
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      fontWeight: 800,
      color: getNode(sel).color,
      marginBottom: 8
    }
  }, getNode(sel).label), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, info[sel])) : /*#__PURE__*/React.createElement(Card, {
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 12,
      fontWeight: 700,
      color: C.green,
      marginBottom: 8
    }
  }, "Why Multi-Agent?"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      lineHeight: 1.8
    }
  }, "Complex queries need ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.orange
    }
  }, "parallel investigation"), " across completely separate knowledge domains. Agents specialise \\u2014 and run simultaneously, so total latency equals the ", /*#__PURE__*/React.createElement("span", {
    style: {
      color: C.accent
    }
  }, "slowest single agent"), ", not the sum.")), /*#__PURE__*/React.createElement(Card, {
    style: {
      marginBottom: 12
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.text,
      marginBottom: 8
    }
  }, "Communication Patterns"), [{
    p: "Parallel",
    d: "All agents run simultaneously. Best for independent sub-tasks.",
    c: C.green
  }, {
    p: "Sequential",
    d: "Agent A → Agent B → Agent C. Each needs prior results.",
    c: C.blue
  }, {
    p: "Hierarchical",
    d: "Root → domain orchestrators → leaf agents. Large KBs.",
    c: C.purple
  }].map(function (r, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        marginBottom: 8
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        fontWeight: 700,
        color: r.c
      }
    }, r.p, ": "), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.muted
      }
    }, r.d));
  })), /*#__PURE__*/React.createElement(Card, null, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.green,
      marginBottom: 6
    }
  }, "Frameworks"), [["LangGraph", "State-machine workflows"], ["AutoGen", "Conversational multi-agent"], ["CrewAI", "Role-based agent crews"], ["Semantic Kernel", "Enterprise orchestration"]].map(function (r, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        display: "flex",
        justifyContent: "space-between",
        marginBottom: 4
      }
    }, /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.green,
        fontFamily: "monospace"
      }
    }, r[0]), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 10,
        color: C.muted
      }
    }, r[1]));
  })))));
}

/* ================================================================
   TAB 8: DECISION GUIDE
   ================================================================ */
function TabDecision() {
  var _s = useState(null);
  var sel = _s[0];
  var setSel = _s[1];
  var patterns = [{
    name: "Naive RAG",
    color: C.blue,
    lat: "80–200ms",
    prec: "65%",
    cost: "$0.02",
    setup: "1d",
    tags: ["Text only", "Simple Q&A", "Prototypes", "Small corpus"]
  }, {
    name: "Rerank",
    color: C.purple,
    lat: "150–400ms",
    prec: "82%",
    cost: "$0.06",
    setup: "2d",
    tags: ["High precision", "Legal/medical", "Large corpus", "Noisy docs"]
  }, {
    name: "Multimodal",
    color: C.pink,
    lat: "200–600ms",
    prec: "78%",
    cost: "$0.15",
    setup: "5d",
    tags: ["Images/video", "PDFs with charts", "Product catalogs", "Cross-modal"]
  }, {
    name: "Graph RAG",
    color: C.orange,
    lat: "200–500ms",
    prec: "85%",
    cost: "$0.12",
    setup: "10d",
    tags: ["Multi-hop questions", "Entity relations", "KG traversal", "Org charts"]
  }, {
    name: "Hybrid RAG",
    color: C.teal,
    lat: "300–700ms",
    prec: "88%",
    cost: "$0.18",
    setup: "14d",
    tags: ["Enterprise KBs", "All query types", "Max accuracy", "Complex domains"]
  }, {
    name: "Agentic RAG",
    color: C.yellow,
    lat: "200–800ms",
    prec: "80%",
    cost: "$0.10",
    setup: "7d",
    tags: ["Diverse queries", "Multi-source", "Adaptive routing", "Heterogeneous data"]
  }, {
    name: "Multi-Agent",
    color: C.green,
    lat: "300–900ms",
    prec: "90%",
    cost: "$0.35",
    setup: "21d",
    tags: ["Parallel research", "Complex synthesis", "3+ sources", "Enterprise intelligence"]
  }];
  var p = sel !== null ? patterns[sel] : null;
  return /*#__PURE__*/React.createElement("div", {
    className: "fade-in"
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 16
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 800,
      color: C.accent
    }
  }, "Decision Guide ", DASH, " Which Pattern?"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 3
    }
  }, "Compare all 7 patterns by latency, precision, and cost. Click any pattern for details.")), /*#__PURE__*/React.createElement(Card, {
    style: {
      maxWidth: 820,
      margin: "0 auto 14px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.text,
      marginBottom: 12
    }
  }, "Precision Comparison"), patterns.map(function (pt, i) {
    var pct = parseInt(pt.prec);
    var active = sel === i;
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      onClick: function () {
        setSel(active ? null : i);
      },
      style: {
        display: "flex",
        alignItems: "center",
        gap: 10,
        marginBottom: 7,
        cursor: "pointer"
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        width: 95,
        fontSize: 9,
        color: active ? pt.color : C.muted,
        textAlign: "right",
        fontFamily: "monospace",
        fontWeight: active ? 700 : 400
      }
    }, pt.name), /*#__PURE__*/React.createElement("div", {
      style: {
        flex: 1,
        height: 20,
        borderRadius: 4,
        background: C.border,
        overflow: "hidden"
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        width: pct + "%",
        height: "100%",
        borderRadius: 4,
        background: pt.color + (active ? "90" : "35"),
        border: "1px solid " + pt.color + (active ? "80" : "40"),
        transition: "all 0.3s"
      }
    })), /*#__PURE__*/React.createElement("div", {
      style: {
        width: 40,
        fontSize: 9,
        fontFamily: "monospace",
        color: active ? pt.color : C.dim
      }
    }, pt.prec), /*#__PURE__*/React.createElement("div", {
      style: {
        width: 50,
        fontSize: 9,
        fontFamily: "monospace",
        color: C.muted
      }
    }, pt.lat.split("–")[0], "ms"), /*#__PURE__*/React.createElement("div", {
      style: {
        width: 40,
        fontSize: 9,
        fontFamily: "monospace",
        color: C.muted
      }
    }, pt.cost));
  }), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      justifyContent: "flex-end",
      gap: 20,
      marginTop: 4
    }
  }, [["Precision", C.text], ["Min latency", C.muted], ["$/1K queries", C.muted]].map(function (h, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        fontSize: 8,
        color: h[1],
        fontFamily: "monospace"
      }
    }, h[0]);
  }))), p ? /*#__PURE__*/React.createElement(Card, {
    color: p.color,
    style: {
      maxWidth: 820,
      margin: "0 auto 14px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 14,
      fontWeight: 800,
      color: p.color,
      marginBottom: 10
    }
  }, p.name), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "grid",
      gridTemplateColumns: "repeat(4,1fr)",
      gap: 12,
      marginBottom: 10
    }
  }, [["Latency", p.lat], ["Precision", p.prec], ["Cost/1K", p.cost], ["Setup", p.setup]].map(function (m, i) {
    return /*#__PURE__*/React.createElement("div", {
      key: i,
      style: {
        textAlign: "center"
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 9,
        color: C.muted,
        marginBottom: 3
      }
    }, m[0]), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 14,
        fontWeight: 700,
        color: p.color
      }
    }, m[1]));
  })), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 9,
      color: C.muted,
      marginBottom: 6
    }
  }, "BEST FOR"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      flexWrap: "wrap",
      gap: 4
    }
  }, p.tags.map(function (t, i) {
    return /*#__PURE__*/React.createElement(Badge, {
      key: i,
      color: p.color
    }, t);
  }))) : null, /*#__PURE__*/React.createElement(Card, {
    style: {
      maxWidth: 820,
      margin: "0 auto 4px"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      fontWeight: 700,
      color: C.accent,
      marginBottom: 10
    }
  }, "The RAG Maturity Path"), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      gap: 6,
      flexWrap: "wrap"
    }
  }, [{
    n: "Naive RAG",
    c: C.blue,
    t: "Start here. Always."
  }, {
    n: "+ Reranking",
    c: C.purple,
    t: "Highest ROI first upgrade"
  }, {
    n: "+ Hybrid search",
    c: C.teal,
    t: "Keyword + semantic"
  }, {
    n: "+ Graph (if needed)",
    c: C.orange,
    t: "Multi-hop fails?"
  }, {
    n: "+ Agents (if needed)",
    c: C.yellow,
    t: "Diverse query types?"
  }, {
    n: "+ Multi-Agent",
    c: C.green,
    t: "3+ data sources?"
  }].map(function (s, i, arr) {
    return /*#__PURE__*/React.createElement(React.Fragment, {
      key: i
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        textAlign: "center"
      }
    }, /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 9,
        fontWeight: 700,
        color: s.c,
        fontFamily: "monospace",
        marginBottom: 2
      }
    }, s.n), /*#__PURE__*/React.createElement("div", {
      style: {
        fontSize: 8,
        color: C.dim
      }
    }, s.t)), i < arr.length - 1 && /*#__PURE__*/React.createElement("div", {
      style: {
        color: C.dim,
        fontSize: 12,
        flexShrink: 0
      }
    }, ARR));
  }))));
}

/* ================================================================
   ROOT APP
   ================================================================ */
function App() {
  var _t = useState(0);
  var tab = _t[0];
  var setTab = _t[1];
  var tabs = ["\\\\uD83D\\\\uDD0D Naive RAG", "\\\\u21A9 Rerank", "\\\\uD83D\\\\uDDBC Multimodal", "\\\\uD83D\\\\uDD78 Graph", "\\\\u26A1 Hybrid", "\\\\uD83E\\\\uDD16 Agentic", "\\\\uD83E\\\\uDDE9 Multi-Agent", "\\\\uD83D\\\\uDCCA Decision Guide"];
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: C.bg,
      minHeight: "100vh",
      padding: "20px 16px",
      fontFamily: "'JetBrains Mono','SF Mono',monospace",
      color: C.text,
      maxWidth: 1000,
      margin: "0 auto"
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      textAlign: "center",
      marginBottom: 20
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 22,
      fontWeight: 800,
      background: "linear-gradient(135deg," + C.accent + "," + C.purple + "," + C.teal + ")",
      WebkitBackgroundClip: "text",
      WebkitTextFillColor: "transparent",
      display: "inline-block"
    }
  }, "RAG Design Patterns"), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 11,
      color: C.muted,
      marginTop: 4
    }
  }, "7 architectural patterns " + DASH + " from Naive RAG to Multi-Agent — click each tab to explore")), /*#__PURE__*/React.createElement(TabBar, {
    tabs: tabs,
    active: tab,
    onChange: setTab
  }), tab === 0 && /*#__PURE__*/React.createElement(TabNaive, null), tab === 1 && /*#__PURE__*/React.createElement(TabRerank, null), tab === 2 && /*#__PURE__*/React.createElement(TabMultimodal, null), tab === 3 && /*#__PURE__*/React.createElement(TabGraph, null), tab === 4 && /*#__PURE__*/React.createElement(TabHybrid, null), tab === 5 && /*#__PURE__*/React.createElement(TabAgentic, null), tab === 6 && /*#__PURE__*/React.createElement(TabMultiAgent, null), tab === 7 && /*#__PURE__*/React.createElement(TabDecision, null));
}
ReactDOM.createRoot(document.getElementById("root")).render(/*#__PURE__*/React.createElement(App, null));
document.getElementById('loading').style.display = 'none';

document.getElementById('loading').style.display = 'none';
</script>
</body>
</html>
"""

RAG_VISUAL_HEIGHT = 1100