"""
Self-contained HTML for the RAG (Retrieval-Augmented Generation) interactive walkthrough.
Covers: RAG pipeline, vector embeddings, chunking strategies, retrieval & reranking,
and context window management including Agentic RAG.
Embed in Streamlit via st.components.v1.html(RAG_VISUAL_HTML, height=RAG_VISUAL_HEIGHT).
"""

RAG_VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; overflow-x: hidden; }
  input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1e1e2e; outline: none; }
  input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #ff6b35; }
  @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
  .zoom-wrap { transform-origin: top center; transition: transform 0.12s ease; }
  .zoom-badge { position:fixed; bottom:18px; right:18px; background:#12121a; border:1px solid #3f3f46; border-radius:8px; padding:6px 12px; font-size:13px; font-family:'JetBrains Mono',monospace; color:#71717a; pointer-events:none; z-index:9999; transition:opacity 0.4s; }
  .zoom-badge span { color:#ff6b35; font-weight:700; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState  = React.useState;
var useEffect = React.useEffect;
var useMemo   = React.useMemo;

var C = {
  bg:"#0a0a0f", card:"#12121a", border:"#1e1e2e",
  accent:"#ff6b35", blue:"#4ecdc4", purple:"#a78bfa",
  yellow:"#fbbf24", text:"#e4e4e7", muted:"#71717a",
  dim:"#3f3f46", red:"#ef4444", green:"#4ade80",
  cyan:"#38bdf8", pink:"#f472b6", orange:"#fb923c",
};

var MUL  = "\\u00D7";
var ARR  = "\\u2192";
var DASH = "\\u2014";
var CHK  = "\\u2713";
var WARN = "\\u26A0";
var LQ   = "\\u201C";
var RQ   = "\\u201D";
var LARR = "\\u2190";
var PLAY = "\\u25B6";
var PAUSE= "\\u23F8";
var BULB = "\\uD83D\\uDCA1";
var TARG = "\\uD83C\\uDFAF";
var UP   = "\\u2191";
var DOWN = "\\u2193";

function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange;
  return (
    <div style={{display:"flex",gap:0,borderBottom:"2px solid "+C.border,marginBottom:24,overflowX:"auto"}}>
      {tabs.map(function(t,i){return(
        <button key={i} onClick={function(){onChange(i);}} style={{
          padding:"12px 18px",background:"none",border:"none",
          borderBottom:active===i?"2px solid "+C.accent:"2px solid transparent",
          color:active===i?C.accent:C.muted,cursor:"pointer",
          fontSize:13,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",
          transition:"all 0.2s",whiteSpace:"nowrap",marginBottom:-2,
        }}>{t}</button>
      );})}
    </div>
  );
}

function Card(props) {
  return (
    <div style={Object.assign({
      background:C.card,borderRadius:10,padding:"18px 22px",
      border:"1px solid "+(props.highlight?C.accent:C.border),
      transition:"border 0.3s",
    },props.style||{})}>
      {props.children}
    </div>
  );
}

function Insight(props) {
  return (
    <div style={Object.assign({
      maxWidth:750,margin:"16px auto 0",
      padding:"16px 22px",background:"rgba(255,107,53,0.06)",
      borderRadius:10,border:"1px solid rgba(255,107,53,0.2)",
    },props.style||{})}>
      <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:6}}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
      <div style={{fontSize:13,color:C.muted,lineHeight:1.8}}>{props.children}</div>
    </div>
  );
}

function SectionTitle(props) {
  return (
    <div style={{textAlign:"center",marginBottom:20}}>
      <div style={{fontSize:21,fontWeight:800,color:C.text,marginBottom:4}}>{props.title}</div>
      <div style={{fontSize:14,color:C.muted}}>{props.subtitle}</div>
    </div>
  );
}


/* ===============================================================
   TAB 1: RAG PIPELINE (animated step-through)
   =============================================================== */
function TabPipeline() {
  var _s = useState(0); var step = _s[0], setStep = _s[1];
  var _a = useState(false); var auto = _a[0], setAuto = _a[1];

  var steps = [
    {tag:"OFFLINE",color:C.blue,   title:"Raw Document Collection",
     desc:"Source documents: PDFs, wikis, emails, code, web pages. Total size can be millions of tokens "+DASH+" far beyond any context window. This is the core problem RAG solves: externalising memory the model can never hold internally."},
    {tag:"OFFLINE",color:C.purple, title:"Chunking: Split Into Pieces",
     desc:"Documents split into chunks (200"+DASH+"500 tokens each) with 10"+DASH+"15% overlap. Too large: embedding averages over too much, retrieval precision drops. Too small: chunks lose context. Overlap prevents answers from being severed at boundaries."},
    {tag:"OFFLINE",color:C.yellow, title:"Embedding: Text "+ARR+" Vectors",
     desc:"Each chunk passes through an encoder model (e.g. text-embedding-3-small). Output: a dense float vector (1536 dims). Semantically similar chunks produce geometrically nearby vectors regardless of exact wording "+DASH+" this is what enables semantic search."},
    {tag:"OFFLINE",color:C.orange, title:"Vector Index: Built and Ready",
     desc:"Vectors stored in a vector database (Pinecone, pgvector, FAISS, Chroma). An HNSW graph built for fast approximate nearest-neighbour (ANN) search. Raw chunk text stored alongside. Index is now queryable in milliseconds at any scale."},
    {tag:"ONLINE", color:C.green,  title:"User Query Arrives",
     desc:"A user asks a question. Raw text is the entry point for the online phase. The same embedding model used at index time is applied to the query, producing a query vector for comparison against all indexed chunk vectors."},
    {tag:"ONLINE", color:C.cyan,   title:"ANN Search "+ARR+" Top-k Chunks",
     desc:"Query vector compared to all chunk vectors via ANN search. Returns top-k candidates (k = 20"+DASH+"50 initially). A cross-encoder reranker then re-scores each (query, chunk) pair jointly "+DASH+" far more accurate than cosine similarity alone "+DASH+" and trims to the best 3"+DASH+"8."},
    {tag:"ONLINE", color:C.pink,   title:"Context Assembly: Stuff the Window",
     desc:"Retrieved chunks inserted into the prompt: [system] + [chunks] + [history] + [query]. Most relevant chunk placed first or last (not buried in the middle) to avoid the lost-in-the-middle effect. This is the only point retrieval results touch the LLM."},
    {tag:"ONLINE", color:C.accent, title:"LLM Reads and Generates",
     desc:"The LLM sees only what is in the context window. No knowledge of the vector index. It reads the retrieved text and generates a grounded answer. If no relevant chunk was retrieved, a well-prompted system will say so rather than hallucinating from frozen weights."},
  ];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(p){return(p+1)%8;});},2500);
    return function(){clearInterval(t);};
  },[auto]);

  var s = steps[step];
  var BW=150, BH=50;

  // Returns SVG group for one pipeline box
  function PBox(label,x,y,si,icon) {
    var active=step===si, done=step>si;
    var col=active?steps[si].color:(done?C.dim:C.dim+"40");
    var bg=active?steps[si].color+"18":(done?"#ffffff05":"transparent");
    return(
      <g key={label} onClick={function(){setStep(si);setAuto(false);}} style={{cursor:"pointer"}}>
        <rect x={x} y={y} width={BW} height={BH} rx={7} fill={bg} stroke={col} strokeWidth={active?2.5:1}/>
        <text x={x+BW/2} y={y+17} textAnchor="middle" dominantBaseline="middle"
          fill={active?steps[si].color:C.muted} fontSize={12} fontWeight={700} fontFamily="monospace">{label}</text>
        <text x={x+BW/2} y={y+37} textAnchor="middle" dominantBaseline="middle"
          fill={active?steps[si].color+"99":C.dim} fontSize={10} fontFamily="monospace">{icon}</text>
      </g>
    );
  }

  var icons = ["[pdf][md][txt]","split+overlap","[0.2,0.8,−0.1]","HNSW index","what is X?","top-k chunks","[sys][doc][q]","grounded answer"];
  var OBX = [["DOCS",20,0],["CHUNK",190,1],["EMBED",360,2],["VECTOR DB",530,3]];
  var NBX = [["QUERY",20,4],["RETRIEVE",190,5],["AUGMENT",360,6],["GENERATE",530,7]];

  return(
    <div>
      <SectionTitle title="The RAG Pipeline" subtitle={"Offline indexing (runs once) + Online retrieval (runs per query) "+DASH+" step through both phases"} />

      <div style={{display:"flex",justifyContent:"center",gap:6,marginBottom:16,flexWrap:"wrap"}}>
        {steps.map(function(s2,i){return(
          <button key={i} onClick={function(){setStep(i);setAuto(false);}} style={{
            padding:"5px 10px",borderRadius:6,
            border:"1.5px solid "+(step===i?s2.color:C.border),
            background:step===i?s2.color+"20":C.card,
            color:step===i?s2.color:C.muted,
            cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"monospace",
          }}>{"0"+(i+1)}</button>
        );})}
        <button onClick={function(){setAuto(!auto);}} style={{
          padding:"5px 12px",borderRadius:6,
          border:"1.5px solid "+(auto?C.yellow:C.border),
          background:auto?C.yellow+"20":C.card,
          color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:13,fontFamily:"monospace",
        }}>{auto?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={760} height={230} viewBox="0 0 760 230" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <defs>
            <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M2 1L8 5L2 9" fill="none" stroke={C.dim} strokeWidth="1.5"/>
            </marker>
            <marker id="ahC" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M2 1L8 5L2 9" fill="none" stroke={C.cyan} strokeWidth="1.5"/>
            </marker>
          </defs>

          {/* Phase banners */}
          <text x={370} y={16} textAnchor="middle" fill={C.blue} fontSize={11} fontWeight={700} fontFamily="monospace">OFFLINE PHASE — runs once at setup</text>
          <line x1={10} y1={100} x2={750} y2={100} stroke={C.border} strokeWidth={1} strokeDasharray="5,4"/>
          <text x={370} y={116} textAnchor="middle" fill={C.green} fontSize={11} fontWeight={700} fontFamily="monospace">ONLINE PHASE — runs per query</text>

          {/* Offline row */}
          {OBX.map(function(b){return PBox(b[0],b[1],26,b[2],icons[b[2]]);}) }
          {[[170,188],[340,358],[510,528]].map(function(p,i){
            return <line key={"oa"+i} x1={p[0]} y1={51} x2={p[1]} y2={51} stroke={C.dim+"60"} strokeWidth={1.5} markerEnd="url(#ah)"/>;
          })}

          {/* VECTOR DB → RETRIEVE connector (L-path) */}
          <path d={"M605,76 L605,100 L265,100 L265,126"} fill="none"
            stroke={step>=5?C.cyan:C.dim+"30"} strokeWidth={step>=5?2:1}
            strokeDasharray="5,3" markerEnd={step>=5?"url(#ahC)":"url(#ah)"}/>
          {step>=5&&<text x={470} y={96} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">ANN lookup</text>}

          {/* Online row */}
          {NBX.map(function(b){return PBox(b[0],b[1],126,b[2],icons[b[2]]);}) }
          {[[170,188],[340,358],[510,528]].map(function(p,i){
            return <line key={"na"+i} x1={p[0]} y1={151} x2={p[1]} y2={151} stroke={C.dim+"60"} strokeWidth={1.5} markerEnd="url(#ah)"/>;
          })}

          {/* Step badge */}
          <rect x={580} y={192} width={160} height={26} rx={6} fill={s.color+"15"} stroke={s.color+"40"}/>
          <text x={660} y={208} textAnchor="middle" fill={s.color} fontSize={11} fontWeight={700} fontFamily="monospace">{"Step "+(step+1)+"/8 "+DASH+" "+s.tag}</text>
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:s.color}}>
        <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
          <span style={{background:s.tag==="OFFLINE"?C.blue+"20":C.green+"20",color:s.tag==="OFFLINE"?C.blue:C.green,fontSize:11,fontWeight:700,fontFamily:"monospace",padding:"2px 8px",borderRadius:4,border:"1px solid "+(s.tag==="OFFLINE"?C.blue+"40":C.green+"40")}}>{s.tag}</span>
          <div style={{fontSize:15,fontWeight:700,color:s.color}}>{s.title}</div>
        </div>
        <div style={{fontSize:13,color:C.muted,lineHeight:1.8}}>{s.desc}</div>
      </Card>

      <Insight icon={TARG} title="The Core Architectural Insight">
        RAG does <span style={{color:C.red,fontWeight:700}}>not</span> change model weights. The LLM is a frozen function throughout. RAG changes what it <span style={{color:C.green,fontWeight:700}}>reads</span> at inference time. Updating your knowledge base means re-indexing, <span style={{color:C.accent,fontWeight:700}}>not retraining</span> {DASH} knowledge is in the index, not the weights.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: VECTOR EMBEDDINGS & COSINE SIMILARITY
   =============================================================== */
function TabEmbeddings() {
  var _q = useState(0); var qsel = _q[0], setQsel = _q[1];

  var queries = [
    {text:"How do I get a refund?",        color:C.accent},
    {text:"What is the return timeline?",   color:C.blue},
    {text:"Explain gradient descent",       color:C.purple},
    {text:"Fed rate decision news",         color:C.yellow},
  ];

  // [label, x, y, sims[4 queries]]
  var chunks = [
    {label:"Refund within 30 days",       x:195, y:75,  sims:[0.93,0.70,0.08,0.05]},
    {label:"Returns take 5-7 days",       x:235, y:135, sims:[0.80,0.92,0.07,0.04]},
    {label:"Shipping non-refundable",     x:155, y:200, sims:[0.68,0.55,0.06,0.03]},
    {label:"Email support@shop.com",      x:280, y:95,  sims:[0.74,0.60,0.05,0.02]},
    {label:"Int\\'l orders up to 14 days",x:185, y:260, sims:[0.55,0.78,0.06,0.03]},
    {label:"Stock market fell 3%",        x:510, y:185, sims:[0.04,0.03,0.10,0.94]},
    {label:"Fed rate unchanged",          x:470, y:260, sims:[0.03,0.04,0.08,0.89]},
    {label:"Backprop through layers",     x:440, y:95,  sims:[0.05,0.04,0.91,0.12]},
    {label:"Learning rate scheduling",    x:500, y:55,  sims:[0.04,0.03,0.88,0.10]},
    {label:"The cat sat on the mat",      x:330, y:345, sims:[0.06,0.05,0.04,0.03]},
  ];

  var q       = queries[qsel];
  var sims    = chunks.map(function(c){return c.sims[qsel];});
  var sorted  = chunks.slice().map(function(c,i){return {i:i,s:c.sims[qsel]};}).sort(function(a,b){return b.s-a.s;});
  var top3    = [sorted[0].i, sorted[1].i, sorted[2].i];

  function simColor(s) {
    if(s>=0.8) return C.green; if(s>=0.5) return C.yellow; if(s>=0.2) return C.orange; return C.dim;
  }

  return(
    <div>
      <SectionTitle title="Vector Embeddings & Cosine Similarity" subtitle={"Select a query "+DASH+" watch which chunks light up in the embedding space"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {queries.map(function(q2,i){var on=qsel===i;return(
          <button key={i} onClick={function(){setQsel(i);}} style={{
            padding:"7px 14px",borderRadius:20,
            border:"1.5px solid "+(on?q2.color:C.border),
            background:on?q2.color+"20":C.card,
            color:on?q2.color:C.muted,cursor:"pointer",
            fontSize:12,fontWeight:700,fontFamily:"monospace",transition:"all 0.2s",
          }}>{LQ+q2.text+RQ}</button>
        );})}
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={660} height={410} viewBox="0 0 660 410" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          {/* Cluster labels */}
          <text x={215} y={40} textAnchor="middle" fill={C.blue+"40"} fontSize={10} fontFamily="monospace">[ Refund / Returns cluster ]</text>
          <text x={480} y={40} textAnchor="middle" fill={C.purple+"40"} fontSize={10} fontFamily="monospace">[ ML / AI cluster ]</text>
          <text x={490} y={320} textAnchor="middle" fill={C.yellow+"40"} fontSize={10} fontFamily="monospace">[ Finance cluster ]</text>
          <text x={330} y={390} textAnchor="middle" fill={C.dim+"60"} fontSize={10} fontFamily="monospace">[ Unrelated ]</text>

          {/* Axis labels */}
          <text x={330} y={405} textAnchor="middle" fill={C.dim} fontSize={10} fontFamily="monospace">Semantic Dimension 1</text>
          <text x={12} y={210} textAnchor="middle" fill={C.dim} fontSize={10} fontFamily="monospace" transform="rotate(-90,12,210)">Semantic Dimension 2</text>

          {/* Query point */}
          <circle cx={310} cy={310} r={14} fill={q.color+"20"} stroke={q.color} strokeWidth={2} style={{animation:"pulse 1.5s infinite"}}/>
          <text x={310} y={314} textAnchor="middle" fill={q.color} fontSize={11} fontWeight={700} fontFamily="monospace">Q</text>
          <text x={310} y={335} textAnchor="middle" fill={q.color} fontSize={9} fontFamily="monospace">[query]</text>

          {/* Similarity lines to top-3 */}
          {top3.map(function(ci){
            var c=chunks[ci]; var s=sims[ci];
            return <line key={"sl"+ci} x1={310} y1={310} x2={c.x} y2={c.y} stroke={q.color} strokeWidth={s*2.5} opacity={s*0.65} strokeDasharray="4,3"/>;
          })}

          {/* Chunk dots */}
          {chunks.map(function(c,i){
            var s=sims[i]; var isTop=top3.indexOf(i)!==-1; var col=simColor(s); var r=isTop?11:7;
            return(
              <g key={"cd"+i}>
                <circle cx={c.x} cy={c.y} r={r+4} fill={col+"12"}/>
                <circle cx={c.x} cy={c.y} r={r} fill={col+"28"} stroke={col} strokeWidth={isTop?2:1}/>
                {isTop&&<text x={c.x} y={c.y+4} textAnchor="middle" fill={col} fontSize={11} fontWeight={800} fontFamily="monospace">{top3.indexOf(i)+1}</text>}
                <text x={c.x+(c.x<310?-16:16)} y={c.y+4} textAnchor={c.x<310?"end":"start"} fill={isTop?col:C.dim} fontSize={9} fontFamily="monospace">{c.label}</text>
                {isTop&&<text x={c.x+(c.x<310?-16:16)} y={c.y+15} textAnchor={c.x<310?"end":"start"} fill={col} fontSize={8} fontWeight={700} fontFamily="monospace">{s.toFixed(2)}</text>}
              </g>
            );
          })}

          {/* Legend */}
          {[{l:">0.8 high match",c:C.green},{l:"0.5-0.8 related",c:C.yellow},{l:"<0.2 unrelated",c:C.dim}].map(function(it,i){return(
            <g key={"leg"+i}>
              <circle cx={30} cy={340+i*17} r={4} fill={it.c+"25"} stroke={it.c} strokeWidth={1}/>
              <text x={42} y={344+i*17} fill={it.c} fontSize={9} fontFamily="monospace">{it.l}</text>
            </g>
          );})}
        </svg>
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:10}}>Top-3 Retrieved Chunks {DASH} {LQ+q.text+RQ}</div>
        {sorted.slice(0,3).map(function(it,rank){var c=chunks[it.i]; var col=simColor(it.s);return(
          <div key={rank} style={{display:"flex",alignItems:"center",gap:12,padding:"10px 0",borderBottom:rank<2?"1px solid "+C.border+"40":"none"}}>
            <div style={{width:26,height:26,borderRadius:6,background:col+"20",border:"1.5px solid "+col,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:800,color:col,flexShrink:0}}>{rank+1}</div>
            <div style={{flex:1,fontSize:12,color:C.text,fontFamily:"monospace"}}>{LQ+c.label+RQ}</div>
            <div style={{textAlign:"right",flexShrink:0}}>
              <div style={{fontSize:19,fontWeight:800,color:col}}>{it.s.toFixed(2)}</div>
              <div style={{fontSize:10,color:C.dim}}>cosine sim</div>
            </div>
          </div>
        );})}
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.purple,marginBottom:8}}>Cosine Similarity Formula</div>
        <div style={{display:"flex",justifyContent:"center"}}>
          <svg width={600} height={90} viewBox="0 0 600 90" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
            <text x={300} y={22} textAnchor="middle" fill={C.purple} fontSize={14} fontWeight={700} fontFamily="monospace">{"cos(Q, C) = (Q \u00B7 C) / (|Q| "+MUL+" |C|)"}</text>
            <text x={300} y={44} textAnchor="middle" fill={C.muted} fontSize={11} fontFamily="monospace">{"Q \u00B7 C = dot product: multiply element-wise, then sum"}</text>
            <text x={300} y={60} textAnchor="middle" fill={C.muted} fontSize={11} fontFamily="monospace">|Q|, |C| = L2 norms {DASH} result ranges from -1.0 to +1.0</text>
            <text x={300} y={74} textAnchor="middle" fill={C.green} fontSize={10} fontFamily="monospace">pre-normalised vectors: cos(Q, C) = Q {"\u00B7"} C (just a dot product)</text>
          </svg>
        </div>
      </Card>

      <Insight>
        The embedding model is <span style={{color:C.purple,fontWeight:700}}>never updated</span> during RAG operation. Each text produces a different output vector from the same frozen model. The key insight: <span style={{color:C.accent,fontWeight:700}}>"refund" and "return policy"</span> end up geometrically close in 1536-dimensional space even though they share no words.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: CHUNKING STRATEGIES
   =============================================================== */
function TabChunking() {
  var _s = useState(0); var sel = _s[0], setSel = _s[1];

  var strats = [
    { name:"Fixed-Size", color:C.accent,
      desc:"Split every N tokens regardless of sentence boundaries. Simple and uniform, but may cut sentences mid-thought. Overlap (10-15%) prevents context loss at edges.",
      chunks:[
        {text:"The refund policy requires submission within 30 days of purchase. Claims must include the",         toks:16, overlap:false},
        {text:"Claims must include the original receipt and order number. Processing takes 5 to 7",               toks:16, overlap:true},
        {text:"Processing takes 5 to 7 business days after approval. International orders may require up to",    toks:16, overlap:true},
        {text:"International orders may require up to 14 days. Digital products are non-refundable once",        toks:16, overlap:true},
      ]
    },
    { name:"Sentence", color:C.blue,
      desc:"Split at sentence boundaries. Each sentence becomes one chunk. Grammatically clean but sizes vary wildly "+DASH+" a 3-word sentence loses all surrounding context.",
      chunks:[
        {text:"The refund policy requires submission within 30 days of purchase.",    toks:12, overlap:false},
        {text:"Claims must include the original receipt and order number.",            toks:10, overlap:false},
        {text:"Processing takes 5 to 7 business days after approval.",                toks:10, overlap:false},
        {text:"International orders may require up to 14 days for currency conversion.",toks:12, overlap:false},
        {text:"Digital products are non-refundable once the download has started.",   toks:11, overlap:false},
        {text:"Damaged goods qualify for full reimbursement with photographic evidence.",toks:10, overlap:false},
      ]
    },
    { name:"Paragraph", color:C.purple,
      desc:"Group natural paragraph breaks together. Highest semantic coherence but chunks can be 800+ tokens "+DASH+" very uneven sizes that are hard to embed precisely.",
      chunks:[
        {text:"The refund policy requires submission within 30 days of purchase. Claims must include the original receipt and order number. Processing takes 5 to 7 business days after approval.",                                         toks:38, overlap:false},
        {text:"International orders may require up to 14 days for currency conversion. Digital products are non-refundable once the download has started. Damaged goods qualify for full reimbursement with photographic evidence.", toks:43, overlap:false},
      ]
    },
    { name:"Semantic", color:C.green,
      desc:"Detect topic shifts via embedding similarity. A new chunk begins when cosine similarity between consecutive sentences drops below a threshold. Aligns with actual topic boundaries.",
      chunks:[
        {text:"The refund policy requires submission within 30 days of purchase. Claims must include the original receipt and order number.",                                   toks:22, overlap:false, note:"topic: submission rules"},
        {text:"Processing takes 5 to 7 business days after approval. International orders may require up to 14 days for currency conversion.",                                toks:24, overlap:false, note:"topic: processing timelines"},
        {text:"Digital products are non-refundable once the download has started. Damaged goods qualify for full reimbursement with photographic evidence.",                  toks:25, overlap:false, note:"topic: special cases"},
      ]
    },
  ];

  var st = strats[sel];
  var avgTok = Math.round(st.chunks.reduce(function(a,c){return a+c.toks;},0)/st.chunks.length);

  return(
    <div>
      <SectionTitle title="Chunking Strategies" subtitle={"How you split documents directly determines retrieval precision "+DASH+" select a strategy to compare"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {strats.map(function(st2,i){var on=sel===i;return(
          <button key={i} onClick={function(){setSel(i);}} style={{
            padding:"8px 18px",borderRadius:20,
            border:"1.5px solid "+(on?st2.color:C.border),
            background:on?st2.color+"20":C.card,
            color:on?st2.color:C.muted,
            cursor:"pointer",fontSize:12,fontWeight:700,fontFamily:"monospace",transition:"all 0.2s",
          }}>{st2.name}</button>
        );})}
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:14,fontWeight:700,color:st.color,marginBottom:6}}>{st.name} Chunking</div>
        <div style={{fontSize:13,color:C.muted,lineHeight:1.7,marginBottom:16}}>{st.desc}</div>
        <div style={{display:"flex",flexDirection:"column",gap:8}}>
          {st.chunks.map(function(c,i){return(
            <div key={i} style={{borderRadius:8,padding:"12px 16px",background:st.color+"10",border:"1px solid "+st.color+(c.overlap?"55":"28"),position:"relative"}}>
              {c.overlap&&<span style={{position:"absolute",top:6,right:10,fontSize:10,color:st.color+"99",fontFamily:"monospace",fontWeight:700}}>OVERLAP</span>}
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                <span style={{fontSize:11,color:st.color,fontFamily:"monospace",fontWeight:700}}>
                  {"CHUNK "+(i+1)+(c.note?" "+DASH+" "+c.note:"")}
                </span>
                <span style={{fontSize:11,color:C.muted,fontFamily:"monospace"}}>{c.toks+" tok"}</span>
              </div>
              <div style={{fontSize:12,color:C.text,lineHeight:1.7,fontStyle:"italic"}}>{LQ}{c.text}{RQ}</div>
            </div>
          );})}
        </div>
        <div style={{display:"flex",gap:24,marginTop:16,paddingTop:14,borderTop:"1px solid "+C.border}}>
          {[
            {l:"CHUNKS",    v:st.chunks.length, s:"total"},
            {l:"AVG SIZE",  v:avgTok,            s:"tokens"},
            {l:"OVERLAP",   v:sel===0?"YES":"NO", s:""},
            {l:"SEMANTIC",  v:sel===3?"YES":"NO", s:"boundary"},
          ].map(function(d,i){return(
            <div key={i} style={{textAlign:"center"}}>
              <div style={{fontSize:10,color:C.muted,marginBottom:2}}>{d.l}</div>
              <div style={{fontSize:23,fontWeight:800,color:st.color}}>{d.v}</div>
              {d.s&&<div style={{fontSize:10,color:C.dim}}>{d.s}</div>}
            </div>
          );})}
        </div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.text,marginBottom:12}}>Strategy Comparison</div>
        {[
          {name:"Fixed-Size", pros:"Simple, uniform, predictable cost",              cons:"Cuts mid-sentence, context breaks at boundary",  c:C.accent},
          {name:"Sentence",   pros:"Grammatically clean, good for fact-dense text",   cons:"Tiny chunks lose context, sizes vary 5-50x",     c:C.blue},
          {name:"Paragraph",  pros:"Highest semantic coherence per chunk",             cons:"20-800 token variance, hard to embed precisely", c:C.purple},
          {name:"Semantic",   pros:"Aligns with topic shifts, best precision",         cons:"Slower: extra embedding pass at indexing time",  c:C.green},
        ].map(function(r,i){return(
          <div key={i} style={{display:"flex",gap:12,padding:"8px 0",borderBottom:i<3?"1px solid "+C.border+"30":"none",alignItems:"flex-start"}}>
            <div style={{width:82,fontSize:11,color:r.c,fontFamily:"monospace",fontWeight:700,flexShrink:0}}>{r.name}</div>
            <div style={{flex:1,fontSize:12,color:C.green,lineHeight:1.6}}>{CHK+" "+r.pros}</div>
            <div style={{flex:1,fontSize:12,color:C.red+"99",lineHeight:1.6}}>{WARN+" "+r.cons}</div>
          </div>
        );})}
      </Card>

      <Insight>
        Chunk size is the <span style={{color:C.accent,fontWeight:700}}>single most impactful decision</span> in a RAG system. 200-500 tokens with 10-15% overlap is the practical starting point. <span style={{color:C.green,fontWeight:700}}>Semantic chunking</span> gives the best retrieval quality at the cost of indexing speed.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: RETRIEVAL & RERANKING
   =============================================================== */
function TabRetrieval() {
  var _r = useState(false); var reranked = _r[0], setReranked = _r[1];
  var _m = useState("hybrid"); var mode = _m[0], setMode = _m[1];

  var query = "How long does an international refund take?";

  var cands = [
    {id:0, text:"International orders may require up to 14 days for currency conversion.", bm25:0.88, vec:0.91, cross:0.96, rb:1, rc:1},
    {id:1, text:"Processing takes 5 to 7 business days after approval.",                   bm25:0.40, vec:0.78, cross:0.82, rb:2, rc:2},
    {id:2, text:"Refund policy requires submission within 30 days of purchase.",           bm25:0.65, vec:0.74, cross:0.61, rb:3, rc:4},
    {id:3, text:"Shipping is non-refundable unless the error was ours.",                   bm25:0.72, vec:0.58, cross:0.44, rb:5, rc:6},
    {id:4, text:"Digital products are non-refundable once downloaded.",                    bm25:0.58, vec:0.56, cross:0.38, rb:6, rc:7},
    {id:5, text:"Claims require original receipt and order number.",                       bm25:0.45, vec:0.62, cross:0.56, rb:4, rc:5},
    {id:6, text:"Damaged goods qualify for full reimbursement with evidence.",             bm25:0.30, vec:0.52, cross:0.42, rb:7, rc:8},
    {id:7, text:"Contact support@shop.com for all refund queries.",                        bm25:0.55, vec:0.49, cross:0.50, rb:8, rc:3},
  ];

  function biScore(c) {
    if(mode==="bm25") return c.bm25;
    if(mode==="vector") return c.vec;
    var bRank=cands.filter(function(x){return x.bm25>c.bm25;}).length+1;
    var vRank=cands.filter(function(x){return x.vec>c.vec;}).length+1;
    return 1/(60+bRank)+1/(60+vRank);
  }

  var sorted = cands.slice().sort(function(a,b){
    if(reranked) return a.rc-b.rc;
    return biScore(b)-biScore(a);
  });

  function rankColor(r){if(r===1)return C.green;if(r===2)return C.yellow;if(r===3)return C.orange;return C.dim;}

  return(
    <div>
      <SectionTitle title="Retrieval & Reranking" subtitle={"Bi-encoder (fast) finds candidates "+DASH+" cross-encoder (precise) reorders them"} />

      <div style={{maxWidth:750,margin:"0 auto 12px",padding:"12px 18px",background:C.accent+"10",border:"1px solid "+C.accent+"30",borderRadius:8}}>
        <div style={{fontSize:11,color:C.muted,marginBottom:4,fontFamily:"monospace"}}>QUERY</div>
        <div style={{fontSize:14,color:C.accent,fontFamily:"monospace",fontWeight:700}}>{LQ+query+RQ}</div>
      </div>

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:14,flexWrap:"wrap"}}>
        <div style={{fontSize:11,color:C.muted,display:"flex",alignItems:"center",marginRight:4}}>Search mode:</div>
        {[["bm25","BM25 keyword",C.yellow],["vector","Vector semantic",C.blue],["hybrid","Hybrid RRF",C.green]].map(function(m){var on=mode===m[0];return(
          <button key={m[0]} onClick={function(){setMode(m[0]);}} style={{padding:"5px 12px",borderRadius:20,border:"1.5px solid "+(on?m[2]:C.border),background:on?m[2]+"20":C.card,color:on?m[2]:C.muted,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"monospace"}}>{m[1]}</button>
        );})}
        <div style={{width:1,background:C.border,margin:"0 6px"}}/>
        <button onClick={function(){setReranked(!reranked);}} style={{padding:"5px 14px",borderRadius:20,border:"1.5px solid "+(reranked?C.purple:C.border),background:reranked?C.purple+"20":C.card,color:reranked?C.purple:C.muted,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"monospace"}}>{reranked?"Cross-encoder ON":"Cross-encoder OFF"}</button>
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 14px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <div style={{fontSize:13,fontWeight:700,color:C.text}}>{reranked?"After cross-encoder reranking":"Bi-encoder / initial ranking"}</div>
          <div style={{fontSize:11,color:C.muted,fontFamily:"monospace"}}>{reranked?"SLOWER "+DASH+" MORE ACCURATE":"FAST "+DASH+" APPROXIMATE"}</div>
        </div>
        {sorted.map(function(c,i){
          var col=rankColor(i+1);
          var moved=reranked&&c.rb!==c.rc;
          var movedUp=c.rb>c.rc;
          return(
            <div key={c.id} style={{display:"flex",alignItems:"center",gap:10,padding:"9px 0",borderBottom:i<sorted.length-1?"1px solid "+C.border+"30":"none",transition:"all 0.4s"}}>
              <div style={{width:26,height:26,borderRadius:6,background:col+"20",border:"1.5px solid "+col,display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:800,color:col,flexShrink:0}}>{i+1}</div>
              <div style={{flex:1,fontSize:12,color:i<3?C.text:C.muted,lineHeight:1.5}}>{c.text}</div>
              {reranked&&moved&&<span style={{fontSize:15,color:movedUp?C.green:C.red,fontWeight:700,flexShrink:0}}>{movedUp?UP:DOWN}</span>}
              <div style={{display:"flex",gap:8,flexShrink:0}}>
                {mode!=="bm25"&&<div style={{textAlign:"center"}}><div style={{fontSize:9,color:C.dim}}>VEC</div><div style={{fontSize:12,color:C.blue,fontFamily:"monospace",fontWeight:700}}>{c.vec.toFixed(2)}</div></div>}
                {mode!=="vector"&&<div style={{textAlign:"center"}}><div style={{fontSize:9,color:C.dim}}>BM25</div><div style={{fontSize:12,color:C.yellow,fontFamily:"monospace",fontWeight:700}}>{c.bm25.toFixed(2)}</div></div>}
                {reranked&&<div style={{textAlign:"center"}}><div style={{fontSize:9,color:C.dim}}>CROSS</div><div style={{fontSize:12,color:C.purple,fontFamily:"monospace",fontWeight:700}}>{c.cross.toFixed(2)}</div></div>}
              </div>
            </div>
          );
        })}
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.text,marginBottom:12}}>Bi-encoder vs Cross-encoder</div>
        <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {name:"Bi-encoder",col:C.blue,  spd:"fast (pre-computed)", acc:"approximate",
             how:"Query and chunk encoded SEPARATELY. Cosine similarity between vectors. All chunk vectors pre-computed at index time "+DASH+" millisecond lookup at query time."},
            {name:"Cross-encoder",col:C.purple,spd:"slow (query-time)",acc:"high precision",
             how:"[Query + Chunk] encoded JOINTLY in one forward pass. Self-attention across both. Score = direct relevance. Cannot pre-compute "+DASH+" must run for every candidate."},
          ].map(function(e,i){return(
            <div key={i} style={{flex:1,minWidth:220,padding:"12px 16px",background:e.col+"08",border:"1px solid "+e.col+"25",borderRadius:8}}>
              <div style={{fontSize:13,color:e.col,fontWeight:700,marginBottom:8}}>{e.name}</div>
              <div style={{fontSize:12,color:C.muted,lineHeight:1.7,marginBottom:8}}>{e.how}</div>
              <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                <span style={{fontSize:11,color:e.col,background:e.col+"15",padding:"2px 6px",borderRadius:4,fontFamily:"monospace"}}>{e.spd}</span>
                <span style={{fontSize:11,color:e.col,background:e.col+"15",padding:"2px 6px",borderRadius:4,fontFamily:"monospace"}}>{e.acc}</span>
              </div>
            </div>
          );})}
        </div>
      </Card>

      <Insight icon={TARG} title="The ANN + Reranker Pattern">
        ANN search handles <span style={{color:C.blue,fontWeight:700}}>scale</span> (millions of chunks in ms). Cross-encoder handles <span style={{color:C.purple,fontWeight:700}}>precision</span> (which 5 of 50 actually matter). Notice chunk #8 (contact email) jumps from rank 8 to rank 3 after reranking {DASH} the bi-encoder missed its relevance because the surface text looks unrelated, but the cross-encoder reads it jointly with the query and sees the connection.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: CONTEXT WINDOW & AGENTIC RAG
   =============================================================== */
function TabContext() {
  var _m = useState("standard"); var mode = _m[0], setMode = _m[1];
  var _rs = useState(0); var rStep = _rs[0], setRStep = _rs[1];
  var _a = useState(false); var auto = _a[0], setAuto = _a[1];

  useEffect(function(){
    if(!auto||mode!=="agentic")return;
    var t=setInterval(function(){setRStep(function(s){return(s+1)%4;});},2000);
    return function(){clearInterval(t);};
  },[auto,mode]);

  var budgets = {
    standard:[
      {label:"System prompt",         tok:500,   pct:4,  color:C.dim},
      {label:"Retrieved chunks (RAG)",tok:6000,  pct:46, color:C.purple},
      {label:"Episodic memory",        tok:600,   pct:5,  color:C.cyan},
      {label:"Conversation history",   tok:3000,  pct:23, color:C.blue},
      {label:"User query",             tok:100,   pct:1,  color:C.dim},
      {label:"Output buffer",          tok:2800,  pct:21, color:C.accent},
    ],
    agentic:[
      {label:"System + instructions",     tok:800,  pct:5,  color:C.dim},
      {label:"Task working memory",       tok:3000, pct:19, color:C.yellow},
      {label:"Retrieved chunks (round N)",tok:4000, pct:25, color:C.purple},
      {label:"Agent scratchpad",          tok:2500, pct:16, color:C.orange},
      {label:"ReAct turns (last 3)",      tok:2400, pct:15, color:C.cyan},
      {label:"Output buffer",             tok:3100, pct:20, color:C.accent},
    ],
  };

  var bud   = budgets[mode];
  var total = bud.reduce(function(a,b){return a+b.tok;},0);

  var reactSteps = [
    {label:"THOUGHT", color:C.yellow, text:"I need international refund timelines. My prior search returned only domestic policy. I should query specifically for international + currency."},
    {label:"ACTION",  color:C.cyan,   text:"search("+LQ+"international refund processing time currency conversion 14 days"+RQ+")"},
    {label:"OBS",     color:C.green,  text:"Found: "+LQ+"International orders may require up to 14 days for currency conversion."+RQ+" "+DASH+" chunk from policy_2024.pdf, section 3.2"},
    {label:"THOUGHT", color:C.yellow, text:"I now have both domestic (5-7 days) and international (up to 14 days). I can answer fully. Writing to scratchpad before generating."},
  ];

  var memLayers = [
    {name:"In-context scratchpad",   scope:"ephemeral per call",   color:C.accent, desc:"Working memory for this forward pass. Wiped when call ends."},
    {name:"Task working memory",      scope:"ephemeral per task",   color:C.yellow, desc:"Compressed findings across retrieval rounds. Serialised JSON between calls."},
    {name:"External vector index",    scope:"persistent",           color:C.purple, desc:"All indexed documents. Vast but retrieval-only. Not visible to agent unless queried."},
    {name:"Episodic memory store",    scope:"persistent per user",  color:C.blue,   desc:"Session summaries, entity facts, user preferences. Retrieved at task start."},
  ];

  return(
    <div>
      <SectionTitle title="Context Window & Agentic RAG" subtitle={"Token budget management in standard RAG vs the iterative Agentic retrieval loop"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:20}}>
        {[["standard","Standard RAG"],["agentic","Agentic RAG"]].map(function(m){var on=mode===m[0];return(
          <button key={m[0]} onClick={function(){setMode(m[0]);setRStep(0);setAuto(false);}} style={{
            padding:"9px 24px",borderRadius:20,
            border:"1.5px solid "+(on?C.accent:C.border),
            background:on?C.accent+"20":C.card,
            color:on?C.accent:C.muted,cursor:"pointer",
            fontSize:13,fontWeight:700,fontFamily:"monospace",
          }}>{m[1]}</button>
        );})}
      </div>

      <div style={{display:"flex",gap:16,justifyContent:"center",flexWrap:"wrap",marginBottom:16}}>

        {/* Token budget card */}
        <Card style={{flex:1,minWidth:280,maxWidth:350}}>
          <div style={{fontSize:13,fontWeight:700,color:C.text,marginBottom:4}}>Context Window Budget</div>
          <div style={{fontSize:11,color:C.muted,fontFamily:"monospace",marginBottom:10}}>{total.toLocaleString()+" tok used (16K window)"}</div>
          <div style={{height:26,borderRadius:6,overflow:"hidden",display:"flex",marginBottom:14}}>
            {bud.map(function(b,i){return(
              <div key={i} style={{width:b.pct+"%",background:b.color+"55",borderRight:"1px solid #0a0a0f",transition:"width 0.5s"}}/>
            );})}
          </div>
          {bud.map(function(b,i){return(
            <div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:7}}>
              <div style={{width:10,height:10,borderRadius:2,background:b.color+"55",border:"1px solid "+b.color,flexShrink:0}}/>
              <div style={{flex:1,fontSize:12,color:C.muted}}>{b.label}</div>
              <div style={{fontSize:12,color:b.color,fontFamily:"monospace",fontWeight:700}}>{b.tok.toLocaleString()}</div>
            </div>
          );})}
        </Card>

        {/* ReAct loop or memory layers */}
        <Card style={{flex:1,minWidth:300,maxWidth:390}}>
          {mode==="standard"?(
            <div>
              <div style={{fontSize:13,fontWeight:700,color:C.text,marginBottom:12}}>Memory Layers</div>
              {memLayers.map(function(ml,i){return(
                <div key={i} style={{padding:"10px 14px",marginBottom:8,borderRadius:8,background:ml.color+"08",border:"1px solid "+ml.color+"25"}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:4}}>
                    <div style={{fontSize:12,fontWeight:700,color:ml.color}}>{ml.name}</div>
                    <span style={{fontSize:10,color:ml.color,background:ml.color+"15",padding:"1px 6px",borderRadius:4,fontFamily:"monospace",flexShrink:0,marginLeft:6}}>{ml.scope}</span>
                  </div>
                  <div style={{fontSize:11,color:C.muted,lineHeight:1.6}}>{ml.desc}</div>
                </div>
              );})}
            </div>
          ):(
            <div>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
                <div style={{fontSize:13,fontWeight:700,color:C.text}}>ReAct Loop (Reason + Act)</div>
                <button onClick={function(){setAuto(!auto);}} style={{padding:"4px 10px",borderRadius:6,border:"1px solid "+(auto?C.yellow:C.border),background:auto?C.yellow+"15":C.card,color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{auto?PAUSE+" stop":PLAY+" auto"}</button>
              </div>
              {reactSteps.map(function(rs,i){var active=rStep===i; var done=rStep>i;return(
                <div key={i} onClick={function(){setRStep(i);setAuto(false);}} style={{
                  display:"flex",gap:10,padding:"10px 12px",marginBottom:6,borderRadius:8,
                  background:active?rs.color+"18":done?rs.color+"06":"transparent",
                  border:"1px solid "+(active?rs.color+"55":done?rs.color+"20":C.border),
                  cursor:"pointer",transition:"all 0.3s",
                }}>
                  <div style={{width:58,flexShrink:0,fontSize:11,fontWeight:700,color:active?rs.color:done?rs.color+"80":C.dim,fontFamily:"monospace",paddingTop:1}}>{rs.label}</div>
                  <div style={{fontSize:11,color:active?C.text:done?C.muted:C.dim,lineHeight:1.6}}>{rs.text}</div>
                </div>
              );})}
              <div style={{marginTop:8,padding:"8px 12px",borderRadius:6,background:C.border+"20"}}>
                <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>
                  Each round: +1,500 tok (chunks + ReAct turn). After ~3 rounds, prior turns are compressed into the scratchpad (500 tok {ARR} 50 tok fact note) to prevent window overflow.
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.text,marginBottom:10}}>
          {mode==="standard"?"Context Budget Rules of Thumb (16K window)":"Agentic Burn Rate & Compression"}
        </div>
        {mode==="standard"?(
          <div style={{display:"flex",gap:20,flexWrap:"wrap",justifyContent:"center"}}>
            {[
              {l:"Chunks",    v:"30-50%",   c:C.purple, n:"highest priority"},
              {l:"History",   v:"≤3,000",   c:C.blue,   n:"compress after turn 8"},
              {l:"Episodic",  v:"≤600 tok", c:C.cyan,   n:"compressed summaries"},
              {l:"Output",    v:"≥2,000",   c:C.accent, n:"always hold back"},
            ].map(function(d,i){return(
              <div key={i} style={{textAlign:"center",minWidth:120}}>
                <div style={{fontSize:10,color:C.muted,marginBottom:2}}>{d.l}</div>
                <div style={{fontSize:21,fontWeight:800,color:d.c}}>{d.v}</div>
                <div style={{fontSize:10,color:C.dim}}>{d.n}</div>
              </div>
            );})}
          </div>
        ):(
          <div>
            <div style={{display:"flex",justifyContent:"center"}}>
              <svg width={700} height={118} viewBox="0 0 700 118" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
                <text x={50} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontFamily="monospace">Round 1</text>
                <text x={200} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontFamily="monospace">Round 3</text>
                <text x={360} y={20} textAnchor="middle" fill={C.muted} fontSize={10} fontFamily="monospace">Round 5</text>
                <text x={520} y={20} textAnchor="middle" fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">Round 7 (compress!)</text>

                {/* Budget bars growing across rounds */}
                {[
                  {x:20,  used:5600,  pct:35, col:C.green},
                  {x:170, used:8200,  pct:51, col:C.yellow},
                  {x:330, used:11000, pct:69, col:C.orange},
                  {x:490, used:13800, pct:86, col:C.red},
                ].map(function(b,i){return(
                  <g key={i}>
                    <rect x={b.x} y={28} width={120} height={12} rx={3} fill={C.border}/>
                    <rect x={b.x} y={28} width={Math.round(b.pct*1.2)} height={12} rx={3} fill={b.col+"60"} stroke={b.col} strokeWidth={0.5}/>
                    <text x={b.x+60} y={53} textAnchor="middle" fill={b.col} fontSize={10} fontFamily="monospace">{b.used.toLocaleString()+" tok"}</text>
                    <text x={b.x+60} y={66} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{b.pct+"% used"}</text>
                  </g>
                );})}

                <text x={350} y={92} textAnchor="middle" fill={C.muted} fontSize={11} fontFamily="monospace">Compress every 3 rounds: 500 tok chunk {ARR} 50 tok fact note (10:1 ratio)</text>
                <text x={350} y={106} textAnchor="middle" fill={C.cyan} fontSize={10} fontFamily="monospace">16K window supports ~8-10 rounds before compression required</text>
              </svg>
            </div>
          </div>
        )}
      </Card>

      <Insight icon={TARG} title={mode==="standard"?"The Budget Principle":"Why Context Is the Critical Variable in Agentic RAG"}>
        {mode==="standard"?(
          <span>Even with a 200K window, you must budget. The act of retrieval <span style={{color:C.accent,fontWeight:700}}>IS</span> the act of deciding what the model reads. Context window = <span style={{color:C.blue,fontWeight:700}}>working memory</span>. External vector store = <span style={{color:C.purple,fontWeight:700}}>long-term memory</span>. Everything not retrieved is invisible to the LLM.</span>
        ):(
          <span>In standard RAG, context is assembled once. In Agentic RAG, the scratchpad <span style={{color:C.yellow,fontWeight:700}}>grows every round</span>. Without compression, early findings get evicted. The agent <span style={{color:C.red,fontWeight:700}}>"forgets"</span> what it found in round 1 by round 5. Structured scratchpad compression is not optional {DASH} it is the architecture.</span>
        )}
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP  (with Ctrl+Scroll zoom)
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var _z = useState(1.0); var zoom = _z[0], setZoom = _z[1];
  var _v = useState(false); var badgeVis = _v[0], setBadgeVis = _v[1];
  var hideTimer = React.useRef(null);

  var tabs = ["RAG Pipeline","Vector Embeddings","Chunking","Retrieval & Reranking","Context & Agentic RAG"];

  useEffect(function() {
    function onWheel(e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZoom(function(z) {
        var next = z + (e.deltaY < 0 ? 0.08 : -0.08);
        return Math.round(Math.min(Math.max(next, 0.4), 2.0) * 100) / 100;
      });
      // show badge, reset hide timer
      setBadgeVis(true);
      if (hideTimer.current) clearTimeout(hideTimer.current);
      hideTimer.current = setTimeout(function() { setBadgeVis(false); }, 1800);
    }
    window.addEventListener("wheel", onWheel, {passive: false});
    return function() {
      window.removeEventListener("wheel", onWheel);
      if (hideTimer.current) clearTimeout(hideTimer.current);
    };
  }, []);

  var pct = Math.round(zoom * 100) + "%";

  return(
    <div>
      <div className="zoom-wrap" style={{transform:"scale("+zoom+")"}}>
        <div style={{background:C.bg,minHeight:"100vh",padding:"24px 16px",fontFamily:"'JetBrains Mono','SF Mono',monospace",color:C.text,maxWidth:960,margin:"0 auto"}}>
          <div style={{textAlign:"center",marginBottom:16}}>
            <div style={{fontSize:26,fontWeight:800,background:"linear-gradient(135deg,"+C.accent+","+C.yellow+")",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",display:"inline-block"}}>
              Retrieval-Augmented Generation
            </div>
            <div style={{fontSize:13,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from pipeline to Agentic RAG"}</div>
            <div style={{fontSize:11,color:C.dim,marginTop:2}}>{"Ctrl + scroll to zoom"}</div>
          </div>
          <TabBar tabs={tabs} active={tab} onChange={setTab} />
          {tab===0 && <TabPipeline />}
          {tab===1 && <TabEmbeddings />}
          {tab===2 && <TabChunking />}
          {tab===3 && <TabRetrieval />}
          {tab===4 && <TabContext />}
        </div>
      </div>

      {/* Zoom level badge — fades in on change, fades out after 1.8s */}
      <div className="zoom-badge" style={{opacity:badgeVis?1:0}}>
        {"zoom "}<span>{pct}</span>
        {zoom!==1.0 && (
          <button onClick={function(){setZoom(1.0);setBadgeVis(false);}} style={{
            marginLeft:8,background:"none",border:"none",
            color:C.muted,cursor:"pointer",fontSize:12,padding:"0 2px",
            fontFamily:"'JetBrains Mono',monospace",
          }}>{"reset"}</button>
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

RAG_VISUAL_HEIGHT = 1100