"""
Self-contained HTML for the Weight Initialisation & Normalisation interactive walkthrough.
Covers: Vanishing/Exploding Gradients, Weight Init Strategies (Zero/Naive/Xavier/He/LeCun),
Batch Normalisation step-by-step, and the Normalisation Family (BN/LN/IN/GN).
Embed in Streamlit via:
    st.components.v1.html(WEIGHTINIT_VISUAL_HTML, height=WEIGHTINIT_VISUAL_HEIGHT)
"""

WEIGHTINIT_VISUAL_HTML = """
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
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState   = React.useState;
var useEffect  = React.useEffect;
var useMemo    = React.useMemo;

var C = {
  bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
  accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
};

var ARR  = "\\u2192";
var DASH = "\\u2014";
var CHK  = "\\u2713";
var WARN = "\\u26A0";
var BULB = "\\uD83D\\uDCA1";
var TARG = "\\uD83C\\uDFAF";
var LQ   = "\\u201C";
var RQ   = "\\u201D";
var MUL  = "\\u00D7";
var LARR = "\\u2190";

/* ── Shared UI components (identical pattern to cnn_visual) ─────────────── */
function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange;
  return (
    <div style={{ display:"flex", gap:0, borderBottom:"2px solid "+C.border, marginBottom:24, overflowX:"auto" }}>
      {tabs.map(function(t,i) {
        return (
          <button key={i} onClick={function(){ onChange(i); }} style={{
            padding:"12px 18px", background:"none", border:"none",
            borderBottom: active===i ? "2px solid "+C.accent : "2px solid transparent",
            color: active===i ? C.accent : C.muted, cursor:"pointer",
            fontSize:11, fontWeight:700, fontFamily:"'JetBrains Mono', monospace",
            transition:"all 0.2s", whiteSpace:"nowrap", marginBottom:-2,
          }}>{t}</button>
        );
      })}
    </div>
  );
}

function Card(props) {
  return (
    <div style={Object.assign({
      background:C.card, borderRadius:10, padding:"18px 22px",
      border:"1px solid "+(props.highlight ? C.accent : C.border),
      transition:"border 0.3s",
    }, props.style || {})}>
      {props.children}
    </div>
  );
}

function Insight(props) {
  return (
    <div style={Object.assign({
      maxWidth:750, margin:"16px auto 0",
      padding:"16px 22px", background:"rgba(255,107,53,0.06)",
      borderRadius:10, border:"1px solid rgba(255,107,53,0.2)",
    }, props.style || {})}>
      <div style={{ fontSize:11, fontWeight:700, color:C.accent, marginBottom:6 }}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
      <div style={{ fontSize:11, color:C.muted, lineHeight:1.8 }}>{props.children}</div>
    </div>
  );
}

function SectionTitle(props) {
  return (
    <div style={{ textAlign:"center", marginBottom:20 }}>
      <div style={{ fontSize:18, fontWeight:800, color:C.text, marginBottom:4 }}>{props.title}</div>
      <div style={{ fontSize:12, color:C.muted }}>{props.subtitle}</div>
    </div>
  );
}


/* ===============================================================
   TAB 1: VANISHING & EXPLODING GRADIENTS
   =============================================================== */
function TabGradients() {
  var _w = useState(0.5); var wVal = _w[0], setWVal = _w[1];
  var N_LAYERS = 10;

  var signals = useMemo(function() {
    var out = [1.0];
    for (var l = 0; l < N_LAYERS; l++) out.push(out[out.length-1] * wVal);
    return out;
  }, [wVal]);

  var finalSig = signals[N_LAYERS];
  var status = finalSig < 0.01 ? "VANISHING" : finalSig > 100 ? "EXPLODING" : "STABLE";
  var statusColor = status==="VANISHING" ? C.blue : status==="EXPLODING" ? C.red : C.green;
  var statusMsg = status==="VANISHING"
    ? "Signal collapses to \u22480 "+DASH+" early layers learn nothing"
    : status==="EXPLODING"
    ? "Signal diverges to \u221e "+DASH+" loss becomes NaN"
    : "Signal stays in stable range "+CHK;

  var BAR_W = 680;
  var maxBar = Math.min(Math.max(...signals), 200);

  return (
    <div>
      <SectionTitle
        title="Vanishing & Exploding Gradients"
        subtitle={"Drag the weight slider to see how signal magnitude propagates through 10 layers"}
      />

      <Card style={{ maxWidth:750, margin:"0 auto 20px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:16, marginBottom:20, flexWrap:"wrap" }}>
          <div style={{ fontSize:11, color:C.muted, minWidth:120 }}>Weight value <span style={{ color:C.accent, fontWeight:700 }}>w = {wVal.toFixed(2)}</span></div>
          <input type="range" min={0.1} max={3.0} step={0.05} value={wVal}
            onChange={function(e){ setWVal(parseFloat(e.target.value)); }}
            style={{ flex:1, minWidth:200, accentColor:C.accent }} />
          <div style={{ fontSize:10, color:statusColor, fontWeight:700, minWidth:100 }}>{status}</div>
        </div>

        <div style={{ display:"flex", gap:12, justifyContent:"center", marginBottom:16, flexWrap:"wrap" }}>
          {[0.3, 0.5, 1.0, 2.0, 2.5].map(function(v) {
            return (
              <button key={v} onClick={function(){ setWVal(v); }} style={{
                padding:"4px 12px", borderRadius:6,
                border:"1px solid "+(Math.abs(wVal-v)<0.01 ? C.accent : C.border),
                background: Math.abs(wVal-v)<0.01 ? C.accent+"20" : C.card,
                color: Math.abs(wVal-v)<0.01 ? C.accent : C.muted,
                cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace"
              }}>{"w="+v}</button>
            );
          })}
        </div>

        <svg width="100%" viewBox={"0 0 "+BAR_W+" 220"} style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          <text x={BAR_W/2} y={18} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Signal Magnitude at Each Layer (z = w^L)</text>
          {signals.map(function(s, l) {
            var barH = Math.max(2, Math.min((s / maxBar) * 160, 160));
            var x = 30 + l * (BAR_W - 60) / N_LAYERS;
            var col = s < 0.01 ? C.blue : s > 100 ? C.red : C.green;
            return (
              <g key={l}>
                <rect x={x-12} y={190-barH} width={24} height={barH} rx={3}
                  fill={col+"30"} stroke={col+"80"} />
                <text x={x} y={205} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"L"+l}</text>
                <text x={x} y={185-barH} textAnchor="middle" fill={col} fontSize={7} fontFamily="monospace">
                  {s >= 100 ? s.toFixed(0) : s >= 1 ? s.toFixed(2) : s.toFixed(3)}
                </text>
              </g>
            );
          })}
          <line x1={20} y1={190} x2={BAR_W-10} y2={190} stroke={C.dim} strokeWidth={1} />
        </svg>
      </Card>

      <div style={{ display:"flex", gap:12, maxWidth:750, margin:"0 auto 16px", flexWrap:"wrap" }}>
        {[
          { label:"Vanishing (w=0.5)", eq:"0.5\u00B9\u2070 \u2248 0.001", col:C.blue, desc:"Early layer gradients \u2248 0. Network cannot learn." },
          { label:"Stable   (w=1.0)", eq:"1.0\u00B9\u2070 = 1.000", col:C.green, desc:"Signal preserved. All layers learn equally." },
          { label:"Exploding (w=2.0)", eq:"2.0\u00B9\u2070 \u2248 1024", col:C.red,  desc:"Gradients overflow. Loss diverges to NaN." },
        ].map(function(d, i) {
          return (
            <div key={i} style={{ flex:1, minWidth:180, background:C.card, borderRadius:8, padding:"12px 14px", border:"1px solid "+d.col+"40" }}>
              <div style={{ fontSize:10, fontWeight:700, color:d.col, marginBottom:4 }}>{d.label}</div>
              <div style={{ fontSize:14, fontWeight:800, color:d.col, fontFamily:"monospace", marginBottom:6 }}>{d.eq}</div>
              <div style={{ fontSize:10, color:C.muted, lineHeight:1.6 }}>{d.desc}</div>
            </div>
          );
        })}
      </div>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.yellow, marginBottom:10 }}>The Symmetry Problem — Why Zeros Init Fails</div>
        <svg width="100%" viewBox="0 0 680 120" style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          <text x={160} y={18} textAnchor="middle" fill={C.red} fontSize={10} fontWeight={700} fontFamily="monospace">Zero Init (w=0)</text>
          {[0,1,2].map(function(i) {
            return (
              <g key={i}>
                <circle cx={60} cy={45+i*25} r={14} fill={C.red+"15"} stroke={C.red+"40"} />
                <text x={60} y={49+i*25} textAnchor="middle" fill={C.red} fontSize={9} fontFamily="monospace">{"x"+i}</text>
                <line x1={74} y1={45+i*25} x2={190} y2={60} stroke={C.dim} strokeWidth={1} />
                <line x1={74} y1={45+i*25} x2={190} y2={85} stroke={C.dim} strokeWidth={1} />
              </g>
            );
          })}
          {[0,1].map(function(i) {
            return (
              <g key={i}>
                <circle cx={200} cy={60+i*25} r={14} fill={C.red+"20"} stroke={C.red} strokeWidth={2} />
                <text x={200} y={64+i*25} textAnchor="middle" fill={C.red} fontSize={8} fontFamily="monospace">0.0</text>
              </g>
            );
          })}
          <text x={200} y={108} textAnchor="middle" fill={C.red} fontSize={8} fontWeight={700} fontFamily="monospace">h\u2081=h\u2082 always</text>

          <line x1={340} y1={60} x2={340} y2={100} stroke={C.dim} strokeWidth={1} strokeDasharray="4,3" />

          <text x={500} y={18} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">Random Init (w~N(0,\u03c3))</text>
          {[0,1,2].map(function(i) {
            return (
              <g key={i}>
                <circle cx={390} cy={45+i*25} r={14} fill={C.green+"15"} stroke={C.green+"40"} />
                <text x={390} y={49+i*25} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">{"x"+i}</text>
                <line x1={404} y1={45+i*25} x2={520} y2={60} stroke={C.green} strokeWidth={1} opacity={0.4} />
                <line x1={404} y1={45+i*25} x2={520} y2={85} stroke={C.green} strokeWidth={1} opacity={0.4} />
              </g>
            );
          })}
          {[{v:"0.43"},{v:"-0.17"}].map(function(d,i) {
            return (
              <g key={i}>
                <circle cx={530} cy={60+i*25} r={14} fill={C.green+"20"} stroke={C.green} strokeWidth={2} />
                <text x={530} y={64+i*25} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{d.v}</text>
              </g>
            );
          })}
          <text x={530} y={108} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={700} fontFamily="monospace">h\u2081\u2260h\u2082 \u2192 specialise</text>
        </svg>
      </Card>

      <Insight>
        The <span style={{color:C.blue,fontWeight:700}}>vanishing gradient problem</span> was the primary reason deep networks failed before 2010. He initialisation (2015) solved it for ReLU networks by compensating for the factor of 2 lost when ReLU zeros negative values.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: WEIGHT INITIALISATION STRATEGIES
   =============================================================== */
function TabWeightInit() {
  var _sel = useState(2); var sel = _sel[0], setSel = _sel[1];
  var N_LAYERS = 8;
  var N_IN = 256;

  var strategies = [
    {
      name:"Zero",
      color: C.red,
      formula:"w = 0",
      use:"NEVER",
      activation:"Any",
      desc:"All neurons identical. Symmetry never breaks. No learning possible.",
      getStd: function(l) { return 0.0; },
    },
    {
      name:"Naive N(0,1)",
      color: C.orange,
      formula:"w ~ N(0, 1)",
      use:"Avoid",
      activation:"Any",
      desc:"Variance grows as n_in per layer. Activations saturate \u2192 vanishing gradient.",
      getStd: function(l) { return Math.pow(Math.sqrt(N_IN), l+1) > 1e6 ? 1e6 : Math.min(Math.pow(Math.sqrt(N_IN), l+1), 1e4); },
    },
    {
      name:"Xavier / Glorot",
      color: C.blue,
      formula:"w ~ N(0, \u221a(2/(n_in+n_out)))",
      use:"Sigmoid, Tanh",
      activation:"Sigmoid, Tanh",
      desc:"Preserves activation variance across layers for symmetric activations. Poor for ReLU.",
      getStd: function(l) { var v = 1.0; for (var i=0;i<l+1;i++) v *= 1.0; return Math.max(0.05, 1.0 + (Math.random()-0.5)*0.1); },
    },
    {
      name:"He / Kaiming",
      color: C.green,
      formula:"w ~ N(0, \u221a(2/n_in))",
      use:"ReLU, Leaky ReLU",
      activation:"ReLU, Leaky ReLU",
      desc:"Accounts for ReLU halving the variance. The modern default for deep networks.",
      getStd: function(l) { return Math.max(0.05, 1.0 + (Math.random()-0.5)*0.08); },
    },
    {
      name:"LeCun",
      color: C.purple,
      formula:"w ~ N(0, \u221a(1/n_in))",
      use:"SELU",
      activation:"SELU",
      desc:"Pairs with SELU for self-normalising networks. Activations auto-converge to mean 0, var 1.",
      getStd: function(l) { return Math.max(0.05, 0.95 + (Math.random()-0.5)*0.06); },
    },
  ];

  var strat = strategies[sel];

  var variances = useMemo(function() {
    var out = [];
    for (var l = 0; l < N_LAYERS; l++) {
      if (sel===0) out.push(0);
      else if (sel===1) out.push(Math.min(Math.pow(N_IN, (l+1)/2), 5000));
      else if (sel===2) out.push(0.9 + l*0.015 + Math.sin(l)*0.05);
      else if (sel===3) out.push(0.95 + Math.sin(l*0.7)*0.08);
      else out.push(0.92 + Math.sin(l*0.5)*0.05);
    }
    return out;
  }, [sel]);

  var maxV = Math.max(...variances, 1);

  return (
    <div>
      <SectionTitle
        title="Weight Initialisation Strategies"
        subtitle="Choose a strategy to see its formula, variance behaviour, and when to use it"
      />

      <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
        {strategies.map(function(s, i) {
          var on = sel===i;
          return (
            <button key={i} onClick={function(){ setSel(i); }} style={{
              padding:"8px 14px", borderRadius:8,
              border:"1.5px solid "+(on ? s.color : C.border),
              background: on ? s.color+"20" : C.card,
              color: on ? s.color : C.muted,
              cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace"
            }}>{s.name}</button>
          );
        })}
      </div>

      <Card highlight={true} style={{ maxWidth:750, margin:"0 auto 16px", borderColor:strat.color }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:16 }}>
          <div>
            <div style={{ fontSize:20, fontWeight:800, color:strat.color }}>{strat.name}</div>
            <div style={{ fontSize:13, color:C.text, fontFamily:"monospace", marginTop:6, marginBottom:8 }}>{strat.formula}</div>
            <div style={{ fontSize:11, color:C.muted, lineHeight:1.7, maxWidth:420 }}>{strat.desc}</div>
          </div>
          <div style={{ display:"flex", gap:16, flexWrap:"wrap" }}>
            <div style={{ textAlign:"center" }}>
              <div style={{ fontSize:8, color:C.muted }}>USE FOR</div>
              <div style={{ fontSize:11, fontWeight:700, color:strat.color, marginTop:4, fontFamily:"monospace" }}>{strat.activation}</div>
            </div>
            <div style={{ textAlign:"center" }}>
              <div style={{ fontSize:8, color:C.muted }}>VERDICT</div>
              <div style={{ fontSize:11, fontWeight:700, color:strat.use==="NEVER"?C.red:strat.use==="Avoid"?C.orange:C.green, marginTop:4, fontFamily:"monospace" }}>{strat.use}</div>
            </div>
          </div>
        </div>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>Activation Variance at Each Layer</div>
        <svg width="100%" viewBox="0 0 680 200" style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          <text x={340} y={16} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">Activation Std Dev per Layer (ideal = ~1.0)</text>
          <line x1={40} y1={160} x2={650} y2={160} stroke={C.dim} strokeWidth={1} />
          <line x1={40} y1={40} x2={40} y2={160} stroke={C.dim} strokeWidth={1} />
          <line x1={40} y1={100} x2={650} y2={100} stroke={C.green+"30"} strokeWidth={1} strokeDasharray="6,4" />
          <text x={32} y={103} textAnchor="end" fill={C.green} fontSize={8} fontFamily="monospace">1.0</text>
          {variances.map(function(v, l) {
            var x = 60 + l*(580/N_LAYERS);
            var clampedV = Math.min(v, maxV);
            var barH = Math.max(2, (clampedV/Math.max(maxV,2)) * 110);
            var yTop = 155 - barH;
            return (
              <g key={l}>
                <rect x={x-14} y={yTop} width={28} height={barH} rx={3} fill={strat.color+"25"} stroke={strat.color+"70"} />
                <text x={x} y={172} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"L"+(l+1)}</text>
                <text x={x} y={yTop-3} textAnchor="middle" fill={strat.color} fontSize={7} fontFamily="monospace">
                  {v >= 1000 ? (v/1000).toFixed(1)+"k" : v.toFixed(2)}
                </text>
              </g>
            );
          })}
          {sel===0 && <text x={340} y={130} textAnchor="middle" fill={C.red} fontSize={13} fontWeight={700} fontFamily="monospace">ALL ZEROS \u2014 Symmetry Never Breaks</text>}
          {sel===1 && <text x={340} y={60} textAnchor="middle" fill={C.orange} fontSize={10} fontWeight={700} fontFamily="monospace">Variance EXPLODES \u2192 saturation, vanishing gradient</text>}
          {(sel===2||sel===3||sel===4) && <text x={340} y={185} textAnchor="middle" fill={strat.color} fontSize={9} fontWeight={700} fontFamily="monospace">Variance stays near 1.0 across all layers \u2713</text>}
        </svg>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>All Strategies at a Glance</div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:10, fontFamily:"monospace" }}>
            <thead>
              <tr>
                {["Init","Formula","Var(w)","For Activation","Status"].map(function(h,i) {
                  return <th key={i} style={{ padding:"8px 10px", textAlign:"left", color:C.muted, borderBottom:"1px solid "+C.border, whiteSpace:"nowrap" }}>{h}</th>;
                })}
              </tr>
            </thead>
            <tbody>
              {[
                { n:"Zeros",          f:"0",                    v:"0",           a:"NEVER",          s:C.red },
                { n:"Naive N(0,1)",   f:"N(0, 1\u00b2)",        v:"1",           a:"Avoid",          s:C.orange },
                { n:"Xavier/Glorot",  f:"N(0, 2/(n_in+n_out))", v:"2/(n+n_out)", a:"Sigmoid, Tanh",  s:C.blue },
                { n:"He / Kaiming",   f:"N(0, 2/n_in)",         v:"2/n_in",      a:"ReLU, L-ReLU",   s:C.green },
                { n:"LeCun",          f:"N(0, 1/n_in)",         v:"1/n_in",      a:"SELU",           s:C.purple },
              ].map(function(r,i) {
                var on = sel===i;
                return (
                  <tr key={i} onClick={function(){ setSel(i); }} style={{ cursor:"pointer", background:on?r.s+"10":"transparent", transition:"background 0.2s" }}>
                    <td style={{ padding:"8px 10px", color:on?r.s:C.text, fontWeight:on?700:400, borderBottom:"1px solid "+C.border+"40" }}>{r.n}</td>
                    <td style={{ padding:"8px 10px", color:r.s, borderBottom:"1px solid "+C.border+"40" }}>{r.f}</td>
                    <td style={{ padding:"8px 10px", color:C.muted, borderBottom:"1px solid "+C.border+"40" }}>{r.v}</td>
                    <td style={{ padding:"8px 10px", color:C.cyan, borderBottom:"1px solid "+C.border+"40" }}>{r.a}</td>
                    <td style={{ padding:"8px 10px", color:r.s, fontWeight:700, borderBottom:"1px solid "+C.border+"40" }}>{i===0?"NEVER":i===1?"Avoid":"\u2713 Good"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Insight icon={TARG} title="Modern Default">
        Use <span style={{color:C.green,fontWeight:700}}>He / Kaiming</span> initialisation for any architecture using ReLU or Leaky ReLU. Use <span style={{color:C.blue,fontWeight:700}}>Xavier / Glorot</span> when using Tanh or Sigmoid. Both are built into PyTorch: <span style={{color:C.yellow,fontFamily:"monospace"}}>nn.init.kaiming_normal_</span> and <span style={{color:C.yellow,fontFamily:"monospace"}}>nn.init.xavier_normal_</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: BATCH NORMALISATION STEP-BY-STEP
   =============================================================== */
function TabBatchNorm() {
  var _step = useState(0); var step = _step[0], setStep = _step[1];
  var _mode = useState("train"); var mode = _mode[0], setMode = _mode[1];

  var rawVals = [2.1, 5.8, 3.4, 7.2, 4.6, 1.9];
  var mu = rawVals.reduce(function(a,b){return a+b;},0) / rawVals.length;
  var variance = rawVals.reduce(function(a,b){return a+Math.pow(b-mu,2);},0) / rawVals.length;
  var std = Math.sqrt(variance + 1e-8);
  var normalised = rawVals.map(function(x){ return (x-mu)/std; });
  var gamma = 1.5, beta = 0.5;
  var scaled = normalised.map(function(xh){ return gamma*xh + beta; });

  var runMean = 4.0, runVar = 2.5;
  var inferNorm = rawVals.map(function(x){ return (x-runMean)/Math.sqrt(runVar+1e-8); });
  var inferOut  = inferNorm.map(function(xh){ return gamma*xh+beta; });

  var steps = [
    { title:"Raw Input Activations", desc:"Before BatchNorm. Features have different scales \u2014 mean \u2248 "+mu.toFixed(2)+", std \u2248 "+std.toFixed(2)+". This shifts with every weight update (internal covariate shift).", vals:rawVals, color:C.muted },
    { title:"Step 1: Compute Batch Mean \u03bc_B", desc:"\u03bc_B = (1/m) \u03a3 x_i = "+mu.toFixed(4)+". Computed per feature, across all m examples in the mini-batch.", vals:rawVals, color:C.blue, highlight:"mean" },
    { title:"Step 2: Compute Batch Variance \u03c3\u00b2_B", desc:"\u03c3\u00b2_B = (1/m) \u03a3 (x_i \u2212 \u03bc_B)\u00b2 = "+variance.toFixed(4)+". Measures spread of the activations.", vals:rawVals, color:C.purple, highlight:"var" },
    { title:"Step 3: Normalise x\u0302_i = (x_i \u2212 \u03bc_B) / \u221a(\u03c3\u00b2_B + \u03b5)", desc:"After normalisation: mean \u2248 0, std \u2248 1. Every feature is now on the same scale, removing scale differences between layers.", vals:normalised, color:C.cyan },
    { title:"Step 4: Scale and Shift y_i = \u03b3 x\u0302_i + \u03b2", desc:"\u03b3 (scale) and \u03b2 (shift) are LEARNABLE parameters. Here \u03b3="+gamma+", \u03b2="+beta+". This lets the network undo normalisation if needed. Gradients flow back through \u03b3 and \u03b2 during training.", vals:scaled, color:C.accent },
  ];

  var displayed = mode==="train" ? steps[step] : { title:"Inference Mode \u2014 Using Running Statistics", desc:"At inference time, there is no mini-batch. Instead, we use running mean and variance accumulated during training via exponential moving average. This makes inference deterministic and works for batch size = 1.", vals:inferOut, color:C.yellow };

  var maxAbsVal = Math.max(...(displayed.vals).map(Math.abs));

  return (
    <div>
      <SectionTitle
        title="Batch Normalisation \u2014 Step by Step"
        subtitle={"Watch each transformation applied to a mini-batch of 6 activations"}
      />

      <div style={{ display:"flex", gap:8, justifyContent:"center", marginBottom:16 }}>
        <button onClick={function(){setMode("train");}} style={{ padding:"8px 20px", borderRadius:8, border:"1.5px solid "+(mode==="train"?C.accent:C.border), background:mode==="train"?C.accent+"20":C.card, color:mode==="train"?C.accent:C.muted, cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace" }}>Training Mode</button>
        <button onClick={function(){setMode("infer");}} style={{ padding:"8px 20px", borderRadius:8, border:"1.5px solid "+(mode==="infer"?C.yellow:C.border), background:mode==="infer"?C.yellow+"20":C.card, color:mode==="infer"?C.yellow:C.muted, cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace" }}>Inference Mode</button>
      </div>

      {mode==="train" && (
        <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:16, flexWrap:"wrap" }}>
          {steps.map(function(s,i) {
            return (
              <button key={i} onClick={function(){setStep(i);}} style={{
                padding:"6px 12px", borderRadius:6,
                border:"1.5px solid "+(step===i ? s.color : C.border),
                background: step===i ? s.color+"20" : C.card,
                color: step===i ? s.color : C.muted,
                cursor:"pointer", fontSize:9, fontWeight:700, fontFamily:"monospace", whiteSpace:"nowrap"
              }}>{"Step "+i}</button>
            );
          })}
        </div>
      )}

      <Card highlight={true} style={{ maxWidth:750, margin:"0 auto 16px", borderColor:displayed.color }}>
        <div style={{ fontSize:13, fontWeight:700, color:displayed.color, marginBottom:6 }}>{displayed.title}</div>
        <div style={{ fontSize:11, color:C.muted, lineHeight:1.7 }}>{displayed.desc}</div>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>Activation Values (mini-batch of 6)</div>
        <svg width="100%" viewBox="0 0 680 180" style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          <line x1={40} y1={90} x2={650} y2={90} stroke={C.dim} strokeWidth={1} />
          <text x={30} y={93} textAnchor="end" fill={C.dim} fontSize={8} fontFamily="monospace">0</text>
          {displayed.vals.map(function(v, i) {
            var x = 80 + i*90;
            var scale = Math.min(70 / Math.max(maxAbsVal, 1), 40);
            var barH = Math.abs(v) * scale;
            var isPos = v >= 0;
            var col = displayed.color;
            return (
              <g key={i}>
                <rect x={x-22} y={isPos ? 90-barH : 90} width={44} height={Math.max(barH,2)} rx={3} fill={col+"25"} stroke={col+"80"} />
                <text x={x} y={isPos ? 85-barH : 98+barH} textAnchor="middle" fill={col} fontSize={10} fontWeight={700} fontFamily="monospace">{v.toFixed(2)}</text>
                <text x={x} y={165} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{"x"+i}</text>
              </g>
            );
          })}
          {step>=1 && mode==="train" && (
            <line x1={40} y1={90 - (mu / maxAbsVal) * 40} x2={650} y2={90 - (mu / maxAbsVal) * 40} stroke={C.blue} strokeWidth={1.5} strokeDasharray="6,4" />
          )}
          {step>=1 && mode==="train" && (
            <text x={655} y={90 - (mu / maxAbsVal) * 40 + 4} fill={C.blue} fontSize={8} fontFamily="monospace">{"\u03bc"}</text>
          )}
        </svg>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>Training vs Inference</div>
        <svg width="100%" viewBox="0 0 680 100" style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          <rect x={10} y={15} width={300} height={70} rx={6} fill={C.accent+"08"} stroke={C.accent+"30"} />
          <text x={160} y={35} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">TRAINING</text>
          <text x={160} y={52} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Uses batch \u03bc_B and \u03c3\u00b2_B</text>
          <text x={160} y={68} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Updates running stats via EMA</text>
          <text x={160} y={84} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">running \u2190 (1-m)\u00b7running + m\u00b7batch</text>

          <rect x={370} y={15} width={300} height={70} rx={6} fill={C.yellow+"08"} stroke={C.yellow+"30"} />
          <text x={520} y={35} textAnchor="middle" fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">INFERENCE</text>
          <text x={520} y={52} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Uses running_mean, running_var</text>
          <text x={520} y={68} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Fixed \u2014 no batch needed</text>
          <text x={520} y={84} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">Works for batch size = 1</text>

          <text x={340} y={58} textAnchor="middle" fill={C.dim} fontSize={16} fontWeight={800}>|</text>
        </svg>
      </Card>

      <Insight>
        <span style={{color:C.accent,fontWeight:700}}>\u03b3 and \u03b2</span> are what make BatchNorm powerful {DASH} the network can learn to <span style={{color:C.green,fontWeight:700}}>undo the normalisation</span> if a layer truly needs a non-standard distribution. Without them, BN would permanently fix every layer to mean=0, var=1.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: NORMALISATION FAMILY
   =============================================================== */
function TabNormFamily() {
  var _sel = useState(0); var sel = _sel[0], setSel = _sel[1];

  var methods = [
    {
      name:"Batch Norm",
      short:"BN",
      color:C.accent,
      axes:"N, H, W (per channel)",
      formula:"\u03bc_c = mean over batch+spatial",
      batch:"Needs batch \u2265 16",
      best:"CNNs with large batch",
      avoid:"Small batch, RNNs, 1 GPU",
      desc:"Normalises across the BATCH dimension. Each channel gets its own \u03bc and \u03c3. Invented by Ioffe & Szegedy (2015). Dominant in CV for large batches.",
      dims:{ n:true, c:false, h:true, w:true },
    },
    {
      name:"Layer Norm",
      short:"LN",
      color:C.blue,
      axes:"C, H, W (per sample)",
      formula:"\u03bc = mean over all features per sample",
      batch:"Works with batch = 1",
      best:"Transformers, RNNs, NLP",
      avoid:"CNNs (ignores channel structure)",
      desc:"Normalises across ALL features within ONE sample. Batch-size independent. Standard in Transformers (BERT, GPT) and all sequence models.",
      dims:{ n:false, c:true, h:true, w:true },
    },
    {
      name:"Instance Norm",
      short:"IN",
      color:C.purple,
      axes:"H, W (per channel per sample)",
      formula:"\u03bc = mean over spatial only",
      batch:"Works with batch = 1",
      best:"Style transfer, GANs",
      avoid:"Classification (loses mean info)",
      desc:"Normalises within each channel of each sample independently. Used in neural style transfer \u2014 it removes instance-level contrast while preserving content.",
      dims:{ n:false, c:false, h:true, w:true },
    },
    {
      name:"Group Norm",
      short:"GN",
      color:C.green,
      axes:"G, H, W (groups of channels per sample)",
      formula:"\u03bc = mean over G channels + spatial",
      batch:"Works with batch = 1",
      best:"Object detection, small batches",
      avoid:"When C is not divisible by G",
      desc:"Splits channels into G groups, normalises within each group. Bridge between BN (G=1) and LN (G=C). Proposed by Yuxin Wu & Kaiming He (2018).",
      dims:{ n:false, c:"group", h:true, w:true },
    },
  ];

  var m = methods[sel];

  return (
    <div>
      <SectionTitle
        title="The Normalisation Family"
        subtitle={"All methods normalise then scale+shift "+DASH+" they differ only in WHICH dimensions are averaged"}
      />

      <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
        {methods.map(function(mt,i) {
          var on = sel===i;
          return (
            <button key={i} onClick={function(){setSel(i);}} style={{
              padding:"8px 16px", borderRadius:8,
              border:"1.5px solid "+(on?mt.color:C.border),
              background: on ? mt.color+"20" : C.card,
              color: on ? mt.color : C.muted,
              cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace"
            }}>{mt.name}</button>
          );
        })}
      </div>

      <Card highlight={true} style={{ maxWidth:750, margin:"0 auto 16px", borderColor:m.color }}>
        <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", flexWrap:"wrap", gap:16 }}>
          <div style={{ flex:1 }}>
            <div style={{ fontSize:20, fontWeight:800, color:m.color }}>{m.name}</div>
            <div style={{ fontSize:11, color:C.muted, fontFamily:"monospace", marginTop:4 }}>Normalise over: <span style={{color:m.color}}>{m.axes}</span></div>
            <div style={{ fontSize:11, color:C.muted, lineHeight:1.7, marginTop:8, maxWidth:400 }}>{m.desc}</div>
          </div>
          <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
            <div><div style={{ fontSize:8, color:C.muted }}>BEST FOR</div><div style={{ fontSize:10, fontWeight:700, color:C.green, marginTop:2, fontFamily:"monospace" }}>{m.best}</div></div>
            <div><div style={{ fontSize:8, color:C.muted }}>AVOID WHEN</div><div style={{ fontSize:10, fontWeight:700, color:C.red, marginTop:2, fontFamily:"monospace" }}>{m.avoid}</div></div>
            <div><div style={{ fontSize:8, color:C.muted }}>BATCH SIZE</div><div style={{ fontSize:10, fontWeight:700, color:m.color, marginTop:2, fontFamily:"monospace" }}>{m.batch}</div></div>
          </div>
        </div>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:14 }}>Which Dimensions Are Averaged? (Tensor: N\u00d7C\u00d7H\u00d7W)</div>
        <svg width="100%" viewBox="0 0 680 240" style={{ background:"#08080d", borderRadius:8, border:"1px solid "+C.border }}>
          {methods.map(function(mt, mi) {
            var bx = 20 + mi*168;
            var active = sel===mi;
            var mc = mt.color;
            var dimLabels = ["N","C","H","W"];
            var dimActive = [mt.dims.n===true, mt.dims.c===true||mt.dims.c==="group", mt.dims.h===true, mt.dims.w===true];
            return (
              <g key={mi} onClick={function(){setSel(mi);}} style={{ cursor:"pointer" }}>
                <rect x={bx} y={10} width={150} height={220} rx={8} fill={active?mc+"08":"transparent"} stroke={active?mc:C.border} strokeWidth={active?2:1} />
                <text x={bx+75} y={30} textAnchor="middle" fill={active?mc:C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{mt.short}</text>
                {dimLabels.map(function(dl, di) {
                  var isOn = dimActive[di];
                  var barW = isOn ? 110 : 60;
                  return (
                    <g key={di}>
                      <text x={bx+10} y={60+di*42} fill={isOn?mc:C.dim} fontSize={9} fontWeight={isOn?700:400} fontFamily="monospace">{dl}</text>
                      <rect x={bx+26} y={47+di*42} width={barW} height={20} rx={4}
                        fill={isOn?mc+"30":"transparent"} stroke={isOn?mc+"80":C.dim+"40"} />
                      {mt.dims.c==="group" && di===1 && (
                        <text x={bx+81} y={61} textAnchor="middle" fill={mc} fontSize={7} fontFamily="monospace">G groups</text>
                      )}
                      {(mt.dims.c!=="group" || di!==1) && isOn && (
                        <text x={bx+81} y={61+di*42} textAnchor="middle" fill={mc} fontSize={8} fontFamily="monospace">{"\u2190 averaged"}</text>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}
          <text x={340} y={230} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">Click a method to highlight its normalisation axes</text>
        </svg>
      </Card>

      <Card style={{ maxWidth:750, margin:"0 auto 16px" }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>When to Use Each Method</div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:10, fontFamily:"monospace" }}>
            <thead>
              <tr>
                {["Method","Normalises Over","Min Batch","Primary Use Case"].map(function(h,i){
                  return <th key={i} style={{ padding:"8px 10px", textAlign:"left", color:C.muted, borderBottom:"1px solid "+C.border }}>{h}</th>;
                })}
              </tr>
            </thead>
            <tbody>
              {[
                { n:"Batch Norm",    c:C.accent,  axes:"N, H, W",    b:"\u226516", use:"CNN image classification" },
                { n:"Layer Norm",   c:C.blue,    axes:"C, H, W",    b:"1",       use:"Transformers, GPT, BERT" },
                { n:"Instance Norm",c:C.purple,  axes:"H, W",       b:"1",       use:"Style transfer, GANs" },
                { n:"Group Norm",   c:C.green,   axes:"G\u00d7H\u00d7W",b:"1",  use:"Detection, small-batch training" },
              ].map(function(r,i) {
                var on = sel===i;
                return (
                  <tr key={i} onClick={function(){setSel(i);}} style={{ cursor:"pointer", background:on?r.c+"10":"transparent" }}>
                    <td style={{ padding:"8px 10px", color:r.c, fontWeight:on?700:400, borderBottom:"1px solid "+C.border+"40" }}>{r.n}</td>
                    <td style={{ padding:"8px 10px", color:C.muted, borderBottom:"1px solid "+C.border+"40", fontFamily:"monospace" }}>{r.axes}</td>
                    <td style={{ padding:"8px 10px", color:C.cyan, borderBottom:"1px solid "+C.border+"40" }}>{r.b}</td>
                    <td style={{ padding:"8px 10px", color:C.text, borderBottom:"1px solid "+C.border+"40" }}>{r.use}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Insight icon={TARG} title="The Key Rule">
        <span style={{color:C.accent,fontWeight:700}}>Batch Norm</span> when you have large batches and spatial data (CNNs). <span style={{color:C.blue,fontWeight:700}}>Layer Norm</span> for everything sequence-based (Transformers, RNNs). <span style={{color:C.green,fontWeight:700}}>Group Norm</span> when batch size is small (object detection, medical imaging). <span style={{color:C.purple,fontWeight:700}}>Instance Norm</span> for style and texture.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t = useState(0); var tab = _t[0], setTab = _t[1];
  var tabs = ["Vanishing / Exploding", "Weight Init Strategies", "Batch Normalisation", "Norm Family"];
  return (
    <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px", fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text, maxWidth:960, margin:"0 auto" }}>
      <div style={{ textAlign:"center", marginBottom:16 }}>
        <div style={{ fontSize:22, fontWeight:800, background:"linear-gradient(135deg,"+C.accent+","+C.yellow+")", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>Weight Init & Normalisation</div>
        <div style={{ fontSize:11, color:C.muted, marginTop:4 }}>{"Making deep networks trainable from the first forward pass "+DASH+" interactive visual walkthrough"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab===0 && <TabGradients />}
      {tab===1 && <TabWeightInit />}
      {tab===2 && <TabBatchNorm />}
      {tab===3 && <TabNormFamily />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

WEIGHTINIT_VISUAL_HEIGHT = 1100