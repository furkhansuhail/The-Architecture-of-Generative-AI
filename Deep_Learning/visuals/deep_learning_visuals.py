"""
Self-contained HTML for an interactive "Perceptron → MLP → CNN → RNN → Transformer" progression.
Includes quiz checkpoints between tabs (unlock next tab when quiz is correct).

Embed in Streamlit via:
    import streamlit as st
    import streamlit.components.v1 as components
    from deep_learning_visuals import DEEP_LEARNING_PATH_VISUAL_HTML, DEEP_LEARNING_PATH_VISUAL_HEIGHT
    components.html(DEEP_LEARNING_PATH_VISUAL_HTML, height=DEEP_LEARNING_PATH_VISUAL_HEIGHT)
"""

DEEP_LEARNING_PATH_VISUAL_HEIGHT = 1300

DEEP_LEARNING_PATH_VISUAL_HTML = r"""
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
  @keyframes slideIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes pulse { 0%,100%{opacity:0.65} 50%{opacity:1} }
</style>
</head>
<body>
<div id="root"></div>

<script type="text/babel">
var useState = React.useState;
var useEffect = React.useEffect;
var useMemo = React.useMemo;

var C = {
  bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
  accent: "#ff6b35", blue: "#4ecdc4", purple: "#a78bfa",
  yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
  dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
  cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
};

var MUL = "\u00D7";
var ARR = "\u2192";
var DASH = "\u2014";
var CHK = "\u2713";
var LOCK = "\uD83D\uDD12";
var BULB = "\uD83D\uDCA1";
var TARG = "\uD83C\uDFAF";

/* ─────────────────────────────────────────────────────────────
   Shared UI bits
   ───────────────────────────────────────────────────────────── */

function Card(props) {
  return (
    <div style={{
      background: C.card,
      border: "1.5px solid " + C.border,
      borderRadius: 14,
      padding: 16,
      boxShadow: "0 10px 30px rgba(0,0,0,0.25)",
      animation: "slideIn 0.35s ease-out",
      ...props.style
    }}>
      {props.children}
    </div>
  );
}

function Pill(props) {
  return (
    <span style={{
      display:"inline-flex", alignItems:"center", gap:6,
      padding:"6px 10px", borderRadius:999,
      border:"1px solid " + (props.border || C.border),
      background: (props.bg || (C.border + "30")),
      color: props.color || C.text,
      fontSize: 11, fontWeight: 700,
      fontFamily: "'JetBrains Mono','SF Mono',monospace"
    }}>
      {props.children}
    </span>
  );
}

function SectionTitle(props) {
  return (
    <div style={{marginBottom:12}}>
      <div style={{
        fontSize: 18, fontWeight: 900,
        fontFamily: "'JetBrains Mono','SF Mono',monospace",
        letterSpacing: "0.01em"
      }}>
        {props.title}
      </div>
      {props.subtitle && (
        <div style={{fontSize: 12, color: C.muted, marginTop: 4, lineHeight: 1.35}}>
          {props.subtitle}
        </div>
      )}
    </div>
  );
}

function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange, unlocked = props.unlocked;
  return (
    <div style={{ display: "flex", gap: 0, borderBottom: "2px solid " + C.border, marginBottom: 18, overflowX: "auto" }}>
      {tabs.map(function(t, i) {
        var ok = unlocked[i];
        var isActive = active === i;
        return (
          <button key={i}
            onClick={function() { if(ok) onChange(i); }}
            title={ok ? "" : "Complete the quiz to unlock"}
            style={{
              padding: "12px 16px",
              background: "none",
              border: "none",
              borderBottom: isActive ? "2px solid " + C.accent : "2px solid transparent",
              color: isActive ? C.accent : (ok ? C.text : C.muted),
              cursor: ok ? "pointer" : "not-allowed",
              fontSize: 12,
              fontWeight: 800,
              fontFamily: "'JetBrains Mono','SF Mono',monospace",
              opacity: ok ? 1 : 0.65,
              whiteSpace: "nowrap"
            }}>
            {ok ? t : (LOCK + " " + t)}
          </button>
        );
      })}
    </div>
  );
}

function InfoRow(props) {
  return (
    <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center", marginBottom: 10}}>
      <Pill border={props.color + "55"} bg={props.color + "18"} color={props.color}>
        {BULB} {props.kicker}
      </Pill>
      <div style={{color:C.muted, fontSize:12, lineHeight:1.35}}>
        {props.text}
      </div>
    </div>
  );
}

function Divider() {
  return <div style={{height:1, background:C.border, margin:"14px 0"}} />;
}

/* ─────────────────────────────────────────────────────────────
   Quiz System (unlock next tab)
   ───────────────────────────────────────────────────────────── */

function QuizCard(props) {
  var q = props.q;
  var solved = props.solved;
  var setSolved = props.setSolved;

  var _a = useState(null); var choice = _a[0]; var setChoice = _a[1];
  var _f = useState(null); var feedback = _f[0]; var setFeedback = _f[1];

  useEffect(function(){
    // reset when question changes
    setChoice(null);
    setFeedback(null);
  }, [q && q.id]);

  function submit() {
    if (choice == null) {
      setFeedback({ ok:false, msg:"Pick an option first." });
      return;
    }
    var ok = choice === q.correct;
    setFeedback({ ok: ok, msg: ok ? ("Correct " + CHK + " — " + q.explain_ok) : ("Not quite — " + q.explain_no) });
    if (ok) setSolved(true);
  }

  return (
    <Card style={{marginTop:16}}>
      <SectionTitle title={"Quiz Checkpoint " + TARG} subtitle="Answer correctly to unlock the next tab." />
      <div style={{fontSize:13, fontWeight:800, marginBottom:10}}>{q.prompt}</div>

      <div style={{display:"grid", gap:8}}>
        {q.options.map(function(opt, idx){
          var selected = choice === idx;
          var disabled = solved;
          return (
            <button key={idx}
              onClick={function(){ if(!disabled) setChoice(idx); }}
              style={{
                textAlign:"left",
                padding:"10px 12px",
                borderRadius:10,
                border:"1.5px solid " + (selected ? C.accent : C.border),
                background: selected ? (C.accent + "18") : "#0b0b12",
                color: selected ? C.accent : C.text,
                cursor: disabled ? "not-allowed" : "pointer",
                fontFamily:"'JetBrains Mono','SF Mono',monospace",
                fontSize: 12,
                fontWeight: 700,
                opacity: disabled ? 0.85 : 1
              }}>
              {String.fromCharCode(65 + idx) + ". "} {opt}
            </button>
          );
        })}
      </div>

      <div style={{display:"flex", gap:10, alignItems:"center", marginTop:12, flexWrap:"wrap"}}>
        <button onClick={submit}
          style={{
            padding:"10px 12px",
            borderRadius:10,
            border:"1.5px solid " + (solved ? (C.green+"80") : C.border),
            background: solved ? (C.green+"18") : C.card,
            color: solved ? C.green : C.text,
            cursor: solved ? "not-allowed" : "pointer",
            fontFamily:"'JetBrains Mono','SF Mono',monospace",
            fontSize: 12,
            fontWeight: 800
          }}>
          {solved ? (CHK + " Unlocked") : "Check Answer"}
        </button>

        {feedback && (
          <div style={{
            flex:1,
            padding:"10px 12px",
            borderRadius:10,
            border:"1.5px solid " + (feedback.ok ? (C.green+"70") : (C.red+"70")),
            background: (feedback.ok ? C.green : C.red) + "10",
            color: feedback.ok ? C.green : C.red,
            fontSize: 12,
            fontWeight: 800,
            lineHeight: 1.35
          }}>
            {feedback.msg}
          </div>
        )}
      </div>
    </Card>
  );
}

/* ─────────────────────────────────────────────────────────────
   Diagrams (SVG)
   ───────────────────────────────────────────────────────────── */

function SvgBox(props) {
  return (
    <svg width="100%" height={props.h || 320} viewBox={props.viewBox || "0 0 860 320"}
      style={{ background:"#08080d", borderRadius: 12, border: "1px solid " + C.border }}>
      {props.children}
    </svg>
  );
}

function Node(props) {
  var r = props.r || 18;
  var fill = (props.fill || "#0b0b12");
  var stroke = props.stroke || C.border;
  var label = props.label || "";
  return (
    <g>
      <circle cx={props.x} cy={props.y} r={r} fill={fill} stroke={stroke} strokeWidth={props.sw || 2}/>
      <text x={props.x} y={props.y + 4} textAnchor="middle"
        fill={props.textColor || C.text}
        fontSize={props.fs || 11}
        fontWeight={props.fw || 900}
        fontFamily="'JetBrains Mono','SF Mono',monospace">
        {label}
      </text>
    </g>
  );
}

function Arrow(props) {
  return (
    <line x1={props.x1} y1={props.y1} x2={props.x2} y2={props.y2}
      stroke={props.color || C.muted}
      strokeWidth={props.w || 2}
      strokeDasharray={props.dash || "0"}
      markerEnd="url(#arrowHead)"
      opacity={props.op == null ? 1 : props.op} />
  );
}

function Defs() {
  return (
    <defs>
      <marker id="arrowHead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill={C.muted} />
      </marker>
    </defs>
  );
}

/* ─────────────────────────────────────────────────────────────
   Tabs content
   ───────────────────────────────────────────────────────────── */

function TabPerceptron() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="1) Single Perceptron"
          subtitle="One neuron = dot product + bias + activation. Great for linear patterns, fails on XOR."
        />
        <InfoRow color={C.cyan} kicker="Core Operation"
          text={"Compute z = w·x + b, then apply an activation (step/sigmoid/ReLU) to get the output."} />
        <SvgBox h={320} viewBox="0 0 860 320">
          <Defs/>
          <text x="24" y="28" fill={C.cyan} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"z = w·x + b  " + ARR + "  a = act(z)"}
          </text>

          <Node x={120} y={90} label="x₁" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />
          <Node x={120} y={150} label="x₂" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />
          <Node x={120} y={210} label="x₃" r={18} fill="#0b0b12" stroke={C.border} textColor={C.text} />

          <Node x={360} y={150} label="Σ" r={26} fill={C.cyan+"10"} stroke={C.cyan} textColor={C.cyan} fs={16} />
          <Node x={520} y={150} label="act" r={28} fill={C.accent+"10"} stroke={C.accent} textColor={C.accent} fs={12} />
          <Node x={700} y={150} label="y" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          <Arrow x1={140} y1={90} x2={330} y2={140} color={C.muted} />
          <Arrow x1={140} y1={150} x2={330} y2={150} color={C.muted} />
          <Arrow x1={140} y1={210} x2={330} y2={160} color={C.muted} />

          <Arrow x1={388} y1={150} x2={492} y2={150} color={C.muted} />
          <Arrow x1={548} y1={150} x2={678} y2={150} color={C.muted} />

          <text x="250" y="92" fill={C.muted} fontSize="10" fontFamily="monospace">weights w</text>
          <text x="410" y="120" fill={C.muted} fontSize="10" fontFamily="monospace">+ bias b</text>
          <text x="595" y="120" fill={C.muted} fontSize="10" fontFamily="monospace">activation</text>

          <g>
            <rect x="24" y="250" width="812" height="52" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="273" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"Limitation: only linear decision boundaries " + DASH + " can't solve XOR with a single line."}
            </text>
            <text x="40" y="292" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: add hidden layers (MLP) to compose non-linear features."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabMLP() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="2) Multi-Layer Perceptron (MLP)"
          subtitle="Hidden layers + non-linear activations let you solve XOR and other non-linear problems."
        />
        <InfoRow color={C.purple} kicker="Key Upgrade"
          text={"Add hidden neurons: output becomes a composition of non-linear functions. This creates curved / piecewise boundaries."} />

        <SvgBox h={330} viewBox="0 0 860 330">
          <Defs/>
          <text x="24" y="28" fill={C.purple} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"Non-linear composition: x " + ARR + " hidden features " + ARR + " y"}
          </text>

          {/* Inputs */}
          <Node x={90} y={90} label="x₁" r={18} fill="#0b0b12" stroke={C.border} />
          <Node x={90} y={160} label="x₂" r={18} fill="#0b0b12" stroke={C.border} />
          <Node x={90} y={230} label="x₃" r={18} fill="#0b0b12" stroke={C.border} />

          {/* Hidden */}
          <Node x={320} y={110} label="h₁" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />
          <Node x={320} y={170} label="h₂" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />
          <Node x={320} y={230} label="h₃" r={20} fill={C.purple+"12"} stroke={C.purple} textColor={C.purple} />

          {/* Output */}
          <Node x={560} y={170} label="act" r={28} fill={C.accent+"10"} stroke={C.accent} textColor={C.accent} fs={12} />
          <Node x={730} y={170} label="y" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          {/* Connections input->hidden */}
          {[
            [90,90,320,110],[90,90,320,170],[90,90,320,230],
            [90,160,320,110],[90,160,320,170],[90,160,320,230],
            [90,230,320,110],[90,230,320,170],[90,230,320,230],
          ].map(function(p,i){
            return <line key={i} x1={p[0]+18} y1={p[1]} x2={p[2]-20} y2={p[3]}
              stroke={C.muted} strokeWidth="1.5" opacity="0.7" />;
          })}

          {/* hidden->output */}
          {[
            [320,110,560,170],[320,170,560,170],[320,230,560,170],
          ].map(function(p,i){
            return <line key={i} x1={p[0]+20} y1={p[1]} x2={p[2]-28} y2={p[3]}
              stroke={C.muted} strokeWidth="2" opacity="0.85" />;
          })}

          <Arrow x1={588} y1={170} x2={708} y2={170} color={C.muted} />

          <g>
            <rect x="24" y="260" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="283" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"MLP solves XOR, but scales poorly for images: too many parameters + no spatial inductive bias."}
            </text>
            <text x="40" y="302" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: use local connectivity + weight sharing (CNN)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabCNN() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="3) CNN (Convolutional Neural Network)"
          subtitle="For images/grids: local receptive fields + weight sharing. Detects edges → textures → shapes."
        />
        <InfoRow color={C.accent} kicker="Inductive Bias"
          text={"A small filter (kernel) slides across space: same weights reused everywhere, capturing local spatial patterns efficiently."} />

        <SvgBox h={340} viewBox="0 0 860 340">
          <Defs/>
          <text x="24" y="28" fill={C.accent} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"(kernel) " + ARR + " feature maps " + ARR + " pooling/downsample " + ARR + " head"}
          </text>

          {/* Image grid */}
          <g transform="translate(36,60)">
            <text x="0" y="-10" fill={C.blue} fontSize="10" fontWeight="800" fontFamily="monospace">INPUT IMAGE</text>
            {Array.from({length: 6}).map(function(_, r){
              return Array.from({length: 6}).map(function(_, c){
                var v = (r===c || r+c===5) ? 1 : 0;
                var col = v ? (C.blue+"aa") : (C.border+"55");
                return <rect key={r+"-"+c} x={c*22} y={r*22} width="20" height="20" rx="4"
                  fill={col} stroke={C.border} />;
              });
            })}
            {/* Kernel overlay */}
            <rect x={22} y={22} width={22*3-2} height={22*3-2} rx="8"
              fill={C.accent+"12"} stroke={C.accent} strokeWidth="2"
              style={{animation:"pulse 1.8s ease-in-out infinite"}} />
            <text x={22} y={22*6+22} fill={C.muted} fontSize="10" fontFamily="monospace">3×3 kernel slides</text>
          </g>

          {/* Arrows */}
          <Arrow x1={220} y1={160} x2={320} y2={160} color={C.muted} />

          {/* Feature map */}
          <g transform="translate(340,76)">
            <text x="0" y="-10" fill={C.cyan} fontSize="10" fontWeight="800" fontFamily="monospace">FEATURE MAP</text>
            {Array.from({length: 4}).map(function(_, r){
              return Array.from({length: 4}).map(function(_, c){
                var hot = (r===1 && c===1) || (r===2 && c===2);
                return <rect key={r+"-"+c} x={c*26} y={r*26} width="24" height="24" rx="5"
                  fill={hot ? (C.cyan+"b0") : (C.border+"55")} stroke={C.border} />;
              });
            })}
            <text x="0" y={4*26+24} fill={C.muted} fontSize="10" fontFamily="monospace">high = pattern found</text>
          </g>

          <Arrow x1={470} y1={160} x2={560} y2={160} color={C.muted} />

          {/* Pooling */}
          <g transform="translate(580,108)">
            <text x="0" y="-10" fill={C.yellow} fontSize="10" fontWeight="800" fontFamily="monospace">POOL/DOWNSAMPLE</text>
            {Array.from({length: 2}).map(function(_, r){
              return Array.from({length: 2}).map(function(_, c){
                return <rect key={r+"-"+c} x={c*34} y={r*34} width="32" height="32" rx="7"
                  fill={(r===0 && c===0) ? (C.yellow+"b0") : (C.border+"55")} stroke={C.border} />;
              });
            })}
            <text x="0" y={2*34+30} fill={C.muted} fontSize="10" fontFamily="monospace">smaller, robust</text>
          </g>

          <Arrow x1={680} y1={160} x2={760} y2={160} color={C.muted} />
          <Node x={805} y={160} label="ŷ" r={22} fill={C.green+"10"} stroke={C.green} textColor={C.green} fs={14} />

          <g>
            <rect x="24" y="274" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="297" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"CNNs are great for space, but not for time/order (language, audio)."}
            </text>
            <text x="40" y="316" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: add recurrence / memory across steps (RNN)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabRNN() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="4) RNN (Recurrent Neural Network)"
          subtitle="For sequences: reuse the same weights at each time step and carry a hidden state (memory)."
        />
        <InfoRow color={C.pink} kicker="Core Idea"
          text={"Hidden state hₜ summarizes the past. Process tokens one-by-one: (xₜ, hₜ₋₁) → hₜ → output."} />

        <SvgBox h={350} viewBox="0 0 860 350">
          <Defs/>
          <text x="24" y="28" fill={C.pink} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"x₁ " + ARR + " h₁ " + ARR + " x₂ " + ARR + " h₂ " + ARR + " x₃ " + ARR + " h₃ ..."}
          </text>

          {/* tokens */}
          <g>
            {/* x boxes */}
            <rect x="70" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="115" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₁</text>

            <rect x="250" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="295" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₂</text>

            <rect x="430" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="475" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₃</text>

            <rect x="610" y="70" width="90" height="42" rx="10" fill={C.border+"25"} stroke={C.border} />
            <text x="655" y="97" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">x₄</text>
          </g>

          {/* RNN cells */}
          <g>
            <g transform="translate(115,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(295,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(475,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
            <g transform="translate(655,170)">
              <circle cx="0" cy="0" r="26" fill={C.pink+"14"} stroke={C.pink} strokeWidth="2"/>
              <text x="0" y="4" fill={C.pink} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">RNN</text>
            </g>
          </g>

          {/* x -> cell arrows */}
          <Arrow x1={115} y1={112} x2={115} y2={142} color={C.muted} />
          <Arrow x1={295} y1={112} x2={295} y2={142} color={C.muted} />
          <Arrow x1={475} y1={112} x2={475} y2={142} color={C.muted} />
          <Arrow x1={655} y1={112} x2={655} y2={142} color={C.muted} />

          {/* hidden state arrows (recurrent) */}
          <path d="M141 170 C 185 170, 225 170, 269 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <path d="M321 170 C 365 170, 405 170, 449 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <path d="M501 170 C 545 170, 585 170, 629 170" fill="none" stroke={C.cyan} strokeWidth="3" markerEnd="url(#arrowHead)" opacity="0.95"/>
          <text x="208" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₁</text>
          <text x="388" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₂</text>
          <text x="568" y="156" fill={C.cyan} fontSize="10" fontWeight="900" fontFamily="monospace">h₃</text>

          {/* outputs */}
          <g>
            <rect x="70" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="115" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₁</text>
            <rect x="250" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="295" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₂</text>
            <rect x="430" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="475" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₃</text>
            <rect x="610" y="230" width="90" height="36" rx="10" fill={C.green+"10"} stroke={C.green+"70"} />
            <text x="655" y="253" fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">y₄</text>
          </g>

          <Arrow x1={115} y1={196} x2={115} y2={230} color={C.muted} />
          <Arrow x1={295} y1={196} x2={295} y2={230} color={C.muted} />
          <Arrow x1={475} y1={196} x2={475} y2={230} color={C.muted} />
          <Arrow x1={655} y1={196} x2={655} y2={230} color={C.muted} />

          <g>
            <rect x="24" y="286" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="309" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"RNNs struggle with very long sequences (vanishing gradients, slow sequential processing)."}
            </text>
            <text x="40" y="328" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"Fix: replace recurrence with self-attention (Transformer)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabTransformer() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="5) Transformer"
          subtitle="Processes all tokens in parallel. Self-attention lets every token attend to every other token."
        />
        <InfoRow color={C.yellow} kicker="Key Mechanism"
          text={"Self-attention computes weighted mixes of tokens using Q/K/V. This captures long-range dependencies without recurrence."} />

        <SvgBox h={420} viewBox="0 0 860 420">
          <Defs/>
          <text x="24" y="28" fill={C.yellow} fontSize="12" fontWeight="900" fontFamily="'JetBrains Mono','SF Mono',monospace">
            {"Tokens " + ARR + " Q,K,V " + ARR + " Attention matrix " + ARR + " context vectors"}
          </text>

          {/* Token row */}
          <g transform="translate(60,70)">
            {["x₁","x₂","x₃","x₄","x₅","x₆"].map(function(t, i){
              var x = i*120;
              return (
                <g key={i}>
                  <rect x={x} y="0" width="88" height="40" rx="10" fill={C.border+"25"} stroke={C.border}/>
                  <text x={x+44} y="26" fill={C.text} fontSize="12" fontWeight="900" textAnchor="middle" fontFamily="monospace">{t}</text>
                </g>
              );
            })}
            <text x="0" y="-10" fill={C.blue} fontSize="10" fontWeight="800" fontFamily="monospace">INPUT TOKENS</text>
          </g>

          {/* Attention matrix block */}
          <g transform="translate(270,140)">
            <rect x="0" y="0" width="320" height="210" rx="14" fill={C.yellow+"0f"} stroke={C.yellow+"aa"} strokeWidth="2"/>
            <text x="16" y="26" fill={C.yellow} fontSize="11" fontWeight="900" fontFamily="monospace">SELF-ATTENTION</text>
            <text x="16" y="46" fill={C.muted} fontSize="10" fontFamily="monospace">softmax(QKᵀ/√d) · V</text>

            {/* Grid */}
            {Array.from({length: 6}).map(function(_, r){
              return Array.from({length: 6}).map(function(_, c){
                var hot = (r===1 && c===4) || (r===2 && c===0) || (r===4 && c===2) || (r===0 && c===0) || (r===5 && c===5);
                var fill = hot ? (C.yellow+"b0") : (C.border+"55");
                return <rect key={r+"-"+c} x={18 + c*46} y={70 + r*22} width="40" height="18" rx="4"
                  fill={fill} stroke={C.border} />;
              });
            })}
            <text x="16" y="200" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"each row = where token attends"}
            </text>
          </g>

          {/* attention lines from tokens to matrix */}
          {[
            [104,110,270,160],[224,110,270,180],[344,110,270,200],
            [464,110,270,220],[584,110,270,240],[704,110,270,260],
          ].map(function(p,i){
            return <line key={i} x1={p[0]} y1={p[1]} x2={p[2]} y2={p[3]}
              stroke={C.muted} strokeWidth="1.5" opacity="0.6" />;
          })}

          {/* output context vectors */}
          <g transform="translate(640,164)">
            <text x="0" y="-10" fill={C.green} fontSize="10" fontWeight="800" fontFamily="monospace">CONTEXT VECTORS</text>
            {["c₁","c₂","c₃","c₄","c₅","c₆"].map(function(t, i){
              return (
                <g key={i}>
                  <rect x="0" y={i*34} width="110" height="28" rx="9" fill={C.green+"10"} stroke={C.green+"70"}/>
                  <text x="55" y={i*34+19} fill={C.green} fontSize="11" fontWeight="900" textAnchor="middle" fontFamily="monospace">{t}</text>
                </g>
              );
            })}
          </g>

          <Arrow x1={590} y1={240} x2={640} y2={240} color={C.muted} />

          {/* Bottom note */}
          <g>
            <rect x="24" y="358" width="812" height="54" rx="10" fill={C.border+"20"} stroke={C.border} />
            <text x="40" y="381" fill={C.text} fontSize="11" fontWeight="800" fontFamily="monospace">
              {"Transformer replaces recurrence with attention " + DASH + " better long-range context + parallel compute."}
            </text>
            <text x="40" y="400" fill={C.muted} fontSize="10" fontFamily="monospace">
              {"This is the backbone of modern LLMs (GPT/BERT/T5 families)."}
            </text>
          </g>
        </SvgBox>
      </Card>
    </div>
  );
}

function TabSummary() {
  return (
    <div>
      <Card>
        <SectionTitle
          title="6) Summary: The Progression"
          subtitle="Each model adds an inductive bias that matches the structure of the data."
        />

        <div style={{display:"grid", gridTemplateColumns:"repeat(1, 1fr)", gap:12}}>
          <Card style={{background:"#0b0b12"}}>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.cyan+"55"} bg={C.cyan+"18"} color={C.cyan}>Perceptron</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Linear decision boundary (dot product + activation)
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.purple+"55"} bg={C.purple+"18"} color={C.purple}>MLP</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Non-linear composition via hidden layers (solves XOR)
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.accent+"55"} bg={C.accent+"18"} color={C.accent}>CNN</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Spatial inductive bias: locality + weight sharing
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.pink+"55"} bg={C.pink+"18"} color={C.pink}>RNN</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Temporal memory via hidden state across time
              </div>
            </div>
            <Divider/>
            <div style={{display:"flex", gap:10, flexWrap:"wrap", alignItems:"center"}}>
              <Pill border={C.yellow+"55"} bg={C.yellow+"18"} color={C.yellow}>Transformer</Pill>
              <div style={{color:C.muted, fontSize:12}}>
                Global context via self-attention (parallel, long-range)
              </div>
            </div>
          </Card>

          <Card style={{background:"#0b0b12"}}>
            <SectionTitle title="One-line takeaway" subtitle="" />
            <div style={{fontFamily:"'JetBrains Mono','SF Mono',monospace", fontSize:12, lineHeight:1.5, color:C.text}}>
              {"Perceptron " + ARR + " MLP " + ARR + " CNN " + ARR + " RNN " + ARR + " Transformer"}
              <div style={{marginTop:10, color:C.muted}}>
                {"Linear " + ARR + " Non-linear " + ARR + " Spatial " + ARR + " Sequential " + ARR + " Global Attention"}
              </div>
            </div>
          </Card>
        </div>
      </Card>
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────
   Quizzes (unlock next tab)
   ───────────────────────────────────────────────────────────── */

var QUIZZES = [
  {
    id: "q0",
    prompt: "Why can a single perceptron NOT solve XOR?",
    options: [
      "Because it has too many parameters",
      "Because it only forms a linear decision boundary",
      "Because it cannot multiply inputs",
      "Because it cannot use bias"
    ],
    correct: 1,
    explain_ok: "XOR is not linearly separable — you need hidden layers / non-linear composition.",
    explain_no: "Think geometry: one line can’t separate XOR’s corners."
  },
  {
    id: "q1",
    prompt: "What’s the key feature that lets an MLP solve XOR?",
    options: [
      "Pooling layers",
      "Self-attention",
      "Hidden layer(s) with non-linear activation",
      "Weight sharing across space"
    ],
    correct: 2,
    explain_ok: "Hidden + non-linearity creates feature composition, enabling non-linear boundaries.",
    explain_no: "MLPs win by adding hidden representations + activation."
  },
  {
    id: "q2",
    prompt: "What makes CNNs efficient for images?",
    options: [
      "They use recurrence across time",
      "Local receptive fields + weight sharing (same filter reused across space)",
      "They avoid activations",
      "They require fixed-length sequences"
    ],
    correct: 1,
    explain_ok: "CNN filters scan local patches and reuse the same weights everywhere.",
    explain_no: "Think: small kernel sliding with shared parameters."
  },
  {
    id: "q3",
    prompt: "What is the RNN’s 'memory' mechanism?",
    options: [
      "Pooling window",
      "Hidden state carried across time steps",
      "Convolution kernel",
      "Positional encoding"
    ],
    correct: 1,
    explain_ok: "RNNs carry a hidden state hₜ that summarizes prior steps.",
    explain_no: "RNNs remember via hidden state, not via filters or pooling."
  },
  {
    id: "q4",
    prompt: "What replaces recurrence in Transformers?",
    options: [
      "A larger convolution kernel",
      "A deeper MLP head",
      "Self-attention over all tokens (parallel)",
      "Max pooling over time"
    ],
    correct: 2,
    explain_ok: "Self-attention lets each token use information from all others without sequential loops.",
    explain_no: "Transformers use attention, not recurrent hidden states."
  }
];

/* ─────────────────────────────────────────────────────────────
   Root App
   ───────────────────────────────────────────────────────────── */

function App() {
  var tabs = ["Perceptron", "MLP", "CNN", "RNN", "Transformer", "Summary"];

  // unlocked[i] indicates whether tab i is accessible
  // tab 0 starts unlocked; each quiz unlocks the next tab
  var _u = useState([true, false, false, false, false, false]);
  var unlocked = _u[0];
  var setUnlocked = _u[1];

  var _t = useState(0);
  var tab = _t[0];
  var setTab = _t[1];

  // quizSolved[k] corresponds to quiz between tab k and k+1 (k=0..4)
  var _qs = useState([false, false, false, false, false]);
  var quizSolved = _qs[0];
  var setQuizSolved = _qs[1];

  function markSolved(k) {
    var nextSolved = quizSolved.slice();
    nextSolved[k] = true;
    setQuizSolved(nextSolved);

    var nextUnlocked = unlocked.slice();
    nextUnlocked[k+1] = true; // unlock next tab
    setUnlocked(nextUnlocked);
  }

  // Convenience: find quiz index for current tab (if any)
  var quizIndex = tab; // quiz after tab 0..4
  var showQuiz = tab >= 0 && tab <= 4; // summary has no quiz

  return (
    <div style={{
      background: C.bg,
      minHeight:"100vh",
      padding:"24px 16px",
      fontFamily:"'JetBrains Mono','SF Mono',monospace",
      color:C.text,
      maxWidth: 980,
      margin:"0 auto"
    }}>
      <div style={{textAlign:"center", marginBottom: 14}}>
        <div style={{
          fontSize: 22,
          fontWeight: 900,
          background: "linear-gradient(135deg," + C.accent + "," + C.yellow + ")",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          display:"inline-block"
        }}>
          Neural Network Evolution
        </div>
        <div style={{fontSize: 11, color: C.muted, marginTop: 4}}>
          {"Interactive progression " + DASH + " Perceptron → MLP → CNN → RNN → Transformer (with quizzes)"}
        </div>
      </div>

      <TabBar tabs={tabs} active={tab} onChange={setTab} unlocked={unlocked} />

      {tab===0 && <TabPerceptron />}
      {tab===1 && <TabMLP />}
      {tab===2 && <TabCNN />}
      {tab===3 && <TabRNN />}
      {tab===4 && <TabTransformer />}
      {tab===5 && <TabSummary />}

      {showQuiz && (
        <QuizCard
          q={QUIZZES[quizIndex]}
          solved={quizSolved[quizIndex]}
          setSolved={function(v){ if(v) markSolved(quizIndex); }}
        />
      )}

      <div style={{marginTop: 16, color: C.muted, fontSize: 10, textAlign:"center"}}>
        {"Tip: quizzes unlock the next tab. If you want free navigation, remove the unlock logic in App()."}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>
"""
# DEEP_LEARNING_PATH_VISUAL_HTML = r"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8"/>
# <style>
# * { box-sizing: border-box; margin: 0; padding: 0; }
# body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; padding: 20px; }
# h2   { color: #38bdf8; margin-bottom: 4px; }
# .subtitle { color: #64748b; margin-bottom: 20px; font-size: 0.9em; }
# .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
# .card { background: #1e2130; border-radius: 12px; padding: 18px; border: 1px solid #2d3148; }
# .card h3 { color: #38bdf8; margin: 0 0 10px; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em; }
# canvas { display: block; }
# .params { background: #12141f; padding: 8px 12px; border-radius: 8px; font-size: 0.8em; color: #94a3b8; margin: 8px 0; line-height: 1.65; }
# .pv { color: #38bdf8; font-weight: bold; }
# .btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }
# button { background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168; border-radius: 6px;
#          padding: 4px 11px; cursor: pointer; font-size: 0.78em; transition: background 0.15s; }
# button:hover { background: #3d4168; }
# button.active { color: #0f1117; font-weight: bold; }
# .chip { display: inline-block; padding: 3px 9px; border-radius: 20px; font-size: 0.75em;
#         margin: 3px; cursor: pointer; border: 1px solid transparent; transition: all 0.15s; }
# .chip:hover { filter: brightness(1.25); }
# .chip.sel { border-color: #fff; }
# .asic-label { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 2px; }
# .asic-val { font-size: 0.82em; color: #cbd5e1; line-height: 1.6; margin-bottom: 10px; }
# .asic-name { font-size: 1.05em; font-weight: bold; margin-bottom: 12px; }
# .slider-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
# .slider-row label { font-size: 0.8em; color: #94a3b8; min-width: 90px; }
# input[type=range] { accent-color: #38bdf8; flex: 1; }
# .vb { font-size: 0.8em; color: #38bdf8; min-width: 32px; }
# .decision-btn { padding: 6px 14px; border-radius: 8px; font-size: 0.8em; margin: 3px; cursor: pointer;
#                 background: #2d3148; color: #e2e8f0; border: 1px solid #3d4168; transition: all 0.15s; }
# .decision-btn:hover { background: #3d4168; }
# .decision-btn.chosen { background: #38bdf8; color: #0f1117; border-color: #38bdf8; font-weight: bold; }
# .rec-card { background: #12141f; border-radius: 8px; padding: 10px 14px; margin: 4px 0;
#             border-left: 3px solid #38bdf8; font-size: 0.82em; }
# .rec-name { color: #38bdf8; font-weight: bold; }
# .rec-reason { color: #94a3b8; font-size: 0.9em; margin-top: 2px; }
# </style>
# </head>
# <body>
# <h2>🧠 Deep Learning Architecture Explorer</h2>
# <p class="subtitle">10 families · 60+ architectures — navigate, inspect ASIC details, trace the timeline, and get recommendations</p>
#
# <div class="grid">
#
#   <!-- ══ PANEL 1 (full-width): Family + Architecture Navigator ══ -->
#   <div class="card" style="grid-column:1/-1;">
#     <h3>Architecture Family Navigator</h3>
#     <div id="familyBtns" class="btn-row"></div>
#     <div class="params" id="familyDesc">Select a family to explore its architectures.</div>
#     <div id="archChips" style="min-height:52px;margin-top:4px;"></div>
#   </div>
#
#   <!-- ══ PANEL 2: ASIC Detail Card ══ -->
#   <div class="card">
#     <h3>ASIC Detail</h3>
#     <div id="asicPanel">
#       <div style="color:#475569;font-size:0.85em;padding:20px 0;">
#         Click any architecture chip above to see its full ASIC breakdown.
#       </div>
#     </div>
#   </div>
#
#   <!-- ══ PANEL 3: Architecture Timeline ══ -->
#   <div class="card">
#     <h3>Architecture Timeline</h3>
#     <div class="params">Key milestones in deep learning history.
#       <span style="color:#38bdf8">Hover</span> a dot for details.
#       Use slider to zoom into an era.
#     </div>
#     <canvas id="cvTimeline" width="420" height="250" style="cursor:crosshair;"></canvas>
#     <div class="slider-row">
#       <label>Era</label>
#       <input type="range" id="eraSlider" min="0" max="3" step="1" value="3">
#       <span class="vb" id="eraVal">All</span>
#     </div>
#     <div class="params" id="timelineInfo">Hover a milestone to see details.</div>
#   </div>
#
#   <!-- ══ PANEL 4 (full-width): Decision Guide ══ -->
#   <div class="card" style="grid-column:1/-1;">
#     <h3>⚡ How to Choose — Interactive Decision Guide</h3>
#     <div class="params">Select your constraints below to get architecture recommendations.</div>
#     <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:8px;">
#       <div>
#         <div class="asic-label" style="margin-bottom:6px;">Data modality</div>
#         <div id="modalityBtns"></div>
#       </div>
#       <div>
#         <div class="asic-label" style="margin-bottom:6px;">Task</div>
#         <div id="taskBtns"></div>
#       </div>
#       <div>
#         <div class="asic-label" style="margin-bottom:6px;">Labelled data</div>
#         <div id="dataBtns"></div>
#       </div>
#       <div>
#         <div class="asic-label" style="margin-bottom:6px;">Compute</div>
#         <div id="computeBtns"></div>
#       </div>
#     </div>
#     <div id="recommendations" style="margin-top:14px;"></div>
#   </div>
#
# </div>
#
# <script>
# // ══ FAMILY DATA ═══════════════════════════════════════════════════════════════
# const FAMILIES = [
#   {
#     id:'ff', label:'Feedforward', icon:'⬡', color:'#c084fc',
#     desc:'The simplest deep networks. Input flows through fully connected layers. Universal approximators and the foundation of all DL.',
#     archs:[
#       {name:'Perceptron',        s:'Single-layer linear classifier. Weighted sum + step activation.',           io:'Feature vec → Binary class',     c:'Only linearly separable problems; historical origin of NNs'},
#       {name:'MLP',               s:'Stacks fully connected layers with non-linear activations. End-to-end backprop.', io:'Feature vec → Any output',  c:'Universal approximator; struggles with spatial/temporal data without CNN/RNN'},
#       {name:'Deep MLP',          s:'MLP with many hidden layers (5+). Requires careful init and normalisation.', io:'Feature vec → Any output',      c:'Highly complex functions; vanishing gradients without BatchNorm or residual connections'},
#       {name:'RBF Network',       s:'Hidden neurons compute Gaussian similarity to learned centres.',             io:'Feature vec → Continuous/Class', c:'Fast closed-form training; local receptive fields; rarely used in modern practice'},
#       {name:'MDN',               s:'MLP outputting parameters of a Gaussian Mixture Model.',                    io:'Feature vec → GMM distribution', c:'Models multi-modal conditional distributions; avoids regression-to-mean'},
#     ]
#   },
#   {
#     id:'cnn', label:'CNNs', icon:'▦', color:'#60a5fa',
#     desc:'CNNs exploit spatial locality and translation invariance via shared-weight filters. Foundation of computer vision.',
#     archs:[
#       {name:'LeNet-5',        s:'5-layer CNN — conv→pool→conv→pool→FC. Pioneered modern CNN structure.',  io:'Greyscale 32×32 → Class',     c:'Designed for digit recognition; historically important; architecturally obsolete'},
#       {name:'AlexNet',        s:'First deep CNN to win ImageNet (2012). Introduced ReLU, Dropout, GPU training.', io:'RGB 224×224 → 1000 classes', c:'Sparked the deep learning revolution; now superseded'},
#       {name:'VGGNet',         s:'Very deep networks using only 3×3 convolutions. Depth drives performance.', io:'RGB 224×224 → Class',         c:'Simple uniform structure; 138M parameters; slow to train'},
#       {name:'GoogLeNet',      s:'Inception modules: parallel 1×1, 3×3, 5×5 paths concatenated.',          io:'RGB 224×224 → Class',         c:'Efficient computation; 1×1 convolutions reduce channels; Inception v3/v4 refined'},
#       {name:'ResNet',         s:'Skip connections: output = F(x) + x. Learns residual functions.',        io:'RGB any size → Class/Features', c:'Enables 50–152+ layers without vanishing gradients; universal backbone'},
#       {name:'DenseNet',       s:'Each layer receives feature maps from ALL previous layers.',              io:'RGB → Class/Features',        c:'Maximum feature reuse; fewer params than ResNet; high memory due to concatenation'},
#       {name:'EfficientNet',   s:'Scales width, depth, and resolution with compound scaling coefficient.', io:'RGB variable → Class',        c:'Best accuracy/efficiency tradeoff; uses NAS; V2 adds Fused-MBConv blocks'},
#       {name:'MobileNet',      s:'Depthwise separable convolutions for drastically fewer FLOPs.',          io:'RGB → Class',                 c:'Edge/mobile deployment; V2 inverted residuals; V3 hard-swish and SE modules'},
#       {name:'U-Net',          s:'Encoder–decoder with skip connections at each resolution level.',        io:'Image → Pixel mask',          c:'Biomedical segmentation; preserves fine spatial detail; works with small datasets'},
#       {name:'FPN',            s:'Multi-scale feature pyramid fusing bottom-up and top-down pathways.',    io:'Image → Multi-scale features', c:'Core of many object detectors; strong at multi-scale detection; used in Mask R-CNN'},
#       {name:'YOLO',           s:'Single-pass detection — predicts boxes and classes from full image.',    io:'Image → Boxes + classes',     c:'Real-time detection; trades accuracy for speed; YOLOv5/v8 popular'},
#       {name:'Deformable Conv', s:'Learnable filter offsets allow sampling non-grid positions.',           io:'Image → Class/Detect/Segment', c:'Adapts to object geometry; improves detection of deformable objects'},
#     ]
#   },
#   {
#     id:'rnn', label:'Recurrent', icon:'↺', color:'#34d399',
#     desc:'Recurrent networks process sequences by maintaining a hidden state. Suited for language, audio, and time-series. Largely superseded by Transformers for long sequences.',
#     archs:[
#       {name:'Vanilla RNN',      s:'Hidden state hₜ = tanh(Wxh·xₜ + Whh·h_{t-1}). Shares weights across time.', io:'Sequence → Sequence/Class', c:'Short-range dependencies; vanishing/exploding gradients over long sequences'},
#       {name:'LSTM',             s:'Cell state + three gates (input, forget, output) for long-range memory.',     io:'Sequence → Sequence/Class', c:'Solves vanishing gradient; learns to retain/forget; slower than GRU'},
#       {name:'GRU',              s:'Simplified LSTM with two gates (reset, update). Merged cell+hidden state.',   io:'Sequence → Sequence/Class', c:'Fewer params than LSTM; often similar performance; faster to train'},
#       {name:'Bidirectional RNN',s:'Two RNNs (forward + backward) — hidden states concatenated at each step.',    io:'Sequence → Sequence/Class', c:'Captures past and future context; cannot stream (needs full sequence)'},
#       {name:'Deep RNN',         s:'Stacks multiple recurrent layers — previous output feeds next layer.',        io:'Sequence → Sequence/Class', c:'More expressive; vanishing gradients harder to manage; use dropout between layers'},
#       {name:'Seq2Seq',          s:'Encoder RNN → context vector → Decoder RNN generates output sequence.',      io:'Sequence → Sequence',       c:'Classic for translation/summarisation; information bottleneck at context vector'},
#       {name:'Seq2Seq+Attention',s:'Decoder attends over all encoder hidden states via learnable alignment.',     io:'Sequence → Sequence',       c:'Eliminates bottleneck; Bahdanau (additive) and Luong (dot-product) variants'},
#       {name:'CTC',              s:'Sequence-to-sequence mapping without frame-level alignment labels.',          io:'Variable seq → Shorter labels', c:'Used in speech recognition and OCR; handles alignment implicitly'},
#     ]
#   },
#   {
#     id:'attn', label:'Transformers', icon:'⚡', color:'#38bdf8',
#     desc:'Transformers replaced recurrence with self-attention, allowing full parallelism and arbitrarily long dependencies. The dominant paradigm for language, vision, audio, and multimodal tasks.',
#     archs:[
#       {name:'Self-Attention',   s:'Each token queries all others via Q·Kᵀ/√d; aggregates values weighted by softmax.', io:'Vectors → Vectors', c:'O(n²) in sequence length; full parallel computation; foundation of all Transformers'},
#       {name:'Multi-Head Attn',  s:'h parallel attention heads in lower-dimensional subspaces; outputs concatenated.',   io:'Vectors → Vectors', c:'Each head learns different relation types; used in every Transformer block'},
#       {name:'Transformer',      s:'Encoder–decoder with MHA + FFN + residual connections + LayerNorm.',                io:'Sequence → Sequence', c:'"Attention is All You Need" (2017); encoder for understanding, decoder for generation'},
#       {name:'BERT',             s:'Bidirectional encoder pre-trained with Masked LM and NSP objectives.',              io:'Text → Embeddings/Class', c:'Rich contextual representations; fine-tunes on downstream tasks; cannot generate'},
#       {name:'GPT',              s:'Causal (left-to-right) Transformer decoder. Next-token prediction pre-training.',   io:'Text → Next token/Seq', c:'Autoregressive generation; scales remarkably with data and compute'},
#       {name:'T5',               s:'Every NLP task framed as text-in → text-out. Full encoder–decoder.',                io:'Text → Text',          c:'Unified framework for any NLP task; large sizes needed for best results'},
#       {name:'ViT',              s:'Image split into patches, each embedded as a token; standard Transformer encoder.', io:'Image patches → Class', c:'Outperforms CNNs at large scale; needs large pre-training; less inductive bias'},
#       {name:'Swin Transformer', s:'Self-attention within local shifted windows; hierarchical merging.',                io:'Image → Class/Features', c:'Linear complexity; strong on detection/segmentation; matches FPN pipelines'},
#       {name:'CLIP',             s:'Joint image + text encoder trained with contrastive loss on 400M pairs.',           io:'Image+Text → Embedding', c:'Zero-shot image classification; powerful visual–language alignment'},
#       {name:'DALL·E / LDM',     s:'Text-conditioned image generation using CLIP encoder + generative model.',          io:'Text prompt → Image',  c:'High-quality diverse images; Stable Diffusion uses latent diffusion for efficiency'},
#       {name:'Longformer',       s:'Sparse attention (sliding window + global tokens) replacing full attention.',       io:'Long document → Output', c:'O(n) complexity; enables books/genomics; global tokens attend everywhere'},
#       {name:'FlashAttention',   s:'IO-aware tiling avoids materialising full N×N attention matrix in HBM.',           io:'Sequence → Sequence',  c:'Not a new architecture — 2–4× faster MHA; lower memory; enables longer context'},
#     ]
#   },
#   {
#     id:'gen', label:'Generative', icon:'✦', color:'#f472b6',
#     desc:'Generative models learn P(X) or P(X|y) and sample new realistic examples. Central to image synthesis, drug discovery, and LLMs.',
#     archs:[
#       {name:'Autoencoder',       s:'Encoder compresses to bottleneck z; decoder reconstructs. MSE/BCE loss.',     io:'Any data → Reconstruction', c:'Learns compressed representation; not generative by itself; decoder can hallucinate'},
#       {name:'VAE',               s:'Encoder outputs distribution q(z|x)=N(μ,σ²). ELBO = recon loss + KL.',       io:'Any data → Generated/Embed', c:'True generative model; smooth interpolable latent space; generations can be blurry'},
#       {name:'GAN',               s:'Generator G produces fakes from noise; Discriminator D distinguishes real/fake.', io:'Noise → Generated samples', c:'Sharp high-quality samples; training unstable (mode collapse); WGAN stabilises'},
#       {name:'Conditional GAN',   s:'Conditions both G and D on extra info y (class label, image, text).',          io:'Noise+Condition → Sample', c:'Generates class-specific samples; used in image-to-image translation (pix2pix)'},
#       {name:'StyleGAN',          s:'Disentangled style space W; coarse/fine features via AdaIN.',                  io:'Noise → High-res image',   c:'State-of-the-art face/scene generation; StyleGAN2/3 address aliasing'},
#       {name:'Normalising Flows', s:'Invertible, differentiable transformations map prior to data distribution.',   io:'Noise → Generated/Density', c:'Exact likelihood; fast inference; limited by invertibility requirement'},
#       {name:'DDPM',              s:'Gradually add noise (forward); train network to reverse step-by-step.',        io:'Noise → (denoising) → Sample', c:'SOTA image quality; stable training; slow sampling — DDIM/consistency accelerate'},
#       {name:'LDM',               s:'Diffusion in VAE latent space. Text conditioning via cross-attention.',        io:'Latent noise → High-res image', c:'Compute-efficient vs pixel-space; backbone of Stable Diffusion; LoRA fine-tuning'},
#       {name:'EBM',               s:'Scalar energy E(x) assigned to every input; samples ∝ exp(-E(x)/T).',         io:'Any data → Energy/Samples', c:'Flexible density estimation; MCMC training is expensive; active research'},
#       {name:'RBM',               s:'Bipartite undirected model. Visible + hidden units; contrastive divergence.',  io:'Binary/Cont → Generated', c:'Pre-DL generative model; stacked RBMs form Deep Belief Networks'},
#     ]
#   },
#   {
#     id:'gnn', label:'GNNs', icon:'◉', color:'#fb923c',
#     desc:'GNNs operate on graph-structured data by passing messages between connected nodes. Used for molecules, social networks, knowledge graphs, and recommendation systems.',
#     archs:[
#       {name:'GCN',             s:'Aggregates neighbour features with normalised adjacency: H = σ(D̃⁻¹ÃHW).',   io:'Graph → Node/Graph embed', c:'Spectral motivation; simple and effective; transductive (fixed graph at inference)'},
#       {name:'GraphSAGE',       s:'Learns aggregation functions over sampled fixed-size neighbourhoods.',         io:'Graph → Node embeddings',  c:'Inductive — generalises to unseen nodes; scalable sampling; used in Pinterest'},
#       {name:'GAT',             s:'Attention coefficients weight neighbour contributions during aggregation.',    io:'Graph → Node/Graph embed', c:'Learns which neighbours matter; interpretable via attention; multi-head for stability'},
#       {name:'MPNN',            s:'Generalised framework: nodes aggregate neighbour messages, then update state.', io:'Graph (edge feats) → Output', c:'Unifying GNN framework; edge features for molecules; may over-smooth with depth'},
#       {name:'Graph Transformer',s:'Transformer self-attention + graph structure (relative position on edges).', io:'Graph → Node/Graph embed', c:'Global structure beyond local neighbourhoods; long-range molecular interactions'},
#       {name:'HetGNN',          s:'Handles multiple node and edge types via type-specific aggregation.',         io:'Heterogeneous graph → Embed', c:'Knowledge graphs and recommendation; requires careful schema design'},
#     ]
#   },
#   {
#     id:'ssl', label:'Self-Supervised', icon:'⟳', color:'#fbbf24',
#     desc:'Self-supervised learning creates supervision from the data itself, removing the need for manual labels. Pre-trained representations transfer to many downstream tasks.',
#     archs:[
#       {name:'SimCLR',  s:'Augments each image twice; maximise agreement between views via contrastive loss.',     io:'Images (unlabelled) → Embed', c:'Requires large batch (many negatives); simple and effective'},
#       {name:'MoCo',    s:'Queue of negatives updated with momentum encoder copy.',                                io:'Images (unlabelled) → Embed', c:'Decouples batch size from negatives; memory efficient; V3 adapts for ViT'},
#       {name:'BYOL',    s:'Online + target networks; online predicts target output. No negatives.',               io:'Images (unlabelled) → Embed', c:'No negatives; avoids collapse via stop-gradient + EMA; strong downstream'},
#       {name:'DINO/DINOv2',s:'Self-distillation: student matches teacher (EMA) under augmentations.',             io:'Images (unlabelled) → Embed', c:'Emergent object segmentation; DINOv2 scales to large diverse datasets'},
#       {name:'MAE',     s:'Masks ~75% of patches; ViT encoder + lightweight decoder reconstructs masked.',       io:'Images (unlabelled) → Embed', c:'Encoder sees only visible patches — very efficient; strong fine-tuning; inspired by BERT'},
#       {name:'SimSiam', s:'Stop-gradient only (no EMA, no negatives). Encoder–predictor asymmetry.',             io:'Images (unlabelled) → Embed', c:'Simplest non-contrastive method; works with small batch sizes'},
#       {name:'CLIP',    s:'Contrastive pre-training on 400M image–text pairs. Aligns vision and language.',      io:'Image+Text → Shared embed', c:'Zero-shot transfer to dozens of tasks; sensitive to distribution shift'},
#     ]
#   },
#   {
#     id:'reg', label:'Regularisation', icon:'⊕', color:'#f87171',
#     desc:'Critical techniques that determine whether a network trains successfully and generalises well. Not architectures in themselves but essential design decisions.',
#     archs:[
#       {name:'BatchNorm',          s:'Normalises activations within each mini-batch. Learns γ (scale) and β (shift).', io:'Any deep network',       c:'Accelerates training; mild regulariser; use LayerNorm for Transformers/small batches'},
#       {name:'LayerNorm',          s:'Normalises across the feature dimension per sample, independent of batch.',      io:'Transformers, RNNs',     c:'Preferred in NLP/sequential models; batch-size independent; more stable'},
#       {name:'Dropout',            s:'Randomly zeros units with probability p during training; scale by 1-p at inference.', io:'MLPs, CNNs, RNNs', c:'Equivalent to ensembling sub-networks; use 0.1–0.5 depending on model size'},
#       {name:'Weight Decay',       s:'Adds λ·‖W‖² penalty to loss. Gaussian prior over weights.',                    io:'All networks',           c:'Use AdamW for correct decoupled weight decay with Adam; most common regulariser'},
#       {name:'Data Augmentation',  s:'Label-preserving transforms: flips, crops, cutout, mixup, cutmix.',             io:'Vision, NLP, Audio',     c:'Most impactful regulariser for vision; Mixup and CutMix mix examples and labels'},
#       {name:'Early Stopping',     s:'Halt training when validation loss stops improving; restore best weights.',      io:'All networks',           c:'Free regularisation; requires validation set; use patience parameter'},
#       {name:'LR Scheduling',      s:'Reduce LR during training: step decay, cosine annealing, linear warmup.',       io:'All networks',           c:'Critical for convergence quality; cosine annealing + warm restarts popular'},
#       {name:'Gradient Clipping',  s:'Clip gradient norm to a maximum value before the parameter update.',            io:'RNNs, Transformers',     c:'Essential for RNNs (exploding gradients); clip-by-norm preferred; threshold 1–5'},
#       {name:'Mixup / CutMix',     s:'Mixup: train on convex combinations of pairs. CutMix: paste crops between images.', io:'Vision (and beyond)', c:'Improves calibration and generalisation; especially useful with ViTs'},
#       {name:'Label Smoothing',    s:'Replace hard one-hot targets with (1-ε)·one-hot + ε/K·uniform.',               io:'Classification tasks',   c:'Prevents overconfidence; improves calibration; ε=0.1 standard in Transformers'},
#       {name:'Stochastic Depth',   s:'Randomly drops entire residual blocks during training.',                        io:'Deep ResNets, ViTs',     c:'Reduces effective depth at training time; drop rate increases with layer depth'},
#     ]
#   },
#   {
#     id:'opt', label:'Optimisers', icon:'▷', color:'#2dd4bf',
#     desc:'The optimiser determines how gradients update weights. Choice significantly affects training speed, stability, and final performance.',
#     archs:[
#       {name:'SGD',           s:'Updates weights in negative gradient direction: w ← w - η∇L.',               io:'All networks',          c:'Simple; good asymptotic convergence; requires careful LR tuning; use with momentum'},
#       {name:'SGD+Momentum',  s:'Velocity vector: v ← βv - η∇L, w ← w + v.',                               io:'All networks',          c:'Dampens oscillations; accelerates in ravines; β=0.9 standard; Nesterov look-ahead better'},
#       {name:'Adam',          s:'Per-parameter running mean (m) and squared gradient (v). Bias-corrected.',   io:'Transformers, GANs, most DL', c:'Fast convergence; little LR tuning; can generalise slightly worse than SGD for CNNs'},
#       {name:'AdamW',         s:'Adam with decoupled L2 weight decay (not mixed into adaptive scaling).',     io:'Transformers, ViTs, LLMs', c:'Standard for large-scale DL; correct weight decay; default for LLM pre-training'},
#       {name:'RMSProp',       s:'Running average of squared gradients normalises per-parameter LR.',          io:'RNNs, RL',              c:'Historically popular for RNNs; often outperformed by Adam in modern practice'},
#       {name:'Lion',          s:'Sign of momentum update: w ← w - η·sign(βm + g).',                        io:'Large-scale vision, LLMs', c:'More memory efficient than Adam (one state vs two); competitive performance'},
#       {name:'LARS / LAMB',   s:'Scales LR by ratio of weight norm to gradient norm, per layer.',            io:'Large-batch CNN/BERT',  c:'Enables very large batch training (16k+) without accuracy loss; LAMB extends to Adam'},
#     ]
#   },
#   {
#     id:'tf', label:'Transfer', icon:'⇉', color:'#a3e635',
#     desc:'Transfer learning leverages models pre-trained on large datasets and adapts them to new tasks efficiently. Essential for most practical deep learning.',
#     archs:[
#       {name:'Feature Extraction', s:'Freeze all pre-trained weights; train only a new task-specific head.',       io:'CNNs, BERT, ViT',   c:'Fastest approach; appropriate when target ≈ source domain; benchmarks representation quality'},
#       {name:'Full Fine-Tuning',   s:'Unfreeze entire pre-trained network; continue training on target data.',     io:'All pre-trained',   c:'Best accuracy with enough target data; risk of catastrophic forgetting; expensive for LLMs'},
#       {name:'Gradual Unfreezing', s:'Unfreeze layers top-to-bottom progressively. Higher layers adapt first.',    io:'CNNs, BERT',        c:'Reduces catastrophic forgetting; ULMFiT popularised for NLP; combine with discriminative LR'},
#       {name:'LoRA',               s:'Injects trainable low-rank matrices ΔW = AB into frozen weight matrices.',   io:'LLMs, Diffusion',   c:'<1% of params trained; no inference latency; adapters merge back; de facto LLM fine-tune'},
#       {name:'Prefix Tuning',      s:'Prepends small trainable token embeddings to the input; weights frozen.',    io:'LLMs',              c:'Extremely parameter-efficient; prompt tuning matches fine-tuning at very large scale (>10B)'},
#       {name:'Adapters',           s:'Small bottleneck modules (down→non-linear→up) after each Transformer block.', io:'BERT, ViT, LLMs', c:'Modular — one adapter per task; share base model; slight inference overhead; precursor to LoRA'},
#       {name:'Instruction FT',     s:'Fine-tunes LLM on diverse (instruction, response) pairs.',                  io:'LLMs',              c:'Makes models instruction-following; combined with RLHF for alignment; InstructGPT, Llama-2'},
#       {name:'RLHF',               s:'Reward model on human preferences; LLM fine-tuned with PPO to maximise reward.', io:'LLMs',        c:'Key alignment technique; PPO optimiser; DPO is a simpler offline alternative'},
#     ]
#   },
# ];
#
# // ══ STATE ═════════════════════════════════════════════════════════════════════
# let selectedFamily = 0;
# let selectedArch   = null;
#
# // ══ FAMILY BUTTONS ════════════════════════════════════════════════════════════
# const familyBtns = document.getElementById('familyBtns');
# FAMILIES.forEach((f,i)=>{
#   const b=document.createElement('button');
#   b.textContent=f.icon+' '+f.label;
#   b.style.borderColor=f.color+'66';
#   if(i===0){ b.classList.add('active'); b.style.background=f.color; b.style.color='#0f1117'; }
#   b.onclick=()=>{ selectedFamily=i; selectedArch=null; renderFamily(); };
#   familyBtns.appendChild(b);
# });
#
# function renderFamily(){
#   const f=FAMILIES[selectedFamily];
#   // Update button styles
#   [...familyBtns.children].forEach((b,i)=>{
#     b.classList.toggle('active',i===selectedFamily);
#     b.style.background=i===selectedFamily?FAMILIES[i].color:'#2d3148';
#     b.style.color=i===selectedFamily?'#0f1117':'#e2e8f0';
#     b.style.borderColor=i===selectedFamily?FAMILIES[i].color:FAMILIES[i].color+'66';
#   });
#   document.getElementById('familyDesc').textContent=f.desc;
#
#   // Chips
#   const container=document.getElementById('archChips');
#   container.innerHTML='';
#   f.archs.forEach((a,ai)=>{
#     const chip=document.createElement('span');
#     chip.className='chip'+(ai===selectedArch?' sel':'');
#     chip.textContent=a.name;
#     chip.style.background=f.color+(ai===selectedArch?'33':'1a');
#     chip.style.color=f.color;
#     chip.style.borderColor=ai===selectedArch?f.color:'transparent';
#     chip.onclick=()=>{ selectedArch=ai; renderFamily(); renderASIC(); };
#     container.appendChild(chip);
#   });
#
#   if(selectedArch!==null) renderASIC();
#   else document.getElementById('asicPanel').innerHTML=
#     '<div style="color:#475569;font-size:0.85em;padding:20px 0;">Click any architecture chip above to see its full ASIC breakdown.</div>';
# }
#
# function renderASIC(){
#   if(selectedArch===null) return;
#   const f=FAMILIES[selectedFamily];
#   const a=f.archs[selectedArch];
#   const c=f.color;
#   document.getElementById('asicPanel').innerHTML=`
#     <div class="asic-name" style="color:${c}">${f.icon} ${a.name}</div>
#     <div class="asic-label">A — Architecture</div>
#     <div class="asic-val" style="color:${c};font-weight:bold;">${a.name} &nbsp;<span style="color:#64748b;font-weight:normal;font-size:0.85em">${f.label} family</span></div>
#     <div class="asic-label">S — Summary</div>
#     <div class="asic-val">${a.s}</div>
#     <div class="asic-label">I — Input / Output</div>
#     <div class="asic-val" style="font-family:monospace;font-size:0.8em;color:#cbd5e1;">${a.io}</div>
#     <div class="asic-label">C — Characteristics</div>
#     <div class="asic-val">${a.c}</div>
#   `;
# }
#
# renderFamily();
#
# // ══ TIMELINE ══════════════════════════════════════════════════════════════════
# const MILESTONES=[
#   {yr:1958,name:'Perceptron',       fam:'ff',   note:'First trainable neural model (Rosenblatt)'},
#   {yr:1986,name:'Backprop (MLP)',   fam:'ff',   note:'Rumelhart et al. popularise backpropagation'},
#   {yr:1989,name:'LeNet',            fam:'cnn',  note:'LeCun — convolutional networks for digit recognition'},
#   {yr:1997,name:'LSTM',             fam:'rnn',  note:'Hochreiter & Schmidhuber — long short-term memory'},
#   {yr:2006,name:'Deep Belief Nets', fam:'gen',  note:'Hinton et al. — deep generative pre-training'},
#   {yr:2012,name:'AlexNet',          fam:'cnn',  note:'ImageNet winner — deep learning revolution begins'},
#   {yr:2013,name:'VAE',              fam:'gen',  note:'Kingma & Welling — variational autoencoder'},
#   {yr:2014,name:'GAN',              fam:'gen',  note:'Goodfellow et al. — generative adversarial network'},
#   {yr:2014,name:'Seq2Seq',          fam:'rnn',  note:'Sutskever et al. — sequence-to-sequence learning'},
#   {yr:2015,name:'ResNet',           fam:'cnn',  note:'He et al. — 152-layer network wins ImageNet'},
#   {yr:2015,name:'BatchNorm',        fam:'reg',  note:'Ioffe & Szegedy — accelerates deep network training'},
#   {yr:2015,name:'Attention (RNN)',  fam:'rnn',  note:'Bahdanau attention — precursor to Transformers'},
#   {yr:2016,name:'YOLO',             fam:'cnn',  note:'Redmon — real-time single-pass detection'},
#   {yr:2017,name:'Transformer',      fam:'attn', note:'"Attention Is All You Need" — Vaswani et al.'},
#   {yr:2017,name:'MobileNet',        fam:'cnn',  note:'Howard et al. — efficient mobile architecture'},
#   {yr:2018,name:'BERT',             fam:'attn', note:'Devlin et al. — bidirectional language pre-training'},
#   {yr:2018,name:'GPT',              fam:'attn', note:'OpenAI — causal language model pre-training'},
#   {yr:2019,name:'EfficientNet',     fam:'cnn',  note:'Tan & Le — compound scaling of width/depth/resolution'},
#   {yr:2020,name:'GPT-3',            fam:'attn', note:'175B param model — few-shot in-context learning'},
#   {yr:2020,name:'DDPM',             fam:'gen',  note:'Ho et al. — denoising diffusion probabilistic models'},
#   {yr:2021,name:'SimCLR / MoCo',   fam:'ssl',  note:'Contrastive self-supervised visual representations'},
#   {yr:2021,name:'ViT',              fam:'attn', note:'Dosovitskiy et al. — vision Transformer patches'},
#   {yr:2021,name:'CLIP',             fam:'attn', note:'Radford et al. — contrastive image–language pre-training'},
#   {yr:2021,name:'DINO',             fam:'ssl',  note:'Caron et al. — self-supervised ViT with knowledge distillation'},
#   {yr:2021,name:'LoRA',             fam:'tf',   note:'Hu et al. — low-rank adaptation of LLMs'},
#   {yr:2022,name:'MAE',              fam:'ssl',  note:'He et al. — masked autoencoder (BERT for vision)'},
#   {yr:2022,name:'Stable Diffusion', fam:'gen',  note:'LDM — latent diffusion; open-source image generation'},
#   {yr:2022,name:'ChatGPT',          fam:'attn', note:'RLHF + instruction fine-tuning for alignment'},
#   {yr:2023,name:'GPT-4',            fam:'attn', note:'Multimodal large language model — SOTA on many benchmarks'},
#   {yr:2023,name:'DINOv2',           fam:'ssl',  note:'Oquab et al. — large-scale self-supervised vision model'},
# ];
#
# const ERA_RANGES=[
#   {label:'Pre-2012', min:1955,max:2012},
#   {label:'2012–2017',min:2012,max:2018},
#   {label:'2018–2021',min:2018,max:2022},
#   {label:'All',      min:1955,max:2024},
# ];
#
# const cvTL=document.getElementById('cvTimeline'),ctxTL=cvTL.getContext('2d');
# let tlHover=-1;
#
# function famColor(id){
#   const f=FAMILIES.find(f=>f.id===id); return f?f.color:'#64748b';
# }
#
# function drawTimeline(){
#   const W=420,H=250,PAD=40,INNER_W=W-2*PAD;
#   ctxTL.clearRect(0,0,W,H);
#   const era=ERA_RANGES[parseInt(document.getElementById('eraSlider').value)];
#   const filtered=MILESTONES.filter(m=>m.yr>=era.min&&m.yr<=era.max);
#   const sx=yr=>PAD+(yr-era.min)/(era.max-era.min)*INNER_W;
#   const LANES=5;
#   const laneH=(H-50)/LANES;
#
#   // Grid lines
#   ctxTL.strokeStyle='#2d3148'; ctxTL.lineWidth=0.4;
#   const step=era.max-era.min<=10?1:era.max-era.min<=20?2:5;
#   for(let yr=Math.ceil(era.min/step)*step;yr<=era.max;yr+=step){
#     const x=sx(yr);
#     ctxTL.beginPath(); ctxTL.moveTo(x,20); ctxTL.lineTo(x,H-20); ctxTL.stroke();
#     ctxTL.fillStyle='#475569'; ctxTL.font='8px sans-serif'; ctxTL.textAlign='center';
#     ctxTL.fillText(yr,x,H-6);
#   }
#
#   // Timeline axis
#   ctxTL.strokeStyle='#3d4168'; ctxTL.lineWidth=1.5;
#   ctxTL.beginPath(); ctxTL.moveTo(PAD,H/2); ctxTL.lineTo(W-PAD,H/2); ctxTL.stroke();
#
#   // Milestones
#   filtered.forEach((m,i)=>{
#     const x=sx(m.yr);
#     const lane=i%LANES;
#     const above=lane<3;
#     const y=above ? (H/2)-30-(lane*28) : (H/2)+30+((lane-3)*28);
#     const col=famColor(m.fam);
#     const isHov=i===tlHover;
#
#     // Connector
#     ctxTL.beginPath(); ctxTL.setLineDash([2,2]);
#     ctxTL.moveTo(x,H/2); ctxTL.lineTo(x,y+(above?12:-12));
#     ctxTL.strokeStyle=col+'66'; ctxTL.lineWidth=0.8; ctxTL.stroke();
#     ctxTL.setLineDash([]);
#
#     // Dot
#     ctxTL.beginPath(); ctxTL.arc(x,y,isHov?7:4.5,0,2*Math.PI);
#     ctxTL.fillStyle=isHov?col:col+'cc'; ctxTL.fill();
#     if(isHov){ ctxTL.strokeStyle='#fff'; ctxTL.lineWidth=1.5; ctxTL.stroke(); }
#
#     // Label
#     ctxTL.fillStyle=isHov?col:'#94a3b8';
#     ctxTL.font=(isHov?'bold ':'')+'8px sans-serif';
#     ctxTL.textAlign='center';
#     ctxTL.fillText(m.name,x,y+(above?-8:18));
#   });
#
#   // Store for hover
#   cvTL._filtered=filtered; cvTL._sx=sx;
# }
#
# cvTL.addEventListener('mousemove',e=>{
#   const rect=cvTL.getBoundingClientRect();
#   const mx=e.clientX-rect.left, my=e.clientY-rect.top;
#   const H=250, LANES=5;
#   const era=ERA_RANGES[parseInt(document.getElementById('eraSlider').value)];
#   const filtered=MILESTONES.filter(m=>m.yr>=era.min&&m.yr<=era.max);
#   const sx=cvTL._sx;
#   let best=-1,bd=Infinity;
#   filtered.forEach((m,i)=>{
#     const x=sx(m.yr);
#     const lane=i%LANES;
#     const above=lane<3;
#     const y=above?(H/2)-30-(lane*28):(H/2)+30+((lane-3)*28);
#     const d=Math.hypot(x-mx,y-my);
#     if(d<bd){bd=d;best=i;}
#   });
#   if(bd<18&&best>=0){
#     const m=filtered[best];
#     document.getElementById('timelineInfo').innerHTML=
#       `<span style="color:${famColor(m.fam)};font-weight:bold">${m.name}</span> (${m.yr}) — ${m.note}`;
#     tlHover=best;
#   } else {
#     document.getElementById('timelineInfo').textContent='Hover a milestone to see details.';
#     tlHover=-1;
#   }
#   drawTimeline();
# });
# cvTL.addEventListener('mouseleave',()=>{ tlHover=-1; drawTimeline(); });
#
# document.getElementById('eraSlider').addEventListener('input',e=>{
#   document.getElementById('eraVal').textContent=ERA_RANGES[parseInt(e.target.value)].label;
#   drawTimeline();
# });
# drawTimeline();
#
# // ══ DECISION GUIDE ════════════════════════════════════════════════════════════
# const DECISIONS={
#   modality:{
#     label:'Data modality',
#     opts:['Images/Video','Text/Language','Audio/Speech','Time Series','Tabular','Graphs']
#   },
#   task:{
#     label:'Task',
#     opts:['Classification','Generation','Detection/Seg','Seq2Seq/Translation','Representation','Graph tasks']
#   },
#   data:{
#     label:'Labelled data',
#     opts:['None (unlabelled)','Very little (<1k)','Moderate (1k–100k)','Large (100k+)']
#   },
#   compute:{
#     label:'Compute',
#     opts:['Edge/Mobile','Single GPU','Multi-GPU','Unlimited']
#   }
# };
#
# const RECS=[
#   {mod:'Images/Video', task:'Classification',       data:'Large (100k+)',     comp:'Multi-GPU',    name:'ResNet / EfficientNet', reason:'Proven backbone; pre-train from scratch or fine-tune from ImageNet'},
#   {mod:'Images/Video', task:'Classification',       data:'Moderate (1k–100k)',comp:'Single GPU',   name:'ViT (pre-trained)',     reason:'Fine-tune a pre-trained ViT with LoRA — strong with limited data'},
#   {mod:'Images/Video', task:'Classification',       data:'Very little (<1k)', comp:'Single GPU',   name:'CLIP linear probe',     reason:'Zero-shot or linear probe on CLIP features — minimal fine-tuning needed'},
#   {mod:'Images/Video', task:'Classification',       data:'None (unlabelled)', comp:'Multi-GPU',    name:'MAE / DINO pre-train',  reason:'Self-supervised pre-training then linear probe on small labelled set'},
#   {mod:'Images/Video', task:'Detection/Seg',        data:'Large (100k+)',     comp:'Multi-GPU',    name:'YOLOv8 / DETR',         reason:'YOLOv8 for speed; DETR/Mask R-CNN for accuracy'},
#   {mod:'Images/Video', task:'Generation',           data:'Large (100k+)',     comp:'Multi-GPU',    name:'Stable Diffusion (LDM)', reason:'Fine-tune via LoRA or DreamBooth on a base diffusion model'},
#   {mod:'Images/Video', task:'Representation',       data:'None (unlabelled)', comp:'Multi-GPU',    name:'DINOv2 / MAE',           reason:'Large-scale self-supervised ViT; features transfer broadly'},
#   {mod:'Text/Language',task:'Classification',       data:'Moderate (1k–100k)',comp:'Single GPU',   name:'BERT (fine-tuned)',      reason:'Fine-tune BERT or RoBERTa; excellent for classification/NER/QA'},
#   {mod:'Text/Language',task:'Generation',           data:'Large (100k+)',     comp:'Multi-GPU',    name:'GPT / LoRA fine-tune',   reason:'Use a GPT-class model; fine-tune with LoRA for efficiency'},
#   {mod:'Text/Language',task:'Seq2Seq/Translation',  data:'Large (100k+)',     comp:'Multi-GPU',    name:'T5 / BART',              reason:'Encoder–decoder Transformers designed for text-to-text tasks'},
#   {mod:'Text/Language',task:'Representation',       data:'None (unlabelled)', comp:'Multi-GPU',    name:'LLM (self-supervised)',   reason:'Pre-train causal or masked LM; BERT-style for bidirectional embeddings'},
#   {mod:'Audio/Speech', task:'Classification',       data:'Large (100k+)',     comp:'Multi-GPU',    name:'Wav2Vec 2.0 / Whisper',  reason:'Self-supervised audio pre-training + fine-tune for downstream task'},
#   {mod:'Audio/Speech', task:'Seq2Seq/Translation',  data:'Large (100k+)',     comp:'Multi-GPU',    name:'Whisper',                reason:'OpenAI Whisper — end-to-end speech-to-text Transformer'},
#   {mod:'Time Series',  task:'Classification',       data:'Moderate (1k–100k)',comp:'Single GPU',   name:'Temporal CNN / LSTM',    reason:'1D CNN for local patterns; LSTM for longer temporal dependencies'},
#   {mod:'Time Series',  task:'Representation',       data:'Large (100k+)',     comp:'Multi-GPU',    name:'Transformer (pos. enc.)',reason:'Self-attention with positional encoding; scales well to long series'},
#   {mod:'Tabular',      task:'Classification',       data:'Moderate (1k–100k)',comp:'Single GPU',   name:'MLP / Gradient Boosting',reason:'DL rarely beats GBMs on pure tabular data; try both'},
#   {mod:'Graphs',       task:'Classification',       data:'Large (100k+)',     comp:'Multi-GPU',    name:'GCN / GAT',              reason:'GCN for transductive; GAT when neighbour importance varies'},
#   {mod:'Graphs',       task:'Representation',       data:'Large (100k+)',     comp:'Multi-GPU',    name:'GraphSAGE / MPNN',       reason:'GraphSAGE for inductive (new nodes); MPNN for molecular graphs'},
#   {mod:'Images/Video', task:'Classification',       data:'Very little (<1k)', comp:'Edge/Mobile',  name:'MobileNet (feature ext.)',reason:'Freeze MobileNet; train only head — fast and edge-deployable'},
# ];
#
# const chosen={modality:null,task:null,data:null,compute:null};
#
# function makeDecisionBtns(containerId, optList, key){
#   const container=document.getElementById(containerId);
#   optList.forEach(opt=>{
#     const b=document.createElement('button');
#     b.className='decision-btn'; b.textContent=opt;
#     b.onclick=()=>{
#       chosen[key]=chosen[key]===opt?null:opt;
#       renderDecision();
#     };
#     container.appendChild(b);
#     container.appendChild(document.createElement('br'));
#   });
# }
#
# makeDecisionBtns('modalityBtns',DECISIONS.modality.opts,'modality');
# makeDecisionBtns('taskBtns',    DECISIONS.task.opts,    'task');
# makeDecisionBtns('dataBtns',    DECISIONS.data.opts,    'data');
# makeDecisionBtns('computeBtns', DECISIONS.compute.opts, 'compute');
#
# function renderDecision(){
#   // Update button styles
#   [['modalityBtns','modality'],['taskBtns','task'],['dataBtns','data'],['computeBtns','compute']]
#     .forEach(([id,key])=>{
#       [...document.getElementById(id).querySelectorAll('.decision-btn')].forEach(b=>{
#         const isChosen=b.textContent===chosen[key];
#         b.classList.toggle('chosen',isChosen);
#       });
#     });
#
#   // Filter recommendations
#   const active=Object.entries(chosen).filter(([,v])=>v!==null);
#   if(!active.length){
#     document.getElementById('recommendations').innerHTML=
#       '<div style="color:#475569;font-size:0.85em;">Select criteria above to get architecture recommendations.</div>';
#     return;
#   }
#
#   let matches=RECS.filter(r=>{
#     if(chosen.modality&&r.mod!==chosen.modality) return false;
#     if(chosen.task    &&r.task!==chosen.task)    return false;
#     if(chosen.data    &&r.data!==chosen.data)    return false;
#     if(chosen.compute &&r.comp!==chosen.compute) return false;
#     return true;
#   });
#
#   // Fallback: relax compute and data constraints
#   if(!matches.length){
#     matches=RECS.filter(r=>{
#       if(chosen.modality&&r.mod!==chosen.modality) return false;
#       if(chosen.task    &&r.task!==chosen.task)    return false;
#       return true;
#     });
#   }
#
#   if(!matches.length){
#     document.getElementById('recommendations').innerHTML=
#       '<div style="color:#475569;font-size:0.85em;">No exact match — try adjusting your criteria.</div>';
#     return;
#   }
#
#   const html=matches.slice(0,4).map(r=>`
#     <div class="rec-card">
#       <div class="rec-name">✦ ${r.name}</div>
#       <div class="rec-reason">${r.reason}</div>
#       <div style="font-size:0.75em;color:#475569;margin-top:4px;">${r.mod} · ${r.task}</div>
#     </div>`).join('');
#   document.getElementById('recommendations').innerHTML=html;
# }
# renderDecision();
#
# // ══ Ctrl+Scroll zoom ══════════════════════════════════════════════════════════
# var ZOOM = 1.0;
# var zoomTimer = null;
#
# function applyZoom(){
#   document.body.style.zoom = ZOOM;
#   clearTimeout(zoomTimer);
#   var toast = document.getElementById('zoom-toast');
#   if(!toast){
#     toast = document.createElement('div');
#     toast.id = 'zoom-toast';
#     toast.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#1e2130;'
#       +'border:1px solid #38bdf8;color:#38bdf8;font-family:monospace;font-size:12px;'
#       +'font-weight:700;padding:8px 14px;border-radius:8px;z-index:9999;'
#       +'pointer-events:none;transition:opacity .3s;';
#     document.body.appendChild(toast);
#   }
#   toast.textContent = 'zoom ' + Math.round(ZOOM * 100) + '%';
#   toast.style.opacity = '1';
#   zoomTimer = setTimeout(function(){ toast.style.opacity = '0'; }, 1200);
# }
#
# document.addEventListener('wheel', function(e){
#   if(!e.ctrlKey && !e.metaKey) return;
#   e.preventDefault();
#   var delta = e.deltaY > 0 ? -0.05 : 0.05;
#   ZOOM = Math.min(3.0, Math.max(0.4, ZOOM + delta));
#   ZOOM = Math.round(ZOOM * 100) / 100;
#   applyZoom();
# }, {passive: false});
# </script>
# </body>
# </html>"""
#
# DEEP_LEARNING_PATH_VISUAL_HEIGHT = 1300