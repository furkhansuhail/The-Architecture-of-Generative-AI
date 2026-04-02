"""
Self-contained HTML for the Backpropagation interactive walkthrough.
Covers: The credit assignment problem, the chain rule, forward pass,
backward pass (delta computation), weight updates, and the vanishing gradient problem.
Embed in Streamlit via st.components.v1.html(BACKPROP_VISUAL_HTML, height=BACKPROP_VISUAL_HEIGHT).
"""

BACKPROP_VISUAL_HTML = """
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
  @keyframes pulse { 0%,100%{opacity:0.5;} 50%{opacity:1;} }
  @keyframes flowRight { 0%{stroke-dashoffset:20;} 100%{stroke-dashoffset:0;} }
  @keyframes flowLeft  { 0%{stroke-dashoffset:0;} 100%{stroke-dashoffset:20;} }
  .flow-fwd { animation: flowRight 0.8s linear infinite; }
  .flow-bwd { animation: flowLeft  0.8s linear infinite; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState   = React.useState;
var useEffect  = React.useEffect;
var useMemo    = React.useMemo;
var useRef     = React.useRef;

/* ── Colour palette (matches CNN visual) ─────────────────────── */
var C = {
  bg:"#0a0a0f", card:"#12121a", border:"#1e1e2e",
  accent:"#ff6b35", blue:"#4ecdc4", purple:"#a78bfa",
  yellow:"#fbbf24", text:"#e4e4e7", muted:"#71717a",
  dim:"#3f3f46", red:"#ef4444", green:"#4ade80",
  cyan:"#38bdf8", pink:"#f472b6", orange:"#fb923c",
};

/* ── Unicode helpers ─────────────────────────────────────────── */
var ARR  = "\u2192";
var LARR = "\u2190";
var UARR = "\u2191";
var DARR = "\u2193";
var DASH = "\u2014";
var PLAY = "\u25B6";
var PAUSE= "\u23F8";
var BULB = "&#128161;";
var TARG = "&#127919;";
var WARN = "\u26A0";
var CHK  = "\u2713";
var PART = "\u2202";   /* ∂ */
var SIGMA= "\u03C3";   /* σ */
var DELTA= "\u03B4";   /* δ */
var ETA  = "\u03B7";   /* η */
var MUL  = "\u00D7";
var LQ   = "\u201C";
var RQ   = "\u201D";

/* ── Maths helpers ───────────────────────────────────────────── */
function sigmoid(z){ return 1/(1+Math.exp(-Math.max(-500,Math.min(500,z)))); }
function sigDeriv(o){ return o*(1-o); }
function fmt(v,d){ return typeof d==="number"?v.toFixed(d):v.toFixed(4); }

/* ── Shared UI components ────────────────────────────────────── */
function TabBar(props){
  var tabs=props.tabs, active=props.active, onChange=props.onChange;
  return (
    <div style={{display:"flex",gap:0,borderBottom:"2px solid "+C.border,marginBottom:24,overflowX:"auto"}}>
      {tabs.map(function(t,i){
        return (
          <button key={i} onClick={function(){onChange(i);}} style={{
            padding:"12px 18px",background:"none",border:"none",
            borderBottom:active===i?"2px solid "+C.accent:"2px solid transparent",
            color:active===i?C.accent:C.muted,cursor:"pointer",
            fontSize:11,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",
            transition:"all 0.2s",whiteSpace:"nowrap",marginBottom:-2,
          }}>{t}</button>
        );
      })}
    </div>
  );
}

function Card(props){
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

function Insight(props){
  return (
    <div style={Object.assign({
      maxWidth:750,margin:"16px auto 0",
      padding:"16px 22px",background:"rgba(255,107,53,0.06)",
      borderRadius:10,border:"1px solid rgba(255,107,53,0.2)",
    },props.style||{})}>
      <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:6}}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
      <div style={{fontSize:11,color:C.muted,lineHeight:1.8}}>{props.children}</div>
    </div>
  );
}

function SectionTitle(props){
  return (
    <div style={{textAlign:"center",marginBottom:20}}>
      <div style={{fontSize:18,fontWeight:800,color:C.text,marginBottom:4}}>{props.title}</div>
      <div style={{fontSize:12,color:C.muted}}>{props.subtitle}</div>
    </div>
  );
}


/* ===============================================================
   TAB 1 — THE PROBLEM: CREDIT ASSIGNMENT
   =============================================================== */
function TabProblem(){
  var _s=useState(0); var step=_s[0], setStep=_s[1];
  var _a=useState(false); var auto=_a[0], setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%4;});},2200);
    return function(){clearInterval(t);};
  },[auto]);

  var stages=[
    {label:"Predict",color:C.blue,   desc:"The network makes a prediction: 0.72"},
    {label:"Measure",color:C.yellow, desc:"Expected was 1. Error = 1 − 0.72 = 0.28. Loss = 0.039"},
    {label:"Problem",color:C.red,    desc:"Who caused this error? The output neuron? Hidden neurons? Every weight?"},
    {label:"Solve",  color:C.green,  desc:"Backpropagation: chain rule distributes exact blame to every weight"},
  ];

  /* network geometry */
  var cx=[80,240,400,560,720];  /* x centres: inputs, H1, H2, output, loss box */
  var hy=[90,170,250];          /* y centres: 3 inputs */
  var hh=[110,210];             /* hidden layer y centres */

  return (
    <div>
      <SectionTitle title="The Credit Assignment Problem" subtitle={"What backpropagation solves "+DASH+" and why it was hard"}/>

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {stages.map(function(s,i){var on=step===i;return(
          <button key={i} onClick={function(){setStep(i);setAuto(false);}} style={{
            padding:"8px 18px",borderRadius:8,
            border:"1.5px solid "+(on?s.color:C.border),
            background:on?s.color+"20":C.card,
            color:on?s.color:C.muted,cursor:"pointer",
            fontSize:10,fontWeight:700,fontFamily:"monospace",transition:"all 0.2s"
          }}>{(i+1)+". "+s.label}</button>
        );})}
        <button onClick={function(){setAuto(!auto);}} style={{padding:"8px 14px",borderRadius:8,border:"1.5px solid "+(auto?C.yellow:C.border),background:auto?C.yellow+"20":C.card,color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:11,fontFamily:"monospace"}}>{auto?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={780} height={310} viewBox="0 0 780 310" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>

          {/* ── connection lines ── */}
          {hh.map(function(hy2,hi){
            return hy.map(function(iy,ii){
              var c=step===0?C.blue:step===1?C.yellow:step===2?C.red:C.green;
              return (<line key={"il"+hi+ii} x1={cx[0]+22} y1={iy} x2={cx[1]-22} y2={hy2}
                stroke={step>=0?c+"40":C.dim} strokeWidth={step>=0?1.5:0.5}
                strokeDasharray={step>=0?"8,4":"4,4"}
                className={step===0?"flow-fwd":step>=3?"flow-bwd":""} />);
            });
          })}
          {[0,1].map(function(oi){
            return hh.map(function(hy2,hi){
              var c=step===0?C.blue:step===1?C.yellow:step===2?C.red:C.green;
              return (<line key={"hl"+oi+hi} x1={cx[1]+22} y1={hy2} x2={cx[2]-22} y2={185}
                stroke={step>=0?c+"40":C.dim} strokeWidth={step>=0?1.5:0.5}
                strokeDasharray={step>=0?"8,4":"4,4"}
                className={step===0?"flow-fwd":step>=3?"flow-bwd":""} />);
            });
          })}

          {/* ── input nodes ── */}
          {[["x\u2081","0"],["x\u2082","1"],["x\u2083","0"]].map(function(d,i){
            return (<g key={"inp"+i}>
              <circle cx={cx[0]} cy={hy[i]} r={22} fill={C.card} stroke={step===0?C.blue:C.dim} strokeWidth={step===0?2:1}/>
              <text x={cx[0]} y={hy[i]-6} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{d[0]}</text>
              <text x={cx[0]} y={hy[i]+8} textAnchor="middle" fill={step===0?C.blue:C.muted} fontSize={13} fontWeight={700} fontFamily="monospace">{d[1]}</text>
            </g>);
          })}

          {/* ── hidden neurons ── */}
          {hh.map(function(hy2,i){
            var active=step>=0;
            var vals=["0.63","0.58"];
            return (<g key={"h"+i}>
              <circle cx={cx[1]} cy={hy2} r={26} fill={step>=3?C.green+"15":step===2?C.red+"12":C.card} stroke={step>=3?C.green:step===2?C.red:step===0?C.blue:C.dim} strokeWidth={step>=2?2:1}/>
              <text x={cx[1]} y={hy2-8} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{"H"+(i+1)}</text>
              <text x={cx[1]} y={hy2+7} textAnchor="middle" fill={step>=0?C.blue:C.dim} fontSize={12} fontWeight={700} fontFamily="monospace">{active?vals[i]:"?"}</text>
              {step===2 && <text x={cx[1]} y={hy2+26} textAnchor="middle" fill={C.red} fontSize={8} fontFamily="monospace">{"blame?"}</text>}
              {step===3 && <text x={cx[1]} y={hy2+26} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">{DELTA+"="+["+0.021","-0.014"][i]}</text>}
            </g>);
          })}

          {/* ── output neuron ── */}
          <circle cx={cx[2]} cy={185} r={28} fill={step>=1?C.yellow+"15":C.card} stroke={step>=1?C.yellow:step===0?C.blue:C.dim} strokeWidth={step>=1?2:1}/>
          <text x={cx[2]} y={176} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Output</text>
          <text x={cx[2]} y={193} textAnchor="middle" fill={step>=0?C.yellow:C.dim} fontSize={14} fontWeight={800} fontFamily="monospace">{step>=0?"0.72":"?"}</text>
          {step===3&&<text x={cx[2]} y={220} textAnchor="middle" fill={C.green} fontSize={9} fontFamily="monospace">{DELTA+"=+0.0439"}</text>}

          {/* ── loss arrow + box ── */}
          <line x1={cx[2]+28} y1={185} x2={cx[3]-10} y2={185} stroke={step>=1?C.yellow:C.dim} strokeWidth={step>=1?2:1}/>
          <polygon points={(cx[3]-6)+",185 "+(cx[3]-14)+",180 "+(cx[3]-14)+",190"} fill={step>=1?C.yellow:C.dim}/>
          <rect x={cx[3]} y={162} width={90} height={46} rx={8} fill={step>=1?C.red+"15":C.card} stroke={step>=1?C.red:C.dim} strokeWidth={step>=1?2:1}/>
          <text x={cx[3]+45} y={179} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Loss</text>
          <text x={cx[3]+45} y={198} textAnchor="middle" fill={step>=1?C.red:C.dim} fontSize={14} fontWeight={800} fontFamily="monospace">{step>=1?"0.039":"?"}</text>

          {/* ── backprop arrow ── */}
          {step>=3&&<g>
            <path d={"M"+cx[3]+" 220 Q"+((cx[3]+cx[2])/2)+" 260 "+cx[2]+" 213"} fill="none" stroke={C.green} strokeWidth={2} strokeDasharray="8,4" className="flow-bwd"/>
            <text x={(cx[3]+cx[2])/2} y={278} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">Gradient flows BACKWARD</text>
          </g>}

          {/* ── step label ── */}
          <rect x={10} y={270} width={760} height={32} rx={6} fill={C.card} stroke={C.border}/>
          <text x={390} y={290} textAnchor="middle" fill={stages[step].color} fontSize={11} fontWeight={700} fontFamily="monospace">{stages[step].desc}</text>

          {/* layer labels */}
          {[["Inputs",cx[0]],["Hidden",cx[1]],["Output",cx[2]]].map(function(d,i){
            return <text key={"ll"+i} x={d[1]} y={12} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{d[0]}</text>;
          })}
        </svg>
      </div>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:12}}>The Central Challenge</div>
        <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {q:"Output error?",a:"Easy — we know expected vs actual",c:C.green},
            {q:"Hidden neuron error?",a:"No expected value — it\u2019s hidden!",c:C.red},
            {q:"Which weight to fix?",a:"Every weight influenced the output",c:C.yellow},
            {q:"By how much?",a:"Proportional to its contribution",c:C.purple},
          ].map(function(d,i){return(
            <div key={i} style={{background:d.c+"10",border:"1px solid "+d.c+"30",borderRadius:8,padding:"12px 16px",minWidth:140,flex:1}}>
              <div style={{fontSize:10,color:d.c,fontWeight:700,marginBottom:4}}>{d.q}</div>
              <div style={{fontSize:10,color:C.muted,lineHeight:1.6}}>{d.a}</div>
            </div>
          );})}
        </div>
      </Card>

      <Insight icon={BULB} title="The Eureka Moment (1986)">
        Rumelhart, Hinton &amp; Williams realised: <span style={{color:C.yellow,fontWeight:700}}>the chain rule of calculus</span> can propagate
        the output error backward through every layer simultaneously, telling each weight exactly how much to change.
        This unlocked training of multi-layer networks and started the deep learning era.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2 — THE CHAIN RULE
   =============================================================== */
function TabChainRule(){
  var _v=useState(0.5); var xVal=_v[0], setXVal=_v[1];

  /* simple 3-step chain: x → z=2x+1 → a=sigmoid(z) → L=(a-1)^2 */
  var z=2*xVal+1;
  var a=sigmoid(z);
  var L=0.5*Math.pow(a-1,2);
  var dLda=-(1-a);
  var dadz=sigDeriv(a);
  var dzdx=2;
  var dLdx=dLda*dadz*dzdx;
  var dLdz_chain=dLda*dadz;

  /* gear analogy positions */
  var gx=[90,250,420,590], gy=130, gr=45;

  return(
    <div>
      <SectionTitle title="The Chain Rule" subtitle={"How gradients multiply through a sequence of operations "+DASH+" the engine of backprop"}/>

      {/* ── interactive slider ── */}
      <Card style={{maxWidth:600,margin:"0 auto 20px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:12}}>
          Interactive: drag x to watch every gradient update live
        </div>
        <div style={{display:"flex",alignItems:"center",gap:16,marginBottom:16}}>
          <span style={{fontSize:10,color:C.muted,minWidth:20}}>x=</span>
          <input type="range" min={-3} max={3} step={0.02} value={xVal}
            onChange={function(e){setXVal(parseFloat(e.target.value));}}
            style={{flex:1,accentColor:C.accent}}/>
          <span style={{fontSize:12,fontWeight:700,color:C.yellow,minWidth:40}}>{fmt(xVal,2)}</span>
        </div>
        <div style={{display:"flex",gap:8,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {label:"x",value:xVal,color:C.cyan},
            {label:"z = 2x+1",value:z,color:C.blue},
            {label:"a = \u03C3(z)",value:a,color:C.accent},
            {label:"L = \u00BD(a-1)\u00B2",value:L,color:C.red},
          ].map(function(d,i){return(
            <div key={i} style={{background:d.color+"10",border:"1px solid "+d.color+"30",borderRadius:8,padding:"10px 14px",textAlign:"center",minWidth:100}}>
              <div style={{fontSize:9,color:d.color,fontWeight:700,marginBottom:4}}>{d.label}</div>
              <div style={{fontSize:18,fontWeight:800,color:d.color}}>{fmt(d.value,4)}</div>
            </div>
          );})}
        </div>
      </Card>

      {/* ── chain rule breakdown ── */}
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.yellow,marginBottom:14}}>
          Chain Rule: {PART+"L/"+PART+"x = ("+PART+"L/"+PART+"a) \u00D7 ("+PART+"a/"+PART+"z) \u00D7 ("+PART+"z/"+PART+"x)"}
        </div>
        <div style={{display:"flex",flexDirection:"column",gap:8}}>
          {[
            {label:PART+"L/"+PART+"a  = \u2212(1\u2212a)",  val:dLda,  color:C.red,    note:"How loss responds to output"},
            {label:PART+"a/"+PART+"z  = \u03C3\u2032(a) = a(1\u2212a)", val:dadz, color:C.accent, note:"Sigmoid sensitivity (max 0.25)"},
            {label:PART+"z/"+PART+"x  = 2",                val:dzdx,  color:C.blue,   note:"Constant slope of z=2x+1"},
            {label:PART+"L/"+PART+"z  = \u00D7 above two", val:dLdz_chain, color:C.purple, note:"Error reaching z"},
            {label:PART+"L/"+PART+"x  = chain product",    val:dLdx,  color:C.green,  note:"Final gradient: nudge x by this"},
          ].map(function(d,i){
            var barW=Math.min(95,Math.abs(d.val)/0.5*60);
            return(
              <div key={i} style={{display:"flex",alignItems:"center",gap:12,background:d.color+"08",borderRadius:8,padding:"8px 12px"}}>
                <div style={{width:230,fontSize:10,color:d.color,fontFamily:"monospace",fontWeight:700}}>{d.label}</div>
                <div style={{width:70,fontSize:13,fontWeight:800,color:d.color,fontFamily:"monospace",textAlign:"right"}}>{fmt(d.val,4)}</div>
                <div style={{flex:1,height:12,background:C.border,borderRadius:6,overflow:"hidden"}}>
                  <div style={{width:barW+"%",height:"100%",background:d.color+"70",borderRadius:6,transition:"width 0.15s"}}/>
                </div>
                <div style={{fontSize:9,color:C.muted,minWidth:160}}>{d.note}</div>
              </div>
            );
          })}
        </div>
        <div style={{marginTop:14,padding:"10px 14px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border,fontFamily:"monospace"}}>
          <div style={{fontSize:9,color:C.muted,marginBottom:4}}>CHAIN PRODUCT</div>
          <div style={{fontSize:11,color:C.text}}>
            <span style={{color:C.red}}>{fmt(dLda,4)}</span>
            <span style={{color:C.dim}}> \u00D7 </span>
            <span style={{color:C.accent}}>{fmt(dadz,4)}</span>
            <span style={{color:C.dim}}> \u00D7 </span>
            <span style={{color:C.blue}}>{fmt(dzdx,1)}</span>
            <span style={{color:C.dim}}> = </span>
            <span style={{color:C.green,fontWeight:700}}>{fmt(dLdx,6)}</span>
          </div>
        </div>
      </Card>

      {/* ── gear analogy svg ── */}
      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={720} height={200} viewBox="0 0 720 200" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={360} y={22} textAnchor="middle" fill={C.muted} fontSize={11} fontWeight={700} fontFamily="monospace">Gear Analogy: Effects Multiply Along the Chain</text>
          {[
            {label:"x",sub:"input",rate:"1x",color:C.cyan},
            {label:"z",sub:"2x+1",rate:"2x",color:C.blue},
            {label:"a",sub:"\u03C3(z)",rate:fmt(dadz,3)+"x",color:C.accent},
            {label:"L",sub:"loss",rate:fmt(dLdz_chain*2,4)+"x",color:C.red},
          ].map(function(d,i){
            var x=gx[i], r=gr;
            return(<g key={i}>
              <circle cx={x} cy={gy} r={r} fill={d.color+"12"} stroke={d.color} strokeWidth={2}/>
              <text x={x} y={gy-8} textAnchor="middle" fill={d.color} fontSize={16} fontWeight={800} fontFamily="monospace">{d.label}</text>
              <text x={x} y={gy+10} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{d.sub}</text>
              <text x={x} y={gy+26} textAnchor="middle" fill={d.color} fontSize={10} fontWeight={700} fontFamily="monospace">{d.rate}</text>
              {i<3&&<line x1={x+r} y1={gy} x2={gx[i+1]-r} y2={gy} stroke={C.dim} strokeWidth={2} strokeDasharray="4,3"/>}
            </g>);
          })}
          <text x={360} y={185} textAnchor="middle" fill={C.muted} fontSize={10} fontFamily="monospace">
            {"Turning x by 1\u00B0 \u2192 L changes by "+fmt(Math.abs(dLdx),4)+" (the chain product)"}
          </text>
        </svg>
      </div>

      <Insight icon={BULB} title="Why This is Powerful">
        The chain rule works for <span style={{color:C.green,fontWeight:700}}>any depth</span>. A 100-layer network just has 100 terms to multiply.
        Each layer computes its own local gradient and passes it backward. No layer needs to know the whole network{DASH}only its immediate neighbours.
        This is what makes backpropagation <span style={{color:C.yellow,fontWeight:700}}>linear in the number of layers</span> (not exponential).
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3 — FORWARD PASS (animated, with live numbers)
   =============================================================== */
function TabForward(){
  /* Fixed network weights (theory-matching) */
  var W={
    w1:0.15, w2:0.20, b1:0.35,   /* H1 from x1,x2 */
    w3:0.25, w4:0.30, b2:0.35,   /* H2 from x1,x2 */
    w5:0.40, w6:0.45, b3:0.60,   /* O  from h1,h2 */
  };
  var X=[0,1];  /* input */
  var expected=1;

  var _s=useState(0); var step=_s[0], setStep=_s[1];
  var _a=useState(false); var auto=_a[0], setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%6;});},1800);
    return function(){clearInterval(t);};
  },[auto]);

  /* computed values */
  var zh1=W.w1*X[0]+W.w2*X[1]+W.b1;
  var h1=sigmoid(zh1);
  var zh2=W.w3*X[0]+W.w4*X[1]+W.b2;
  var h2=sigmoid(zh2);
  var zo =W.w5*h1+W.w6*h2+W.b3;
  var yhat=sigmoid(zo);
  var loss=0.5*Math.pow(expected-yhat,2);

  var steps=[
    {title:"Start — Inputs enter",        color:C.cyan,    highlight:"input"},
    {title:"Hidden Neuron H1 fires",       color:C.blue,    highlight:"h1"},
    {title:"Hidden Neuron H2 fires",       color:C.purple,  highlight:"h2"},
    {title:"Output Neuron fires",          color:C.accent,  highlight:"out"},
    {title:"Compare to expected value",    color:C.yellow,  highlight:"loss"},
    {title:"Loss computed \u2014 ready for backprop!", color:C.red, highlight:"done"},
  ];

  /* positions */
  var NX=140, NH=340, NO=520, NLoss=660;
  var IY=[70,160,250], HY=[100,200], OY=150;

  function neuron(x,y,r,label,val,col,active,showVal){
    return(
      <g>
        <circle cx={x} cy={y} r={r} fill={active?col+"20":C.card}
          stroke={active?col:C.dim} strokeWidth={active?2.5:1}/>
        <text x={x} y={y-7} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{label}</text>
        <text x={x} y={y+8} textAnchor="middle" fill={active?col:C.dim}
          fontSize={showVal?13:11} fontWeight={700} fontFamily="monospace">
          {showVal?fmt(val,4):"?"}
        </text>
      </g>
    );
  }

  function conn(x1,y1,x2,y2,w,active,col){
    return(
      <g>
        <line x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={active?col+"60":C.dim+"40"} strokeWidth={active?2:0.8}
          strokeDasharray={active?"8,3":"4,4"}
          className={active?"flow-fwd":""}/>
        <text x={(x1+x2)/2} y={(y1+y2)/2-6}
          textAnchor="middle" fill={active?col:C.dim} fontSize={9} fontFamily="monospace">
          {active?fmt(w,2):""}
        </text>
      </g>
    );
  }

  var hl=steps[step].highlight;
  var showI=step>=0, showH1=step>=1, showH2=step>=2, showO=step>=3, showLoss=step>=4;

  return(
    <div>
      <SectionTitle title="Forward Pass" subtitle={"Data flows left "+ARR+" right. No weights change. Pure computation."}/>

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:14,flexWrap:"wrap"}}>
        {steps.map(function(s,i){var on=step===i;return(
          <button key={i} onClick={function(){setStep(i);setAuto(false);}} style={{
            padding:"6px 14px",borderRadius:8,
            border:"1.5px solid "+(on?s.color:C.border),
            background:on?s.color+"20":C.card,
            color:on?s.color:C.muted,cursor:"pointer",
            fontSize:10,fontWeight:700,fontFamily:"monospace"
          }}>{i+1}</button>
        );})}
        <button onClick={function(){setAuto(!auto);}} style={{padding:"6px 12px",borderRadius:8,border:"1.5px solid "+(auto?C.yellow:C.border),background:auto?C.yellow+"20":C.card,color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{auto?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={780} height={300} viewBox="0 0 780 300" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>

          {/* input → H1 */}
          {IY.map(function(iy,i){return conn(NX+22,iy,NH-24,HY[0],i===0?W.w1:W.w2,showH1&&hl==="h1"||step>1,C.blue);})}
          {/* input → H2 */}
          {IY.map(function(iy,i){return conn(NX+22,iy,NH-24,HY[1],i===0?W.w3:W.w4,showH2&&hl==="h2"||step>2,C.purple);})}
          {/* H1 → O */}
          {conn(NH+24,HY[0],NO-24,OY,W.w5,showO,C.accent)}
          {/* H2 → O */}
          {conn(NH+24,HY[1],NO-24,OY,W.w6,showO,C.accent)}

          {/* inputs */}
          {IY.map(function(iy,i){
            var v=[X[0],X[1],"—"][i];
            var lbl=["x\u2081","x\u2082","bias"][i];
            return neuron(NX,iy,22,lbl,v,C.cyan,showI,true);
          })}

          {/* hidden neurons */}
          {neuron(NH,HY[0],26,"H1",h1,C.blue,showH1,showH1)}
          {neuron(NH,HY[1],26,"H2",h2,C.purple,showH2,showH2)}

          {/* output neuron */}
          {neuron(NO,OY,28,"Output",yhat,C.accent,showO,showO)}

          {/* loss arrow + box */}
          {showLoss&&<>
            <line x1={NO+28} y1={OY} x2={NLoss-10} y2={OY} stroke={C.yellow} strokeWidth={2}/>
            <polygon points={(NLoss-6)+","+OY+" "+(NLoss-14)+","+(OY-5)+" "+(NLoss-14)+","+(OY+5)} fill={C.yellow}/>
          </>}
          <rect x={NLoss} y={OY-30} width={98} height={62} rx={8}
            fill={showLoss?C.red+"15":C.card} stroke={showLoss?C.red:C.dim} strokeWidth={showLoss?2:1}/>
          <text x={NLoss+49} y={OY-14} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Loss (MSE)</text>
          <text x={NLoss+49} y={OY+8} textAnchor="middle" fill={showLoss?C.red:C.dim} fontSize={16} fontWeight={800} fontFamily="monospace">{showLoss?fmt(loss,5):"?"}</text>
          <text x={NLoss+49} y={OY+24} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{"exp=1, got "+fmt(yhat,3)}</text>

          {/* layer labels */}
          {[["INPUTS",NX],["HIDDEN",NH],["OUTPUT",NO],["LOSS",NLoss+49]].map(function(d,i){
            return <text key={"ll"+i} x={d[1]} y={14} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{d[0]}</text>;
          })}

          {/* step banner */}
          <rect x={10} y={268} width={760} height={26} rx={6} fill={C.card} stroke={C.border}/>
          <text x={390} y={284} textAnchor="middle" fill={steps[step].color} fontSize={11} fontWeight={700} fontFamily="monospace">{steps[step].title}</text>
        </svg>
      </div>

      {/* formula card that updates with step */}
      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:steps[step].color}}>
        <div style={{fontSize:11,fontWeight:700,color:steps[step].color,marginBottom:10}}>{"Step "+(step+1)+" Calculation"}</div>
        {step<=0&&<div style={{fontSize:11,color:C.muted,fontFamily:"monospace",lineHeight:2}}>
          Input x\u2081 = <span style={{color:C.cyan}}>{X[0]}</span>,&nbsp;
          x\u2082 = <span style={{color:C.cyan}}>{X[1]}</span>,&nbsp;
          Expected = <span style={{color:C.green}}>1</span>
        </div>}
        {step===1&&<div style={{fontSize:11,color:C.muted,fontFamily:"monospace",lineHeight:2}}>
          z_H1 = ({W.w1}{MUL}{X[0]}) + ({W.w2}{MUL}{X[1]}) + {W.b1} = <span style={{color:C.blue}}>{fmt(zh1,4)}</span><br/>
          H1 = \u03C3({fmt(zh1,4)}) = <span style={{color:C.blue,fontWeight:700}}>{fmt(h1,4)}</span>
        </div>}
        {step===2&&<div style={{fontSize:11,color:C.muted,fontFamily:"monospace",lineHeight:2}}>
          z_H2 = ({W.w3}{MUL}{X[0]}) + ({W.w4}{MUL}{X[1]}) + {W.b2} = <span style={{color:C.purple}}>{fmt(zh2,4)}</span><br/>
          H2 = \u03C3({fmt(zh2,4)}) = <span style={{color:C.purple,fontWeight:700}}>{fmt(h2,4)}</span>
        </div>}
        {step===3&&<div style={{fontSize:11,color:C.muted,fontFamily:"monospace",lineHeight:2}}>
          z_O = ({W.w5}{MUL}{fmt(h1,4)}) + ({W.w6}{MUL}{fmt(h2,4)}) + {W.b3} = <span style={{color:C.accent}}>{fmt(zo,4)}</span><br/>
          \u0177 = \u03C3({fmt(zo,4)}) = <span style={{color:C.accent,fontWeight:700}}>{fmt(yhat,4)}</span>
        </div>}
        {step===4&&<div style={{fontSize:11,color:C.muted,fontFamily:"monospace",lineHeight:2}}>
          Error = expected \u2212 \u0177 = 1 \u2212 {fmt(yhat,4)} = <span style={{color:C.red}}>{fmt(expected-yhat,4)}</span><br/>
          Loss = \u00BD\u00D7error\u00B2 = \u00BD\u00D7{fmt(Math.pow(expected-yhat,2),4)} = <span style={{color:C.red,fontWeight:700}}>{fmt(loss,5)}</span>
        </div>}
        {step===5&&<div style={{fontSize:11,color:C.muted,lineHeight:1.8}}>
          Forward pass complete. We know the output (<span style={{color:C.accent}}>{fmt(yhat,4)}</span>) and the loss (<span style={{color:C.red}}>{fmt(loss,5)}</span>).
          All intermediate values are cached. Now the <span style={{color:C.green,fontWeight:700}}>backward pass</span> can begin.
        </div>}
      </Card>

      <Insight icon={TARG} title="What Gets Cached">
        During the forward pass, every neuron stores its <span style={{color:C.blue,fontWeight:700}}>pre-activation z</span> and
        <span style={{color:C.accent,fontWeight:700}}> output a</span>. The backward pass needs these to compute
        the sigmoid derivative &sigma;&prime;(a) = a(1&minus;a). If you don&apos;t cache them, you&apos;d have to recompute the entire forward pass for every gradient{DASH}extremely wasteful.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4 — BACKWARD PASS (the heart of backprop)
   =============================================================== */
function TabBackward(){
  var _s=useState(0); var step=_s[0], setStep=_s[1];
  var _a=useState(false); var auto=_a[0], setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%7;});},2200);
    return function(){clearInterval(t);};
  },[auto]);

  /* Pre-computed values (same as forward pass) */
  var X=[0,1], W={w1:0.15,w2:0.20,b1:0.35,w3:0.25,w4:0.30,b2:0.35,w5:0.40,w6:0.45,b3:0.60};
  var expected=1;
  var zh1=W.w1*X[0]+W.w2*X[1]+W.b1, h1=sigmoid(zh1);
  var zh2=W.w3*X[0]+W.w4*X[1]+W.b2, h2=sigmoid(zh2);
  var zo=W.w5*h1+W.w6*h2+W.b3,     yhat=sigmoid(zo);
  var err=expected-yhat;
  var dO=err*sigDeriv(yhat);
  var errH1=dO*W.w5, dH1=errH1*sigDeriv(h1);
  var errH2=dO*W.w6, dH2=errH2*sigDeriv(h2);
  var gw5=dO*h1, gw6=dO*h2, gb3=dO;
  var gw1=dH1*X[0], gw2=dH1*X[1], gb1=dH1;
  var gw3=dH2*X[0], gw4=dH2*X[1], gb2=dH2;

  var steps=[
    {title:"Output delta: \u03B4_o = error \u00D7 \u03C3\u2032(\u0177)",         color:C.red,    hi:"out"},
    {title:"Gradients for output weights w5, w6, b3",            color:C.orange,  hi:"outW"},
    {title:"Error flows back to H1: \u03B4_o \u00D7 w5",         color:C.blue,    hi:"h1err"},
    {title:"H1 delta: \u03B4_H1 = backprop_error \u00D7 \u03C3\u2032(H1)",       color:C.blue,   hi:"h1d"},
    {title:"Error flows back to H2: \u03B4_o \u00D7 w6",         color:C.purple,  hi:"h2err"},
    {title:"H2 delta: \u03B4_H2 = backprop_error \u00D7 \u03C3\u2032(H2)",       color:C.purple, hi:"h2d"},
    {title:"All deltas computed \u2014 ready to update weights!", color:C.green,  hi:"done"},
  ];
  var hl=steps[step].highlight||steps[step].hi;

  /* layout */
  var NX=100, NH=290, NO=490, NLOS=640;
  var IY=[75,155,235], HY=[100,205], OY=155;

  function nCirc(x,y,r,lbl,val,col,active){
    return(<g>
      <circle cx={x} cy={y} r={r} fill={active?col+"20":C.card} stroke={active?col:C.dim} strokeWidth={active?2.5:1}/>
      <text x={x} y={y-7} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{lbl}</text>
      <text x={x} y={y+8} textAnchor="middle" fill={active?col:C.dim} fontSize={12} fontWeight={700} fontFamily="monospace">{val}</text>
    </g>);
  }

  return(
    <div>
      <SectionTitle title="Backward Pass" subtitle={"Error flows RIGHT "+ARR+" LEFT"+DASH+" every weight learns its exact blame"}/>

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:14,flexWrap:"wrap"}}>
        {steps.map(function(s,i){var on=step===i;return(
          <button key={i} onClick={function(){setStep(i);setAuto(false);}} style={{
            padding:"6px 12px",borderRadius:8,
            border:"1.5px solid "+(on?s.color:C.border),
            background:on?s.color+"20":C.card,
            color:on?s.color:C.muted,cursor:"pointer",
            fontSize:10,fontWeight:700,fontFamily:"monospace"
          }}>{i+1}</button>
        );})}
        <button onClick={function(){setAuto(!auto);}} style={{padding:"6px 12px",borderRadius:8,border:"1.5px solid "+(auto?C.yellow:C.border),background:auto?C.yellow+"20":C.card,color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:10,fontFamily:"monospace"}}>{auto?PAUSE:PLAY}</button>
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={780} height={320} viewBox="0 0 780 320" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>

          {/* static forward connections (faint) */}
          {IY.map(function(iy,i){return(<>
            <line key={"il1"+i} x1={NX+22} y1={iy} x2={NH-24} y2={HY[0]} stroke={C.dim+"30"} strokeWidth={0.8}/>
            <line key={"il2"+i} x1={NX+22} y1={iy} x2={NH-24} y2={HY[1]} stroke={C.dim+"30"} strokeWidth={0.8}/>
          </>);})}
          <line x1={NH+24} y1={HY[0]} x2={NO-24} y2={OY} stroke={C.dim+"30"} strokeWidth={0.8}/>
          <line x1={NH+24} y1={HY[1]} x2={NO-24} y2={OY} stroke={C.dim+"30"} strokeWidth={0.8}/>

          {/* ── backward gradient arrows ── */}

          {/* delta_o arrow from loss */}
          {step>=0&&<>
            <line x1={NLOS-6} y1={OY} x2={NO+30} y2={OY} stroke={C.red} strokeWidth={2.5} strokeDasharray="8,3" className="flow-bwd"/>
            <text x={(NLOS+NO)/2-30} y={OY-10} textAnchor="middle" fill={C.red} fontSize={9} fontFamily="monospace">{"\u03B4_o="+fmt(dO,4)}</text>
          </>}

          {/* w5 gradient arrow */}
          {(hl==="outW"||hl==="done"||step>=1)&&<>
            <path d={"M"+NO+" "+(OY-30)+" Q"+(NO+NH)/2+" "+(HY[0]-50)+" "+NH+" "+(HY[0]-30)}
              fill="none" stroke={C.orange} strokeWidth={2} strokeDasharray="6,3" className="flow-bwd"/>
            <text x={(NO+NH)/2} y={HY[0]-60} textAnchor="middle" fill={C.orange} fontSize={9} fontFamily="monospace">{"gw5="+fmt(gw5,4)}</text>
          </>}
          {(hl==="outW"||hl==="done"||step>=1)&&<>
            <path d={"M"+NO+" "+(OY+30)+" Q"+(NO+NH)/2+" "+(HY[1]+50)+" "+NH+" "+(HY[1]+30)}
              fill="none" stroke={C.orange} strokeWidth={2} strokeDasharray="6,3" className="flow-bwd"/>
            <text x={(NO+NH)/2} y={HY[1]+68} textAnchor="middle" fill={C.orange} fontSize={9} fontFamily="monospace">{"gw6="+fmt(gw6,4)}</text>
          </>}

          {/* error to H1 */}
          {step>=2&&<>
            <line x1={NO-24} y1={OY} x2={NH+24} y2={HY[0]}
              stroke={C.blue} strokeWidth={2.5} strokeDasharray="8,3" className="flow-bwd"/>
            <text x={(NO+NH)/2-20} y={(OY+HY[0])/2-14} textAnchor="middle" fill={C.blue} fontSize={9} fontFamily="monospace">{"err="+fmt(errH1,4)}</text>
          </>}
          {/* error to H2 */}
          {step>=4&&<>
            <line x1={NO-24} y1={OY} x2={NH+24} y2={HY[1]}
              stroke={C.purple} strokeWidth={2.5} strokeDasharray="8,3" className="flow-bwd"/>
            <text x={(NO+NH)/2-20} y={(OY+HY[1])/2+14} textAnchor="middle" fill={C.purple} fontSize={9} fontFamily="monospace">{"err="+fmt(errH2,4)}</text>
          </>}

          {/* gradient arrows from H1 to inputs */}
          {step>=3&&IY.map(function(iy,i){
            var gw=[gw1,gw2,gb1][i];
            return(<line key={"hig"+i} x1={NH-24} y1={HY[0]} x2={NX+22} y2={iy}
              stroke={C.blue} strokeWidth={1.5} strokeDasharray="5,3" className="flow-bwd" opacity={0.7}/>);
          })}
          {step>=5&&IY.map(function(iy,i){
            return(<line key={"h2g"+i} x1={NH-24} y1={HY[1]} x2={NX+22} y2={iy}
              stroke={C.purple} strokeWidth={1.5} strokeDasharray="5,3" className="flow-bwd" opacity={0.7}/>);
          })}

          {/* input nodes */}
          {[["x\u2081","0"],["x\u2082","1"],["bias","1"]].map(function(d,i){
            return nCirc(NX,IY[i],22,d[0],d[1],C.cyan,true);
          })}

          {/* hidden neurons */}
          {nCirc(NH,HY[0],26,"H1",fmt(h1,3),C.blue,step>=3)}
          {step>=3&&<text x={NH} y={HY[0]+42} textAnchor="middle" fill={C.blue} fontSize={9} fontWeight={700} fontFamily="monospace">{"\u03B4="+fmt(dH1,4)}</text>}
          {nCirc(NH,HY[1],26,"H2",fmt(h2,3),C.purple,step>=5)}
          {step>=5&&<text x={NH} y={HY[1]+42} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">{"\u03B4="+fmt(dH2,4)}</text>}

          {/* output neuron */}
          {nCirc(NO,OY,28,"Output",fmt(yhat,3),C.red,step>=0)}
          {step>=0&&<text x={NO} y={OY+44} textAnchor="middle" fill={C.red} fontSize={9} fontWeight={700} fontFamily="monospace">{"\u03B4_o="+fmt(dO,4)}</text>}

          {/* loss box */}
          <rect x={NLOS} y={OY-28} width={100} height={56} rx={8} fill={C.red+"10"} stroke={C.red} strokeWidth={2}/>
          <text x={NLOS+50} y={OY-12} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Loss</text>
          <text x={NLOS+50} y={OY+8} textAnchor="middle" fill={C.red} fontSize={14} fontWeight={800} fontFamily="monospace">{"0.0289"}</text>
          <text x={NLOS+50} y={OY+24} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">err={fmt(err,4)}</text>

          {/* layer labels */}
          {[["INPUTS",NX],["HIDDEN",NH],["OUTPUT",NO],["LOSS",NLOS+50]].map(function(d,i){
            return <text key={"ll"+i} x={d[1]} y={14} textAnchor="middle" fill={C.muted} fontSize={9} fontWeight={700} fontFamily="monospace">{d[0]}</text>;
          })}

          {/* direction label */}
          <text x={390} y={290} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">
            {step>=6?"All \u03B4s computed! \u2190 Gradients ready for weight update":"Gradient direction: "+(hl==="out"||hl==="outW"?"Output layer":hl.includes("h1")?"Hidden layer H1":hl.includes("h2")?"Hidden layer H2":"")}
          </text>

          {/* step banner */}
          <rect x={10} y={300} width={760} height={14} rx={4} fill={C.card} stroke={C.border}/>
          <text x={390} y={311} textAnchor="middle" fill={steps[step].color} fontSize={9} fontWeight={700} fontFamily="monospace">{steps[step].title}</text>
        </svg>
      </div>

      {/* formula card */}
      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:steps[step].color}}>
        <div style={{fontSize:11,fontWeight:700,color:steps[step].color,marginBottom:10}}>{"Step "+(step+1)+" Formula"}</div>
        {step===0&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          error = expected \u2212 \u0177 = 1 \u2212 <span style={{color:C.accent}}>{fmt(yhat,4)}</span> = <span style={{color:C.red}}>{fmt(err,4)}</span><br/>
          \u03C3\u2032(\u0177) = \u0177(1\u2212\u0177) = {fmt(yhat,4)}\u00D7{fmt(1-yhat,4)} = <span style={{color:C.yellow}}>{fmt(sigDeriv(yhat),4)}</span><br/>
          \u03B4_o = error \u00D7 \u03C3\u2032(\u0177) = {fmt(err,4)}\u00D7{fmt(sigDeriv(yhat),4)} = <span style={{color:C.red,fontWeight:700}}>{fmt(dO,5)}</span>
        </div>}
        {step===1&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          \u2202L/\u2202w5 = \u03B4_o \u00D7 H1 = {fmt(dO,4)}\u00D7{fmt(h1,4)} = <span style={{color:C.orange,fontWeight:700}}>{fmt(gw5,5)}</span><br/>
          \u2202L/\u2202w6 = \u03B4_o \u00D7 H2 = {fmt(dO,4)}\u00D7{fmt(h2,4)} = <span style={{color:C.orange,fontWeight:700}}>{fmt(gw6,5)}</span><br/>
          \u2202L/\u2202b3 = \u03B4_o = <span style={{color:C.orange,fontWeight:700}}>{fmt(gb3,5)}</span>
        </div>}
        {step===2&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          Error flowing back to H1 = \u03B4_o \u00D7 w5<br/>
          = {fmt(dO,4)} \u00D7 {W.w5} = <span style={{color:C.blue,fontWeight:700}}>{fmt(errH1,5)}</span><br/>
          <span style={{color:C.cyan,fontSize:10}}>Why w5? It\u2019s the connection strength. Higher w5 = H1 had more influence on the error.</span>
        </div>}
        {step===3&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          \u03C3\u2032(H1) = H1(1\u2212H1) = {fmt(h1,4)}\u00D7{fmt(1-h1,4)} = <span style={{color:C.blue}}>{fmt(sigDeriv(h1),4)}</span><br/>
          \u03B4_H1 = errH1 \u00D7 \u03C3\u2032(H1) = {fmt(errH1,4)}\u00D7{fmt(sigDeriv(h1),4)} = <span style={{color:C.blue,fontWeight:700}}>{fmt(dH1,6)}</span><br/>
          \u2202L/\u2202w2 = \u03B4_H1 \u00D7 x2 = {fmt(dH1,6)}\u00D7{X[1]} = <span style={{color:C.blue}}>{fmt(gw2,6)}</span>
        </div>}
        {step===4&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          Error flowing back to H2 = \u03B4_o \u00D7 w6<br/>
          = {fmt(dO,4)} \u00D7 {W.w6} = <span style={{color:C.purple,fontWeight:700}}>{fmt(errH2,5)}</span>
        </div>}
        {step===5&&<div style={{fontSize:11,fontFamily:"monospace",lineHeight:2,color:C.muted}}>
          \u03C3\u2032(H2) = {fmt(h2,4)}\u00D7{fmt(1-h2,4)} = <span style={{color:C.purple}}>{fmt(sigDeriv(h2),4)}</span><br/>
          \u03B4_H2 = {fmt(errH2,4)}\u00D7{fmt(sigDeriv(h2),4)} = <span style={{color:C.purple,fontWeight:700}}>{fmt(dH2,6)}</span><br/>
          \u2202L/\u2202w4 = \u03B4_H2 \u00D7 x2 = {fmt(dH2,6)}\u00D7{X[1]} = <span style={{color:C.purple}}>{fmt(gw4,6)}</span>
        </div>}
        {step===6&&<div style={{fontSize:11,color:C.muted,lineHeight:1.8}}>
          Every weight now has a gradient telling it exactly how to change.
          Output layer: <span style={{color:C.orange}}>\u03B4_o = {fmt(dO,4)}</span>.&nbsp;
          Hidden H1: <span style={{color:C.blue}}>\u03B4_H1 = {fmt(dH1,5)}</span>.&nbsp;
          Hidden H2: <span style={{color:C.purple}}>\u03B4_H2 = {fmt(dH2,5)}</span>.&nbsp;
          <span style={{color:C.green,fontWeight:700}}>Next: update the weights.</span>
        </div>}
      </Card>

      <Insight icon={BULB} title="Two Delta Equations — One Pattern">
        Output: <span style={{color:C.red,fontFamily:"monospace"}}>\u03B4 = (expected\u2212actual) \u00D7 \u03C3\u2032(output)</span><br/>
        Hidden: <span style={{color:C.blue,fontFamily:"monospace"}}>\u03B4 = \u03C3\u2032(output) \u00D7 \u03A3(\u03B4_next \u00D7 w_connecting)</span><br/>
        The same intuition: <span style={{color:C.yellow,fontWeight:700}}>"how wrong is downstream" \u00D7 "how sensitive am I"</span>.
        The chain rule handles the rest automatically.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5 — WEIGHT UPDATE (gradient descent + learning rate)
   =============================================================== */
function TabUpdate(){
  var _lr=useState(0.5); var lr=_lr[0], setLr=_lr[1];

  /* same fixed network */
  var W={w1:0.15,w2:0.20,b1:0.35,w3:0.25,w4:0.30,b2:0.35,w5:0.40,w6:0.45,b3:0.60};
  var X=[0,1], expected=1;
  var h1=sigmoid(W.w1*X[0]+W.w2*X[1]+W.b1);
  var h2=sigmoid(W.w3*X[0]+W.w4*X[1]+W.b2);
  var yhat=sigmoid(W.w5*h1+W.w6*h2+W.b3);
  var err=expected-yhat;
  var dO=err*sigDeriv(yhat);
  var dH1=(dO*W.w5)*sigDeriv(h1);
  var dH2=(dO*W.w6)*sigDeriv(h2);

  /* new weights */
  var nW={
    w1:W.w1+lr*dH1*X[0], w2:W.w2+lr*dH1*X[1], b1:W.b1+lr*dH1,
    w3:W.w3+lr*dH2*X[0], w4:W.w4+lr*dH2*X[1], b2:W.b2+lr*dH2,
    w5:W.w5+lr*dO*h1,    w6:W.w6+lr*dO*h2,    b3:W.b3+lr*dO,
  };

  /* new loss */
  var nh1=sigmoid(nW.w1*X[0]+nW.w2*X[1]+nW.b1);
  var nh2=sigmoid(nW.w3*X[0]+nW.w4*X[1]+nW.b2);
  var nyhat=sigmoid(nW.w5*nh1+nW.w6*nh2+nW.b3);
  var nloss=0.5*Math.pow(expected-nyhat,2);
  var lossOld=0.5*Math.pow(err,2);
  var improvement=(lossOld-nloss)/lossOld*100;

  /* loss curve for viz — sweep lr 0 to 2 */
  var lrCurve=[];
  for(var li=0;li<=40;li++){
    var lr_=li*0.05;
    var nw5_=W.w5+lr_*dO*h1, nw6_=W.w6+lr_*dO*h2, nb3_=W.b3+lr_*dO;
    var nh1_=sigmoid(W.w1*X[0]+W.w2*X[1]+W.b1);
    var nh2_=sigmoid(W.w3*X[0]+W.w4*X[1]+W.b2);
    var ny_=sigmoid(nw5_*nh1_+nw6_*nh2_+nb3_);
    lrCurve.push({lr:lr_,loss:0.5*Math.pow(expected-ny_,2)});
  }

  var weightsData=[
    {name:"w1 (x1\u2192H1)",before:W.w1,grad:dH1*X[0],after:nW.w1,color:C.blue,   note:"x1=0, no change"},
    {name:"w2 (x2\u2192H1)",before:W.w2,grad:dH1*X[1],after:nW.w2,color:C.blue,   note:""},
    {name:"b1 (H1 bias)", before:W.b1,grad:dH1,       after:nW.b1,color:C.blue,   note:""},
    {name:"w3 (x1\u2192H2)",before:W.w3,grad:dH2*X[0],after:nW.w3,color:C.purple, note:"x1=0, no change"},
    {name:"w4 (x2\u2192H2)",before:W.w4,grad:dH2*X[1],after:nW.w4,color:C.purple, note:""},
    {name:"b2 (H2 bias)", before:W.b2,grad:dH2,       after:nW.b2,color:C.purple, note:""},
    {name:"w5 (H1\u2192O)", before:W.w5,grad:dO*h1,    after:nW.w5,color:C.accent, note:""},
    {name:"w6 (H2\u2192O)", before:W.w6,grad:dO*h2,    after:nW.w6,color:C.accent, note:""},
    {name:"b3 (O bias)",  before:W.b3,grad:dO,         after:nW.b3,color:C.accent, note:""},
  ];

  /* loss curve svg */
  var svgW=400, svgH=120;
  var maxL=Math.max.apply(null,lrCurve.map(function(d){return d.loss;}))||0.1;
  function lx(lr_){return 20+lr_/2*(svgW-40);}
  function ly(l){return svgH-15-Math.min((l/maxL)*(svgH-30),svgH-30);}
  var pathD=lrCurve.map(function(d,i){return(i===0?"M":"L")+lx(d.lr)+","+ly(d.loss);}).join(" ");
  var curX=lx(lr), curY=ly(lrCurve[Math.round(lr/0.05)].loss);

  return(
    <div>
      <SectionTitle title="Weight Update — Gradient Descent" subtitle={"w_new = w_old + "+ETA+" \u00D7 \u03B4 \u00D7 input"}/>

      <Card style={{maxWidth:600,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:12}}>
          {"Learning Rate "+ETA+" = "+fmt(lr,2)+" (drag to see effect on loss)"}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:16,marginBottom:4}}>
          <span style={{fontSize:10,color:C.muted}}>0.01</span>
          <input type="range" min={0.01} max={2.0} step={0.01} value={lr}
            onChange={function(e){setLr(parseFloat(e.target.value));}}
            style={{flex:1}}/>
          <span style={{fontSize:10,color:C.muted}}>2.0</span>
        </div>
        <div style={{display:"flex",gap:16,justifyContent:"center",marginTop:12,flexWrap:"wrap"}}>
          {[
            {l:"Old Loss",v:fmt(lossOld,5),c:C.red},
            {l:"New Loss",v:fmt(nloss,5),c:nloss<lossOld?C.green:C.red},
            {l:"Improvement",v:fmt(improvement,1)+"%",c:improvement>0?C.green:C.red},
            {l:"New output \u0177",v:fmt(nyhat,4),c:C.accent},
          ].map(function(d,i){return(
            <div key={i} style={{textAlign:"center"}}>
              <div style={{fontSize:9,color:C.muted,marginBottom:4}}>{d.l}</div>
              <div style={{fontSize:18,fontWeight:800,color:d.c}}>{d.v}</div>
            </div>
          );})}
        </div>
        {lr>1.2&&<div style={{marginTop:12,padding:"8px 12px",background:C.red+"10",borderRadius:6,border:"1px solid "+C.red+"30",fontSize:10,color:C.red}}>
          {WARN+" Learning rate too high! Loss may diverge or oscillate. Typical range: 0.001\u20130.1"}
        </div>}
        {lr<0.05&&<div style={{marginTop:12,padding:"8px 12px",background:C.yellow+"10",borderRadius:6,border:"1px solid "+C.yellow+"30",fontSize:10,color:C.yellow}}>
          Learning rate very small. Network will converge, but training will be slow.
        </div>}
      </Card>

      {/* loss curve */}
      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={svgW+40} height={svgH+20} viewBox={"0 0 "+(svgW+40)+" "+(svgH+20)} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={(svgW+40)/2} y={14} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">Loss after update vs Learning Rate</text>
          <path d={pathD} fill="none" stroke={C.blue} strokeWidth={2}/>
          <line x1={curX} y1={20} x2={curX} y2={svgH} stroke={C.accent} strokeWidth={2} strokeDasharray="4,3"/>
          <circle cx={curX} cy={curY} r={6} fill={C.accent} stroke={C.bg} strokeWidth={2}/>
          <text x={curX+8} y={curY-4} fill={C.accent} fontSize={8} fontFamily="monospace">{"\u03B7="+fmt(lr,2)}</text>
          <text x={20} y={svgH+14} fill={C.muted} fontSize={8} fontFamily="monospace">0</text>
          <text x={svgW} y={svgH+14} fill={C.muted} fontSize={8} fontFamily="monospace">2.0</text>
          <text x={(svgW+40)/2} y={svgH+14} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{ETA}</text>
        </svg>
      </div>

      {/* weight table */}
      <Card style={{maxWidth:750,margin:"0 auto 16px",overflowX:"auto"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>All 9 Weight Updates ("+ETA+"={fmt(lr,2)})"</div>
        <div style={{display:"flex",flexDirection:"column",gap:4}}>
          <div style={{display:"flex",gap:8,padding:"4px 8px"}}>
            {["Weight","Old Value","Gradient","Change","New Value",""].map(function(h,i){return(
              <div key={i} style={{flex:i===0?1.6:i===5?0.8:1,fontSize:9,color:C.muted,fontFamily:"monospace",fontWeight:700,textAlign:i>0?"right":"left"}}>{h}</div>
            );})}
          </div>
          {weightsData.map(function(w,i){
            var change=lr*w.grad;
            var noChange=Math.abs(change)<1e-9;
            return(
              <div key={i} style={{display:"flex",gap:8,padding:"6px 8px",background:w.color+"08",borderRadius:6,border:"1px solid "+w.color+"20"}}>
                <div style={{flex:1.6,fontSize:10,color:w.color,fontFamily:"monospace",fontWeight:700}}>{w.name}</div>
                <div style={{flex:1,fontSize:10,color:C.muted,fontFamily:"monospace",textAlign:"right"}}>{fmt(w.before,4)}</div>
                <div style={{flex:1,fontSize:10,color:w.grad>=0?C.green:C.red,fontFamily:"monospace",textAlign:"right"}}>{fmt(w.grad,5)}</div>
                <div style={{flex:1,fontSize:10,color:noChange?C.dim:change>=0?C.green:C.red,fontFamily:"monospace",textAlign:"right"}}>{noChange?"0.0000":fmt(change,5)}</div>
                <div style={{flex:1,fontSize:10,color:noChange?C.muted:w.color,fontFamily:"monospace",textAlign:"right",fontWeight:700}}>{fmt(w.after,4)}</div>
                <div style={{flex:0.8,fontSize:8,color:C.dim,fontFamily:"monospace",textAlign:"right"}}>{w.note}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight icon={TARG} title="Why the 1/2 in the Loss?">
        Loss = <span style={{color:C.red,fontFamily:"monospace"}}>(1/2)(expected \u2212 actual)\u00B2</span>.
        The &frac12; cancels the 2 from the power rule derivative, giving a clean <span style={{color:C.yellow,fontWeight:700}}>-(expected - actual)</span> rather than -2(expected - actual).
        It doesn&apos;t change which minimum we find{DASH}just keeps the gradients tidier.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 6 — VANISHING GRADIENTS
   =============================================================== */
function TabVanishing(){
  var _d=useState(6); var depth=_d[0], setDepth=_d[1];
  var _act=useState("sigmoid"); var act=_act[0], setAct=_act[1];

  /* compute gradient magnitude through `depth` layers */
  /* sigmoid max deriv = 0.25, relu deriv = 1 (for active neurons) */
  var deriv=act==="sigmoid"?0.25:1.0;
  var gradients=[];
  var g=1.0;
  for(var li=0;li<depth;li++){g*=deriv;gradients.push(g);}

  /* bar chart */
  var barW=560, barH=180;
  var maxG=Math.max.apply(null,gradients)||1;

  return(
    <div>
      <SectionTitle title="The Vanishing Gradient Problem" subtitle={"Why sigmoid fails in deep networks "+DASH+" and why ReLU fixed it"}/>

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {["sigmoid","relu"].map(function(a){var on=act===a;var col=a==="sigmoid"?C.red:C.green;return(
          <button key={a} onClick={function(){setAct(a);}} style={{
            padding:"8px 20px",borderRadius:20,
            border:"1.5px solid "+(on?col:C.border),
            background:on?col+"20":C.card,
            color:on?col:C.muted,cursor:"pointer",
            fontSize:11,fontWeight:700,fontFamily:"monospace"
          }}>{a==="sigmoid"?"\u03C3(z) — Sigmoid":"ReLU — max(0,z)"}</button>
        );})}
      </div>

      <Card style={{maxWidth:600,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:10}}>
          {"Network Depth: "+depth+" layers (drag)"}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:16}}>
          <span style={{fontSize:10,color:C.muted}}>1</span>
          <input type="range" min={1} max={12} step={1} value={depth}
            onChange={function(e){setDepth(parseInt(e.target.value));}}
            style={{flex:1}}/>
          <span style={{fontSize:10,color:C.muted}}>12</span>
        </div>
        <div style={{marginTop:12,display:"flex",gap:16,justifyContent:"center"}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:9,color:C.muted}}>MAX DERIVATIVE</div>
            <div style={{fontSize:20,fontWeight:800,color:act==="sigmoid"?C.red:C.green}}>
              {act==="sigmoid"?"0.25":"1.0"}
            </div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:9,color:C.muted}}>AFTER {depth} LAYERS</div>
            <div style={{fontSize:20,fontWeight:800,color:gradients[depth-1]<0.001?C.red:gradients[depth-1]>0.5?C.green:C.yellow}}>
              {gradients[depth-1]<1e-6?gradients[depth-1].toExponential(1):fmt(gradients[depth-1],4)}
            </div>
          </div>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:9,color:C.muted}}>% OF ORIGINAL</div>
            <div style={{fontSize:20,fontWeight:800,color:gradients[depth-1]<0.01?C.red:C.green}}>
              {(gradients[depth-1]*100).toFixed(4)+"%"}
            </div>
          </div>
        </div>
        {act==="sigmoid"&&depth>=4&&<div style={{marginTop:10,padding:"8px 12px",background:C.red+"10",borderRadius:6,border:"1px solid "+C.red+"30",fontSize:10,color:C.red}}>
          {WARN+" After "+depth+" sigmoid layers, the gradient is "+fmt(gradients[depth-1]*100,4)+"% of its original value. Early layers barely learn."}
        </div>}
        {act==="relu"&&<div style={{marginTop:10,padding:"8px 12px",background:C.green+"10",borderRadius:6,border:"1px solid "+C.green+"30",fontSize:10,color:C.green}}>
          {CHK+" ReLU preserves gradient magnitude (derivative = 1 for active neurons). Depth doesn\u2019t kill learning!"}
        </div>}
      </Card>

      {/* bar chart */}
      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={barW+80} height={barH+60} viewBox={"0 0 "+(barW+80)+" "+(barH+60)} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border}}>
          <text x={(barW+80)/2} y={16} textAnchor="middle" fill={C.muted} fontSize={11} fontWeight={700} fontFamily="monospace">Gradient Magnitude at Each Layer (backpropagating rightward)</text>
          {gradients.map(function(g,i){
            var bw=Math.max(2,barW/depth-6);
            var x=40+i*(barW/depth);
            var bh=Math.max(2,(g/1.0)*barH);
            var col=act==="sigmoid"?
              (g<0.001?C.red:g<0.01?C.orange:g<0.1?C.yellow:C.blue):
              C.green;
            return(<g key={i}>
              <rect x={x} y={barH+30-bh} width={bw} height={bh} rx={3} fill={col+"50"} stroke={col} strokeWidth={1}/>
              <text x={x+bw/2} y={barH+46} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{"L"+(i+1)}</text>
              {bh>16&&<text x={x+bw/2} y={barH+30-bh-4} textAnchor="middle" fill={col} fontSize={7} fontFamily="monospace">
                {g<0.001?g.toExponential(1):fmt(g,3)}
              </text>}
            </g>);
          })}
          {/* y axis label */}
          <text x={18} y={barH/2+30} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace" transform={"rotate(-90 18 "+(barH/2+30)+")"}></text>
          <text x={26} y={36} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">1.0</text>
          <line x1={36} y1={30} x2={36} y2={barH+30} stroke={C.border} strokeWidth={1}/>
          <line x1={36} y1={barH+30} x2={barW+46} y2={barH+30} stroke={C.border} strokeWidth={1}/>
          <text x={(barW+80)/2} y={barH+60} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Layer number (signal flows leftward, gradient flows rightward)</text>
        </svg>
      </div>

      {/* sigmoid vs relu comparison */}
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Sigmoid vs ReLU Side by Side</div>
        <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {name:"Sigmoid \u03C3(z)", formula:"1/(1+e\u207B\u1D59)", range:"(0,1)", maxDeriv:"0.25", problem:"Gradient shrinks 4x per layer. Dead in 5+ layers.", color:C.red, fix:false},
            {name:"ReLU max(0,z)", formula:"z if z>0 else 0", range:"[0,\u221E)", maxDeriv:"1.0", problem:"No vanishing! Dead neuron problem (negative z).", color:C.green, fix:true},
          ].map(function(d,i){return(
            <div key={i} style={{flex:1,minWidth:240,background:d.color+"08",border:"1px solid "+d.color+"25",borderRadius:10,padding:"14px"}}>
              <div style={{fontSize:13,fontWeight:800,color:d.color,marginBottom:8}}>{d.name}</div>
              <div style={{fontSize:10,fontFamily:"monospace",color:C.muted,lineHeight:2}}>
                <div>Formula: <span style={{color:d.color}}>{d.formula}</span></div>
                <div>Output range: <span style={{color:d.color}}>{d.range}</span></div>
                <div>Max derivative: <span style={{color:d.color,fontWeight:700}}>{d.maxDeriv}</span></div>
              </div>
              <div style={{marginTop:8,fontSize:10,color:C.muted,borderLeft:"3px solid "+d.color+"50",paddingLeft:8,lineHeight:1.6}}>{d.problem}</div>
              {d.fix&&<div style={{marginTop:8,fontSize:10,color:C.green,fontWeight:700}}>{CHK+" Modern standard for hidden layers"}</div>}
            </div>
          );})}
        </div>
        <div style={{marginTop:14}}>
          <div style={{fontSize:10,fontWeight:700,color:C.muted,marginBottom:8}}>Modern Activation Variants</div>
          <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
            {[
              {name:"LeakyReLU",desc:"0.01z for z<0. Fixes dead neurons.",c:C.cyan},
              {name:"GELU",desc:"Used in BERT, GPT. Smooth approximation of ReLU.",c:C.purple},
              {name:"ELU",desc:"Exponential for z<0. Smooth + negative outputs.",c:C.yellow},
              {name:"Swish",desc:"z\u00D7\u03C3(z). Used in EfficientNet, T5.",c:C.orange},
            ].map(function(d,i){return(
              <div key={i} style={{background:d.c+"10",border:"1px solid "+d.c+"25",borderRadius:8,padding:"8px 12px",flex:1,minWidth:140}}>
                <div style={{fontSize:10,color:d.c,fontWeight:700,marginBottom:3}}>{d.name}</div>
                <div style={{fontSize:9,color:C.muted}}>{d.desc}</div>
              </div>
            );})}
          </div>
        </div>
      </Card>

      <Insight icon={WARN} title="The Vanishing Gradient in Numbers">
        Sigmoid derivative is at most <span style={{color:C.yellow,fontWeight:700}}>0.25</span>.
        After 8 layers: 0.25\u2078 = <span style={{color:C.red,fontWeight:700}}>0.0000153</span> (0.0015% of original).
        The first layers receive near-zero gradients and their weights barely change.
        ReLU&apos;s derivative is 1 for active neurons, so gradients flow through
        <span style={{color:C.green,fontWeight:700}}> unchanged</span> — this is the key reason deep networks became practical post-2012.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App(){
  var _t=useState(0); var tab=_t[0], setTab=_t[1];
  var tabs=["Credit Assignment","Chain Rule","Forward Pass","Backward Pass","Weight Update","Vanishing Gradients"];
  return(
    <div style={{background:C.bg,minHeight:"100vh",padding:"24px 16px",fontFamily:"'JetBrains Mono','SF Mono',monospace",color:C.text,maxWidth:960,margin:"0 auto"}}>
      <div style={{textAlign:"center",marginBottom:16}}>
        <div style={{fontSize:22,fontWeight:800,background:"linear-gradient(135deg,"+C.accent+","+C.purple+")",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",display:"inline-block"}}>
          Backpropagation
        </div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from the problem to the vanishing gradient"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab}/>
      {tab===0&&<TabProblem/>}
      {tab===1&&<TabChainRule/>}
      {tab===2&&<TabForward/>}
      {tab===3&&<TabBackward/>}
      {tab===4&&<TabUpdate/>}
      {tab===5&&<TabVanishing/>}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
</body>
</html>
"""

BACKPROP_VISUAL_HEIGHT = 1100



#
# # ─────────────────────────────────────────────────────────────────────────────
# # VISUAL HTML
# # ─────────────────────────────────────────────────────────────────────────────
#
# PROMPT_TUNING_VISUAL_HTML = r"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8"/>
# <title>Prompt Tuning — PEFT Visual</title>
# <style>
# @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Sora:wght@300;400;600;700&display=swap');
#
# *{box-sizing:border-box;margin:0;padding:0;}
# body{
#   font-family:'Sora',sans-serif;
#   background:#07070e;
#   color:#e4e4e7;
#   padding:22px;
#   min-height:100vh;
# }
# h2{font-family:'JetBrains Mono',monospace;font-size:1.18em;color:#f97316;margin-bottom:3px;letter-spacing:.02em;}
# .subtitle{color:#3f3f46;font-size:0.78em;margin-bottom:20px;font-family:'JetBrains Mono',monospace;}
#
# .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
# .full{grid-column:1/-1;}
# .col3{grid-template-columns:1fr 1fr 1fr;}
#
# .card{
#   background:#0f0f1a;
#   border:1px solid #1e1e30;
#   border-radius:14px;
#   padding:18px;
#   box-shadow:0 8px 36px rgba(0,0,0,.45);
# }
# .card h3{
#   font-family:'JetBrains Mono',monospace;
#   font-size:.76em;font-weight:800;
#   text-transform:uppercase;letter-spacing:.1em;
#   color:#f97316;margin-bottom:12px;
# }
# canvas{display:block;border-radius:8px;}
#
# .info-box{
#   background:#0a0a14;border:1px solid #1e1e30;border-radius:8px;
#   padding:8px 12px;font-size:.76em;color:#94a3b8;
#   line-height:1.75;margin-top:8px;font-family:'JetBrains Mono',monospace;
# }
# .hl {color:#f97316;font-weight:700;}
# .hl2{color:#34d399;font-weight:700;}
# .hl3{color:#60a5fa;font-weight:700;}
# .hl4{color:#facc15;font-weight:700;}
#
# .btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
# button{
#   background:#1a1a2e;color:#71717a;
#   border:1px solid #2d2d44;border-radius:6px;
#   padding:5px 13px;cursor:pointer;
#   font-family:'JetBrains Mono',monospace;
#   font-size:.73em;font-weight:700;
#   transition:all .15s;
# }
# button:hover{background:#252540;color:#e4e4e7;}
# button.active{background:#f97316;color:#07070e;border-color:#f97316;}
#
# .row{display:flex;align-items:center;gap:10px;margin:6px 0;}
# .row label{font-family:'JetBrains Mono',monospace;font-size:.73em;color:#52525b;min-width:90px;}
# input[type=range]{
#   flex:1;height:4px;border-radius:2px;
#   -webkit-appearance:none;appearance:none;
#   background:#1e1e30;outline:none;cursor:pointer;
# }
# input[type=range]::-webkit-slider-thumb{
#   -webkit-appearance:none;width:14px;height:14px;
#   border-radius:50%;background:#f97316;cursor:pointer;
# }
# .val{font-family:'JetBrains Mono',monospace;font-size:.73em;color:#f97316;min-width:46px;text-align:right;}
#
# /* family tree */
# .tree-wrap{overflow-x:auto;padding-bottom:4px;}
#
# /* quiz */
# .quiz-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px;}
# .q-card{background:#0a0a14;border:1px solid #1e1e30;border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:7px;}
# .q-num{font-family:'JetBrains Mono',monospace;font-size:.65em;color:#2d2d44;text-transform:uppercase;letter-spacing:.1em;}
# .q-prompt{font-family:'JetBrains Mono',monospace;font-size:.76em;color:#cbd5e1;line-height:1.55;font-weight:700;}
# .q-opt{background:#0f0f1a;border:1px solid #2d2d44;border-radius:6px;padding:6px 10px;font-family:'JetBrains Mono',monospace;font-size:.71em;color:#52525b;cursor:pointer;text-align:left;transition:all .15s;line-height:1.4;width:100%;}
# .q-opt:hover:not([disabled]){background:#1a1a2e;color:#e4e4e7;border-color:#3f3f56;}
# .q-opt.correct{border-color:#34d399;color:#34d399;background:#051a10!important;}
# .q-opt.wrong  {border-color:#f87171;color:#f87171;background:#1a0808!important;}
# .q-feedback{font-family:'JetBrains Mono',monospace;font-size:.69em;line-height:1.55;padding:6px 8px;border-radius:6px;margin-top:2px;display:none;}
# .q-feedback.show{display:block;}
# .q-feedback.ok {background:#051a10;border:1px solid #34d39940;color:#34d399;}
# .q-feedback.bad{background:#1a0808;border:1px solid #f8717140;color:#f87171;}
# .score-bar{display:flex;align-items:center;gap:14px;margin-top:14px;padding:10px 16px;background:#0a0a14;border:1px solid #1e1e30;border-radius:8px;font-family:'JetBrains Mono',monospace;font-size:.78em;}
# .pip{width:11px;height:11px;border-radius:50%;background:#1e1e30;border:1px solid #2d2d44;transition:all .3s;flex-shrink:0;}
# .pip.ok {background:#34d399;border-color:#34d399;box-shadow:0 0 6px #34d39966;}
# .pip.bad{background:#f87171;border-color:#f87171;}
#
# /* legend chip */
# .chip{display:inline-block;padding:2px 8px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.68em;font-weight:700;margin:2px;}
# .chip-orange{background:#f9731622;border:1px solid #f9731655;color:#f97316;}
# .chip-green {background:#34d39922;border:1px solid #34d39955;color:#34d399;}
# .chip-blue  {background:#60a5fa22;border:1px solid #60a5fa55;color:#60a5fa;}
# .chip-gray  {background:#52525b22;border:1px solid #52525b55;color:#71717a;}
# </style>
# </head>
# <body>
#
# <h2>🔵 Prompt Tuning — Parameter-Efficient Fine-Tuning (PEFT)</h2>
# <p class="subtitle">Soft prompts &middot; Frozen model &middot; Embedding space &middot; Family tree &middot; Parameter math &middot; Quiz</p>
#
# <div class="grid">
#
#   <!-- ══════ PANEL 1: Architecture — The Core Mechanism ══════ -->
#   <div class="card full">
#     <h3>① Architecture — Soft Prompt Prepended to a Frozen LLM</h3>
#     <canvas id="cvArch" width="900" height="230"></canvas>
#     <div class="btn-row" style="margin-top:10px;">
#       <button id="btnHard"  class="active" onclick="setMode('hard')">Hard Prompt (text)</button>
#       <button id="btnSoft"  onclick="setMode('soft')">Soft Prompt (learned vectors)</button>
#       <button id="btnAnim"  onclick="toggleAnim()">▶ Animate Forward Pass</button>
#     </div>
#     <div class="info-box" id="archInfo">—</div>
#   </div>
#
#   <!-- ══════ PANEL 2: Embedding Space ══════ -->
#   <div class="card">
#     <h3>② Embedding Space — Hard vs Soft Tokens</h3>
#     <canvas id="cvEmbed" width="430" height="230"></canvas>
#     <div class="row"><label>soft token k</label><input type="range" id="slK" min="1" max="8" step="1" value="3"><span class="val" id="slKv">3</span></div>
#     <div class="info-box" id="embedInfo">—</div>
#   </div>
#
#   <!-- ══════ PANEL 3: Parameter Efficiency ══════ -->
#   <div class="card">
#     <h3>③ Parameter Efficiency — How Few is Few?</h3>
#     <canvas id="cvParam" width="430" height="230"></canvas>
#     <div class="row"><label>model size</label>
#       <input type="range" id="slModel" min="0" max="3" step="1" value="1">
#       <span class="val" id="slModelv">770M</span>
#     </div>
#     <div class="row"><label>soft tokens k</label>
#       <input type="range" id="slPK" min="1" max="100" step="1" value="20">
#       <span class="val" id="slPKv">20</span>
#     </div>
#     <div class="info-box" id="paramInfo">—</div>
#   </div>
#
#   <!-- ══════ PANEL 4: PEFT Family Tree ══════ -->
#   <div class="card full">
#     <h3>④ Prompt-Based PEFT Family Tree — WHERE Soft Tokens Are Inserted</h3>
#     <canvas id="cvFamily" width="900" height="240"></canvas>
#     <div class="btn-row" style="margin-top:10px;">
#       <button id="btnFamHard"  class="active"  onclick="setFamily('hard')">Hard Prompting</button>
#       <button id="btnFamPT"   onclick="setFamily('pt')">Prompt Tuning</button>
#       <button id="btnFamPre"  onclick="setFamily('prefix')">Prefix Tuning</button>
#       <button id="btnFamP2"   onclick="setFamily('p2')">P-Tuning v2</button>
#     </div>
#     <div class="info-box" id="familyInfo">—</div>
#   </div>
#
#   <!-- ══════ PANEL 5: Accuracy vs Scale ══════ -->
#   <div class="card">
#     <h3>⑤ Accuracy vs Model Scale</h3>
#     <canvas id="cvScale" width="430" height="230"></canvas>
#     <div class="info-box" id="scaleInfo">—</div>
#   </div>
#
#   <!-- ══════ PANEL 6: Training Loop ══════ -->
#   <div class="card">
#     <h3>⑥ Training Loop — Gradient Flow</h3>
#     <canvas id="cvGrad" width="430" height="230"></canvas>
#     <div class="btn-row">
#       <button id="btnGradPlay" onclick="toggleGrad()">▶ Run Training</button>
#       <button onclick="resetGrad()">↺ Reset</button>
#     </div>
#     <div class="info-box" id="gradInfo">—</div>
#   </div>
#
#   <!-- ══════ QUIZ ══════ -->
#   <div class="card full">
#     <h3>⑦ Knowledge Check</h3>
#     <div class="quiz-grid" id="quizGrid"></div>
#     <div class="score-bar" id="scoreBar"><span id="scoreText">Answer all 6 questions to see your score.</span></div>
#   </div>
#
# </div>
#
# <script>
# // ─── shared palette ──────────────────────────────────────────────────────────
# const C = {
#   bg:      '#07070e',
#   card:    '#0f0f1a',
#   border:  '#1e1e30',
#   orange:  '#f97316',
#   green:   '#34d399',
#   blue:    '#60a5fa',
#   yellow:  '#facc15',
#   purple:  '#c084fc',
#   red:     '#f87171',
#   dim:     '#3f3f56',
#   dimmer:  '#1e1e30',
#   text:    '#e4e4e7',
#   muted:   '#52525b',
#   mono:    "'JetBrains Mono',monospace",
# };
#
# function mono(ctx,sz,bold){ctx.font=(bold?'bold ':'')+sz+'px '+C.mono;}
# function fill(ctx,color){ctx.fillStyle=color;}
# function stroke(ctx,color,w=1){ctx.strokeStyle=color;ctx.lineWidth=w;}
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 1 — Architecture
# // ─────────────────────────────────────────────────────────────────────────────
# const cvA = document.getElementById('cvArch');
# const ctxA = cvA.getContext('2d');
# let archMode = 'hard';
# let animT = 0, animRunning = false, animRaf = null;
#
# function setMode(m){
#   archMode = m;
#   ['Hard','Soft'].forEach(x=>{
#     document.getElementById('btn'+x).classList.toggle('active', m===x.toLowerCase());
#   });
#   animT = 0;
#   drawArch();
# }
#
# function toggleAnim(){
#   if(animRunning){ animRunning=false; cancelAnimationFrame(animRaf); drawArch(); return; }
#   document.getElementById('btnAnim').textContent='⏸ Pause';
#   animRunning = true;
#   function loop(){
#     animT = (animT+0.012) % 1;
#     drawArch();
#     if(animRunning) animRaf = requestAnimationFrame(loop);
#     else document.getElementById('btnAnim').textContent='▶ Animate Forward Pass';
#   }
#   loop();
# }
#
# function drawArch(){
#   const W=900, H=230;
#   const ctx=ctxA;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091500'); ctx.fillRect(0,0,W,H);
#
#   const isSoft = archMode==='soft';
#   const PAD=22, blockH=54, blockY=(H-blockH)/2;
#   const tokenH=38, tokenY=blockY+(blockH-tokenH)/2;
#
#   // ── section labels ────────────────────────────────────────────────────────
#   mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='center';
#
#   // Hard prompt tokens
#   const hardTokens = isSoft
#     ? ['[?]','[?]','[?]']       // soft → 3 soft tokens shown
#     : ['Classify','the','sentiment',':','Great','movie','!'];
#   const softCount  = isSoft ? 3 : 0;
#   const realTokens = ['Great','movie','!'];
#
#   const allTokens   = isSoft ? [...Array(softCount).fill(null), ...realTokens] : hardTokens;
#   const totalT      = allTokens.length;
#   const tokW        = Math.min(72, (W*0.42 - PAD*2)/totalT - 6);
#   const tokGap      = 5;
#   const tokenAreaW  = totalT * (tokW+tokGap);
#   const tokenAreaX  = PAD;
#
#   // draw tokens
#   allTokens.forEach((tok,i)=>{
#     const tx = tokenAreaX + i*(tokW+tokGap);
#     const isSoftTok = isSoft && i < softCount;
#     const pulse = isSoftTok && animRunning
#       ? 0.5 + 0.5*Math.sin(animT*Math.PI*2 - i*0.8) : 0;
#
#     ctx.beginPath();
#     ctx.roundRect(tx, tokenY, tokW, tokenH, 6);
#     const baseColor = isSoftTok ? `rgba(249,115,22,${0.15+pulse*0.25})`
#                                 : 'rgba(96,165,250,0.1)';
#     fill(ctx, baseColor);
#     ctx.fill();
#     stroke(ctx, isSoftTok ? C.orange : C.blue, isSoftTok?1.5:1);
#     ctx.stroke();
#
#     mono(ctx,9,true); ctx.textAlign='center';
#     fill(ctx, isSoftTok ? C.orange : C.blue);
#     ctx.fillText(isSoftTok ? `p${i+1}` : tok, tx+tokW/2, tokenY+tokenH/2+4);
#
#     // label above
#     mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
#     ctx.fillText(isSoftTok?'soft':'token', tx+tokW/2, tokenY-5);
#   });
#
#   // bracket label
#   if(isSoft && softCount>0){
#     const bx1=tokenAreaX, bx2=tokenAreaX+softCount*(tokW+tokGap)-tokGap;
#     const by = tokenY+tokenH+12;
#     ctx.beginPath(); ctx.moveTo(bx1,by); ctx.lineTo(bx1,by+5); ctx.lineTo(bx2,by+5); ctx.lineTo(bx2,by);
#     stroke(ctx,C.orange,1.2); ctx.stroke();
#     mono(ctx,8.5,true); fill(ctx,C.orange); ctx.textAlign='center';
#     ctx.fillText('Trainable soft prompt P ∈ ℝᵏˣᵈ', (bx1+bx2)/2, by+16);
#   }
#   if(isSoft){
#     const bx1=tokenAreaX+softCount*(tokW+tokGap), bx2=tokenAreaX+totalT*(tokW+tokGap)-tokGap;
#     const by = tokenY+tokenH+12;
#     ctx.beginPath(); ctx.moveTo(bx1,by); ctx.lineTo(bx1,by+5); ctx.lineTo(bx2,by+5); ctx.lineTo(bx2,by);
#     stroke(ctx,C.blue,1); ctx.stroke();
#     mono(ctx,8.5,false); fill(ctx,C.blue); ctx.textAlign='center';
#     ctx.fillText('Frozen token embeddings', (bx1+bx2)/2, by+16);
#   }
#
#   // ── Arrow ────────────────────────────────────────────────────────────────
#   const arrowX = tokenAreaX + tokenAreaW + 8;
#   const midY   = H/2;
#   const arrowLen = 30;
#
#   // animated particle along arrow
#   if(animRunning){
#     const px = arrowX + animT * arrowLen;
#     ctx.beginPath(); ctx.arc(px, midY, 4, 0, Math.PI*2);
#     fill(ctx,'#f97316aa'); ctx.fill();
#   }
#
#   ctx.beginPath(); ctx.moveTo(arrowX,midY); ctx.lineTo(arrowX+arrowLen,midY);
#   stroke(ctx,C.dim,1.5); ctx.stroke();
#   ctx.beginPath(); ctx.moveTo(arrowX+arrowLen,midY-5); ctx.lineTo(arrowX+arrowLen+7,midY); ctx.lineTo(arrowX+arrowLen,midY+5);
#   fill(ctx,C.dim); ctx.fill();
#
#   // ── Transformer block ────────────────────────────────────────────────────
#   const tbX = arrowX + arrowLen + 7;
#   const tbW = 210;
#   const tbH = blockH+30;
#   const tbY = (H-tbH)/2;
#
#   ctx.beginPath(); ctx.roundRect(tbX,tbY,tbW,tbH,10);
#   fill(ctx,'#131326'); ctx.fill();
#   stroke(ctx,C.dimmer,1); ctx.stroke();
#
#   // frozen snowflake indicator
#   mono(ctx,10,false); fill(ctx,'#60a5fa88'); ctx.textAlign='center';
#   ctx.fillText('❄', tbX+tbW-14, tbY+16);
#
#   // inner layers representation
#   const layerCount = 5;
#   for(let l=0;l<layerCount;l++){
#     const lx = tbX+18 + l*(tbW-36)/layerCount;
#     const lw = (tbW-36)/layerCount - 4;
#     // attention bars
#     for(let r=0;r<3;r++){
#       const ry = tbY+14+r*16;
#       ctx.beginPath(); ctx.roundRect(lx,ry,lw,11,3);
#       const glow = animRunning ? 0.4+0.4*Math.sin(animT*Math.PI*2 - l*0.6) : 0.25;
#       fill(ctx, `rgba(52,211,153,${glow*0.3})`); ctx.fill();
#       stroke(ctx, `rgba(52,211,153,${glow})`,0.8); ctx.stroke();
#     }
#   }
#
#   mono(ctx,8.5,true); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText('TRANSFORMER LAYERS', tbX+tbW/2, tbY+tbH-8);
#   mono(ctx,8,false); fill(ctx,C.dimmer);
#   ctx.fillText('ALL WEIGHTS FROZEN  ❄', tbX+tbW/2, tbY+tbH+12);
#
#   // ── Arrow out ────────────────────────────────────────────────────────────
#   const outX = tbX+tbW+5;
#   if(animRunning){
#     const px2 = outX + animT*35;
#     ctx.beginPath(); ctx.arc(px2,midY,4,0,Math.PI*2);
#     fill(ctx,'#34d39988'); ctx.fill();
#   }
#   ctx.beginPath(); ctx.moveTo(outX,midY); ctx.lineTo(outX+35,midY);
#   stroke(ctx,C.dim,1.5); ctx.stroke();
#   ctx.beginPath(); ctx.moveTo(outX+35,midY-5); ctx.lineTo(outX+42,midY); ctx.lineTo(outX+35,midY+5);
#   fill(ctx,C.dim); ctx.fill();
#
#   // ── Output box ────────────────────────────────────────────────────────────
#   const obX=outX+42, obW=140, obH=54;
#   const obY=(H-obH)/2;
#   ctx.beginPath(); ctx.roundRect(obX,obY,obW,obH,10);
#   fill(ctx,'rgba(52,211,153,0.08)'); ctx.fill();
#   stroke(ctx,'rgba(52,211,153,0.5)',1.2); ctx.stroke();
#   mono(ctx,9,true); fill(ctx,C.green); ctx.textAlign='center';
#   ctx.fillText('OUTPUT LOGITS', obX+obW/2, obY+obH/2-2);
#   mono(ctx,8,false); fill(ctx,C.muted);
#   ctx.fillText('task prediction', obX+obW/2, obY+obH/2+12);
#
#   // ── Gradient feedback arrow (only soft mode) ───────────────────────────
#   if(isSoft && animRunning){
#     const gAlpha = 0.3+0.7*Math.sin(animT*Math.PI*2+1);
#     const startX = obX, endX = tokenAreaX+softCount*(tokW+tokGap)/2;
#     const arcY   = H-14;
#     ctx.beginPath();
#     ctx.moveTo(startX+10,obY+obH);
#     ctx.quadraticCurveTo((startX+endX)/2, arcY+20, endX, tokenY+tokenH);
#     ctx.setLineDash([4,4]);
#     stroke(ctx,`rgba(249,115,22,${gAlpha})`,1.5); ctx.stroke();
#     ctx.setLineDash([]);
#     // label
#     mono(ctx,8,true); fill(ctx,`rgba(249,115,22,${gAlpha})`); ctx.textAlign='center';
#     ctx.fillText('∇ gradient ONLY to soft prompt', (startX+endX)/2, arcY+12);
#   }
#
#   // ── Title / legend ────────────────────────────────────────────────────────
#   mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='left';
#   ctx.fillText(isSoft
#     ? '⊕ Soft tokens are free-floating vectors in ℝᵈ — not constrained to real words'
#     : '⊕ Hard prompt: every token must be a real vocabulary item', PAD, H-6);
#
#   document.getElementById('archInfo').innerHTML = isSoft
#     ? `<span class="hl">Soft Prompt P</span> = [p₁,p₂,...,pₖ] each pᵢ ∈ ℝᵈ&emsp;|&emsp;Prepended to every input&emsp;|&emsp;Model weights: <span class="hl2">ALL FROZEN ❄</span><br>Gradients flow <span class="hl">ONLY</span> to P during training. The model learns to interpret these learned vectors as task instructions.`
#     : `<span class="hl3">Hard Prompt</span>: carefully written text prefix — constrained to real words, human-intensive, hits a quality ceiling.<br>Every token maps to a fixed embedding vector. You <span class="hl">cannot</span> move those vectors — they are fixed in the embedding table.`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 2 — Embedding Space
# // ─────────────────────────────────────────────────────────────────────────────
# const cvE = document.getElementById('cvEmbed');
# const ctxE = cvE.getContext('2d');
# let embedK = 3;
#
# // fixed word embedding positions (2d projection)
# const wordEmbeds = [
#   {x:0.15,y:0.28,label:'great',color:'#60a5fa'},
#   {x:0.22,y:0.60,label:'movie',color:'#60a5fa'},
#   {x:0.78,y:0.70,label:'terrible',color:'#60a5fa'},
#   {x:0.68,y:0.20,label:'classify',color:'#60a5fa'},
#   {x:0.40,y:0.45,label:'review',color:'#60a5fa'},
#   {x:0.55,y:0.72,label:'bad',color:'#60a5fa'},
#   {x:0.88,y:0.40,label:'good',color:'#60a5fa'},
#   {x:0.30,y:0.80,label:'film',color:'#60a5fa'},
# ];
#
# // soft token positions (learned, NOT constrained to grid)
# const softPositions = [
#   {x:0.52,y:0.18},
#   {x:0.35,y:0.32},
#   {x:0.64,y:0.42},
#   {x:0.20,y:0.48},
#   {x:0.75,y:0.55},
#   {x:0.44,y:0.62},
#   {x:0.58,y:0.78},
#   {x:0.28,y:0.15},
# ];
#
# document.getElementById('slK').addEventListener('input',function(){
#   embedK=+this.value;
#   document.getElementById('slKv').textContent=embedK;
#   drawEmbed();
# });
#
# function drawEmbed(){
#   const W=430,H=230;
#   const ctx=ctxE;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);
#
#   const PAD=30;
#   const pw=W-2*PAD, ph=H-2*PAD;
#
#   // grid lines
#   for(let i=0;i<=4;i++){
#     const gx=PAD+i*pw/4, gy=PAD+i*ph/4;
#     ctx.beginPath(); ctx.moveTo(gx,PAD); ctx.lineTo(gx,PAD+ph);
#     stroke(ctx,C.dimmer,0.5); ctx.stroke();
#     ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(PAD+pw,gy);
#     stroke(ctx,C.dimmer,0.5); ctx.stroke();
#   }
#
#   // axis labels
#   mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText('← Embedding Dimension 1 →', PAD+pw/2, H-4);
#   ctx.save(); ctx.translate(10,PAD+ph/2); ctx.rotate(-Math.PI/2);
#   ctx.fillText('← Embedding Dimension 2 →',0,0); ctx.restore();
#
#   // word embeddings (blue, fixed grid)
#   wordEmbeds.forEach(w=>{
#     const wx=PAD+w.x*pw, wy=PAD+w.y*ph;
#     ctx.beginPath(); ctx.arc(wx,wy,5,0,Math.PI*2);
#     fill(ctx,'#60a5fa22'); ctx.fill();
#     stroke(ctx,'#60a5fa',1.5); ctx.stroke();
#     mono(ctx,8,false); fill(ctx,'#60a5fa88'); ctx.textAlign='left';
#     ctx.fillText(w.label, wx+7, wy+3);
#   });
#
#   // vocabulary boundary (conceptual hull)
#   ctx.beginPath();
#   ctx.ellipse(PAD+pw*0.5,PAD+ph*0.5, pw*0.42, ph*0.4, 0.2, 0, Math.PI*2);
#   ctx.setLineDash([4,4]);
#   stroke(ctx,'#60a5fa33',1); ctx.stroke();
#   ctx.setLineDash([]);
#   mono(ctx,8,false); fill(ctx,'#60a5fa44'); ctx.textAlign='center';
#   ctx.fillText('vocabulary manifold', PAD+pw*0.5, PAD+ph*0.92);
#
#   // soft prompts (orange, anywhere)
#   for(let i=0;i<embedK;i++){
#     const sp=softPositions[i];
#     const sx=PAD+sp.x*pw, sy=PAD+sp.y*ph;
#     // glow
#     const grad=ctx.createRadialGradient(sx,sy,0,sx,sy,18);
#     grad.addColorStop(0,'rgba(249,115,22,0.35)');
#     grad.addColorStop(1,'rgba(249,115,22,0)');
#     ctx.beginPath(); ctx.arc(sx,sy,18,0,Math.PI*2);
#     fill(ctx,grad); ctx.fill();
#     // dot
#     ctx.beginPath(); ctx.arc(sx,sy,6,0,Math.PI*2);
#     fill(ctx,C.orange); ctx.fill();
#     stroke(ctx,'#fff3',1); ctx.stroke();
#     mono(ctx,8.5,true); fill(ctx,C.orange); ctx.textAlign='left';
#     ctx.fillText(`p${i+1}`, sx+8, sy+3);
#   }
#
#   // legend
#   mono(ctx,8,false); ctx.textAlign='left';
#   fill(ctx,'#60a5fa'); ctx.fillRect(PAD,PAD+3,8,8);
#   fill(ctx,'#60a5fa88'); ctx.fillText('Real word embeddings (fixed)',PAD+12,PAD+11);
#   fill(ctx,C.orange); ctx.beginPath(); ctx.arc(PAD+4,PAD+22,4,0,Math.PI*2); ctx.fill();
#   fill(ctx,'#f9731688'); ctx.fillText('Soft prompt vectors (learnable, free in ℝᵈ)',PAD+12,PAD+25);
#
#   document.getElementById('embedInfo').innerHTML =
#     `<span class="hl">${embedK} soft token${embedK>1?'s':''}</span> shown — each lives at an <span class="hl">arbitrary location</span> in ℝᵈ, <span class="hl3">not</span> constrained to the vocabulary manifold.<br>Real word embeddings (blue) are fixed. Soft tokens (orange) are free parameters — gradient descent moves them wherever needed.`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 3 — Parameter Efficiency
# // ─────────────────────────────────────────────────────────────────────────────
# const cvP = document.getElementById('cvParam');
# const ctxP = cvP.getContext('2d');
#
# const MODEL_SIZES = [
#   {label:'T5-Small',  params:60e6,  d:512},
#   {label:'T5-Base',   params:250e6, d:768},
#   {label:'T5-Large',  params:770e6, d:1024},
#   {label:'T5-11B',    params:11e9,  d:1024},
# ];
# let pmModel=1, pmK=20;
#
# document.getElementById('slModel').addEventListener('input',function(){
#   pmModel=+this.value;
#   document.getElementById('slModelv').textContent=MODEL_SIZES[pmModel].label;
#   drawParam();
# });
# document.getElementById('slPK').addEventListener('input',function(){
#   pmK=+this.value;
#   document.getElementById('slPKv').textContent=pmK;
#   drawParam();
# });
#
# function drawParam(){
#   const W=430,H=230;
#   const ctx=ctxP;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);
#
#   const m=MODEL_SIZES[pmModel];
#   const softParams = pmK * m.d;
#   const pct = (softParams / m.params * 100);
#   const fullH=H-50, barW=30, PAD=30;
#
#   // total params bar
#   const tb={x:PAD, w:barW, h:fullH, y:20};
#   ctx.beginPath(); ctx.roundRect(tb.x,tb.y,tb.w,tb.h,6);
#   fill(ctx,'#1a1a30'); ctx.fill();
#   stroke(ctx,C.dimmer,1); ctx.stroke();
#   // soft prompt portion inside
#   const softH=Math.max(3, tb.h * Math.min(pct/100, 1));
#   ctx.beginPath(); ctx.roundRect(tb.x, tb.y+tb.h-softH, tb.w, softH, 3);
#   fill(ctx,'#f97316'); ctx.fill();
#
#   mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText(m.label, tb.x+tb.w/2, tb.y+tb.h+12);
#   ctx.fillText('total', tb.x+tb.w/2, tb.y+tb.h+22);
#
#   mono(ctx,8.5,true); fill(ctx,C.orange);
#   ctx.fillText(`${pct<0.001?pct.toFixed(6):pct.toFixed(4)}%`, tb.x+tb.w/2, tb.y+tb.h-softH-5);
#
#   // big number display
#   const numX=PAD+60;
#
#   function fmtN(n){
#     if(n>=1e9) return (n/1e9).toFixed(1)+'B';
#     if(n>=1e6) return (n/1e6).toFixed(0)+'M';
#     if(n>=1e3) return (n/1e3).toFixed(0)+'K';
#     return n+'';
#   }
#
#   mono(ctx,9,false); fill(ctx,C.muted); ctx.textAlign='left';
#   ctx.fillText('Model architecture:', numX, 36);
#   mono(ctx,11,true); fill(ctx,C.blue);
#   ctx.fillText(`${m.label}   d=${m.d}`, numX, 52);
#
#   mono(ctx,9,false); fill(ctx,C.muted);
#   ctx.fillText('Total model parameters:', numX, 76);
#   mono(ctx,13,true); fill(ctx,C.text);
#   ctx.fillText(fmtN(m.params), numX, 95);
#
#   mono(ctx,9,false); fill(ctx,C.muted);
#   ctx.fillText(`Soft prompt:  k=${pmK} × d=${m.d}`, numX, 118);
#   mono(ctx,13,true); fill(ctx,C.orange);
#   ctx.fillText(fmtN(softParams)+' trainable', numX, 137);
#
#   mono(ctx,9,false); fill(ctx,C.muted);
#   ctx.fillText('Percentage of total:', numX, 160);
#   mono(ctx,16,true); fill(ctx,C.orange);
#   const pctStr = pct < 0.001
#     ? pct.toExponential(2)+'%'
#     : pct.toFixed(4)+'%';
#   ctx.fillText(pctStr, numX, 182);
#
#   // bar compare: full FT vs soft prompt
#   const compX=numX+185, compW=W-compX-PAD;
#   const bars=[
#     {label:'Full FT',  val:m.params,  color:C.red},
#     {label:'Soft PT',  val:softParams, color:C.orange},
#   ];
#   const maxV=m.params;
#   bars.forEach((b,i)=>{
#     const bY=60+i*70, bH=24;
#     const bW=Math.max(2,(b.val/maxV)*(compW));
#     ctx.beginPath(); ctx.roundRect(compX,bY,compW,bH,4);
#     fill(ctx,'#1a1a30'); ctx.fill();
#     ctx.beginPath(); ctx.roundRect(compX,bY,bW,bH,4);
#     fill(ctx,b.color+'33'); ctx.fill();
#     stroke(ctx,b.color,1); ctx.stroke();
#     mono(ctx,8.5,true); fill(ctx,b.color); ctx.textAlign='left';
#     ctx.fillText(b.label, compX+6, bY+16);
#     ctx.textAlign='right';
#     ctx.fillText(fmtN(b.val), compX+compW-6, bY+16);
#   });
#
#   mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='left';
#   ctx.fillText('Parameter comparison', compX, 50);
#
#   document.getElementById('paramInfo').innerHTML=
#     `<span class="hl">Soft Prompt</span>: P ∈ ℝ^{${pmK} × ${m.d}} = <span class="hl">${fmtN(softParams)}</span> params &emsp;vs&emsp; `+
#     `<span class="hl3">Full fine-tuning</span>: <span class="hl3">${fmtN(m.params)}</span> params<br>`+
#     `Only <span class="hl">${pctStr}</span> of model weights are touched. At 11B scale, matches full FT quality.`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 4 — Family Tree
# // ─────────────────────────────────────────────────────────────────────────────
# const cvF = document.getElementById('cvFamily');
# const ctxF = cvF.getContext('2d');
# let famMode = 'hard';
#
# const FAM_INFO = {
#   hard:   { name:'Hard Prompting', color:'#60a5fa', trainable:0, layers:'input (text only)', note:'No training. Carefully written text prefixes. Constrained to vocabulary.' },
#   pt:     { name:'Prompt Tuning',  color:'#f97316', trainable:1, layers:'input embedding only', note:'Soft tokens prepended to input ONLY. Fewest params. Best simplicity-to-quality ratio at ≥1B scale.' },
#   prefix: { name:'Prefix Tuning', color:'#c084fc', trainable:2, layers:'K,V of EVERY attention layer', note:'Soft tokens added to Key and Value projections at every layer. More steering power, more params.' },
#   p2:     { name:'P-Tuning v2',   color:'#facc15', trainable:3, layers:'all layers (deep prefix)', note:'Layer-specific prefix parameters on every layer. Most powerful prompt-based method.' },
# };
#
# function setFamily(m){
#   famMode=m;
#   ['Hard','PT','Pre','P2'].forEach(x=>{
#     document.getElementById('btnFam'+x).classList.toggle('active', m===x.toLowerCase());
#   });
#   drawFamily();
# }
# // fix button ids
# ['hard','pt','prefix','p2'].forEach(m=>{
#   const id = m==='hard'?'btnFamHard':m==='pt'?'btnFamPT':m==='prefix'?'btnFamPre':'btnFamP2';
#   document.getElementById(id).onclick=()=>setFamily(m);
# });
#
# function drawFamily(){
#   const W=900,H=240;
#   const ctx=ctxF;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);
#
#   // Show transformer cross-section: embedding + N layers + output
#   const PAD=30;
#   const layerCount=6;
#   const layerW=90, layerH=34, layerGap=10;
#   const startX = PAD+60;
#   const totalW = layerCount*(layerW+layerGap)+layerW;
#   const midY = H/2;
#
#   const fi = FAM_INFO[famMode];
#
#   // INPUT EMBEDDING block
#   const embX=PAD, embY=midY-22, embW=55, embH=44;
#   ctx.beginPath(); ctx.roundRect(embX,embY,embW,embH,8);
#   fill(ctx,'#13132a'); ctx.fill();
#   stroke(ctx, famMode==='pt'||famMode==='hard' ? fi.color : C.dim, famMode==='pt'?2:1); ctx.stroke();
#   mono(ctx,8,true); fill(ctx, famMode==='pt'||famMode==='hard' ? fi.color : C.muted); ctx.textAlign='center';
#   ctx.fillText('EMBED', embX+embW/2, embY+embH/2-4);
#   ctx.fillText('LAYER', embX+embW/2, embY+embH/2+7);
#
#   // soft tokens at input (prompt tuning and hard)
#   if(famMode==='hard'){
#     // text tokens above embed
#     ['t₁','t₂','...','tₙ'].forEach((t,i)=>{
#       const tx=embX-6+i*14, ty=embY-22;
#       ctx.beginPath(); ctx.roundRect(tx,ty,11,14,3);
#       fill(ctx,'#60a5fa22'); ctx.fill(); stroke(ctx,C.blue,0.8); ctx.stroke();
#       mono(ctx,7,false); fill(ctx,C.blue); ctx.textAlign='center';
#       ctx.fillText(t,tx+5.5,ty+10);
#     });
#   }
#   if(famMode==='pt'){
#     ['p₁','p₂','p₃'].forEach((p,i)=>{
#       const tx=embX-4+i*16, ty=embY-26;
#       ctx.beginPath(); ctx.roundRect(tx,ty,13,16,3);
#       fill(ctx,'#f9731633'); ctx.fill(); stroke(ctx,C.orange,1.5); ctx.stroke();
#       mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='center';
#       ctx.fillText(p,tx+6.5,ty+11);
#     });
#     mono(ctx,7,false); fill(ctx,C.orange); ctx.textAlign='center';
#     ctx.fillText('soft', embX+26, embY-30);
#   }
#
#   // Transformer layers
#   for(let l=0;l<layerCount;l++){
#     const lx=startX+l*(layerW+layerGap);
#     const ly=midY-layerH/2-5;
#
#     // outer frame
#     ctx.beginPath(); ctx.roundRect(lx,ly,layerW,layerH+10,8);
#     fill(ctx,'#13132a'); ctx.fill();
#     stroke(ctx,C.dim,0.8); ctx.stroke();
#
#     // Attention sublayer
#     const attnY=ly+4, attnH=14;
#     ctx.beginPath(); ctx.roundRect(lx+4,attnY,layerW-8,attnH,4);
#
#     const showPrefix = (famMode==='prefix'||famMode==='p2');
#     const attnColor = showPrefix ? fi.color : '#34d39933';
#     const attnBorder = showPrefix ? fi.color : '#34d39966';
#     fill(ctx,attnColor+'22'); ctx.fill(); stroke(ctx,attnBorder,showPrefix?1.5:0.8); ctx.stroke();
#     mono(ctx,7,true); fill(ctx,showPrefix?fi.color:C.green); ctx.textAlign='center';
#     ctx.fillText('Attention K,V', lx+layerW/2, attnY+attnH/2+3);
#
#     // prefix tokens on K,V
#     if(showPrefix && l<(famMode==='p2'?layerCount:layerCount)){
#       const px = lx+layerW/2-14;
#       const py = attnY-14;
#       ['k̃','ṽ'].forEach((s,si)=>{
#         const sx=px+si*15, sy2=py;
#         ctx.beginPath(); ctx.roundRect(sx,sy2,12,11,2);
#         fill(ctx,fi.color+'33'); ctx.fill(); stroke(ctx,fi.color,1); ctx.stroke();
#         mono(ctx,7,true); fill(ctx,fi.color); ctx.textAlign='center';
#         ctx.fillText(s,sx+6,sy2+8);
#       });
#     }
#
#     // FFN sublayer
#     const ffnY=attnY+attnH+2, ffnH=14;
#     ctx.beginPath(); ctx.roundRect(lx+4,ffnY,layerW-8,ffnH,4);
#     fill(ctx,'#3f3f5622'); ctx.fill(); stroke(ctx,C.dim,0.5); ctx.stroke();
#     mono(ctx,7,false); fill(ctx,C.muted); ctx.textAlign='center';
#     ctx.fillText('FFN', lx+layerW/2, ffnY+ffnH/2+3);
#
#     // layer label
#     mono(ctx,7.5,false); fill(ctx,C.dim); ctx.textAlign='center';
#     ctx.fillText(`L${l+1}`, lx+layerW/2, ly+layerH+18);
#
#     // connections
#     if(l>0){
#       const px=startX+l*(layerW+layerGap)-layerGap;
#       ctx.beginPath(); ctx.moveTo(px,midY); ctx.lineTo(lx,midY);
#       stroke(ctx,C.dimmer,1); ctx.stroke();
#     }
#   }
#
#   // Arrow from embed to L1
#   ctx.beginPath(); ctx.moveTo(embX+embW,midY); ctx.lineTo(startX,midY);
#   stroke(ctx,C.dim,1.2); ctx.stroke();
#
#   // Output head
#   const outX=startX+layerCount*(layerW+layerGap)+4;
#   ctx.beginPath(); ctx.moveTo(outX-2,midY); ctx.lineTo(outX+40,midY);
#   stroke(ctx,C.dim,1.2); ctx.stroke();
#   ctx.beginPath(); ctx.roundRect(outX+40,midY-22,60,44,8);
#   fill(ctx,'rgba(52,211,153,0.1)'); ctx.fill();
#   stroke(ctx,C.green,1); ctx.stroke();
#   mono(ctx,8,true); fill(ctx,C.green); ctx.textAlign='center';
#   ctx.fillText('OUTPUT', outX+70, midY-4); ctx.fillText('HEAD', outX+70, midY+10);
#
#   // Bottom labels
#   const labs=[
#     {x:embX+embW/2,  label:'Embedding',    sub:'input layer'},
#     {x:startX+layerCount*(layerW+layerGap)/2-20, label:`${layerCount} Transformer Layers`, sub:'attention + FFN'},
#     {x:outX+70,      label:'Output',       sub:'head'},
#   ];
#   labs.forEach(l=>{
#     mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
#   });
#
#   // Annotation
#   mono(ctx,9,true); fill(ctx,fi.color); ctx.textAlign='left';
#   ctx.fillText(`◉ ${fi.name}`, PAD, H-14);
#   mono(ctx,8,false); fill(ctx,C.muted);
#   ctx.fillText(`  Soft tokens at: ${fi.layers}`, PAD+160, H-14);
#
#   document.getElementById('familyInfo').innerHTML=
#     `<span style="color:${fi.color};font-weight:700">${fi.name}</span>: ${fi.note}`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 5 — Accuracy vs Scale
# // ─────────────────────────────────────────────────────────────────────────────
# const cvS = document.getElementById('cvScale');
# const ctxS = cvS.getContext('2d');
#
# const scaleData = [
#   // [model_size_M, full_ft_acc, prompt_tune_acc, few_shot_acc]
#   [60,   92, 56, 45],
#   [250,  94, 72, 62],
#   [770,  95, 88, 76],
#   [3000, 96, 93, 86],
#   [11000,96, 96, 91],
# ];
#
# function drawScale(){
#   const W=430,H=230,ctx=ctxS;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);
#
#   const PAD=40, PB=35, PR=30;
#   const cW=W-PAD-PR, cH=H-PAD-PB;
#
#   // axes
#   ctx.beginPath(); ctx.moveTo(PAD,PAD); ctx.lineTo(PAD,PAD+cH); ctx.lineTo(PAD+cW,PAD+cH);
#   stroke(ctx,C.dimmer,1.2); ctx.stroke();
#
#   const xVals=scaleData.map(d=>Math.log10(d[0]));
#   const xMin=xVals[0], xMax=xVals[xVals.length-1];
#   const xS=v=>(PAD + (Math.log10(v)-xMin)/(xMax-xMin)*cW);
#   const yS=v=>(PAD+cH - (v-40)/(60)*cH);
#
#   // grid
#   [60,70,80,90,100].forEach(y=>{
#     const gy=yS(y);
#     ctx.beginPath(); ctx.moveTo(PAD,gy); ctx.lineTo(PAD+cW,gy);
#     stroke(ctx,C.dimmer,0.5); ctx.stroke();
#     mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='right';
#     ctx.fillText(y+'%',PAD-4,gy+3);
#   });
#
#   // x labels
#   scaleData.forEach(d=>{
#     mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='center';
#     const label = d[0]>=1000?Math.round(d[0]/1000)+'B':d[0]+'M';
#     ctx.fillText(label, xS(d[0]), PAD+cH+12);
#   });
#
#   mono(ctx,8,false); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText('Model size (log scale)', PAD+cW/2, H-4);
#   ctx.save(); ctx.translate(10,PAD+cH/2); ctx.rotate(-Math.PI/2);
#   ctx.fillText('Accuracy (%)',0,0); ctx.restore();
#
#   const series=[
#     {idx:1, color:C.red,    label:'Full Fine-Tuning'},
#     {idx:2, color:C.orange, label:'Prompt Tuning'},
#     {idx:3, color:C.blue,   label:'Few-Shot (GPT-3 style)'},
#   ];
#   series.forEach(s=>{
#     ctx.beginPath();
#     scaleData.forEach((d,i)=>{
#       const px=xS(d[0]),py=yS(d[s.idx]);
#       i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
#     });
#     stroke(ctx,s.color,2); ctx.stroke();
#     // dots
#     scaleData.forEach(d=>{
#       ctx.beginPath(); ctx.arc(xS(d[0]),yS(d[s.idx]),4,0,Math.PI*2);
#       fill(ctx,s.color); ctx.fill();
#     });
#   });
#
#   // legend
#   series.forEach((s,i)=>{
#     const lx=PAD+5, ly=PAD+8+i*16;
#     ctx.beginPath(); ctx.moveTo(lx,ly); ctx.lineTo(lx+18,ly);
#     stroke(ctx,s.color,2); ctx.stroke();
#     ctx.beginPath(); ctx.arc(lx+9,ly,3,0,Math.PI*2);
#     fill(ctx,s.color); ctx.fill();
#     mono(ctx,8,false); fill(ctx,s.color); ctx.textAlign='left';
#     ctx.fillText(s.label, lx+22, ly+3);
#   });
#
#   // annotation at convergence
#   const cx=xS(11000), cy=yS(96);
#   ctx.beginPath(); ctx.moveTo(cx,cy-8); ctx.lineTo(cx+30,cy-28);
#   stroke(ctx,'#f9731688',1); ctx.stroke();
#   mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='left';
#   ctx.fillText('≈ Full FT quality', cx+32, cy-26);
#   ctx.fillText('at 11B scale!', cx+32, cy-14);
#
#   document.getElementById('scaleInfo').innerHTML=
#     `At small scales, Prompt Tuning <span class="hl">underperforms</span> full fine-tuning. At <span class="hl">≥1B params</span>, the gap closes rapidly.<br>At <span class="hl">11B+</span>, Prompt Tuning matches full FT with <span class="hl">0.001% of parameters</span>.`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // PANEL 6 — Gradient Flow / Training Loop
# // ─────────────────────────────────────────────────────────────────────────────
# const cvG = document.getElementById('cvGrad');
# const ctxG = cvG.getContext('2d');
# let gradRunning=false, gradRaf=null, gradT=0, gradEpoch=0, gradLoss=[];
# const INIT_LOSS=2.8;
#
# function resetGrad(){
#   gradRunning=false; cancelAnimationFrame(gradRaf);
#   gradT=0; gradEpoch=0; gradLoss=[];
#   document.getElementById('btnGradPlay').textContent='▶ Run Training';
#   drawGrad();
# }
#
# function toggleGrad(){
#   if(gradRunning){
#     gradRunning=false; cancelAnimationFrame(gradRaf);
#     document.getElementById('btnGradPlay').textContent='▶ Run Training';
#     return;
#   }
#   gradRunning=true;
#   document.getElementById('btnGradPlay').textContent='⏸ Pause';
#   function loop(){
#     gradT=(gradT+0.02)%1;
#     if(gradT<0.02){ gradEpoch++; gradLoss.push(Math.max(0.08, INIT_LOSS*Math.exp(-gradEpoch*0.18)+0.08)); if(gradLoss.length>80) gradLoss.shift(); }
#     drawGrad();
#     if(gradRunning) gradRaf=requestAnimationFrame(loop);
#     else document.getElementById('btnGradPlay').textContent='▶ Run Training';
#   }
#   loop();
# }
#
# function drawGrad(){
#   const W=430,H=230,ctx=ctxG;
#   ctx.clearRect(0,0,W,H);
#   fill(ctx,'#09091a'); ctx.fillRect(0,0,W,H);
#
#   const midY=H*0.44;
#
#   // ── Architecture diagram ─────────────────────────────────────────────────
#   const blks=[
#     {x:20,  w:52, h:38, label:'Soft\nPrompt', color:C.orange, trainable:true},
#     {x:82,  w:52, h:38, label:'Embed\nLayer',  color:C.dim,   trainable:false},
#     {x:144, w:62, h:38, label:'Transformer\nLayers', color:C.green, trainable:false},
#     {x:216, w:52, h:38, label:'Output\nHead',  color:C.blue,  trainable:false},
#   ];
#
#   blks.forEach(b=>{
#     const by=midY-b.h/2;
#     ctx.beginPath(); ctx.roundRect(b.x,by,b.w,b.h,7);
#     fill(ctx, b.trainable ? b.color+'22' : '#13132a'); ctx.fill();
#     stroke(ctx, b.trainable ? b.color : C.dim, b.trainable?2:0.8); ctx.stroke();
#     mono(ctx,7.5, b.trainable); fill(ctx,b.trainable?b.color:C.muted); ctx.textAlign='center';
#     const lines=b.label.split('\n');
#     lines.forEach((l,i)=>ctx.fillText(l, b.x+b.w/2, by+b.h/2+(i-lines.length/2+0.5)*10));
#     if(!b.trainable){
#       mono(ctx,7,false); fill(ctx,'#60a5fa55'); ctx.textAlign='center';
#       ctx.fillText('❄frozen',b.x+b.w/2,by+b.h+10);
#     }
#   });
#
#   // arrows (forward pass)
#   blks.forEach((b,i)=>{
#     if(i===blks.length-1) return;
#     const nb=blks[i+1];
#     const ax=b.x+b.w, ay=midY;
#     ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(nb.x,ay);
#     stroke(ctx,C.dimmer,1); ctx.stroke();
#     const pulse = gradRunning && gradT < 0.5 ? gradT*2 : 0;
#     if(gradRunning){
#       const px=ax+pulse*(nb.x-ax);
#       ctx.beginPath(); ctx.arc(px,ay,3.5,0,Math.PI*2);
#       fill(ctx,'#34d39988'); ctx.fill();
#     }
#   });
#
#   // Loss computation
#   const lossX=285, lossY=midY-18, lossW=50, lossH=36;
#   ctx.beginPath(); ctx.roundRect(lossX,lossY,lossW,lossH,7);
#   const lossVal = gradLoss.length>0 ? gradLoss[gradLoss.length-1] : INIT_LOSS;
#   const lossColor = lossVal < 0.5 ? C.green : lossVal<1.5?C.yellow:C.red;
#   fill(ctx,lossColor+'22'); ctx.fill(); stroke(ctx,lossColor,1.5); ctx.stroke();
#   mono(ctx,8,true); fill(ctx,lossColor); ctx.textAlign='center';
#   ctx.fillText('LOSS', lossX+lossW/2, lossY+14);
#   ctx.fillText(lossVal.toFixed(3), lossX+lossW/2, lossY+28);
#
#   ctx.beginPath(); ctx.moveTo(blks[3].x+blks[3].w,midY); ctx.lineTo(lossX,midY);
#   stroke(ctx,C.dimmer,1); ctx.stroke();
#
#   // Gradient back arrow (only to soft prompt!)
#   if(gradRunning && gradT > 0.5){
#     const gProg=(gradT-0.5)*2;
#     const startX=lossX, endX=blks[0].x+blks[0].w/2;
#     const arcH=H*0.55;
#     ctx.beginPath();
#     const cp1x=(startX+endX)/2, cp1y=arcH;
#     const px=startX+(endX-startX)*gProg;
#     // just draw the gradient path up to progress
#     ctx.save();
#     ctx.setLineDash([4,4]);
#     ctx.beginPath();
#     // quadratic to point
#     for(let s=0;s<gProg;s+=0.02){
#       const qx=Math.pow(1-s,2)*startX+2*(1-s)*s*cp1x+s*s*endX;
#       const qy=Math.pow(1-s,2)*midY+2*(1-s)*s*arcH+s*s*(midY+28);
#       if(s===0) ctx.moveTo(qx,qy); else ctx.lineTo(qx,qy);
#     }
#     stroke(ctx,'#f97316bb',1.8); ctx.stroke();
#     ctx.setLineDash([]);
#     ctx.restore();
#
#     // ∇ label
#     const labProg=Math.min(gProg,0.5);
#     const lqx=Math.pow(1-labProg,2)*startX+2*(1-labProg)*labProg*cp1x+labProg*labProg*endX;
#     const lqy=Math.pow(1-labProg,2)*midY+2*(1-labProg)*labProg*arcH+labProg*labProg*(midY+28);
#     ctx.beginPath(); ctx.arc(lqx,lqy,4,0,Math.PI*2);
#     fill(ctx,C.orange+'cc'); ctx.fill();
#   }
#
#   mono(ctx,7.5,true); fill(ctx,C.orange); ctx.textAlign='center';
#   ctx.fillText('∇ gradient → only here', blks[0].x+blks[0].w/2, midY+28);
#
#   // ── Loss curve ────────────────────────────────────────────────────────────
#   const cX=350, cY=18, cW=W-cX-10, cH=H-40;
#   fill(ctx,'#0d0d1a'); ctx.fillRect(cX,cY,cW,cH);
#   stroke(ctx,C.dimmer,0.8); ctx.strokeRect(cX,cY,cW,cH);
#
#   mono(ctx,7.5,false); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText('Training Loss', cX+cW/2, cY+10);
#
#   if(gradLoss.length>1){
#     const maxL=INIT_LOSS, minL=0;
#     ctx.beginPath();
#     gradLoss.forEach((l,i)=>{
#       const px=cX+1+(i/(Math.max(gradLoss.length-1,1)))*(cW-2);
#       const py=cY+cH-2-(l-minL)/(maxL-minL)*(cH-4);
#       i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);
#     });
#     stroke(ctx,C.orange,2); ctx.stroke();
#     gradLoss.forEach((l,i)=>{}); // lineTo done
#     ctx.lineTo(cX+1+(gradLoss.length-1)/(Math.max(gradLoss.length-1,1))*(cW-2), cY+cH-2);
#     ctx.lineTo(cX+1,cY+cH-2); ctx.closePath();
#     fill(ctx,'rgba(249,115,22,0.08)'); ctx.fill();
#   }
#
#   mono(ctx,7,false); fill(ctx,C.muted); ctx.textAlign='center';
#   ctx.fillText('epoch →', cX+cW/2, cY+cH+10);
#
#   document.getElementById('gradInfo').innerHTML=
#     `Epoch <span class="hl">${gradEpoch}</span> &nbsp;|&nbsp; `+
#     `Loss: <span class="${lossVal<0.5?'hl2':lossVal<1.5?'hl4':'hl'}">${lossVal.toFixed(4)}</span><br>`+
#     `Gradients flow backwards through the frozen transformer but <span class="hl">only update</span> the soft prompt P. All model weights remain <span class="hl2">unchanged ❄</span>.`;
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // QUIZ
# // ─────────────────────────────────────────────────────────────────────────────
# const QUIZ=[
#   {q:'What are "soft prompts" in Prompt Tuning?',
#    opts:['Carefully written text instructions','Learnable floating-point vectors prepended to input','Low-rank weight matrices added to attention','Extra layers added on top of the model'],
#    ans:1,
#    fb:['Incorrect — that describes hard prompting.','✓ Correct! Soft prompts are free-floating vectors pᵢ ∈ ℝᵈ — not constrained to real words.','Incorrect — that describes LoRA.','Incorrect — no new layers are added.']},
#   {q:'Which model weights are updated during Prompt Tuning?',
#    opts:['All weights (like full fine-tuning)','Only the attention Q,K,V matrices','Only the soft prompt matrix P','The embedding table E only'],
#    ans:2,
#    fb:['Incorrect — full FT updates all weights.','Incorrect — that is closer to LoRA/Prefix Tuning.','✓ Correct! ONLY P ∈ ℝ^{k × d} receives gradient updates.','Incorrect — the embedding table is frozen.']},
#   {q:'At what model scale does Prompt Tuning match full fine-tuning quality?',
#    opts:['60M parameters','250M parameters','≥1B parameters','It never matches full fine-tuning'],
#    ans:2,
#    fb:['Incorrect — at 60M the gap is large.','Incorrect — still a noticeable gap at 250M.','✓ Correct! Lester et al. showed parity at ≥1B, especially ≥11B.','Incorrect — at sufficient scale it matches.']},
#   {q:'How does Prefix Tuning differ from Prompt Tuning?',
#    opts:['Prefix Tuning trains the full model','Prefix Tuning adds soft tokens to every attention K,V layer','Prefix Tuning uses text tokens instead of vectors','Prefix Tuning requires more data'],
#    ans:1,
#    fb:['Incorrect.','✓ Correct! Prefix Tuning prepends soft tokens to K and V projections at EVERY transformer layer, not just input.','Incorrect — both use continuous vectors.','Incorrect — this is not the distinction.']},
#   {q:'If d=1024 and k=20, how many trainable parameters does Prompt Tuning add?',
#    opts:['1,024','20,480','204,800','1,048,576'],
#    ans:1,
#    fb:['Incorrect — that is d alone.','✓ Correct! P ∈ ℝ^{20 × 1024} = 20 × 1024 = 20,480 parameters.','Incorrect.','Incorrect.']},
#   {q:'Why can\'t hard prompts be placed "anywhere" in embedding space?',
#    opts:['Hard prompts are too long','Every token must correspond to a real vocabulary item with a fixed embedding','Hard prompts cannot attend to input tokens','Hard prompts update model weights'],
#    ans:1,
#    fb:['Incorrect — length is not the constraint.','✓ Correct! Tokens map to fixed embedding vectors. You cannot place them at arbitrary ℝᵈ locations.','Incorrect.','Incorrect — hard prompts involve no training.']},
# ];
#
# let quizAnswers=new Array(QUIZ.length).fill(null);
# let quizDone=0;
#
# function buildQuiz(){
#   const grid=document.getElementById('quizGrid');
#   grid.innerHTML='';
#   QUIZ.forEach((q,qi)=>{
#     const card=document.createElement('div'); card.className='q-card';
#     card.innerHTML=`<div class="q-num">Question ${qi+1}</div><div class="q-prompt">${q.q}</div>`;
#     q.opts.forEach((o,oi)=>{
#       const btn=document.createElement('button'); btn.className='q-opt';
#       btn.textContent=o;
#       btn.onclick=()=>answerQ(qi,oi,btn,card);
#       card.appendChild(btn);
#     });
#     const fb=document.createElement('div'); fb.className='q-feedback'; fb.id=`fb${qi}`;
#     card.appendChild(fb);
#     grid.appendChild(card);
#   });
# }
#
# function answerQ(qi,oi,btn,card){
#   if(quizAnswers[qi]!==null) return;
#   quizAnswers[qi]=oi;
#   quizDone++;
#   const q=QUIZ[qi];
#   const correct=oi===q.ans;
#   btn.classList.add(correct?'correct':'wrong');
#   card.querySelectorAll('.q-opt').forEach((b,i)=>{
#     b.disabled=true;
#     if(i===q.ans) b.classList.add('correct');
#   });
#   const fb=document.getElementById(`fb${qi}`);
#   fb.textContent=q.fb[oi]; fb.className=`q-feedback show ${correct?'ok':'bad'}`;
#
#   // score
#   const sc=quizAnswers.filter(a=>a!==null&&QUIZ[quizAnswers.indexOf(a)]&&a===QUIZ[quizAnswers.indexOf(a)].ans).length;
#   const correctCount = QUIZ.filter((q,i)=>quizAnswers[i]===q.ans).length;
#   const bar=document.getElementById('scoreBar');
#   bar.innerHTML=`<span id="scoreText">Score: <span style="color:${correctCount===QUIZ.length?'#34d399':'#f97316'};font-weight:700">${correctCount}/${QUIZ.length}</span></span>`;
#   QUIZ.forEach((_,i)=>{
#     const pip=document.createElement('span'); pip.className='pip'+(quizAnswers[i]===null?'':quizAnswers[i]===QUIZ[i].ans?' ok':' bad');
#     bar.appendChild(pip);
#   });
#   if(quizDone===QUIZ.length){
#     const t=document.createElement('span');
#     t.style.cssText='margin-left:8px;color:#34d399;font-weight:700;';
#     t.textContent=correctCount===QUIZ.length?'🎉 Perfect! You understand Prompt Tuning!':correctCount>=4?'👍 Great work!':'Keep reviewing the panels above.';
#     bar.appendChild(t);
#   }
# }
#
# // ─────────────────────────────────────────────────────────────────────────────
# // INIT
# // ─────────────────────────────────────────────────────────────────────────────
# drawArch();
# drawEmbed();
# document.getElementById('slModelv').textContent = MODEL_SIZES[pmModel].label;
# drawParam();
# drawFamily();
# drawScale();
# drawGrad();
# buildQuiz();
#
# // Ctrl+Scroll zoom
# var ZOOM=1.0, zoomTimer=null;
# function applyZoom(){
#   document.body.style.zoom=ZOOM;
#   clearTimeout(zoomTimer);
#   var t=document.getElementById('zt');
#   if(!t){t=document.createElement('div');t.id='zt';t.style.cssText='position:fixed;bottom:18px;right:18px;background:#0f0f1a;border:1px solid #f97316;color:#f97316;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;padding:6px 12px;border-radius:7px;z-index:9999;pointer-events:none;transition:opacity .3s;';document.body.appendChild(t);}
#   t.textContent='zoom '+Math.round(ZOOM*100)+'%'; t.style.opacity='1';
#   zoomTimer=setTimeout(()=>t.style.opacity='0',1200);
# }
# document.addEventListener('wheel',e=>{
#   if(!e.ctrlKey&&!e.metaKey)return; e.preventDefault();
#   ZOOM=Math.min(3,Math.max(0.4,Math.round((ZOOM+(e.deltaY>0?-0.05:0.05))*100)/100));
#   applyZoom();
# },{passive:false});
# </script>
# </body>
# </html>
# """
#
# PROMPT_TUNING_VISUAL_HEIGHT = 1800

