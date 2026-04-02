"""
Optimisers & Learning Rate Strategies interactive visual.
Embed via: st.components.v1.html(OPTIMISERS_VISUAL_HTML, height=OPTIMISERS_VISUAL_HEIGHT)
"""

OPTIMISERS_VISUAL_HTML = """<!DOCTYPE html>
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

var ARR  = "\u2192";
var LARR = "\u2190";
var DASH = "\u2014";
var CHK  = "\u2713";
var WARN = "\u26A0";
var BULB = "!";
var TARG = ">";
var MUL  = "\u00D7";
var ETA  = "\u03B7";
var BETA = "\u03B2";
var RHO  = "\u03C1";
var EPS  = "\u03B5";
var GAMMA= "\u03B3";
var SQRT = "\u221A";
var NABLA= "\u2207";
var PI   = "\u03C0";
var NORM = "\u2016";
var TIMES= "\u00D7";

function TabBar(props) {
  var tabs = props.tabs, active = props.active, onChange = props.onChange;
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

function Card(props) {
  return (
    <div style={Object.assign({
      background:C.card,borderRadius:10,padding:"18px 22px",
      border:"1px solid "+(props.highlight?C.accent:C.border),
    },props.style||{})}>
      {props.children}
    </div>
  );
}

function Insight(props) {
  return (
    <div style={Object.assign({
      maxWidth:750,margin:"16px auto 0",padding:"16px 22px",
      background:"rgba(255,107,53,0.06)",borderRadius:10,
      border:"1px solid rgba(255,107,53,0.2)",
    },props.style||{})}>
      <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:6}}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
      <div style={{fontSize:11,color:C.muted,lineHeight:1.8}}>{props.children}</div>
    </div>
  );
}

function SectionTitle(props) {
  return (
    <div style={{textAlign:"center",marginBottom:20}}>
      <div style={{fontSize:18,fontWeight:800,color:C.text,marginBottom:4}}>{props.title}</div>
      <div style={{fontSize:12,color:C.muted}}>{props.subtitle}</div>
    </div>
  );
}

/* ---- TAB 1: LOSS LANDSCAPE ---- */
function TabLossLandscape() {
  var _s = useState(0); var sel = _s[0], setSel = _s[1];
  var obstacles = [
    {name:"Local Minimum",color:C.orange,
     desc:"A valley that is NOT the global lowest point. Gradient is zero so SGD stops, but the solution is suboptimal. Less common than saddle points in deep networks.",
     fix:"Momentum and warm restarts help escape. In practice deep nets rarely get stuck in truly bad local minima."},
    {name:"Saddle Point",color:C.red,
     desc:"Gradient is zero but it is NOT a minimum. Curves DOWN in some directions and UP in others. The dominant obstacle in deep networks.",
     fix:"Momentum and mini-batch noise help escape. Adaptive optimisers are particularly effective here."},
    {name:"Plateau",color:C.purple,
     desc:"A wide flat region where gradients are near zero. Training stalls. Common with saturating activations like Sigmoid and Tanh.",
     fix:"Larger lr, ReLU activations, BatchNorm, and adaptive optimisers (Adam scales up small gradients)."},
    {name:"Ravine",color:C.yellow,
     desc:"A narrow valley curving sharply in one dimension. Vanilla SGD oscillates across the narrow axis and progresses slowly along it.",
     fix:"Momentum dampens cross-axis oscillations. Adaptive per-parameter lr (Adam) also resolves this."},
    {name:"Cliff / Spike",color:C.pink,
     desc:"A sudden sharp rise in the loss surface. One mini-batch near a cliff produces a massive gradient that throws weights far away.",
     fix:"Gradient clipping (clip-by-norm) caps the update size. Essential for RNNs and transformers."},
  ];
  var ob = obstacles[sel];
  var W=680, H=200;
  var xs = []; for(var i=0;i<100;i++) xs.push(i*W/99);
  var ys = xs.map(function(x,i){
    var t=i/99;
    return H*0.15+H*0.45*Math.pow(Math.sin(t*Math.PI*1.5+0.3),2)
      +H*0.15*Math.exp(-Math.pow((t-0.35)*8,2))
      -H*0.12*Math.exp(-Math.pow((t-0.65)*10,2))
      +H*0.08*Math.exp(-Math.pow((t-0.85)*12,2));
  });
  var pts = xs.map(function(x,i){return x+","+ys[i];}).join(" ");
  var lms = [
    {x:0.35,label:"Saddle",color:C.red,idx:1},
    {x:0.50,label:"Plateau",color:C.purple,idx:2},
    {x:0.65,label:"Local Min",color:C.orange,idx:0},
    {x:0.78,label:"Cliff",color:C.pink,idx:4},
    {x:0.92,label:"Global Min",color:C.green,idx:-1},
  ];
  return (
    <div>
      <SectionTitle title="The Loss Landscape"
        subtitle={"What gradient descent must navigate "+DASH+" click each obstacle"}/>
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Loss Surface (1D slice)</div>
        <svg width="100%" viewBox={"0 0 "+W+" "+(H+44)} style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <polyline points={pts} fill="none" stroke={C.accent} strokeWidth={2.5}/>
          {lms.map(function(lm,i){
            var xi=Math.round(lm.x*99); var ly=ys[xi]; var lx=xs[xi]; var on=sel===lm.idx;
            return (
              <g key={i} style={{cursor:"pointer"}} onClick={function(){if(lm.idx>=0)setSel(lm.idx);}}>
                <circle cx={lx} cy={ly} r={on?9:6} fill={lm.color+(on?"":"50")} stroke={lm.color} strokeWidth={on?2:1}/>
                <line x1={lx} y1={ly+10} x2={lx} y2={H+14} stroke={lm.color+"50"} strokeWidth={1} strokeDasharray="3,3"/>
                <rect x={lx-32} y={H+17} width={64} height={18} rx={4}
                  fill={lm.color+(on?"30":"15")} stroke={lm.color+(on?"80":"40")}/>
                <text x={lx} y={H+30} textAnchor="middle" fill={lm.color}
                  fontSize={8} fontWeight={700} fontFamily="monospace">{lm.label}</text>
              </g>
            );
          })}
          <text x={10} y={18} fill={C.muted} fontSize={9} fontFamily="monospace">Loss</text>
          <line x1={0} y1={H+5} x2={W} y2={H+5} stroke={C.dim} strokeWidth={1}/>
        </svg>
      </Card>
      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {obstacles.map(function(o,i){var on=sel===i;return (
          <button key={i} onClick={function(){setSel(i);}} style={{
            padding:"6px 14px",borderRadius:8,fontSize:10,fontWeight:700,fontFamily:"monospace",cursor:"pointer",
            border:"1.5px solid "+(on?o.color:C.border),background:on?o.color+"20":C.card,color:on?o.color:C.muted,
          }}>{o.name}</button>
        );})}
      </div>
      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:ob.color}}>
        <div style={{fontSize:15,fontWeight:800,color:ob.color,marginBottom:8}}>{ob.name}</div>
        <div style={{fontSize:11,color:C.text,lineHeight:1.8,marginBottom:10}}>{ob.desc}</div>
        <div style={{fontSize:10,color:C.muted,borderLeft:"3px solid "+ob.color+"60",paddingLeft:10}}>
          <span style={{color:C.green,fontWeight:700}}>Fix: </span>{ob.fix}
        </div>
      </Card>
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Batch Size Shapes the Trajectory</div>
        <svg width="100%" viewBox="0 0 680 130" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          {[
            {label:"Full GD (B=N)",color:C.blue,noise:1,desc:"Smooth, slow. One update/epoch."},
            {label:"Mini-batch (B=64)",color:C.green,noise:10,desc:"Best balance. GPU-parallel."},
            {label:"SGD (B=1)",color:C.orange,noise:26,desc:"Very noisy. Helps escape traps."},
          ].map(function(bm,bi){
            var bx=20+bi*228; var ptsl=[];
            for(var k=0;k<28;k++){
              var px=bx+14+k*6.8;
              var py=105-k*2.0+Math.sin(k*(bi+1)*0.9)*(bm.noise*0.6);
              ptsl.push(px+","+Math.max(18,Math.min(108,py)));
            }
            return (
              <g key={bi}>
                <rect x={bx} y={8} width={215} height={115} rx={6} fill={bm.color+"06"} stroke={bm.color+"30"}/>
                <text x={bx+107} y={24} textAnchor="middle" fill={bm.color} fontSize={9} fontWeight={700} fontFamily="monospace">{bm.label}</text>
                <polyline points={ptsl.join(" ")} fill="none" stroke={bm.color} strokeWidth={2}/>
                <text x={bx+107} y={118} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{bm.desc}</text>
              </g>
            );
          })}
        </svg>
      </Card>
      <Insight>
        The dominant obstacle in modern deep learning is <span style={{color:C.red,fontWeight:700}}>saddle points</span>, not local minima {DASH} the gradient is zero but the loss can still decrease in some direction. Mini-batch noise and momentum both help escape. <span style={{color:C.yellow,fontWeight:700}}>Ravines</span> are why plain SGD is slow: it wastes steps oscillating across the narrow axis.
      </Insight>
    </div>
  );
}

/* ---- TAB 2: OPTIMISERS ---- */
function TabOptimisers() {
  var _s=useState(3); var sel=_s[0], setSel=_s[1];
  var _st=useState(0); var step=_st[0], setStep=_st[1];
  var opts = [
    {name:"SGD",color:C.red,hasMom:false,hasAdapt:false,
     f1:"w "+LARR+" w "+DASH+" "+ETA+" "+NABLA+"L(w)",f2:"",
     params:ETA+"=0.01 to 0.1",
     pro:"Simple, memory-free, well-understood theory.",
     con:"Same lr for all params. Slow in ravines. No memory.",
     use:"Simple convex problems, linear models.",
     steps:["Compute gradient g = "+NABLA+"L(w)","Scale by lr: "+ETA+" x g","Subtract: w "+LARR+" w "+DASH+" "+ETA+"g"]},
    {name:"SGD+Momentum",color:C.orange,hasMom:true,hasAdapt:false,
     f1:"v = "+BETA+"v + (1"+DASH+BETA+")g",f2:"w "+LARR+" w "+DASH+" "+ETA+"v",
     params:ETA+"=0.01, "+BETA+"=0.9",
     pro:"Dampens ravine oscillation. Faster convergence along valleys.",
     con:"Still one global lr. Tuning "+BETA+" matters.",
     use:"CNNs with carefully tuned learning rate.",
     steps:["Compute gradient g","Update velocity: v = "+BETA+"v + (1"+DASH+BETA+")g","Move in velocity direction (smoothed)"]},
    {name:"RMSProp",color:C.blue,hasMom:false,hasAdapt:true,
     f1:"s = "+RHO+"s + (1"+DASH+RHO+")g"+TIMES+"g",f2:"w "+LARR+" w "+DASH+" ("+ETA+"/"+SQRT+"s) "+TIMES+" g",
     params:ETA+"=0.001, "+RHO+"=0.9",
     pro:"Per-parameter lr. Handles non-stationary gradients.",
     con:"No momentum. Sensitive to initial lr.",
     use:"RNNs, non-stationary objectives.",
     steps:["Compute gradient g","Track EMA of g^2: s = "+RHO+"s + (1"+DASH+RHO+")g^2","Divide step by "+SQRT+"s (normalise by recent magnitude)"]},
    {name:"Adam",color:C.green,hasMom:true,hasAdapt:true,
     f1:"m = "+BETA+"1*m + (1"+DASH+BETA+"1)*g  [momentum]",
     f2:"v = "+BETA+"2*v + (1"+DASH+BETA+"2)*g^2  [rms]    w "+LARR+" w "+DASH+" "+ETA+"*m_hat/("+SQRT+"v_hat+"+EPS+")",
     params:ETA+"=0.001, "+BETA+"1=0.9, "+BETA+"2=0.999",
     pro:"Momentum + adaptive lr + bias correction. Fast and robust.",
     con:"L2 regularisation is broken inside Adam. Use AdamW.",
     use:"General default for almost all deep learning.",
     steps:["Compute gradient g","1st moment: m = "+BETA+"1*m + (1"+DASH+BETA+"1)*g","2nd moment: v = "+BETA+"2*v + (1"+DASH+BETA+"2)*g^2","Bias-correct: m_hat=m/(1"+DASH+BETA+"1^t), v_hat=v/(1"+DASH+BETA+"2^t)","Step: "+ETA+"*m_hat/("+SQRT+"v_hat+"+EPS+")"]},
    {name:"AdamW",color:C.purple,hasMom:true,hasAdapt:true,
     f1:"[All Adam steps]",
     f2:"THEN: w "+LARR+" w "+DASH+" "+ETA+GAMMA+"w  (weight decay SEPARATE)",
     params:ETA+"=0.001, "+GAMMA+"(wd)=0.01",
     pro:"Correct weight decay decoupled from adaptive scaling.",
     con:"One extra hyperparameter "+GAMMA+" to tune.",
     use:"Transformers (BERT, GPT, ViT). Weight decay matters.",
     steps:["All 5 Adam steps first","Apply weight decay SEPARATELY: w "+LARR+" w "+DASH+" "+ETA+GAMMA+"w","Decay rate is always exactly "+GAMMA+" (not gradient-scaled)"]},
  ];
  var opt=opts[sel];
  return (
    <div>
      <SectionTitle title="Optimiser Comparison"
        subtitle={"Each optimiser adds one idea on top of the previous "+DASH+" click to explore"}/>
      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {opts.map(function(o,i){var on=sel===i;return(
          <button key={i} onClick={function(){setSel(i);setStep(0);}} style={{
            padding:"8px 14px",borderRadius:8,fontSize:10,fontWeight:700,fontFamily:"monospace",cursor:"pointer",
            border:"1.5px solid "+(on?o.color:C.border),background:on?o.color+"20":C.card,color:on?o.color:C.muted,
          }}>{o.name}</button>
        );})}
      </div>
      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:opt.color}}>
        <div style={{fontSize:20,fontWeight:800,color:opt.color,marginBottom:8}}>{opt.name}</div>
        <div style={{display:"flex",gap:8,marginBottom:10,flexWrap:"wrap"}}>
          <span style={{fontSize:9,padding:"3px 8px",borderRadius:20,fontFamily:"monospace",fontWeight:700,
            background:opt.hasMom?C.green+"20":C.dim+"20",color:opt.hasMom?C.green:C.dim}}>
            {opt.hasMom?CHK+" Momentum":"x  Momentum"}
          </span>
          <span style={{fontSize:9,padding:"3px 8px",borderRadius:20,fontFamily:"monospace",fontWeight:700,
            background:opt.hasAdapt?C.cyan+"20":C.dim+"20",color:opt.hasAdapt?C.cyan:C.dim}}>
            {opt.hasAdapt?CHK+" Adaptive LR":"x  Adaptive LR"}
          </span>
        </div>
        <div style={{background:"#08080d",borderRadius:6,padding:"10px 14px",marginBottom:10,border:"1px solid "+C.border}}>
          <div style={{fontSize:9,color:C.muted,marginBottom:4}}>UPDATE RULE</div>
          <div style={{fontSize:10,color:opt.color,fontFamily:"monospace",lineHeight:1.9}}>{opt.f1}</div>
          {opt.f2&&<div style={{fontSize:10,color:opt.color,fontFamily:"monospace",lineHeight:1.9}}>{opt.f2}</div>}
        </div>
        <div style={{fontSize:9,color:C.muted,fontFamily:"monospace",marginBottom:10}}>{"Default params: "+opt.params}</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <div style={{borderLeft:"3px solid "+C.green+"60",paddingLeft:10}}>
            <div style={{fontSize:8,color:C.green,fontWeight:700,marginBottom:3}}>ADVANTAGE</div>
            <div style={{fontSize:10,color:C.muted,lineHeight:1.6}}>{opt.pro}</div>
          </div>
          <div style={{borderLeft:"3px solid "+C.red+"60",paddingLeft:10}}>
            <div style={{fontSize:8,color:C.red,fontWeight:700,marginBottom:3}}>LIMITATION</div>
            <div style={{fontSize:10,color:C.muted,lineHeight:1.6}}>{opt.con}</div>
          </div>
        </div>
        <div style={{marginTop:10,fontSize:10,color:C.cyan,fontFamily:"monospace"}}>
          <span style={{color:C.muted}}>Use when: </span>{opt.use}
        </div>
      </Card>
      <div style={{display:"flex",gap:12,maxWidth:750,margin:"0 auto 16px",flexWrap:"wrap"}}>
        <Card style={{flex:2,minWidth:280}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Step-by-Step Update</div>
          {opt.steps.map(function(s,i){return(
            <div key={i} onClick={function(){setStep(i);}} style={{
              display:"flex",gap:10,alignItems:"flex-start",padding:"8px 10px",
              borderRadius:6,marginBottom:4,cursor:"pointer",
              background:step===i?opt.color+"15":C.card,
              border:"1px solid "+(step===i?opt.color+"60":C.border+"40"),transition:"all 0.2s",
            }}>
              <div style={{width:20,height:20,borderRadius:"50%",background:opt.color+(step===i?"":"30"),
                color:step===i?"#000":opt.color,fontSize:9,fontWeight:800,
                display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>{i+1}</div>
              <div style={{fontSize:10,color:step===i?C.text:C.muted,lineHeight:1.5}}>{s}</div>
            </div>
          );})}
        </Card>
        <Card style={{flex:1,minWidth:200}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Ravine Behaviour</div>
          <svg width="100%" viewBox="0 0 240 185" style={{background:"#08080d",borderRadius:6,border:"1px solid "+C.border}}>
            <ellipse cx={120} cy={92} rx={105} ry={78} fill="none" stroke={C.dim} strokeWidth={1} strokeDasharray="4,3"/>
            <ellipse cx={120} cy={92} rx={72} ry={48} fill="none" stroke={C.dim+"60"} strokeWidth={1} strokeDasharray="4,3"/>
            <ellipse cx={120} cy={92} rx={38} ry={20} fill="none" stroke={C.dim+"40"} strokeWidth={1} strokeDasharray="4,3"/>
            {sel<=1&&(function(){var p=[];for(var k=0;k<20;k++){var px=28+k*4,py=92+Math.sin(k*3.2)*(38*Math.exp(-k*0.06));p.push(px+","+py);}return<polyline points={p.join(" ")} fill="none" stroke={C.red} strokeWidth={2}/>;})()}
            {sel>=1&&sel<=2&&(function(){var p=[];for(var k=0;k<22;k++){var px=28+k*4,py=92+Math.sin(k*2.2)*(18*Math.exp(-k*0.10));p.push(px+","+py);}return<polyline points={p.join(" ")} fill="none" stroke={C.orange} strokeWidth={2}/>;})()}
            {sel>=3&&(function(){var p=[];for(var k=0;k<24;k++){var px=28+k*4,py=92+Math.sin(k*1.1)*(8*Math.exp(-k*0.14));p.push(px+","+py);}return<polyline points={p.join(" ")} fill="none" stroke={C.green} strokeWidth={2}/>;})()}
            <circle cx={120} cy={92} r={7} fill={C.yellow}/>
            <text x={120} y={178} textAnchor="middle" fill={C.yellow} fontSize={8} fontFamily="monospace">Global Min</text>
          </svg>
          <div style={{fontSize:9,color:C.dim,marginTop:6,textAlign:"center"}}>
            <span style={{color:C.red}}>- SGD  </span>
            <span style={{color:C.orange}}>- Mom  </span>
            <span style={{color:C.green}}>- Adam</span>
          </div>
        </Card>
      </div>
      <Insight icon={TARG} title="The Modern Default">
        <span style={{color:C.green,fontWeight:700}}>Adam</span> (lr=0.001) works out of the box for most tasks. Use <span style={{color:C.purple,fontWeight:700}}>AdamW</span> whenever weight decay matters (transformers, large models). For well-tuned CNNs, <span style={{color:C.orange,fontWeight:700}}>SGD+Momentum</span> with a cosine schedule often beats Adam in final accuracy.
      </Insight>
    </div>
  );
}

/* ---- TAB 3: ADAPTIVE LR ---- */
function TabAdaptive() {
  var _lr=useState(0.01); var lr=_lr[0], setLr=_lr[1];
  var _rho=useState(0.9); var rho=_rho[0], setRho=_rho[1];
  var N_GRAD=30;
  var grads=[0.8,0.9,0.75,0.85,0.9,0.7,0.8,0.75,0.9,0.85,
             0.06,0.08,0.05,0.07,0.06,0.08,0.05,0.06,0.07,0.05,
             0.5,0.6,0.55,0.65,0.5,0.6,0.55,0.5,0.6,0.55];
  var adagrad=useMemo(function(){var G=0,r=[];grads.forEach(function(g){G+=g*g;r.push(lr/Math.sqrt(G+1e-8));});return r;},[lr]);
  var rmsprop=useMemo(function(){var s=0,r=[];grads.forEach(function(g){s=rho*s+(1-rho)*g*g;r.push(lr/Math.sqrt(s+1e-8));});return r;},[lr,rho]);
  var maxR=Math.max(lr*4,adagrad[0]||0.1,rmsprop[0]||0.1);
  return (
    <div>
      <SectionTitle title="Adaptive Learning Rate Mechanics"
        subtitle={"Why one global lr is not enough "+DASH+" and how AdaGrad, RMSProp, Adam fix it"}/>
      <div style={{display:"flex",gap:12,maxWidth:750,margin:"0 auto 16px",flexWrap:"wrap"}}>
        <Card style={{flex:1,minWidth:220}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Controls</div>
          <div style={{marginBottom:14}}>
            <div style={{fontSize:10,color:C.muted,marginBottom:4}}>Base lr = <span style={{color:C.accent}}>{lr.toFixed(3)}</span></div>
            <input type="range" min={0.001} max={0.05} step={0.001} value={lr} onChange={function(e){setLr(parseFloat(e.target.value));}} style={{width:"100%",accentColor:C.accent}}/>
          </div>
          <div style={{marginBottom:14}}>
            <div style={{fontSize:10,color:C.muted,marginBottom:4}}>RMSProp {RHO} = <span style={{color:C.blue}}>{rho.toFixed(2)}</span></div>
            <input type="range" min={0.5} max={0.99} step={0.01} value={rho} onChange={function(e){setRho(parseFloat(e.target.value));}} style={{width:"100%",accentColor:C.blue}}/>
          </div>
          <div style={{background:"#08080d",borderRadius:6,padding:"10px",border:"1px solid "+C.border}}>
            <div style={{fontSize:9,color:C.muted,marginBottom:6}}>GRADIENT PATTERN</div>
            <div style={{fontSize:9,color:C.dim,lineHeight:1.9}}>
              <span style={{color:C.orange}}>Steps 1-10: </span>Large (~0.8)<br/>
              <span style={{color:C.purple}}>Steps 11-20: </span>Sparse small (~0.06)<br/>
              <span style={{color:C.cyan}}>Steps 21-30: </span>Medium resumes (~0.55)
            </div>
          </div>
        </Card>
        <Card style={{flex:2,minWidth:320}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Effective Learning Rate Over Time</div>
          <svg width="100%" viewBox="0 0 420 170" style={{background:"#08080d",borderRadius:6,border:"1px solid "+C.border}}>
            <line x1={35} y1={150} x2={410} y2={150} stroke={C.dim} strokeWidth={1}/>
            <line x1={35} y1={20}  x2={35}  y2={150} stroke={C.dim} strokeWidth={1}/>
            <line x1={35} y1={150-(lr/maxR)*120} x2={410} y2={150-(lr/maxR)*120} stroke={C.accent+"50"} strokeWidth={1} strokeDasharray="5,4"/>
            {adagrad.map(function(r,i){if(i+1>=N_GRAD)return null;var x1=35+i*(370/N_GRAD),y1=150-Math.min((r/maxR)*120,125),x2=35+(i+1)*(370/N_GRAD),y2=150-Math.min((adagrad[i+1]/maxR)*120,125);return<line key={"a"+i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.red} strokeWidth={2} opacity={0.85}/>;})}
            {rmsprop.map(function(r,i){if(i+1>=N_GRAD)return null;var x1=35+i*(370/N_GRAD),y1=150-Math.min((r/maxR)*120,125),x2=35+(i+1)*(370/N_GRAD),y2=150-Math.min((rmsprop[i+1]/maxR)*120,125);return<line key={"r"+i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={C.blue} strokeWidth={2} opacity={0.85}/>;})}
            <text x={50} y={14} fill={C.red} fontSize={8} fontFamily="monospace">-- AdaGrad (lr dies)</text>
            <text x={210} y={14} fill={C.blue} fontSize={8} fontFamily="monospace">-- RMSProp (recovers)</text>
            {[10,20].map(function(v,i){var xv=35+(v/N_GRAD)*370;return(
              <g key={i}><line x1={xv} y1={18} x2={xv} y2={150} stroke={C.dim} strokeWidth={1} strokeDasharray="3,3"/>
              <text x={xv} y={163} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{i===0?"sparse":"resume"}</text></g>
            );})}
          </svg>
          <div style={{fontSize:9,color:C.muted,marginTop:6,lineHeight:1.6}}>
            <span style={{color:C.red,fontWeight:700}}>AdaGrad: </span>lr collapses after large gradients and never recovers.{" "}
            <span style={{color:C.blue,fontWeight:700}}>RMSProp: </span>lr adapts back up when gradients become small again.
          </div>
        </Card>
      </div>
      <div style={{display:"flex",gap:12,maxWidth:750,margin:"0 auto 16px",flexWrap:"wrap"}}>
        {[
          {name:"AdaGrad",color:C.red,year:"2011",
           f:"G = G + g^2  (accumulate all)\nw "+LARR+" w "+DASH+" ("+ETA+"/"+SQRT+"G) * g",
           idea:"Rare features have small G "+ARR+" large step. Frequent "+ARR+" small step.",
           problem:"G only grows "+ARR+" lr "+ARR+" 0. Learning stops on long runs.",
           verdict:"Use only for sparse NLP embeddings."},
          {name:"RMSProp",color:C.blue,year:"2012",
           f:"s = "+RHO+"*s + (1"+DASH+RHO+")*g^2  (EMA)\nw "+LARR+" w "+DASH+" ("+ETA+"/"+SQRT+"s) * g",
           idea:"Exponential forgetting fixes AdaGrad. Recent magnitude only.",
           problem:"No momentum. Superseded by Adam in most tasks.",
           verdict:"Good for RNNs and non-stationary problems."},
          {name:"Adam",color:C.green,year:"2015",
           f:"m="+BETA+"1*m+(1"+DASH+BETA+"1)*g  [direction]\nv="+BETA+"2*v+(1"+DASH+BETA+"2)*g^2 [magnitude]",
           idea:"RMSProp + Momentum + bias correction. Adapts direction AND magnitude.",
           problem:"L2 reg couples with adaptive scaling. Use AdamW to fix.",
           verdict:"Default for almost everything. lr=0.001 works out of the box."},
        ].map(function(d,i){return(
          <Card key={i} style={{flex:1,minWidth:190}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
              <div style={{fontSize:13,fontWeight:800,color:d.color}}>{d.name}</div>
              <div style={{fontSize:9,color:C.dim,fontFamily:"monospace"}}>{d.year}</div>
            </div>
            <div style={{background:"#08080d",padding:"8px 10px",borderRadius:6,border:"1px solid "+C.border,marginBottom:8}}>
              {d.f.split("\n").map(function(f,fi){return<div key={fi} style={{fontSize:9,color:d.color,fontFamily:"monospace",lineHeight:1.8}}>{f}</div>;})}
            </div>
            <div style={{fontSize:9,color:C.text,marginBottom:5,lineHeight:1.5}}><span style={{color:C.green,fontWeight:700}}>Idea: </span>{d.idea}</div>
            <div style={{fontSize:9,color:C.muted,marginBottom:5,lineHeight:1.5}}><span style={{color:C.red,fontWeight:700}}>Problem: </span>{d.problem}</div>
            <div style={{fontSize:9,color:C.cyan,lineHeight:1.5}}><span style={{color:C.yellow,fontWeight:700}}>Verdict: </span>{d.verdict}</div>
          </Card>
        );})}
      </div>
      <Insight>
        AdaGrad's insight was right: <span style={{color:C.orange,fontWeight:700}}>rare parameters deserve larger updates</span>. Its fatal flaw was accumulating gradients forever {DASH} the denominator grows without bound and learning stops. RMSProp fixed this with exponential forgetting. Adam combined RMSProp with momentum and bias correction into the most robust general-purpose optimiser ever designed.
      </Insight>
    </div>
  );
}

/* ---- TAB 4: LR SCHEDULES ---- */
function TabSchedules() {
  var _s=useState(2); var sel=_s[0], setSel=_s[1];
  var _T=useState(100); var T=_T[0], setT=_T[1];
  var _w=useState(10); var wu=_w[0], setWu=_w[1];
  var lrMin=0.0001, lrMax=0.01;
  var schedules=[
    {name:"Step Decay",color:C.orange,short:"Step",
     desc:"Multiply lr by "+GAMMA+" every k epochs. Simple staircase. Widely used for CNNs.",
     formula:ETA+"t = "+ETA+"0 x "+GAMMA+"^floor(t/k)   e.g. "+GAMMA+"=0.1 every 30 epochs",
     use:"Classic CNNs (VGG, ResNet). Good when training will stall periodically.",
     fn:function(t){return lrMax*Math.pow(0.1,Math.floor(t/(T/3)));}},
    {name:"Exponential",color:C.yellow,short:"Exp",
     desc:"Smooth continuous decay. lr decreases by a fixed ratio every step.",
     formula:ETA+"t = "+ETA+"0 x "+GAMMA+"^t     "+GAMMA+"=0.97 per epoch",
     use:"Simple continuous decay. Less common than cosine for modern deep learning.",
     fn:function(t){return lrMax*Math.pow(0.97,t);}},
    {name:"Cosine Annealing",color:C.blue,short:"Cosine",
     desc:"lr follows a cosine curve from max to min. Smooth, no hard drops. Can be restarted (SGDR).",
     formula:ETA+"t = "+ETA+"min + (1/2)("+ETA+"max"+DASH+ETA+"min)(1+cos("+PI+"t/T))",
     use:"Computer vision, NLP. Default in many modern training recipes.",
     fn:function(t){return lrMin+0.5*(lrMax-lrMin)*(1+Math.cos(Math.PI*t/T));}},
    {name:"Warmup + Cosine",color:C.green,short:"Warmup",
     desc:"Linear warmup from near-zero, then cosine decay. Standard recipe for transformers.",
     formula:"0"+ARR+"wu: linear ramp    wu"+ARR+"T: cosine decay",
     use:"Transformers (BERT, GPT, ViT). Any large model with Adam.",
     fn:function(t){if(t<wu)return lrMax*(t/wu);return lrMin+0.5*(lrMax-lrMin)*(1+Math.cos(Math.PI*(t-wu)/(T-wu)));}},
    {name:"1-Cycle",color:C.pink,short:"1-Cycle",
     desc:"lr rises then falls in one big cycle. Momentum goes in reverse. Can achieve super-convergence.",
     formula:"Phase 1: lr rises base"+ARR+"max    Phase 2: lr falls max"+ARR+"min/10",
     use:"fastai, super-convergence. When training speed matters most.",
     fn:function(t){var m=T/2;if(t<m)return lrMin+(lrMax-lrMin)*(t/m);if(t<T*0.9)return lrMax+(lrMin-lrMax)*((t-m)/(T*0.4));return lrMin*0.1;}},
  ];
  var sc=schedules[sel]; var N_PTS=100;
  var lrV=[]; for(var i=0;i<N_PTS;i++) lrV.push(sc.fn(i*T/N_PTS));
  var maxLR=Math.max.apply(null,lrV),minLR=Math.min.apply(null,lrV),range=maxLR-minLR+1e-10;
  return (
    <div>
      <SectionTitle title="Learning Rate Schedules"
        subtitle={"Large lr to explore, small lr to converge "+DASH+" schedules automate this transition"}/>
      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
        {schedules.map(function(s,i){var on=sel===i;return(
          <button key={i} onClick={function(){setSel(i);}} style={{
            padding:"8px 14px",borderRadius:8,fontSize:10,fontWeight:700,fontFamily:"monospace",cursor:"pointer",
            border:"1.5px solid "+(on?s.color:C.border),background:on?s.color+"20":C.card,color:on?s.color:C.muted,
          }}>{s.short}</button>
        );})}
      </div>
      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:sc.color}}>
        <div style={{fontSize:16,fontWeight:800,color:sc.color,marginBottom:6}}>{sc.name}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7,marginBottom:8}}>{sc.desc}</div>
        <div style={{background:"#08080d",padding:"8px 12px",borderRadius:6,border:"1px solid "+C.border,marginBottom:8}}>
          <div style={{fontSize:10,color:sc.color,fontFamily:"monospace",lineHeight:1.8}}>{sc.formula}</div>
        </div>
        <div style={{fontSize:10,color:C.cyan}}><span style={{color:C.muted}}>Use for: </span>{sc.use}</div>
      </Card>
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12,flexWrap:"wrap",gap:8}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text}}>Live Learning Rate Curve</div>
          <div style={{display:"flex",gap:16,flexWrap:"wrap",alignItems:"center"}}>
            <div style={{fontSize:9,color:C.muted}}>Steps: <span style={{color:sc.color}}>{T}</span>
              <input type="range" min={20} max={200} step={10} value={T} onChange={function(e){setT(parseInt(e.target.value));}} style={{marginLeft:8,width:80,accentColor:sc.color}}/>
            </div>
            {sel===3&&<div style={{fontSize:9,color:C.muted}}>Warmup: <span style={{color:C.green}}>{wu}</span>
              <input type="range" min={2} max={30} step={1} value={wu} onChange={function(e){setWu(parseInt(e.target.value));}} style={{marginLeft:8,width:60,accentColor:C.green}}/>
            </div>}
          </div>
        </div>
        <svg width="100%" viewBox="0 0 680 185" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <line x1={45} y1={165} x2={665} y2={165} stroke={C.dim} strokeWidth={1}/>
          <line x1={45} y1={20}  x2={45}  y2={165} stroke={C.dim} strokeWidth={1}/>
          <text x={35} y={25} fill={C.muted} fontSize={7} fontFamily="monospace" textAnchor="end">{lrMax.toFixed(4)}</text>
          <text x={35} y={165} fill={C.muted} fontSize={7} fontFamily="monospace" textAnchor="end">{lrMin.toFixed(4)}</text>
          <text x={355} y={180} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{"Training Steps (0 "+ARR+" "+T+")"}</text>
          {lrV.map(function(v,i){if(i+1>=N_PTS)return null;var x1=45+i*(615/N_PTS),y1=165-((v-minLR)/range)*135,x2=45+(i+1)*(615/N_PTS),y2=165-((lrV[i+1]-minLR)/range)*135;return<line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={sc.color} strokeWidth={2.5}/>;})}
          {sel===3&&wu>0&&(function(){var wx=45+(wu/N_PTS)*615;return(<g><line x1={wx} y1={18} x2={wx} y2={165} stroke={C.green+"70"} strokeWidth={1.5} strokeDasharray="4,3"/><text x={wx+4} y={32} fill={C.green} fontSize={8} fontFamily="monospace">warmup end</text></g>);})()}
          {sel===0&&[T/3,2*T/3].map(function(v,i){var wx=45+(v/T)*615;return(<g key={i}><line x1={wx} y1={18} x2={wx} y2={165} stroke={C.orange+"60"} strokeWidth={1} strokeDasharray="4,3"/><text x={wx+3} y={32} fill={C.orange} fontSize={8} fontFamily="monospace">{TIMES+"0.1"}</text></g>);})}
        </svg>
      </Card>
      <Insight icon={TARG} title="The Transformer Recipe">
        <span style={{color:C.green,fontWeight:700}}>Warmup + Cosine decay</span> is the standard schedule for all transformer training. Why warmup? At step 0, weights are random and gradients are unreliable {DASH} a large initial lr makes catastrophic updates. Warmup lets the optimiser accumulate reliable gradient statistics first, then trains fast, then converges precisely.
      </Insight>
    </div>
  );
}

/* ---- TAB 5: GRADIENT CLIPPING ---- */
function TabClipping() {
  var _c=useState(1.0); var clipVal=_c[0], setClipVal=_c[1];
  var _m=useState("norm"); var method=_m[0], setMethod=_m[1];
  var clipGrads=[0.3,0.4,0.35,0.5,8.5,0.4,0.3,0.45,0.38,7.2,0.42,0.36,0.4,0.33,5.8,0.45,0.38];
  var maxG=Math.max.apply(null,clipGrads);
  var clipped=clipGrads.map(function(g){
    if(method==="value") return Math.min(g,clipVal);
    if(method==="norm"){var n=Math.sqrt(clipGrads.reduce(function(a,b){return a+b*b;},0));return n>clipVal?g*(clipVal/n):g;}
    return g;
  });
  return (
    <div>
      <SectionTitle title="Gradient Clipping"
        subtitle={"Preventing catastrophic updates at loss cliffs "+DASH+" essential for RNNs and transformers"}/>
      <div style={{display:"flex",gap:12,maxWidth:750,margin:"0 auto 16px",flexWrap:"wrap"}}>
        <Card style={{flex:1,minWidth:220}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Method</div>
          {[
            {k:"none",label:"No Clipping",color:C.red,desc:"Gradient unchanged. Explosive updates can crash training."},
            {k:"value",label:"Clip by Value",color:C.orange,desc:"Each element clipped to [-c,c]. Changes gradient direction."},
            {k:"norm",label:"Clip by Norm",color:C.green,desc:"Scale entire vector if ||g|| > c. Preserves direction. "+CHK},
          ].map(function(m){var on=method===m.k;return(
            <div key={m.k} onClick={function(){setMethod(m.k);}} style={{
              padding:"10px",borderRadius:8,marginBottom:6,cursor:"pointer",
              background:on?m.color+"15":C.card,border:"1px solid "+(on?m.color:C.border+"40"),transition:"all 0.2s",
            }}>
              <div style={{fontSize:10,fontWeight:700,color:on?m.color:C.muted,marginBottom:4}}>{m.label}</div>
              <div style={{fontSize:9,color:C.dim,lineHeight:1.5}}>{m.desc}</div>
            </div>
          );})}
          <div style={{marginTop:10}}>
            <div style={{fontSize:10,color:C.muted,marginBottom:4}}>Threshold c = <span style={{color:C.accent}}>{clipVal.toFixed(1)}</span></div>
            <input type="range" min={0.5} max={10} step={0.5} value={clipVal} onChange={function(e){setClipVal(parseFloat(e.target.value));}} style={{width:"100%",accentColor:C.accent}}/>
          </div>
        </Card>
        <Card style={{flex:2,minWidth:300}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Gradient Magnitude per Step</div>
          <svg width="100%" viewBox="0 0 420 185" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
            <line x1={30} y1={155} x2={410} y2={155} stroke={C.dim} strokeWidth={1}/>
            <line x1={30} y1={18}  x2={30}  y2={155} stroke={C.dim} strokeWidth={1}/>
            {method!=="none"&&<line x1={30} y1={155-(clipVal/maxG)*130} x2={410} y2={155-(clipVal/maxG)*130} stroke={C.yellow} strokeWidth={1.5} strokeDasharray="6,4"/>}
            {method!=="none"&&<text x={412} y={159-(clipVal/maxG)*130} fill={C.yellow} fontSize={8} fontFamily="monospace">{"c="+clipVal}</text>}
            {clipGrads.map(function(g,i){
              var xc=42+i*(355/clipGrads.length),rawH=(g/maxG)*128,clipH=(clipped[i]/maxG)*128,isSpike=g>clipVal*1.5;
              return(<g key={i}>
                <rect x={xc-7} y={155-rawH} width={14} height={rawH} rx={2} fill={isSpike?C.red+"25":C.muted+"15"} stroke={isSpike?C.red+"70":C.dim+"40"}/>
                {method!=="none"&&<rect x={xc-7} y={155-clipH} width={14} height={clipH} rx={2} fill={C.green+"40"} stroke={C.green+"80"}/>}
                {isSpike&&method==="none"&&<text x={xc} y={155-rawH-4} textAnchor="middle" fill={C.red} fontSize={10} fontFamily="monospace">!</text>}
              </g>);
            })}
            <text x={32} y={12} fill={C.muted} fontSize={8} fontFamily="monospace">{NORM+"g"+NORM}</text>
            <text x={220} y={172} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">Step</text>
          </svg>
          <div style={{fontSize:10,color:C.muted,marginTop:8}}>
            Spikes at steps 4, 9, 14 reach {maxG.toFixed(1)}.{" "}
            {method==="none"&&<span style={{color:C.red}}>NOT clipped {DASH} training may crash.</span>}
            {method==="value"&&<span style={{color:C.orange}}>Clipped to {clipVal} per element. Direction changes.</span>}
            {method==="norm"&&<span style={{color:C.green}}>Scaled to norm={clipVal}. Direction preserved. {CHK}</span>}
          </div>
        </Card>
      </div>
      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Clip by Value vs Clip by Norm</div>
        <svg width="100%" viewBox="0 0 680 165" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <text x={170} y={16} textAnchor="middle" fill={C.orange} fontSize={10} fontWeight={700} fontFamily="monospace">Clip by Value</text>
          <rect x={20} y={22} width={300} height={135} rx={6} fill={C.orange+"05"} stroke={C.orange+"30"}/>
          <line x1={50} y1={90} x2={320} y2={90} stroke={C.dim} strokeWidth={1}/>
          <line x1={170} y1={30} x2={170} y2={150} stroke={C.dim} strokeWidth={1}/>
          <polyline points="30,148 85,148 85,90 255,90 255,32 310,32" fill="none" stroke={C.orange} strokeWidth={2}/>
          <polyline points="30,148 310,32" fill="none" stroke={C.dim} strokeWidth={1} strokeDasharray="4,3"/>
          <text x={85} y={158} textAnchor="middle" fill={C.orange} fontSize={8} fontFamily="monospace">-c</text>
          <text x={255} y={158} textAnchor="middle" fill={C.orange} fontSize={8} fontFamily="monospace">+c</text>
          <text x={170} y={112} textAnchor="middle" fill={C.orange} fontSize={8} fontFamily="monospace">flat zone</text>
          <text x={90} y={78} fill={C.red} fontSize={9} fontFamily="monospace">{WARN+" direction changes"}</text>

          <text x={510} y={16} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">Clip by Norm</text>
          <rect x={360} y={22} width={300} height={135} rx={6} fill={C.green+"05"} stroke={C.green+"30"}/>
          <line x1={390} y1={90} x2={660} y2={90} stroke={C.dim} strokeWidth={1}/>
          <line x1={510} y1={30} x2={510} y2={150} stroke={C.dim} strokeWidth={1}/>
          <circle cx={510} cy={90} r={46} fill="none" stroke={C.green+"30"} strokeWidth={1} strokeDasharray="3,3"/>
          <line x1={440} y1={90} x2={510} y2={38} stroke={C.cyan} strokeWidth={2}/>
          <polygon points="510,38 503,50 517,50" fill={C.cyan}/>
          <line x1={440} y1={90} x2={510} y2={58} stroke={C.green} strokeWidth={2}/>
          <polygon points="510,58 503,66 517,66" fill={C.green}/>
          <text x={518} y={36} fill={C.cyan} fontSize={8} fontFamily="monospace">original g</text>
          <text x={518} y={56} fill={C.green} fontSize={8} fontFamily="monospace">clipped g</text>
          <text x={510} y={148} textAnchor="middle" fill={C.green} fontSize={8} fontFamily="monospace">same direction, smaller magnitude</text>
          <text x={380} y={135} fill={C.green} fontSize={9} fontFamily="monospace">{CHK+" direction preserved"}</text>
        </svg>
      </Card>
      <Insight icon={TARG} title="When to Use Gradient Clipping">
        Always prefer <span style={{color:C.green,fontWeight:700}}>clip-by-norm</span> (not clip-by-value) with threshold 1.0{DASH}5.0. It is essentially free and prevents catastrophic updates from rare loss cliffs. <span style={{color:C.cyan,fontWeight:700}}>RNNs always need it</span> (exploding gradients through time). Transformers use clip_norm=1.0 by default. For CNNs with BatchNorm it is less critical but still good practice.
      </Insight>
    </div>
  );
}

/* ---- ROOT ---- */
function App() {
  var _t=useState(0); var tab=_t[0], setTab=_t[1];
  var tabs=["Loss Landscape","Optimisers","Adaptive LR","LR Schedules","Gradient Clipping"];
  return (
    <div style={{background:C.bg,minHeight:"100vh",padding:"24px 16px",
      fontFamily:"'JetBrains Mono','SF Mono',monospace",color:C.text,maxWidth:960,margin:"0 auto"}}>
      <div style={{textAlign:"center",marginBottom:16}}>
        <div style={{fontSize:22,fontWeight:800,
          background:"linear-gradient(135deg,"+C.accent+","+C.yellow+")",
          WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",display:"inline-block"}}>
          Optimisers & Learning Rate Strategies
        </div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>
          {"From SGD to Adam "+DASH+" making gradient descent work in practice"}
        </div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab}/>
      {tab===0&&<TabLossLandscape/>}
      {tab===1&&<TabOptimisers/>}
      {tab===2&&<TabAdaptive/>}
      {tab===3&&<TabSchedules/>}
      {tab===4&&<TabClipping/>}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
</script>
</body>
</html>"""

OPTIMISERS_VISUAL_HEIGHT = 1150