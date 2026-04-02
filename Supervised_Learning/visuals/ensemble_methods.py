"""
Self-contained HTML visual for Ensemble Methods.
5 interactive tabs: Overview & Strategies, Bias-Variance Space, Bagging & Random Forest,
Boosting (AdaBoost + Gradient Boosting), Stacking Pipeline.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(ENSEMBLE_VISUAL_HTML, height=ENSEMBLE_VISUAL_HEIGHT, scrolling=True)
"""

ENSEMBLE_VISUAL_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#4ecdc4;cursor:pointer;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .3s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 16px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:10px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#4ecdc4;color:#4ecdc4;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;margin-bottom:4px;color:#e4e4e7;}
.section-title p{font-size:11px;color:#71717a;}
.insight{max-width:750px;margin:16px auto 0;padding:16px 22px;background:rgba(78,205,196,.06);border-radius:10px;border:1px solid rgba(78,205,196,.2);}
.ins-title{font-size:11px;font-weight:700;color:#4ecdc4;margin-bottom:6px;}
.ins-body{font-size:11px;color:#71717a;line-height:1.8;}
</style>
</head>
<body>
<div id="app" style="max-width:960px;margin:0 auto;padding:24px 16px;"></div>
<script>
/* ─── PALETTE ─── */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",
  accent:"#4ecdc4",blue:"#38bdf8",purple:"#a78bfa",
  yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",
  dim:"#3f3f46",red:"#ef4444",green:"#4ade80",
  orange:"#fb923c"};

/* ─── HELPERS ─── */
function hex(c,a){
  var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
  return 'rgba('+r+','+g+','+b+','+a+')';
}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function card(inner,extra){
  return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';
}
function sectionTitle(t,s){
  return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';
}
function insight(icon,title,body){
  return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';
}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx
    +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
}
function sliderRow(action,val,min,max,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:80px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||520)+' '+(h||340)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=520,VH=340,PL=52,PR=18,PT=18,PB=44;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
  var xm=xmax||10, ym=ymax||10;
  var xts=xticks||[0,2,4,6,8,10], yts=yticks||[0,2,4,6,8,10];
  var o='';
  xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
  yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="12" font-family="monospace">'+v+'</text>';});
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}

/* ─── STATE ─── */
var S={
  tab:0,
  nTrees:1,        /* bagging: 1..20 */
  boostStep:0,     /* adaboost round 0..4 */
  gbStep:0,        /* gradient boosting step 0..3 */
  boostMode:0,     /* 0=adaboost, 1=gradient boosting */
  stackStep:0      /* stacking pipeline step 0..5 */
};

/* ══════════════════════════════════════════════════════
   TAB 0 — OVERVIEW & STRATEGIES
══════════════════════════════════════════════════════ */
function renderOverview(){
  var out=sectionTitle('Ensemble Methods','Many weak learners → one strong predictor. Diversity is the engine.');

  /* Three strategy cards */
  out+='<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  var strategies=[
    {
      name:'Bagging',icon:'🎲',color:C.blue,
      subtitle:'Parallel + Average',
      formula:'F(x) = (1/M) Σ hₘ(x)',
      goal:'↓ Variance',goalColor:C.blue,
      points:['Bootstrap-sample training data','Train M models independently','Average (regression) or vote (classification)','No sequential dependency','Example: Random Forest'],
      bias:'Low (deep trees)','variance':'High → Low after averaging',
      example:'Random Forest'
    },
    {
      name:'Boosting',icon:'🚀',color:C.orange,
      subtitle:'Sequential + Weighted',
      formula:'F(x) = Σ αₘ hₘ(x)',
      goal:'↓ Bias',goalColor:C.orange,
      points:['Train models one at a time','Each corrects previous errors','Weight learners by accuracy','Focuses on hard examples','Example: XGBoost, LightGBM'],
      bias:'High → Low',variance:'Controlled via lr',
      example:'XGBoost / LightGBM'
    },
    {
      name:'Stacking',icon:'🏗️',color:C.purple,
      subtitle:'Learn to Combine',
      formula:'F(x) = meta(h₁(x), h₂(x)…)',
      goal:'Optimal blend',goalColor:C.purple,
      points:['Train diverse base learners','Generate OOF predictions','Meta-learner learns to combine','Captures complementary strengths','Most complex, marginal gains'],
      bias:'Depends on meta-model',variance:'Depends on meta-model',
      example:'Blending in Kaggle'
    }
  ];

  strategies.forEach(function(s){
    out+='<div style="flex:1 1 220px;">';
    out+='<div class="card" style="margin:0;border-color:'+hex(s.color,.3)+';">';
    out+='<div style="font-size:20px;margin-bottom:6px;">'+s.icon+'</div>';
    out+='<div style="font-size:13px;font-weight:800;color:'+s.color+';margin-bottom:2px;">'+s.name+'</div>';
    out+='<div style="font-size:9px;color:'+C.muted+';margin-bottom:10px;">'+s.subtitle+'</div>';
    out+='<div style="background:#08080d;border-radius:6px;padding:8px 10px;margin-bottom:10px;border:1px solid '+C.border+';">';
    out+='<div style="font-size:9px;font-family:monospace;color:'+s.color+';">'+s.formula+'</div>';
    out+='</div>';
    out+='<div style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:9px;font-weight:700;background:'+hex(s.goalColor,.12)+';color:'+s.goalColor+';border:1px solid '+hex(s.goalColor,.3)+';margin-bottom:10px;">Goal: '+s.goal+'</div>';
    s.points.forEach(function(p,i){
      out+='<div style="display:flex;gap:6px;font-size:9px;color:'+(i===s.points.length-1?s.color:C.muted)+';padding:2px 0;">'
        +'<span style="color:'+s.color+';flex-shrink:0;">'+(i===s.points.length-1?'★':'·')+'</span>'
        +p+'</div>';
    });
    out+='</div></div>';
  });
  out+='</div>';

  /* ERM comparison table */
  out+=card(
    div('font-size:10px;color:'+C.muted+';margin-bottom:12px;','ENSEMBLE METHODS AS ERM — LOSS FUNCTION COMPARISON')
    +'<div style="overflow-x:auto;">'
    +'<table style="width:100%;border-collapse:collapse;font-size:9px;">'
    +'<thead><tr>'
    +['Method','Hypothesis Class','Loss Function','Optimisation'].map(function(h){
      return '<th style="text-align:left;padding:6px 8px;border-bottom:1px solid '+C.border+';color:'+C.muted+';font-weight:700;">'+h+'</th>';
    }).join('')
    +'</tr></thead><tbody>'
    +[
      ['Random Forest','(1/M) Σ hₘ(x)',     'None at ensemble level', 'Average of independent trees',    C.blue],
      ['AdaBoost',     'Σ αₘ hₘ(x)',         'Exponential  exp(−y·F(x))', 'Greedy stage-wise',           C.orange],
      ['Grad. Boosting','F₀ + Σ γₘ hₘ(x)',  'Any differentiable L(y, F)', 'Gradient descent in func. space', C.green],
      ['Stacking',     'g(h₁(x), h₂…hₖ(x))','Meta-learner loss',          'Two-level training',          C.purple],
    ].map(function(r){
      return '<tr style="border-bottom:1px solid '+C.border+';">'
        +'<td style="padding:6px 8px;font-weight:700;color:'+r[4]+';">'+r[0]+'</td>'
        +'<td style="padding:6px 8px;font-family:monospace;color:'+C.accent+';font-size:8.5px;">'+r[1]+'</td>'
        +'<td style="padding:6px 8px;color:'+C.muted+';font-size:8.5px;">'+r[2]+'</td>'
        +'<td style="padding:6px 8px;color:'+C.dim+';font-size:8.5px;">'+r[3]+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table></div>'
  );

  /* Inductive biases */
  out+='<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  [
    {icon:'🌐',title:'Diversity',body:'Independent errors cancel. Models wrong in different ways average toward correct.',c:C.blue},
    {icon:'👥',title:'Wisdom of Crowds',body:'Aggregated mediocre predictors beat a single expert — if they disagree on errors.',c:C.green},
    {icon:'🎯',title:'Sequential Correction',body:'Each new model concentrates on examples that previous models failed on.',c:C.orange},
    {icon:'📉',title:'Smoothness',body:'Averaging over many jagged, high-variance trees smooths the decision boundary.',c:C.purple},
  ].forEach(function(b){
    out+='<div style="flex:1 1 160px;"><div class="card" style="margin:0;padding:14px 16px;">'
      +'<div style="font-size:18px;margin-bottom:6px;">'+b.icon+'</div>'
      +'<div style="font-size:10px;font-weight:700;color:'+b.c+';margin-bottom:4px;">'+b.title+'</div>'
      +'<div style="font-size:9px;color:'+C.muted+';line-height:1.6;">'+b.body+'</div>'
      +'</div></div>';
  });
  out+='</div>';

  out+=insight('&#9889;','The Fundamental Insight',
    'Ensemble methods work because of one key property: <span style="color:'+C.accent+';font-weight:700;">diversity</span>. '
    +'If all models make the same mistakes, averaging them does nothing. '
    +'But if they fail independently, averaging reduces error by a factor of M. '
    +'This is why Random Forest adds feature randomness (↓ correlation ρ) '
    +'and Boosting forces each model to correct the previous one\'s blind spots.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 1 — BIAS–VARIANCE SPACE
══════════════════════════════════════════════════════ */
function renderBiasVariance(){
  /* Bias-variance curves: x = complexity (0..10), Bias↓, Var↑, Total = U-shape */
  var N=60;
  var biasData=[], varData=[], totalData=[];
  for(var i=0;i<=N;i++){
    var x=i/N*10;
    var b=Math.pow(Math.max(0,4.5-x*0.9),1.5)*0.18;
    var v=Math.pow(x*0.28,1.6)*0.06;
    var noise=0.3;
    biasData.push({x:x, y:Math.min(9.5,b)});
    varData.push({x:x,  y:Math.min(9.5,v)});
    totalData.push({x:x, y:Math.min(9.5,b+v+noise)});
  }

  function polyLine(data,xmax,ymax,col,w){
    var pts=data.map(function(d){return sx(d.x,xmax).toFixed(1)+','+sy(d.y,ymax).toFixed(1);}).join(' ');
    return '<polyline points="'+pts+'" fill="none" stroke="'+col+'" stroke-width="'+(w||2)+'" stroke-linejoin="round"/>';
  }

  var sv=plotAxes('Model Complexity →','Error',10,10,[0,2,4,6,8,10],[0,2,4,6,8,10]);
  var xmax=10,ymax=10;

  /* curves */
  sv+=polyLine(biasData,xmax,ymax,C.red,2.5);
  sv+=polyLine(varData,xmax,ymax,C.blue,2.5);
  sv+=polyLine(totalData,xmax,ymax,C.yellow,2);

  /* labels */
  sv+='<text x="'+sx(1.5,xmax)+'" y="'+sy(5.5,ymax)+'" fill="'+C.red+'" font-size="10" font-family="monospace">Bias²</text>';
  sv+='<text x="'+sx(6.5,xmax)+'" y="'+sy(5.2,ymax)+'" fill="'+C.blue+'" font-size="10" font-family="monospace">Variance</text>';
  sv+='<text x="'+sx(3.5,xmax)+'" y="'+sy(2.5,ymax)+'" fill="'+C.yellow+'" font-size="10" font-family="monospace">Total Error</text>';
  sv+='<text x="'+sx(4.8,xmax)+'" y="'+sy(9.5,ymax)+'" fill="'+C.accent+'" font-size="9" font-family="monospace">← Sweet Spot →</text>';

  /* vertical bands for models */
  var models=[
    {x:1.2, label:'Linear', color:C.muted},
    {x:3.5, label:'Shallow Tree', color:C.muted},
    {x:6.5, label:'Deep Tree', color:C.muted},
    {x:9.2, label:'Full Tree', color:C.muted},
  ];
  models.forEach(function(m){
    var lx=sx(m.x,xmax);
    sv+='<line x1="'+lx.toFixed(1)+'" y1="'+PT+'" x2="'+lx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+m.color+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.4"/>';
    sv+='<text x="'+lx.toFixed(1)+'" y="'+(PT+PH+26)+'" text-anchor="middle" fill="'+m.color+'" font-size="8.5" font-family="monospace" opacity="0.7">'+m.label+'</text>';
  });

  /* Bagging annotation: high complexity → reduce variance */
  var bx=sx(9,xmax), by=sy(8.2,ymax);
  sv+='<rect x="'+(bx-55)+'" y="'+(by-14)+'" width="112" height="32" rx="4" fill="#0a0a0f" opacity="0.9" stroke="'+hex(C.blue,.5)+'" stroke-width="1"/>';
  sv+='<text x="'+(bx)+'" y="'+(by-2)+'" text-anchor="middle" fill="'+C.blue+'" font-size="9" font-family="monospace" font-weight="700">Bagging</text>';
  sv+='<text x="'+(bx)+'" y="'+(by+11)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">deep trees → ↓ variance</text>';
  sv+='<line x1="'+(bx-20)+'" y1="'+by+'" x2="'+sx(7.5,xmax).toFixed(1)+'" y2="'+sy(7.5,ymax).toFixed(1)+'" stroke="'+C.blue+'" stroke-width="1" marker-end="url(#arr)" opacity="0.6" stroke-dasharray="3,2"/>';

  /* Boosting annotation: low complexity → reduce bias */
  var bx2=sx(1.5,xmax), by2=sy(2.2,ymax);
  sv+='<rect x="'+(bx2-55)+'" y="'+(by2-14)+'" width="120" height="32" rx="4" fill="#0a0a0f" opacity="0.9" stroke="'+hex(C.orange,.5)+'" stroke-width="1"/>';
  sv+='<text x="'+(bx2)+'" y="'+(by2-2)+'" text-anchor="middle" fill="'+C.orange+'" font-size="9" font-family="monospace" font-weight="700">Boosting</text>';
  sv+='<text x="'+(bx2)+'" y="'+(by2+11)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">shallow trees → ↓ bias</text>';
  sv+='<line x1="'+(bx2+30)+'" y1="'+by2+'" x2="'+sx(3.2,xmax).toFixed(1)+'" y2="'+sy(3.5,ymax).toFixed(1)+'" stroke="'+C.orange+'" stroke-width="1" opacity="0.6" stroke-dasharray="3,2"/>';

  var out=sectionTitle('Bias–Variance Decomposition','Where ensemble methods live — and why they work');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Error = Bias² + Variance + Noise')
    +svgBox(sv)
    +'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;">'
    +[{c:C.red,l:'Bias²'},{c:C.blue,l:'Variance'},{c:C.yellow,l:'Total Error'},{c:C.blue,l:'Bagging target'},{c:C.orange,l:'Boosting target'}].map(function(i){
      return '<div style="display:flex;align-items:center;gap:5px;font-size:9px;color:'+C.muted+';">'
        +'<div style="width:20px;height:3px;background:'+i.c+';border-radius:2px;"></div>'+i.l+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  /* Variance reduction formula */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','WHY BAGGING REDUCES VARIANCE');
  out+='<div style="background:#08080d;border-radius:6px;padding:10px 12px;margin-bottom:8px;font-family:monospace;font-size:9px;line-height:2;">'
    +'<div style="color:'+C.muted+';">M models, correlation ρ, variance σ²:</div>'
    +'<div style="color:'+C.blue+';font-size:10px;">Var[avg] = ρσ² + (1−ρ)σ²/M</div>'
    +'<div style="color:'+C.muted+';margin-top:4px;">M→∞: Var → ρσ²  (irreducible floor)</div>'
    +'<div style="color:'+C.green+';margin-top:4px;">RF lowers ρ via feature randomness ✓</div>'
    +'</div>';
  out+=statRow('Uncorrelated (ρ=0)','σ²/M → 0',C.green);
  out+=statRow('Perfectly correlated (ρ=1)','σ² (no gain!)',C.red);
  out+=statRow('Random Forest (ρ≈0.05)','~0.05σ² + σ²/M',C.accent);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','PARADIGM COMPARISON');
  [
    {name:'Bagging',starts:'High-var, low-bias',reduces:'Variance',base:'Deep trees',parallel:true,c:C.blue},
    {name:'Boosting',starts:'High-bias, low-var',reduces:'Bias',base:'Shallow trees',parallel:false,c:C.orange},
  ].forEach(function(p){
    out+='<div style="padding:8px 10px;margin-bottom:8px;border-radius:6px;border-left:3px solid '+p.c+';background:'+hex(p.c,.04)+';">'
      +'<div style="font-size:10px;font-weight:700;color:'+p.c+';margin-bottom:4px;">'+p.name+'</div>'
      +statRow('Starts with',p.starts,C.muted)
      +statRow('Reduces',p.reduces,p.c)
      +statRow('Base learner',p.base,C.muted)
      +statRow('Training',p.parallel?'Parallel ⚡':'Sequential ↗',p.parallel?C.green:C.orange)
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MODEL COMPARISON TABLE');
  [
    ['Single deep tree','HIGH','Low'],
    ['Bagged trees','Low','→ Low'],
    ['Random Forest','Low','→ Very Low'],
    ['AdaBoost stumps','→ Low','Controlled'],
    ['Gradient Boosting','Very Low','Low (w/ reg.)'],
  ].map(function(r){
    return '<div style="display:flex;justify-content:space-between;font-size:9px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+C.muted+';flex:1;">'+r[0]+'</span>'
      +'<span style="color:'+C.red+';width:60px;text-align:center;">'+r[1]+'</span>'
      +'<span style="color:'+C.blue+';width:60px;text-align:right;">'+r[2]+'</span>'
      +'</div>';
  }).forEach(function(row){out+=row;});
  out+='<div style="display:flex;justify-content:space-between;font-size:8px;padding:4px 0;color:'+C.dim+';">'
    +'<span style="flex:1;">Model</span><span style="width:60px;text-align:center;">Bias</span><span style="width:60px;text-align:right;">Variance</span></div>';
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127919;','The Fundamental Tradeoff',
    'Bias² and Variance pull in opposite directions. Ensemble methods are a principled escape: '
    +'<span style="color:'+C.blue+';font-weight:700;">Bagging</span> keeps bias low by using expressive trees, '
    +'then <em>averages away</em> their variance. '
    +'<span style="color:'+C.orange+';font-weight:700;">Boosting</span> starts simple and <em>accumulates complexity</em>, '
    +'driving bias toward zero while a learning rate controls variance. '
    +'Neither approach can eliminate noise σ² — the irreducible floor.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 2 — BAGGING & RANDOM FOREST
══════════════════════════════════════════════════════ */
function renderBagging(){
  var M=S.nTrees;

  /* Simulate M bootstrap samples and their "accuracy" */
  /* Using deterministic pseudo-random for reproducibility */
  function lcg(seed){return (seed*1664525+1013904223)&0xffffffff;}
  var seed=42;
  var treeAccs=[];
  for(var i=0;i<20;i++){seed=lcg(seed); treeAccs.push(0.68+((seed>>>0)/4294967296)*0.24);}

  /* ensemble accuracy (improves with M) */
  var ensAcc=0;
  var ensVar=0;
  for(var t=0;t<M;t++) ensAcc+=treeAccs[t];
  ensAcc/=M;
  /* ensemble accuracy converges toward 0.92 with variance reduction */
  var ensAccFinal=0.92-(0.92-ensAcc)*Math.exp(-M/5);

  /* SVG: accuracy vs n_estimators curve */
  var sv='';
  var xmax=20, ymax=1.0;
  var pxl=55,pxr=15,pxt=15,pxb=40;
  var pw=520-pxl-pxr, ph=280-pxt-pxb;
  function bsx(x){return pxl+(x/xmax)*pw;}
  function bsy(y){return pxt+ph-(y/ymax)*ph;}

  /* grid */
  [0,5,10,15,20].forEach(function(v){
    sv+='<line x1="'+bsx(v).toFixed(1)+'" y1="'+pxt+'" x2="'+bsx(v).toFixed(1)+'" y2="'+(pxt+ph)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+bsx(v).toFixed(1)+'" y="'+(pxt+ph+14)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">'+v+'</text>';
  });
  [0.7,0.75,0.8,0.85,0.9,0.95,1.0].forEach(function(v){
    sv+='<line x1="'+pxl+'" y1="'+bsy(v).toFixed(1)+'" x2="'+(pxl+pw)+'" y2="'+bsy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+(pxl-6)+'" y="'+(bsy(v)+4).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="10" font-family="monospace">'+(v*100).toFixed(0)+'%</text>';
  });
  sv+='<line x1="'+pxl+'" y1="'+pxt+'" x2="'+pxl+'" y2="'+(pxt+ph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+pxl+'" y1="'+(pxt+ph)+'" x2="'+(pxl+pw)+'" y2="'+(pxt+ph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<text x="'+(pxl+pw/2)+'" y="'+(pxt+ph+30)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace">n_estimators (trees)</text>';
  sv+='<text x="10" y="'+(pxt+ph/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="11" font-family="monospace" transform="rotate(-90,10,'+(pxt+ph/2)+')">Accuracy</text>';

  /* single tree baseline */
  var singleAcc=0.76;
  sv+='<line x1="'+pxl+'" y1="'+bsy(singleAcc).toFixed(1)+'" x2="'+(pxl+pw)+'" y2="'+bsy(singleAcc).toFixed(1)+'" stroke="'+C.red+'" stroke-width="1.5" stroke-dasharray="6,3"/>';
  sv+='<text x="'+(pxl+pw-4)+'" y="'+(bsy(singleAcc)-5).toFixed(1)+'" text-anchor="end" fill="'+C.red+'" font-size="10" font-family="monospace">Single Tree</text>';

  /* ensemble curve */
  var pts='';
  for(var n=1;n<=20;n++){
    var acc=0.92-(0.92-0.75)*Math.exp(-n/4.5);
    pts+=bsx(n).toFixed(1)+','+bsy(acc).toFixed(1)+' ';
  }
  sv+='<polyline points="'+pts.trim()+'" fill="none" stroke="'+C.blue+'" stroke-width="2.5" stroke-linejoin="round"/>';

  /* Current position dot */
  var curAcc=0.92-(0.92-0.75)*Math.exp(-M/4.5);
  sv+='<circle cx="'+bsx(M).toFixed(1)+'" cy="'+bsy(curAcc).toFixed(1)+'" r="7" fill="'+C.accent+'" stroke="#0a0a0f" stroke-width="2"/>';
  sv+='<text x="'+(bsx(M)+12)+'" y="'+(bsy(curAcc)+4).toFixed(1)+'" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">M='+M+': '+(curAcc*100).toFixed(1)+'%</text>';

  /* individual tree dots */
  for(var t=0;t<M&&t<20;t++){
    sv+='<circle cx="'+(bsx(t+1)+0).toFixed(1)+'" cy="'+bsy(treeAccs[t]).toFixed(1)+'" r="3.5" fill="'+hex(C.blue,.5)+'" stroke="'+C.blue+'" stroke-width="1"/>';
  }

  /* OOB annotation */
  sv+='<rect x="'+bsx(10)+'" y="'+pxt+'" width="130" height="38" rx="4" fill="#0a0a0f" opacity="0.9" stroke="'+hex(C.green,.4)+'" stroke-width="1"/>';
  sv+='<text x="'+(bsx(10)+65).toFixed(1)+'" y="'+(pxt+13)+'" text-anchor="middle" fill="'+C.green+'" font-size="9" font-family="monospace" font-weight="700">OOB Score ≈ CV Score</text>';
  sv+='<text x="'+(bsx(10)+65).toFixed(1)+'" y="'+(pxt+26)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">36.8% samples unused per tree</text>';

  var out=sectionTitle('Bagging & Random Forests','Bootstrap Aggregating — parallel training → variance reduction');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Ensemble Accuracy vs Number of Trees')
    +svgBox(sv,520,280)
    +sliderRow('nTrees',M,1,20,1,'n_estimators',0)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.yellow+';">← 1 tree (high variance)</span>'
    +'<span style="color:'+C.green+';">20 trees (stable) →</span></div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','CURRENT ENSEMBLE (M='+M+')');
  out+=statRow('Test accuracy',(curAcc*100).toFixed(1)+'%',C.accent);
  out+=statRow('Single tree baseline','76.0%',C.red);
  out+=statRow('Gain vs single',('+'+((curAcc-0.76)*100).toFixed(1)+'%'),C.green);
  out+=statRow('Variance reduction','1/M ≈ 1/'+M,C.blue);
  out+=statRow('OOB samples per tree','~36.8%',C.yellow);
  out+=statRow('Trees with OOB on x₁','~'+(M*(0.368)).toFixed(1)+' / '+M,C.muted);
  out+='</div>';

  /* Bootstrap process */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','BOOTSTRAP PROCESS');
  out+='<div style="font-size:9px;color:'+C.muted+';line-height:2.0;">';
  out+='<div style="color:'+C.text+';font-weight:700;margin-bottom:4px;">Training: [x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈]</div>';
  out+='<div style="color:'+C.blue+';">B₁: [x₂ x₁ x₅ x₂ x₈ x₃] → Tree₁</div>';
  out+='<div style="color:'+C.purple+';">B₂: [x₄ x₇ x₃ x₄ x₁ x₆] → Tree₂</div>';
  out+='<div style="color:'+C.green+';">B₃: [x₁ x₃ x₈ x₅ x₇ x₃] → Tree₃</div>';
  out+='<div style="color:'+C.muted+';">⋮</div>';
  out+='<div style="margin-top:6px;padding:6px 8px;background:#08080d;border-radius:4px;border:1px solid '+C.border+';">'
    +'<span style="color:'+C.accent+';">ŷ = avg(ŷ₁, ŷ₂, ŷ₃, …) </span>'
    +'<span style="color:'+C.muted+';">or majority vote</span></div>';
  out+='</div></div>';

  /* RF vs Bagging */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','RANDOM FOREST EXTRAS');
  [
    {icon:'🎲',lbl:'Feature subsampling',desc:'Only √p features per split',c:C.green},
    {icon:'↓ρ',lbl:'Lower tree correlation',desc:'Reduces ρ → lower var floor',c:C.blue},
    {icon:'🆓',lbl:'OOB validation',desc:'Free ≈ cross-validation score',c:C.accent},
    {icon:'📊',lbl:'Feature importance',desc:'MDI or permutation (MDA)',c:C.purple},
  ].forEach(function(r){
    out+='<div style="display:flex;align-items:flex-start;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+r.c+';font-weight:700;font-size:9px;width:24px;flex-shrink:0;">'+r.icon+'</span>'
      +'<div><div style="font-size:9px;color:'+C.text+';">'+r.lbl+'</div><div style="font-size:8px;color:'+C.muted+';">'+r.desc+'</div></div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  /* Feature randomness SVG */
  out+=card(
    div('font-size:10px;font-weight:700;color:'+C.text+';margin-bottom:12px;','Feature Randomness — How Random Forest Decorrelates Trees')
    +'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    +'<div style="flex:1 1 300px;">'
    +'<div style="font-size:9px;color:'+C.red+';font-weight:700;margin-bottom:6px;">WITHOUT feature randomness (Bagging):</div>'
    +'<div style="background:#08080d;border-radius:6px;padding:10px;font-size:8.5px;font-family:monospace;line-height:2;">'
    +'<div style="color:'+C.muted+';">All features: [f₁ f₂ f₃ f₄ f₅ f₆ f₇ f₈]</div>'
    +'<div style="color:'+C.red+';">Tree 1 root → always f₃ (best feature)</div>'
    +'<div style="color:'+C.red+';">Tree 2 root → always f₃ (best feature)</div>'
    +'<div style="color:'+C.red+';">Trees highly correlated  ρ ≈ 0.8</div>'
    +'<div style="color:'+C.dim+';margin-top:4px;">Var[avg] ≈ 0.8σ²  (barely improves!)</div>'
    +'</div></div>'
    +'<div style="flex:1 1 300px;">'
    +'<div style="font-size:9px;color:'+C.green+';font-weight:700;margin-bottom:6px;">WITH feature randomness (Random Forest):</div>'
    +'<div style="background:#08080d;border-radius:6px;padding:10px;font-size:8.5px;font-family:monospace;line-height:2;">'
    +'<div style="color:'+C.muted+';">All features: [f₁ f₂ f₃ f₄ f₅ f₆ f₇ f₈]</div>'
    +'<div style="color:'+C.green+';">Tree 1 node → sample [f₂ f₅ f₇] → splits f₅</div>'
    +'<div style="color:'+C.green+';">Tree 2 node → sample [f₁ f₄ f₈] → splits f₁</div>'
    +'<div style="color:'+C.green+';">Diverse trees  ρ ≈ 0.05</div>'
    +'<div style="color:'+C.accent+';margin-top:4px;">Var[avg] ≈ 0.05σ²  (major reduction!)</div>'
    +'</div></div>'
    +'</div>'
  );

  out+=insight('&#127795;','Why Random Forest Beats Simple Bagging',
    'The key innovation is feature subsampling at each split. Without it, all trees split on the same dominant feature first, '
    +'creating highly correlated trees (ρ ≈ 0.8) that cancel little variance when averaged. '
    +'By restricting each split to a random <span style="color:'+C.accent+';font-family:monospace;">√p</span> features, '
    +'trees explore different feature combinations → lower ρ → dramatically lower ensemble variance. '
    +'OOB evaluation is a free bonus: each tree\'s 36.8% unused samples give an unbiased test error estimate.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 3 — BOOSTING (AdaBoost + Gradient Boosting)
══════════════════════════════════════════════════════ */
var ADABOOST_ROUNDS=[
  {
    title:'Round 0 — Equal weights',
    desc:'All n=10 training examples start with equal weight w = 1/10. The first weak learner trains on this uniform distribution.',
    eps:null, alpha:null,
    points:[
      {x:1.5,y:7,c:1,w:1,ok:true},{x:2.5,y:8,c:1,w:1,ok:true},{x:3.5,y:7.5,c:1,w:1,ok:true},
      {x:4.5,y:8.5,c:1,w:1,ok:true},{x:5.5,y:7.2,c:1,w:1,ok:true},
      {x:1.5,y:2,c:0,w:1,ok:true},{x:2.5,y:3,c:0,w:1,ok:true},{x:3.5,y:2.5,c:0,w:1,ok:true},
      {x:4.5,y:1.8,c:0,w:1,ok:true},{x:5.5,y:3.2,c:0,w:1,ok:true},
    ],
    splitY:5.0
  },
  {
    title:'Round 1 — h₁ misclassifies 2 points',
    desc:'h₁ (horizontal split at y=5) gets ε₁=0.2. α₁ = 0.69. Weights increase on misclassified points (⊙).',
    eps:0.2, alpha:0.69,
    points:[
      {x:1.5,y:7,c:1,w:1,ok:true},{x:2.5,y:8,c:1,w:1,ok:true},{x:3.5,y:7.5,c:1,w:1,ok:true},
      {x:4.5,y:8.5,c:1,w:1,ok:true},{x:5.5,y:7.2,c:1,w:1,ok:true},
      {x:1.5,y:2,c:0,w:1,ok:true},{x:2.5,y:3,c:0,w:1,ok:true},{x:3.5,y:2.5,c:0,w:1,ok:true},
      {x:4.5,y:4.2,c:0,w:2.8,ok:false},{x:5.5,y:4.5,c:0,w:2.8,ok:false},
    ],
    splitY:5.0
  },
  {
    title:'Round 2 — h₂ focuses on upweighted points',
    desc:'h₂ trains on reweighted distribution. New mistakes happen elsewhere. α₂ = 0.55 (ε₂=0.28).',
    eps:0.28, alpha:0.55,
    points:[
      {x:1.5,y:7,c:1,w:1,ok:true},{x:2.5,y:8,c:1,w:1,ok:true},{x:3.5,y:7.5,c:1,w:1,ok:true},
      {x:4.5,y:8.5,c:1,w:2.5,ok:false},{x:5.5,y:7.2,c:1,w:2.5,ok:false},
      {x:1.5,y:2,c:0,w:1,ok:true},{x:2.5,y:3,c:0,w:1,ok:true},{x:3.5,y:2.5,c:0,w:1,ok:true},
      {x:4.5,y:4.2,c:0,w:1.5,ok:true},{x:5.5,y:4.5,c:0,w:1.5,ok:true},
    ],
    splitY:6.0
  },
  {
    title:'Round 3 — h₃ corrects latest errors',
    desc:'h₃ shifts focus to the current mistakes. Each round moves the weight distribution. ε₃=0.15, α₃=0.92.',
    eps:0.15, alpha:0.92,
    points:[
      {x:1.5,y:7,c:1,w:1,ok:true},{x:2.5,y:8,c:1,w:1,ok:true},{x:3.5,y:7.5,c:1,w:1,ok:true},
      {x:4.5,y:8.5,c:1,w:1,ok:true},{x:5.5,y:7.2,c:1,w:1,ok:true},
      {x:1.5,y:2,c:0,w:2.5,ok:false},{x:2.5,y:3,c:0,w:2.5,ok:false},{x:3.5,y:2.5,c:0,w:1,ok:true},
      {x:4.5,y:4.2,c:0,w:1,ok:true},{x:5.5,y:4.5,c:0,w:1,ok:true},
    ],
    splitY:5.5
  },
  {
    title:'Round 4 — Final weighted ensemble',
    desc:'F(x) = sign(α₁h₁ + α₂h₂ + α₃h₃ + α₄h₄). All examples correctly classified. Ensemble beats any single weak learner.',
    eps:0.05, alpha:1.47,
    points:[
      {x:1.5,y:7,c:1,w:1,ok:true},{x:2.5,y:8,c:1,w:1,ok:true},{x:3.5,y:7.5,c:1,w:1,ok:true},
      {x:4.5,y:8.5,c:1,w:1,ok:true},{x:5.5,y:7.2,c:1,w:1,ok:true},
      {x:1.5,y:2,c:0,w:1,ok:true},{x:2.5,y:3,c:0,w:1,ok:true},{x:3.5,y:2.5,c:0,w:1,ok:true},
      {x:4.5,y:4.2,c:0,w:1,ok:true},{x:5.5,y:4.5,c:0,w:1,ok:true},
    ],
    splitY:5.0
  }
];

var GB_STEPS=[
  {
    title:'Step 0 — F₀: Constant prediction',
    desc:'Gradient Boosting initialises with the best constant prediction F₀ = mean(y). Residuals rᵢ = yᵢ − F₀.',
    preds:[5,5,5,5,5,5,5,5],
    true_y:[2,4,3,7,8,6,9,5],
    stage:'init'
  },
  {
    title:'Step 1 — h₁ fits residuals',
    desc:'Compute pseudo-residuals (negative gradient of MSE). Fit tree h₁ to residuals. Update F₁ = F₀ + η·h₁.',
    preds:[3,4.5,3.5,6.5,7.5,5.5,8.0,5],
    true_y:[2,4,3,7,8,6,9,5],
    stage:'step1'
  },
  {
    title:'Step 2 — h₂ fits remaining residuals',
    desc:'New residuals are smaller. h₂ fits what h₁ got wrong. F₂ = F₁ + η·h₂. Each step shrinks errors.',
    preds:[2.2,4.1,3.1,6.9,7.9,5.9,8.8,5],
    true_y:[2,4,3,7,8,6,9,5],
    stage:'step2'
  },
  {
    title:'Step 3 — Ensemble after M iterations',
    desc:'F_M = F₀ + Σ η·hₘ. Predictions converge to truth. Learning rate η controls step size (small lr + many trees wins).',
    preds:[2.0,4.0,3.0,7.0,8.0,6.0,9.0,5.0],
    true_y:[2,4,3,7,8,6,9,5],
    stage:'final'
  }
];

function renderBoosting(){
  var mode=S.boostMode;
  var out=sectionTitle('Boosting','Sequential error correction — each model fixes the mistakes of the last');

  /* Mode selector */
  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:20px;">';
  out+=btnSel(0,mode,C.orange,'🎲 AdaBoost — Sample Weights','boostMode');
  out+=btnSel(1,mode,C.green,'📈 Gradient Boosting — Residuals','boostMode');
  out+='</div>';

  if(mode===0){
    /* ADABOOST */
    var step=S.boostStep;
    var round=ADABOOST_ROUNDS[step];

    /* SVG */
    var sv='';
    var xmax=7, ymax=10;
    var apl=45,apr=15,apt=15,apb=35;
    var apw=420-apl-apr, aph=260-apt-apb;
    function asx(x){return apl+(x/xmax)*apw;}
    function asy(y){return apt+aph-(y/ymax)*aph;}

    /* grid */
    sv+='<rect x="0" y="0" width="420" height="260" fill="#08080d"/>';
    [0,2,4,6].forEach(function(v){
      sv+='<line x1="'+asx(v).toFixed(1)+'" y1="'+apt+'" x2="'+asx(v).toFixed(1)+'" y2="'+(apt+aph)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    });
    [0,2,4,6,8,10].forEach(function(v){
      sv+='<line x1="'+apl+'" y1="'+asy(v).toFixed(1)+'" x2="'+(apl+apw)+'" y2="'+asy(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    });
    sv+='<line x1="'+apl+'" y1="'+apt+'" x2="'+apl+'" y2="'+(apt+aph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
    sv+='<line x1="'+apl+'" y1="'+(apt+aph)+'" x2="'+(apl+apw)+'" y2="'+(apt+aph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';

    /* split line */
    var sy_split=asy(round.splitY);
    sv+='<line x1="'+apl+'" y1="'+sy_split.toFixed(1)+'" x2="'+(apl+apw)+'" y2="'+sy_split.toFixed(1)+'" stroke="'+C.accent+'" stroke-width="2" stroke-dasharray="8,4"/>';
    sv+='<rect x="'+(apl+4)+'" y="'+(sy_split-14)+'" width="100" height="14" rx="3" fill="#0a0a0f" opacity="0.9"/>';
    sv+='<text x="'+(apl+8)+'" y="'+(sy_split-4)+'" fill="'+C.accent+'" font-size="9" font-family="monospace">y ≤ '+round.splitY+' ?</text>';

    /* points */
    round.points.forEach(function(p){
      var cx=asx(p.x), cy=asy(p.y);
      var col=p.c===1?C.blue:C.orange;
      var r=Math.min(14, 4+p.w*2.5);
      sv+='<circle cx="'+cx.toFixed(1)+'" cy="'+cy.toFixed(1)+'" r="'+r.toFixed(1)+'" fill="'+col+'" opacity="0.85" stroke="'+(p.ok?'#0a0a0f':C.red)+'" stroke-width="'+(p.ok?1:2.5)+'"/>';
      if(!p.ok) sv+='<text x="'+cx.toFixed(1)+'" y="'+(cy-r-3).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="12" font-weight="700">⊙</text>';
      sv+='<text x="'+cx.toFixed(1)+'" y="'+(cy+3).toFixed(1)+'" text-anchor="middle" fill="#0a0a0f" font-size="8" font-weight="700">'+p.w.toFixed(0)+'×</text>';
    });

    /* class legend */
    sv+='<circle cx="'+(apl+8)+'" cy="'+(apt+aph-12)+'" r="5" fill="'+C.blue+'"/>';
    sv+='<text x="'+(apl+16)+'" y="'+(apt+aph-8)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">class +1</text>';
    sv+='<circle cx="'+(apl+80)+'" cy="'+(apt+aph-12)+'" r="5" fill="'+C.orange+'"/>';
    sv+='<text x="'+(apl+88)+'" y="'+(apt+aph-8)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">class -1</text>';
    sv+='<text x="'+(apl+160)+'" y="'+(apt+aph-8)+'" fill="'+C.red+'" font-size="9" font-family="monospace">⊙ misclassified (upweighted)</text>';

    out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
    out+='<div style="flex:1 1 380px;">';

    /* step buttons */
    var stepBtns='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;">';
    for(var si=0;si<ADABOOST_ROUNDS.length;si++){
      stepBtns+=btnSel(si,step,C.orange,'Round '+si,'boostStep');
    }
    stepBtns+='</div>';

    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',round.title)
      +stepBtns
      +'<svg width="100%" viewBox="0 0 420 260" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv+'</svg>'
      +'<div style="font-size:9px;color:'+C.muted+';margin-top:8px;line-height:1.7;">'+round.desc+'</div>'
    );
    out+='</div>';

    out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','ADABOOST ROUND '+step+' STATS');
    if(round.eps!==null){
      out+=statRow('Weighted error ε'+step, round.eps.toFixed(2), round.eps<0.3?C.green:C.yellow);
      out+=statRow('Learner weight α'+step, round.alpha.toFixed(2), C.accent);
      out+=statRow('Misclassified', round.points.filter(function(p){return !p.ok;}).length+' / '+round.points.length, C.red);
      out+=statRow('Max sample weight', Math.max.apply(null,round.points.map(function(p){return p.w;})).toFixed(1)+'×', C.orange);
    } else {
      out+=statRow('Init weights', '1/10 = 0.1 each', C.muted);
      out+=statRow('Total weight', '1.0 (normalised)', C.accent);
      out+=statRow('Misclassified', '0 / '+round.points.length, C.green);
    }
    out+='</div>';

    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ADABOOST ALGORITHM');
    out+='<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';line-height:2;background:#08080d;padding:10px;border-radius:6px;">'
      +'<div style="color:'+C.text+';">for m = 1..M:</div>'
      +'<div style="padding-left:12px;color:'+C.blue+';">1. Fit hₘ on weighted data</div>'
      +'<div style="padding-left:12px;color:'+C.orange+';">2. εₘ = Σ wᵢ·1[hₘ≠yᵢ]</div>'
      +'<div style="padding-left:12px;color:'+C.accent+';">3. αₘ = ½ log((1−ε)/ε)</div>'
      +'<div style="padding-left:12px;color:'+C.green+';">4. wᵢ ← wᵢ·exp(−αₘyᵢhₘ)</div>'
      +'<div style="padding-left:12px;color:'+C.purple+';">5. normalise Σwᵢ = 1</div>'
      +'<div style="color:'+C.yellow+';margin-top:4px;">F(x)=sign(Σ αₘ hₘ(x))</div>'
      +'</div>';
    out+='</div>';

    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ALPHA INTERPRETATION');
    [
      {cond:'ε → 0',alpha:'α → ∞',desc:'Nearly perfect → dominates vote',c:C.green},
      {cond:'ε = 0.5',alpha:'α = 0',desc:'Random → completely ignored',c:C.yellow},
      {cond:'ε > 0.5',alpha:'α < 0',desc:'Worse than random → reversed!',c:C.red},
    ].forEach(function(r){
      out+='<div style="padding:5px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
        +'<div style="font-size:9px;font-family:monospace;color:'+r.c+';">'+r.cond+' → α: '+r.alpha+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';">'+r.desc+'</div></div>';
    });
    out+='</div>';
    out+='</div></div>';

  } else {
    /* GRADIENT BOOSTING */
    var gbstep=S.gbStep;
    var gbround=GB_STEPS[gbstep];

    /* Residuals chart */
    var trueY=gbround.true_y;
    var preds=gbround.preds;
    var N2=trueY.length;
    var svgW=520,svgH=240;
    var bpl=50,bpr=15,bpt=20,bpb=30;
    var bpw=svgW-bpl-bpr, bph=svgH-bpt-bpb;
    function gbx(i){return bpl+(i+0.5)/(N2)*bpw;}
    function gby(v){return bpt+bph-(v/10)*bph;}

    var sv2='<rect x="0" y="0" width="'+svgW+'" height="'+svgH+'" fill="#08080d"/>';
    /* grid lines */
    [0,2,4,6,8,10].forEach(function(v){
      sv2+='<line x1="'+bpl+'" y1="'+gby(v).toFixed(1)+'" x2="'+(bpl+bpw)+'" y2="'+gby(v).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
      sv2+='<text x="'+(bpl-6)+'" y="'+(gby(v)+4).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="10" font-family="monospace">'+v+'</text>';
    });
    sv2+='<line x1="'+bpl+'" y1="'+bpt+'" x2="'+bpl+'" y2="'+(bpt+bph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
    sv2+='<line x1="'+bpl+'" y1="'+(bpt+bph)+'" x2="'+(bpl+bpw)+'" y2="'+(bpt+bph)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';

    /* bars: true y */
    for(var i=0;i<N2;i++){
      var bw=bpw/N2*0.35;
      var bx_=gbx(i)-bw-1;
      var by_=gby(trueY[i]);
      var bh_=gby(0)-by_;
      sv2+='<rect x="'+bx_.toFixed(1)+'" y="'+by_.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+bh_.toFixed(1)+'" fill="'+hex(C.green,.7)+'" rx="2"/>';
      /* predicted */
      var px_=gbx(i)+1;
      var py_=gby(preds[i]);
      var ph_=gby(0)-py_;
      sv2+='<rect x="'+px_.toFixed(1)+'" y="'+py_.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+ph_.toFixed(1)+'" fill="'+hex(C.orange,.7)+'" rx="2"/>';
      /* residual line */
      var resid=trueY[i]-preds[i];
      var mid=gbx(i);
      sv2+='<line x1="'+mid.toFixed(1)+'" y1="'+gby(trueY[i]).toFixed(1)+'" x2="'+mid.toFixed(1)+'" y2="'+gby(preds[i]).toFixed(1)+'" stroke="'+C.red+'" stroke-width="2" stroke-dasharray="4,2"/>';
      sv2+='<text x="'+mid.toFixed(1)+'" y="'+(Math.min(gby(trueY[i]),gby(preds[i]))-3).toFixed(1)+'" text-anchor="middle" fill="'+C.red+'" font-size="8" font-family="monospace">'+(resid>=0?'+':'')+resid.toFixed(1)+'</text>';
    }

    /* legend */
    sv2+='<rect x="'+(bpl+4)+'" y="'+(bpt+4)+'" width="8" height="8" fill="'+hex(C.green,.7)+'" rx="1"/>';
    sv2+='<text x="'+(bpl+14)+'" y="'+(bpt+12)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">True y</text>';
    sv2+='<rect x="'+(bpl+72)+'" y="'+(bpt+4)+'" width="8" height="8" fill="'+hex(C.orange,.7)+'" rx="1"/>';
    sv2+='<text x="'+(bpl+82)+'" y="'+(bpt+12)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Predicted F_M</text>';
    sv2+='<line x1="'+(bpl+180)+'" y1="'+(bpt+8)+'" x2="'+(bpl+195)+'" y2="'+(bpt+8)+'" stroke="'+C.red+'" stroke-width="2" stroke-dasharray="4,2"/>';
    sv2+='<text x="'+(bpl+198)+'" y="'+(bpt+12)+'" fill="'+C.muted+'" font-size="9" font-family="monospace">Residual</text>';

    var gbStepBtns='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;">';
    for(var si2=0;si2<GB_STEPS.length;si2++){
      gbStepBtns+=btnSel(si2,gbstep,C.green,'Step '+si2,'gbStep');
    }
    gbStepBtns+='</div>';

    out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
    out+='<div style="flex:1 1 380px;">';
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',gbround.title)
      +gbStepBtns
      +'<svg width="100%" viewBox="0 0 '+svgW+' '+svgH+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv2+'</svg>'
      +'<div style="font-size:9px;color:'+C.muted+';margin-top:8px;line-height:1.7;">'+gbround.desc+'</div>'
    );
    out+='</div>';

    out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

    var totalResid=trueY.reduce(function(s,v,i){return s+Math.pow(v-preds[i],2);},0)/N2;
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','STEP '+gbstep+' METRICS');
    out+=statRow('MSE (current)',totalResid.toFixed(3), totalResid<0.1?C.green:totalResid<1?C.yellow:C.orange);
    out+=statRow('Max residual', Math.max.apply(null,trueY.map(function(v,i){return Math.abs(v-preds[i]);})).toFixed(2),C.red);
    out+=statRow('Avg prediction', (preds.reduce(function(a,b){return a+b;})/N2).toFixed(2), C.accent);
    out+=statRow('Trees added', gbstep, C.blue);
    out+='</div>';

    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GRADIENT BOOSTING ALGORITHM');
    out+='<div style="font-size:8.5px;font-family:monospace;color:'+C.muted+';line-height:2;background:#08080d;padding:10px;border-radius:6px;">'
      +'<div style="color:'+C.yellow+';">F₀(x) = argmin Σ L(yᵢ, γ)</div>'
      +'<div style="color:'+C.text+';">for m = 1..M:</div>'
      +'<div style="padding-left:12px;color:'+C.red+';">rᵢ = −∂L/∂F(xᵢ) | F=Fₘ₋₁</div>'
      +'<div style="padding-left:12px;color:'+C.blue+';">fit hₘ to residuals {rᵢ}</div>'
      +'<div style="padding-left:12px;color:'+C.green+';">γₘ = argmin Σ L(yᵢ, Fₘ₋₁+γhₘ)</div>'
      +'<div style="padding-left:12px;color:'+C.accent+';">Fₘ = Fₘ₋₁ + η·γₘ·hₘ</div>'
      +'<div style="color:'+C.purple+';margin-top:4px;">return F_M(x)</div>'
      +'</div>';
    out+='</div>';

    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GBM vs AdaBoost');
    [
      ['Loss function','Any differentiable L','Exponential loss only'],
      ['Flexibility','MSE, MAE, Log-loss…','Fixed to exp(−yF)'],
      ['Approach','Gradient of loss','Reweight samples'],
      ['Robustness','High (with Huber)','Low (sensitive to outliers)'],
    ].forEach(function(r){
      out+='<div style="display:flex;gap:4px;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
        +'<span style="color:'+C.muted+';width:70px;flex-shrink:0;">'+r[0]+'</span>'
        +'<span style="color:'+C.green+';flex:1;">GBM: '+r[1]+'</span>'
        +'<span style="color:'+C.orange+';flex:1;">Ada: '+r[2]+'</span>'
        +'</div>';
    });
    out+='</div>';
    out+='</div></div>';
  }

  out+=insight('&#128640;','Why Boosting Reduces Bias',
    'Each new tree in AdaBoost sees <span style="color:'+C.orange+';font-weight:700;">upweighted hard examples</span>. '
    +'Each new tree in Gradient Boosting fits the <span style="color:'+C.green+';font-weight:700;">negative gradient of the loss</span> — the direction of steepest descent in function space. '
    +'This turns ensemble learning into <em>gradient descent over functions</em>. '
    +'The learning rate η controls step size: small η + many trees always beats large η + few trees. '
    +'GBM\'s ability to optimise any differentiable loss makes it the most flexible ensemble method.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   TAB 4 — STACKING PIPELINE
══════════════════════════════════════════════════════ */
var STACK_STEPS=[
  {title:'Step 0 — Raw training data',desc:'We have X_train with labels y_train. Goal: combine diverse models optimally. Naive approach (train on same data) would be overfit.'},
  {title:'Step 1 — k-Fold CV on base learners',desc:'Split training data into k=5 folds. For each fold i, train all base learners on remaining k-1 folds, then predict on fold i.'},
  {title:'Step 2 — Out-of-Fold predictions',desc:'After CV, each base learner has produced n out-of-fold (OOF) predictions — honest predictions, since each was tested on unseen data.'},
  {title:'Step 3 — Stack OOF predictions → Z_train',desc:'Stack OOF predictions as new feature columns. Z_train shape: (n_samples × n_base_learners). These are honest, unbiased predictions.'},
  {title:'Step 4 — Train meta-learner on Z_train',desc:'Train a simple meta-learner (logistic regression, ridge) on Z_train with y_train as target. It learns "when to trust which model".'},
  {title:'Step 5 — Test-time prediction',desc:'Retrain base learners on full training data. For X_test: base learners predict → meta-learner combines → final ŷ. Typically 0.5–2% better than best single model.'},
];

function renderStacking(){
  var step=S.stackStep;
  var sr=STACK_STEPS[step];

  /* Pipeline SVG */
  var svgW=680,svgH=340;
  var sv='<rect x="0" y="0" width="'+svgW+'" height="'+svgH+'" fill="#08080d"/>';

  var models=[
    {name:'Random Forest',col:C.blue},
    {name:'XGBoost',col:C.orange},
    {name:'Logistic Reg.',col:C.purple},
    {name:'Ridge Regr.',col:C.green},
  ];

  /* X_train box */
  sv+='<rect x="20" y="130" width="100" height="80" rx="8" fill="'+hex(C.accent,.12)+'" stroke="'+C.accent+'" stroke-width="'+(step>=0?2:0.5)+'"/>';
  sv+='<text x="70" y="166" text-anchor="middle" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">X_train</text>';
  sv+='<text x="70" y="180" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">y_train</text>';
  sv+='<text x="70" y="194" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">n×p</text>';

  /* k-fold CV boxes */
  if(step>=1){
    for(var f=0;f<5;f++){
      var fy=50+f*42;
      var fc=f===2?C.yellow:C.muted;
      sv+='<rect x="140" y="'+fy+'" width="72" height="32" rx="4" fill="'+hex(fc,.08)+'" stroke="'+fc+'" stroke-width="1"/>';
      sv+='<text x="176" y="'+(fy+14)+'" text-anchor="middle" fill="'+fc+'" font-size="9" font-family="monospace">Fold '+(f+1)+' '+(f===2?'← test':'')+'</text>';
      sv+='<text x="176" y="'+(fy+26)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">'+(f===2?'predict':'train')+'</text>';
      /* arrow from X_train */
      sv+='<line x1="120" y1="170" x2="140" y2="'+(fy+16)+'" stroke="'+hex(C.accent,.3)+'" stroke-width="1" stroke-dasharray="3,2"/>';
    }
  }

  /* Base learner boxes */
  var blx=step>=1?230:145;
  models.forEach(function(m,i){
    var by=50+i*68;
    var active=step>=1;
    sv+='<rect x="'+blx+'" y="'+by+'" width="110" height="52" rx="6" fill="'+hex(m.col,(active?.12:.04))+'" stroke="'+m.col+'" stroke-width="'+(active?1.5:0.5)+'"/>';
    sv+='<text x="'+(blx+55)+'" y="'+(by+20)+'" text-anchor="middle" fill="'+m.col+'" font-size="9" font-family="monospace" font-weight="700">'+m.name+'</text>';
    sv+='<text x="'+(blx+55)+'" y="'+(by+34)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">base learner '+(i+1)+'</text>';
    sv+='<text x="'+(blx+55)+'" y="'+(by+47)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">'+(active?'OOF preds ↗':'')+'</text>';
    if(step<1 && step>=0){
      sv+='<line x1="120" y1="170" x2="'+blx+'" y2="'+(by+26)+'" stroke="'+hex(C.accent,.3)+'" stroke-width="1" stroke-dasharray="3,2"/>';
    }
  });

  /* OOF predictions → Z_train */
  if(step>=2){
    var zx=step>=3?420:360;
    sv+='<rect x="'+zx+'" y="90" width="90" height="170" rx="8" fill="'+hex(C.yellow,.08)+'" stroke="'+C.yellow+'" stroke-width="1.5"/>';
    sv+='<text x="'+(zx+45)+'" y="114" text-anchor="middle" fill="'+C.yellow+'" font-size="10" font-family="monospace" font-weight="700">Z_train</text>';
    sv+='<text x="'+(zx+45)+'" y="128" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">n × k columns</text>';
    models.forEach(function(m,i){
      var col_y=145+i*28;
      sv+='<rect x="'+(zx+10)+'" y="'+col_y+'" width="70" height="22" rx="3" fill="'+hex(m.col,.12)+'" stroke="'+m.col+'" stroke-width="1"/>';
      sv+='<text x="'+(zx+45)+'" y="'+(col_y+14)+'" text-anchor="middle" fill="'+m.col+'" font-size="8" font-family="monospace">OOF '+m.name.slice(0,3)+'</text>';
    });
    /* arrows from base learners to Z_train */
    models.forEach(function(m,i){
      sv+='<line x1="'+(blx+110)+'" y1="'+(50+i*68+26)+'" x2="'+zx+'" y2="'+(50+i*68+26)+'" stroke="'+m.col+'" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>';
    });
  }

  /* Meta-learner */
  if(step>=4){
    var mx=step>=4?530:600;
    sv+='<rect x="'+mx+'" y="120" width="110" height="110" rx="8" fill="'+hex(C.accent,.12)+'" stroke="'+C.accent+'" stroke-width="2"/>';
    sv+='<text x="'+(mx+55)+'" y="148" text-anchor="middle" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">Meta</text>';
    sv+='<text x="'+(mx+55)+'" y="162" text-anchor="middle" fill="'+C.accent+'" font-size="10" font-family="monospace" font-weight="700">Learner</text>';
    sv+='<text x="'+(mx+55)+'" y="178" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">Ridge / Logistic</text>';
    sv+='<text x="'+(mx+55)+'" y="193" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">trained on Z_train</text>';
    sv+='<text x="'+(mx+55)+'" y="208" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">learns "trust levels"</text>';
    /* arrow from Z to meta */
    sv+='<line x1="510" y1="175" x2="'+mx+'" y2="175" stroke="'+C.accent+'" stroke-width="2" marker-end="url(#arw)"/>';
    sv+='<polygon points="'+(mx-6)+',170 '+mx+',175 '+(mx-6)+',180" fill="'+C.accent+'"/>';
  }

  /* Test-time output */
  if(step>=5){
    sv+='<rect x="530" y="270" width="110" height="44" rx="6" fill="'+hex(C.green,.12)+'" stroke="'+C.green+'" stroke-width="2"/>';
    sv+='<text x="585" y="290" text-anchor="middle" fill="'+C.green+'" font-size="10" font-family="monospace" font-weight="700">Final ŷ</text>';
    sv+='<text x="585" y="306" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">test predictions</text>';
    sv+='<line x1="585" y1="230" x2="585" y2="270" stroke="'+C.green+'" stroke-width="2"/>';
    sv+='<polygon points="580,264 585,270 590,264" fill="'+C.green+'"/>';
  }

  /* step indicator */
  sv+='<rect x="10" y="305" width="660" height="26" rx="4" fill="'+hex(C.accent,.06)+'" stroke="'+hex(C.accent,.2)+'" stroke-width="1"/>';
  sv+='<text x="340" y="322" text-anchor="middle" fill="'+C.accent+'" font-size="10" font-family="monospace">'+sr.title+'</text>';

  var stepBtns='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:10px;">';
  for(var si=0;si<STACK_STEPS.length;si++){
    stepBtns+=btnSel(si,step,C.purple,'Step '+si,'stackStep');
  }
  stepBtns+='</div>';

  var out=sectionTitle('Stacking — Learning to Combine','Train a meta-learner to find the optimal blend of base models');
  out+='<div style="max-width:750px;margin:0 auto 16px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',sr.title)
    +stepBtns
    +'<svg width="100%" viewBox="0 0 680 340" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+sv+'</svg>'
    +'<div style="font-size:9px;color:'+C.muted+';margin-top:8px;line-height:1.7;">'+sr.desc+'</div>'
  );
  out+='</div>';

  out+='<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  [
    {title:'Why k-fold CV?',icon:'🔄',body:'If base learners predict on training data they\'ve seen, predictions are overfit → meta-learner trains on leakage → inflated performance. OOF predictions give honest test-like predictions for the meta-learner.',c:C.blue},
    {title:'Meta-learner choice',icon:'🧠',body:'Keep it simple: Ridge or Logistic Regression. Complex meta-learners overfit the small OOF feature matrix. The base learners do the heavy lifting; meta-learner just learns relative trust.',c:C.accent},
    {title:'When to stack',icon:'🏆',body:'Stack when: you have multiple trained models available, data is large enough for k-fold CV (n > 500), and you need the last 0.5–2% accuracy gain. Usually overkill for production systems.',c:C.purple},
  ].forEach(function(b){
    out+='<div style="flex:1 1 210px;"><div class="card" style="margin:0;padding:14px 16px;">'
      +'<div style="font-size:18px;margin-bottom:6px;">'+b.icon+'</div>'
      +'<div style="font-size:10px;font-weight:700;color:'+b.c+';margin-bottom:6px;">'+b.title+'</div>'
      +'<div style="font-size:9px;color:'+C.muted+';line-height:1.6;">'+b.body+'</div>'
      +'</div></div>';
  });
  out+='</div>';

  /* When to use what */
  out+=card(
    div('font-size:10px;color:'+C.muted+';margin-bottom:12px;','QUICK DECISION GUIDE — WHICH ENSEMBLE TO USE?')
    +'<div style="display:flex;gap:12px;flex-wrap:wrap;">'
    +[
      {title:'Random Forest',cases:['Fast to train/tune','Noisy data / outliers','Feature importance needed','Good out-of-the-box baseline','n < ~500k samples'],c:C.blue,icon:'🌲'},
      {title:'Gradient Boosting',cases:['Highest accuracy on tabular','Kaggle / production ML','Large n (LightGBM → millions)','Complex feature interactions','Willing to tune extensively'],c:C.orange,icon:'🚀'},
      {title:'Stacking',cases:['Squeezing last 0.5–2%','Multiple trained models ready','Sufficient data for k-fold CV','Competition setting','Usually overkill for prod'],c:C.purple,icon:'🏗️'},
      {title:'Single Tree',cases:['Interpretability required','Tiny dataset (n < 100)','Decision rules must be human-readable','Regulatory constraints','Explainability > accuracy'],c:C.muted,icon:'🌿'},
    ].map(function(col){
      return '<div style="flex:1 1 140px;background:#08080d;border-radius:8px;padding:12px 14px;border:1px solid '+hex(col.c,.2)+';">'
        +'<div style="font-size:14px;margin-bottom:4px;">'+col.icon+'</div>'
        +'<div style="font-size:10px;font-weight:700;color:'+col.c+';margin-bottom:8px;">'+col.title+'</div>'
        +col.cases.map(function(c){
          return '<div style="display:flex;gap:5px;font-size:8.5px;color:'+C.muted+';padding:2px 0;">'
            +'<span style="color:'+col.c+';">✓</span>'+c+'</div>';
        }).join('')
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#128202;','Stacking vs Voting',
    'Simple <strong style="color:'+C.muted+';">voting/averaging</strong> assumes all models are equally good — it treats every learner the same. '
    +'<strong style="color:'+C.purple+';">Stacking</strong> learns the optimal weights: maybe Random Forest is better for low-income customers while XGBoost excels at high-income ones. '
    +'The meta-learner can discover these conditional relationships. '
    +'In practice, stacking gains are modest (0.5–2%) but consistent across competitions.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════ */
var TABS=[
  '🌲 Overview',
  '⚖️ Bias–Variance',
  '🎲 Bagging & RF',
  '🚀 Boosting',
  '🏗️ Stacking'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.orange+','+C.purple+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Ensemble Methods</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;','Bagging · Boosting · Stacking — combining weak learners into a stronger predictor')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderOverview();
  else if(S.tab===1) html+=renderBiasVariance();
  else if(S.tab===2) html+=renderBagging();
  else if(S.tab===3) html+=renderBoosting();
  else if(S.tab===4) html+=renderStacking();
  html+='</div></div>';
  return html;
}

function render(){
  document.getElementById('app').innerHTML=renderApp();
  bindEvents();
}

function bindEvents(){
  document.querySelectorAll('[data-action]').forEach(function(el){
    var action=el.getAttribute('data-action');
    var idx=parseInt(el.getAttribute('data-idx'));
    var tag=el.tagName.toLowerCase();
    if(tag==='button'){
      el.addEventListener('click',function(){
        if(action==='tab')        {S.tab=idx; render();}
        else if(action==='boostStep'){S.boostStep=idx; render();}
        else if(action==='gbStep')  {S.gbStep=idx;    render();}
        else if(action==='boostMode'){S.boostMode=idx; S.boostStep=0; S.gbStep=0; render();}
        else if(action==='stackStep'){S.stackStep=idx; render();}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='nTrees') {S.nTrees=Math.round(val); render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

ENSEMBLE_VISUAL_HEIGHT = 1150