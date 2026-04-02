"""
Self-contained HTML visual for Neural Networks.
5 interactive tabs: The Neuron & Forward Pass, Activation Functions,
Backpropagation Step-by-Step, Loss Curves & Training Dynamics,
Regularisation & When to Use NNs.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(NN_VISUAL_HTML, height=NN_VISUAL_HEIGHT, scrolling=True)
"""

NN_VISUAL_HTML = r"""<!DOCTYPE html>
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
    +'" style="padding:8px 14px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';
}
function sliderRow(action,val,min,max,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:90px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:56px;font-weight:700;">'+dv+'</div>'
    +'</div>';
}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+(color||C.accent)+';font-weight:700;">'+val+'</span></div>';
}
function svgBox(inner,w,h){
  return '<svg width="100%" viewBox="0 0 '+(w||440)+' '+(h||280)+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';
}

/* ─── SVG PLOT SCAFFOLD ─── */
var VW=440,VH=260,PL=46,PR=16,PT=14,PB=36;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmin,xmax){return PL+(x-xmin)/(xmax-xmin)*PW;}
function sy(y,ymin,ymax){return PT+PH-(y-ymin)/(ymax-ymin)*PH;}
function plotAxes(xl,yl,xmin,xmax,ymin,ymax,xticks,yticks){
  var o='';
  (xticks||[]).forEach(function(v){
    o+='<line x1="'+sx(v,xmin,xmax).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xmin,xmax).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+sx(v,xmin,xmax).toFixed(1)+'" y="'+(PT+PH+14)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5" font-family="monospace">'+v+'</text>';
  });
  (yticks||[]).forEach(function(v){
    o+='<line x1="'+PL+'" y1="'+sy(v,ymin,ymax).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ymin,ymax).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    o+='<text x="'+(PL-5)+'" y="'+(sy(v,ymin,ymax)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8.5" font-family="monospace">'+v+'</text>';
  });
  /* zero line */
  if(ymin<0&&ymax>0){
    var zy=sy(0,ymin,ymax);
    o+='<line x1="'+PL+'" y1="'+zy.toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+zy.toFixed(1)+'" stroke="'+C.dim+'" stroke-width="1.2"/>';
  }
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}
function polyline(pts,col,w,dash){
  var d=pts.map(function(p,i){return (i===0?'M':'L')+p[0].toFixed(1)+','+p[1].toFixed(1);}).join(' ');
  return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="'+(w||2)+'"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
}

/* ─── STATE ─── */
var S={
  tab:0,
  /* Tab 0 — forward pass */
  w1:0.8, w2:-0.5, b:0.3,
  actFn:0,           /* 0=ReLU,1=sigmoid,2=tanh */
  showLinear:false,  /* show no-activation comparison */
  /* Tab 1 — activation functions */
  actTab:0,          /* which fn to highlight */
  zVal:1.5,          /* slider for single z value */
  /* Tab 2 — backprop */
  bpStep:0,          /* 0..6 step through forward+backward */
  bpLR:0.1,
  /* Tab 3 — loss curves */
  lossMode:0,        /* 0=healthy,1=overfit,2=underfit,3=earlystop */
  epoch:40,
  /* Tab 4 — regularisation */
  regMode:0          /* 0=dropout,1=L2,2=compare */
};

/* ══════════════════════════════════════════════════════════
   ACTIVATION FUNCTION MATH
══════════════════════════════════════════════════════════ */
function sigmoid(z){return 1/(1+Math.exp(-z));}
function relu(z){return Math.max(0,z);}
function leakyRelu(z){return z>=0?z:0.01*z;}
function tanhFn(z){return Math.tanh(z);}
function gelu(z){return z*sigmoid(1.702*z);}
function sigmoidGrad(z){var s=sigmoid(z);return s*(1-s);}
function reluGrad(z){return z>0?1:0;}
function leakyReluGrad(z){return z>0?1:0.01;}
function tanhGrad(z){var t=Math.tanh(z);return 1-t*t;}
function geluGrad(z){
  var s=sigmoid(1.702*z);
  return s+z*1.702*s*(1-s);
}

var ACT_FNS=[relu,sigmoid,tanhFn,leakyRelu,gelu];
var ACT_GRADS=[reluGrad,sigmoidGrad,tanhGrad,leakyReluGrad,geluGrad];
var ACT_NAMES=['ReLU','Sigmoid','Tanh','Leaky ReLU','GELU'];
var ACT_COLS=[C.green,C.orange,C.blue,C.yellow,C.purple];
var ACT_FORMULAE=[
  'max(0, z)',
  '1 / (1 + e\u207b\u1d63)',
  '(e\u1d63 \u2212 e\u207b\u1d63) / (e\u1d63 + e\u207b\u1d63)',
  'max(0.01z, z)',
  'z \u00b7 \u03c3(1.702z)'
];
var ACT_PROBLEMS=[
  'Dying ReLU: neurons stuck at 0 when z<0 for all inputs',
  'Vanishing gradient: \u03c3\'(z)\u226a0.25 at extremes \u2014 early layers barely learn',
  'Still saturates at extremes despite being zero-centred',
  'Fixes dying ReLU with small slope \u03b1=0.01 for negative z',
  'Smooth everywhere, best for Transformers (BERT, GPT)'
];
var ACT_USE=[
  'Default for feedforward hidden layers',
  'Output layer (binary classification) only',
  'LSTMs / RNNs, some output layers',
  'When dying ReLU is a problem',
  'Default in modern Transformers'
];

/* ══════════════════════════════════════════════════════════
   TAB 0 — THE NEURON & FORWARD PASS
══════════════════════════════════════════════════════════ */
function renderForwardPass(){
  var w1=S.w1, w2=S.w2, b=S.b;
  var afn=ACT_FNS[S.actFn];
  var aName=ACT_NAMES[S.actFn];

  /* fixed input values */
  var x1=1.0, x2=0.5;
  var z=w1*x1+w2*x2+b;
  var a=afn(z);

  /* Network diagram: 2 inputs → 3 hidden → 2 hidden → 1 output */
  /* We'll show only the single neuron computation in detail */
  var NW=440, NH=300;
  var sv='';

  /* ── draw a small 2→3→1 network ── */
  var layers=[
    {x:60,  nodes:[{y:90},{y:150},{y:210}]},      /* input layer: 3 nodes but show 2 active */
    {x:175, nodes:[{y:80},{y:130},{y:180},{y:230}]},
    {x:290, nodes:[{y:100},{y:155},{y:210}]},
    {x:400, nodes:[{y:155}]}
  ];
  var lCols=[C.blue, C.accent, C.purple, C.orange];
  var lLabels=['Input','Hidden 1','Hidden 2','Output'];

  /* connection lines (faint) */
  layers.forEach(function(layer,li){
    if(li===layers.length-1) return;
    var nextLayer=layers[li+1];
    layer.nodes.forEach(function(n){
      nextLayer.nodes.forEach(function(m){
        sv+='<line x1="'+layer.x+'" y1="'+n.y+'" x2="'+nextLayer.x+'" y2="'+m.y
          +'" stroke="'+C.border+'" stroke-width="0.8" opacity="0.6"/>';
      });
    });
  });

  /* highlight one path: x1→hidden1[1]→hidden2[1]→output */
  var path=[[60,90],[175,130],[290,155],[400,155]];
  for(var pi=0;pi<path.length-1;pi++){
    sv+='<line x1="'+path[pi][0]+'" y1="'+path[pi][1]+'" x2="'+path[pi+1][0]+'" y2="'+path[pi+1][1]
      +'" stroke="'+C.yellow+'" stroke-width="2" opacity="0.7"/>';
  }

  /* nodes */
  layers.forEach(function(layer,li){
    layer.nodes.forEach(function(n,ni){
      var isHighlight=(li===0&&ni===0)||(li===1&&ni===1)||(li===2&&ni===1)||(li===3&&ni===0);
      sv+='<circle cx="'+layer.x+'" cy="'+n.y+'" r="'+(isHighlight?14:11)+'"'
        +' fill="'+hex(lCols[li],isHighlight?0.3:0.12)+'" stroke="'+lCols[li]+'" stroke-width="'+(isHighlight?2:1)+'"/>';
      if(isHighlight&&li>0){
        var txt=li===3?('a='+a.toFixed(2)):li===1?('z='+z.toFixed(1)):'';
        if(txt) sv+='<text x="'+layer.x+'" y="'+(n.y+3.5)+'" text-anchor="middle" fill="'+C.text+'" font-size="7" font-weight="700">'+txt+'</text>';
      }
      if(li===0&&ni<2){
        sv+='<text x="'+layer.x+'" y="'+(n.y+3.5)+'" text-anchor="middle" fill="'+C.text+'" font-size="8" font-weight="700">x'+(ni+1)+'</text>';
      }
    });
    /* layer label */
    sv+='<text x="'+layer.x+'" y="'+(NH-8)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5">'+lLabels[li]+'</text>';
  });

  /* weight annotation on highlight path */
  sv+='<text x="'+(path[0][0]+path[1][0])/2+'" y="'+(path[0][1]+path[1][1])/2-6+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8.5" font-weight="700">w\u2081='+w1.toFixed(2)+'</text>';

  /* formula annotation */
  sv+='<rect x="4" y="4" width="200" height="48" rx="5" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<text x="8" y="17" fill="'+C.muted+'" font-size="8.5" font-family="monospace">z = '+w1.toFixed(2)+'&#215;'+x1+' + ('+w2.toFixed(2)+')&#215;'+x2+' + '+b.toFixed(2)+'</text>';
  sv+='<text x="8" y="30" fill="'+C.muted+'" font-size="8.5" font-family="monospace">z = '+z.toFixed(4)+'</text>';
  sv+='<text x="8" y="44" fill="'+C.accent+'" font-size="8.5" font-family="monospace" font-weight="700">a = '+aName+'(z) = '+a.toFixed(4)+'</text>';

  /* activation gate icon at highlighted node */
  var gx=175, gy=90;
  sv+='<rect x="'+(gx+18)+'" y="'+(gy-12)+'" width="56" height="22" rx="4" fill="'+hex(C.accent,0.15)+'" stroke="'+C.accent+'" stroke-width="1"/>';
  sv+='<text x="'+(gx+46)+'" y="'+(gy+4)+'" text-anchor="middle" fill="'+C.accent+'" font-size="8" font-weight="700">'+aName+'(z)</text>';

  /* ── linear vs non-linear comparison strip ── */
  var linSv='';
  if(S.showLinear){
    var LW=440, LH=80;
    linSv+='<rect x="0" y="0" width="'+LW+'" height="'+LH+'" fill="#08080d" rx="8"/>';
    var lbls=['With '+aName+' (nonlinear)','Without activation (linear)'];
    var vals=[a,(w1*x1+w2*x2+b)];
    var cols=[C.green,C.red];
    lbls.forEach(function(lbl,i){
      var barW=Math.min(Math.abs(vals[i])/3*300,280);
      var by=12+i*34;
      linSv+='<text x="4" y="'+(by+14)+'" fill="'+cols[i]+'" font-size="8.5" font-family="monospace">'+lbl+'</text>';
      linSv+='<rect x="4" y="'+(by+18)+'" width="'+barW.toFixed(0)+'" height="10" rx="2" fill="'+hex(cols[i],0.6)+'"/>';
      linSv+='<text x="'+(barW+8).toFixed(0)+'" y="'+(by+27)+'" fill="'+cols[i]+'" font-size="8">'+vals[i].toFixed(4)+'</text>';
    });
  }

  var out=sectionTitle('The Neuron & Forward Pass','z = Wx+b (linear), then a = \u03c3(z) (nonlinearity) \u2014 repeated across every layer');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','2\u21923\u21922\u21921 Network (highlighted path)')
    +svgBox(sv,NW,NH)
    +sliderRow('w1',w1,-2,2,0.05,'w\u2081',2)
    +sliderRow('w2',w2,-2,2,0.05,'w\u2082',2)
    +sliderRow('b',b,-2,2,0.05,'bias b',2)
    +'<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;">'
    +['ReLU','Sigmoid','Tanh'].map(function(nm,i){
      return btnSel(i,S.actFn,ACT_COLS[i],nm,'actFn');
    }).join('')
    +'</div>'
    +'<div style="margin-top:10px;">'
    +'<button data-action="toggleLinear" style="padding:7px 14px;border-radius:7px;font-size:9.5px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.yellow,S.showLinear?0.15:0.05)+';border:1.5px solid '+(S.showLinear?C.yellow:C.border)+';'
    +'color:'+(S.showLinear?C.yellow:C.muted)+';cursor:pointer;">'
    +(S.showLinear?'\u2714 ':'+ ')+'Compare: with vs without activation</button>'
    +'</div>'
    +(S.showLinear?'<div style="margin-top:8px;">'+svgBox(linSv,440,80)+'</div>':'')
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','SINGLE NEURON (x\u2081=1.0, x\u2082=0.5)');
  out+=statRow('w\u2081 \u00d7 x\u2081',(w1*x1).toFixed(4),C.blue);
  out+=statRow('w\u2082 \u00d7 x\u2082',(w2*x2).toFixed(4),C.blue);
  out+=statRow('bias b',b.toFixed(4),C.dim);
  out+=statRow('Pre-activation z',z.toFixed(4),C.yellow);
  out+=statRow('Activation ('+aName+')',a.toFixed(4),C.accent);
  out+=statRow('Neuron fires?',a>0?'Yes':'No',a>0?C.green:C.red);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','LAYER FORMULA');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    'z\u02E1 = W\u02E1 a\u02E1\u207B\u00B9 + b\u02E1 &nbsp;<span style="color:'+C.yellow+';">\u2190 linear</span><br>'
    +'a\u02E1 = \u03c3(z\u02E1) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:'+C.accent+';">\u2190 nonlinear</span><br>'
    +'<br>'
    +'Without \u03c3:<br>'
    +'W\u00b2(W\u00b9x+b\u00b9)+b\u00b2 = W\'x+b\'<br>'
    +'<span style="color:'+C.red+';">\u2192 any depth = one linear layer!</span>'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','HISTORICAL ORIGIN');
  [
    {era:'1943',name:'McCulloch-Pitts',desc:'Binary threshold \u2014 IS a logic gate',c:C.dim},
    {era:'1958',name:'Perceptron',     desc:'Learnable weights, still binary',       c:C.muted},
    {era:'1986',name:'Modern neuron',  desc:'Smooth \u03c3(z): enables backprop',    c:C.accent},
  ].forEach(function(r){
    out+='<div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="font-size:8px;color:'+C.dim+';width:30px;flex-shrink:0;padding-top:2px;">'+r.era+'</div>'
      +'<div><div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.name+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.desc+'</div></div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#129504;','Why Nonlinearity is Everything',
    'A deep network with no activation functions collapses to <span style="color:'+C.red+';font-weight:700;">a single linear layer</span>, '
    +'regardless of depth. W\u00b2(W\u00b9x+b\u00b9)+b\u00b2 = W\'x+b\' — the matrices just multiply together. '
    +'Nonlinear activations break this collapse. Each layer then builds a genuinely new, richer representation. '
    +'<span style="color:'+C.accent+';font-weight:700;">Drag the w\u2081, w\u2082, b sliders</span> and '
    +'toggle the activation function to see z and a update in real time.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 1 — ACTIVATION FUNCTIONS
══════════════════════════════════════════════════════════ */
function renderActivations(){
  var ai=S.actTab;
  var zv=S.zVal;
  var afn=ACT_FNS[ai], agrd=ACT_GRADS[ai];

  var xmin=-4, xmax=4, ymin=-1.5, ymax=3.5;
  var gmin=-0.1, gmax=1.1;

  /* ── activation curve + gradient ── */
  var sv=plotAxes('z','\u03c3(z)',xmin,xmax,ymin,ymax,
    [-4,-3,-2,-1,0,1,2,3,4],[-1,0,1,2,3]);

  /* shaded region for saturation */
  if(ai===1||ai===2){
    /* sigmoid/tanh saturate at extremes */
    var satPts1=[sx(-4,xmin,xmax),PT,sx(-2,xmin,xmax),PT,sx(-2,xmin,xmax),PT+PH,sx(-4,xmin,xmax),PT+PH];
    var satPts2=[sx(2,xmin,xmax),PT,sx(4,xmin,xmax),PT,sx(4,xmin,xmax),PT+PH,sx(2,xmin,xmax),PT+PH];
    var satPath1='M'+satPts1[0]+','+satPts1[1]+' L'+satPts1[2]+','+satPts1[3]+' L'+satPts1[4]+','+satPts1[5]+' L'+satPts1[6]+','+satPts1[7]+' Z';
    var satPath2='M'+satPts2[0]+','+satPts2[1]+' L'+satPts2[2]+','+satPts2[3]+' L'+satPts2[4]+','+satPts2[5]+' L'+satPts2[6]+','+satPts2[7]+' Z';
    sv+='<path d="'+satPath1+'" fill="'+hex(C.red,0.06)+'"/>';
    sv+='<path d="'+satPath2+'" fill="'+hex(C.red,0.06)+'"/>';
    sv+='<text x="'+sx(-3,xmin,xmax).toFixed(1)+'" y="'+(PT+12)+'" fill="'+C.red+'" font-size="8" text-anchor="middle">saturates</text>';
    sv+='<text x="'+sx(3,xmin,xmax).toFixed(1)+'" y="'+(PT+12)+'" fill="'+C.red+'" font-size="8" text-anchor="middle">saturates</text>';
  }
  if(ai===0||ai===3){
    /* dead zone for ReLU/LeakyReLU */
    var deadPath='M'+sx(-4,xmin,xmax).toFixed(1)+','+PT+' L'+sx(0,xmin,xmax).toFixed(1)+','+PT
      +' L'+sx(0,xmin,xmax).toFixed(1)+','+(PT+PH)+' L'+sx(-4,xmin,xmax).toFixed(1)+','+(PT+PH)+' Z';
    sv+='<path d="'+deadPath+'" fill="'+hex(ai===0?C.red:C.yellow,0.06)+'"/>';
    sv+='<text x="'+sx(-2,xmin,xmax).toFixed(1)+'" y="'+(PT+12)+'" fill="'+(ai===0?C.red:C.yellow)+'" font-size="8" text-anchor="middle">'+(ai===0?'dead zone':'leaky')+'</text>';
  }

  /* draw all functions faintly */
  ACT_FNS.forEach(function(fn,i){
    if(i===ai) return;
    var pts=[];
    for(var zz=xmin;zz<=xmax;zz+=0.08){
      var v=fn(zz);
      if(v<ymin||v>ymax) return;
      pts.push([sx(zz,xmin,xmax),sy(v,ymin,ymax)]);
    }
    if(pts.length>1) sv+=polyline(pts,ACT_COLS[i],1,'4,3');
  });

  /* draw selected function bold */
  var mainPts=[];
  for(var zz=xmin;zz<=xmax;zz+=0.04){
    var v=afn(zz);
    if(v>=ymin&&v<=ymax) mainPts.push([sx(zz,xmin,xmax),sy(v,ymin,ymax)]);
  }
  if(mainPts.length>1) sv+=polyline(mainPts,ACT_COLS[ai],3);

  /* vertical cursor at zv */
  var cvx=sx(zv,xmin,xmax);
  var cvy=sy(afn(zv),ymin,ymax);
  sv+='<line x1="'+cvx.toFixed(1)+'" y1="'+PT+'" x2="'+cvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+cvy.toFixed(1)+'" r="5" fill="'+ACT_COLS[ai]+'" stroke="#0a0a0f" stroke-width="2"/>';

  /* gradient SVG */
  var gv=plotAxes('z','gradient \u03c3\'(z)',xmin,xmax,gmin,gmax,
    [-4,-2,0,2,4],[0,0.25,0.5,0.75,1.0]);
  var gradPts=[];
  for(var zz=xmin;zz<=xmax;zz+=0.04){
    var gval=agrd(zz);
    if(gval>=gmin&&gval<=gmax) gradPts.push([sx(zz,xmin,xmax),sy(gval,gmin,gmax)]);
  }
  if(gradPts.length>1) gv+=polyline(gradPts,ACT_COLS[ai],2.5);
  /* cursor */
  var gcvx=sx(zv,xmin,xmax);
  var gcvy=sy(Math.max(gmin,Math.min(gmax,agrd(zv))),gmin,gmax);
  gv+='<line x1="'+gcvx.toFixed(1)+'" y1="'+PT+'" x2="'+gcvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  gv+='<circle cx="'+gcvx.toFixed(1)+'" cy="'+gcvy.toFixed(1)+'" r="5" fill="'+ACT_COLS[ai]+'" stroke="#0a0a0f" stroke-width="2"/>';

  var aval=afn(zv).toFixed(4);
  var gval2=agrd(zv).toFixed(4);

  var out=sectionTitle('Activation Functions','The shape of \u03c3(z) determines whether gradients flow — or vanish');
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ACT_NAMES.forEach(function(nm,i){
    out+=btnSel(i,ai,ACT_COLS[i],nm,'actTab');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',ACT_NAMES[ai]+' — '+ACT_FORMULAE[ai])
    +svgBox(sv,VW,VH)
    +'<div style="margin-top:8px;">'+svgBox(gv,VW,160)+'</div>'
    +sliderRow('zVal',zv,-4,4,0.05,'z value',2)
    +'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    +ACT_NAMES.map(function(nm,i){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+(i===ai?C.text:C.muted)+';">'
        +'<div style="width:14px;height:2px;background:'+ACT_COLS[i]+'"></div>'+nm+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT z = '+zv.toFixed(2));
  out+=statRow(ACT_NAMES[ai]+'(z)',aval,ACT_COLS[ai]);
  out+=statRow('Gradient \u03c3\'(z)',gval2,Math.abs(parseFloat(gval2))>0.1?C.green:C.red);
  out+=statRow('Gradient flows?',parseFloat(gval2)>0.01?'Yes':'Barely / No',parseFloat(gval2)>0.01?C.green:C.red);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+ACT_COLS[ai]+';margin-bottom:8px;font-weight:700;',ACT_NAMES[ai]);
  out+=div('padding:6px 8px;border-radius:5px;border-left:3px solid '+ACT_COLS[ai]+';margin-bottom:6px;',
    div('font-size:8.5px;color:'+C.muted+';',ACT_PROBLEMS[ai])
  );
  out+=statRow('Use for',ACT_USE[ai],ACT_COLS[ai]);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','VANISHING GRADIENT PROBLEM');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Gradient of layer l = gradient from above <span style="color:'+C.red+';">&#215;</span> \u03c3\'(z\u02E1). '
    +'Sigmoid max gradient = <span style="color:'+C.red+';font-weight:700;">0.25</span>. '
    +'With 10 layers: 0.25\u00b9\u2070 \u2248 <span style="color:'+C.red+';font-weight:700;">0.000001</span>. '
    +'Early layers receive near-zero gradients \u2014 they barely update. '
    +'<span style="color:'+C.green+';font-weight:700;">ReLU</span> gradient = 1 for z&gt;0 \u2014 no vanishing.'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:6px;','MAX GRADIENT BY FUNCTION');
  [
    {n:'ReLU',     g:'1.0 (for z>0)',   c:C.green},
    {n:'GELU',     g:'\u22481.0',        c:C.purple},
    {n:'Tanh',     g:'1.0 (at z=0)',    c:C.blue},
    {n:'Leaky',    g:'1.0 (for z>0)',   c:C.yellow},
    {n:'Sigmoid',  g:'0.25 (at z=0)',   c:C.red},
  ].forEach(function(r){
    out+=statRow(r.n,r.g,r.c);
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','ReLU Solved the Vanishing Gradient (For Hidden Layers)',
    'Sigmoid and Tanh both have gradients that shrink to near-zero for large |z|. '
    +'With many layers, this shrinkage compounds: '
    +'<span style="color:'+C.red+';font-family:monospace;">0.25\u00b9\u2070 \u2248 10\u207b\u2076</span>. '
    +'ReLU gradient is exactly 1 for positive z — gradients flow all the way to early layers unchanged. '
    +'<span style="color:'+C.purple+';font-weight:700;">GELU</span> is the modern default for Transformers: '
    +'smooth everywhere, no hard zero, works better in attention layers.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 2 — BACKPROPAGATION STEP-BY-STEP
══════════════════════════════════════════════════════════ */
var BP_STEPS=[
  {title:'Forward: z\u00b9 = W\u00b9x + b\u00b9',       phase:'forward', layer:1},
  {title:'Forward: a\u00b9 = ReLU(z\u00b9)',             phase:'forward', layer:1},
  {title:'Forward: z\u00b2 = W\u00b2a\u00b9 + b\u00b2', phase:'forward', layer:2},
  {title:'Forward: \u0177 = \u03c3(z\u00b2)  \u2192 compute Loss', phase:'loss', layer:2},
  {title:'Backward: \u2202L/\u2202W\u00b2 = \u2202L/\u2202\u0177 \u00d7 \u03c3\'(z\u00b2) \u00d7 a\u00b9\u1d40', phase:'backward', layer:2},
  {title:'Backward: propagate \u2202L/\u2202a\u00b9 = W\u00b2\u1d40 \u00d7 \u2202L/\u2202z\u00b2', phase:'backward', layer:1},
  {title:'Backward: \u2202L/\u2202W\u00b9 = \u2202L/\u2202z\u00b9 \u00d7 x\u1d40 \u2192 update W\u00b9', phase:'update', layer:1},
];
var BP_COLS={forward:C.blue, loss:C.red, backward:C.orange, update:C.green};

function renderBackprop(){
  var step=S.bpStep;
  var lr=S.bpLR;

  /* concrete numbers for a 2→2→1 network */
  var x=[0.8, -0.3];
  var W1=[[0.5,-0.2],[0.3,0.7]];
  var b1=[0.1,-0.1];
  var W2=[0.6,0.8];
  var b2=0.2;
  var y_true=1.0;

  /* forward */
  var z1=[W1[0][0]*x[0]+W1[0][1]*x[1]+b1[0], W1[1][0]*x[0]+W1[1][1]*x[1]+b1[1]];
  var a1=[relu(z1[0]),relu(z1[1])];
  var z2=W2[0]*a1[0]+W2[1]*a1[1]+b2;
  var yhat=sigmoid(z2);
  var loss=-(y_true*Math.log(yhat)+(1-y_true)*Math.log(1-yhat));

  /* backward */
  var dL_dyhat=-(y_true/yhat-(1-y_true)/(1-yhat));
  var dL_dz2=yhat-y_true; /* BCE + sigmoid simplifies */
  var dL_dW2=[dL_dz2*a1[0], dL_dz2*a1[1]];
  var dL_db2=dL_dz2;
  var dL_da1=[W2[0]*dL_dz2, W2[1]*dL_dz2];
  var dL_dz1=[dL_da1[0]*reluGrad(z1[0]), dL_da1[1]*reluGrad(z1[1])];
  var dL_dW1=[[dL_dz1[0]*x[0],dL_dz1[0]*x[1]],[dL_dz1[1]*x[0],dL_dz1[1]*x[1]]];

  /* updated weights */
  var W1new=[[W1[0][0]-lr*dL_dW1[0][0], W1[0][1]-lr*dL_dW1[0][1]],
             [W1[1][0]-lr*dL_dW1[1][0], W1[1][1]-lr*dL_dW1[1][1]]];
  var W2new=[W2[0]-lr*dL_dW2[0], W2[1]-lr*dL_dW2[1]];

  /* ── flow diagram SVG ── */
  var FW=440, FH=220;
  var fv='';

  /* nodes: x → z1 → a1 → z2 → yhat → L */
  var nodes=[
    {label:'x', sublabel:'['+x[0]+', '+x[1]+']', x:38, y:110, col:C.blue, active:step>=0},
    {label:'z\u00b9', sublabel:'['+z1[0].toFixed(2)+','+z1[1].toFixed(2)+']', x:108, y:110, col:C.yellow, active:step>=0},
    {label:'a\u00b9', sublabel:'['+a1[0].toFixed(2)+','+a1[1].toFixed(2)+']', x:188, y:110, col:C.accent, active:step>=1},
    {label:'z\u00b2', sublabel:z2.toFixed(4), x:268, y:110, col:C.yellow, active:step>=2},
    {label:'\u0177', sublabel:yhat.toFixed(4), x:340, y:110, col:C.orange, active:step>=3},
    {label:'L', sublabel:loss.toFixed(4), x:406, y:110, col:C.red, active:step>=3},
  ];

  /* arrows */
  var arrowLabels=['W\u00b9','ReLU','W\u00b2','\u03c3','BCE'];
  var arrowCols=[
    step>=0?C.blue:C.dim,
    step>=1?C.accent:C.dim,
    step>=2?C.blue:C.dim,
    step>=3?C.orange:C.dim,
    step>=3?C.red:C.dim
  ];
  for(var ni=0;ni<nodes.length-1;ni++){
    var n1=nodes[ni],n2=nodes[ni+1];
    var active=(step>=ni);
    fv+='<line x1="'+(n1.x+24)+'" y1="'+n1.y+'" x2="'+(n2.x-24)+'" y2="'+n2.y
      +'" stroke="'+arrowCols[ni]+'" stroke-width="'+(active?2:1)+'" '
      +(active?'':'stroke-dasharray="4,3"')
      +' marker-end="url(#arr'+(active?'a':'d')+')" opacity="'+(active?1:0.3)+'"/>';
    fv+='<text x="'+(n1.x+24+(n2.x-24-n1.x-24)/2).toFixed(0)+'" y="'+(n1.y-10)+'" text-anchor="middle" fill="'+arrowCols[ni]+'" font-size="8.5" font-weight="700">'+arrowLabels[ni]+'</text>';
  }

  /* backward gradient arrows (step>=4) */
  if(step>=4){
    var gradPairs=[[406,340],[340,268],[268,188],[188,108],[108,38]];
    var gradLabels=['\u2202L/\u2202\u0177','\u2202L/\u2202z\u00b2','\u2202L/\u2202a\u00b9','\u2202L/\u2202z\u00b9','\u2202L/\u2202W\u00b9'];
    var gradActive=[step>=4,step>=4,step>=5,step>=5,step>=6];
    gradPairs.forEach(function(gp,gi){
      if(!gradActive[gi]) return;
      fv+='<path d="M'+gp[0]+','+(110+20)+' L'+gp[1]+','+(110+20)+'" stroke="'+C.orange
        +'" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arro)" opacity="0.8"/>';
      fv+='<text x="'+((gp[0]+gp[1])/2).toFixed(0)+'" y="'+(110+38)+'" text-anchor="middle" fill="'+C.orange+'" font-size="7.5">'+gradLabels[gi]+'</text>';
    });
  }

  /* arrowhead markers */
  fv='<defs>'
    +'<marker id="arra" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="'+C.blue+'"/></marker>'
    +'<marker id="arrd" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="'+C.dim+'"/></marker>'
    +'<marker id="arro" markerWidth="6" markerHeight="4" refX="0" refY="2" orient="auto"><path d="M6,0 L0,2 L6,4" fill="'+C.orange+'"/></marker>'
    +'</defs>'+fv;

  /* nodes */
  nodes.forEach(function(n){
    fv+='<circle cx="'+n.x+'" cy="'+n.y+'" r="22" fill="'+hex(n.col,n.active?0.2:0.05)+'" stroke="'+n.col+'" stroke-width="'+(n.active?2:1)+'" opacity="'+(n.active?1:0.4)+'"/>';
    fv+='<text x="'+n.x+'" y="'+(n.y-5)+'" text-anchor="middle" fill="'+n.col+'" font-size="10" font-weight="700">'+n.label+'</text>';
    fv+='<text x="'+n.x+'" y="'+(n.y+9)+'" text-anchor="middle" fill="'+C.muted+'" font-size="7">'+n.sublabel+'</text>';
  });

  /* phase label */
  var curStep=BP_STEPS[step];
  var phaseCol=BP_COLS[curStep.phase];
  fv+='<rect x="4" y="4" width="432" height="22" rx="4" fill="'+hex(phaseCol,0.12)+'" stroke="'+phaseCol+'" stroke-width="1"/>';
  fv+='<text x="218" y="19" text-anchor="middle" fill="'+phaseCol+'" font-size="9.5" font-weight="700">Step '+(step+1)+'/7: '+curStep.title+'</text>';

  var out=sectionTitle('Backpropagation — The Chain Rule','Forward pass caches values; backward pass propagates gradients \u2014 O(1\u00d7 forward pass) cost');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','2\u21922\u21921 Network — Concrete Values')
    +svgBox(fv,FW,FH)
    +'<div style="margin-top:10px;display:flex;justify-content:center;gap:8px;">'
    +'<button data-action="bpPrev" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step>0?0.12:0.04)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Back</button>'
    +'<button data-action="bpNext" style="padding:9px 20px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step<6?0.12:0.04)+';border:1.5px solid '+(step<6?C.accent:C.border)+';'
    +'color:'+(step<6?C.accent:C.dim)+';cursor:'+(step<6?'pointer':'default')+';">Next \u2192</button>'
    +'</div>'
    +sliderRow('bpLR',lr,0.01,1.0,0.01,'learn rate \u03b1',2)
    +'<div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:8px;">'
    +BP_STEPS.map(function(s,i){
      return '<div style="width:28px;height:8px;border-radius:4px;background:'+(i<=step?BP_COLS[s.phase]:C.dim)+';opacity:'+(i===step?1:0.5)+';"></div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  if(step<=3){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.blue+';margin-bottom:10px;font-weight:700;','FORWARD PASS VALUES');
    out+=statRow('x\u2081, x\u2082', x[0]+', '+x[1], C.blue);
    out+=statRow('z\u00b9', '['+z1[0].toFixed(4)+', '+z1[1].toFixed(4)+']', step>=0?C.yellow:C.dim);
    out+=statRow('a\u00b9 = ReLU(z\u00b9)', '['+a1[0].toFixed(4)+', '+a1[1].toFixed(4)+']', step>=1?C.accent:C.dim);
    out+=statRow('z\u00b2', z2.toFixed(4), step>=2?C.yellow:C.dim);
    out+=statRow('\u0177 = \u03c3(z\u00b2)', yhat.toFixed(4), step>=3?C.orange:C.dim);
    out+=statRow('BCE Loss', loss.toFixed(4), step>=3?C.red:C.dim);
    out+='</div>';
  } else {
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.orange+';margin-bottom:10px;font-weight:700;','BACKWARD PASS GRADIENTS');
    out+=statRow('\u2202L/\u2202\u0177', dL_dyhat.toFixed(4), step>=4?C.orange:C.dim);
    out+=statRow('\u2202L/\u2202z\u00b2', dL_dz2.toFixed(4), step>=4?C.orange:C.dim);
    out+=statRow('\u2202L/\u2202W\u00b2', '['+dL_dW2[0].toFixed(3)+', '+dL_dW2[1].toFixed(3)+']', step>=4?C.orange:C.dim);
    out+=statRow('\u2202L/\u2202a\u00b9', '['+dL_da1[0].toFixed(3)+', '+dL_da1[1].toFixed(3)+']', step>=5?C.orange:C.dim);
    out+=statRow('\u2202L/\u2202W\u00b9[0]', '['+dL_dW1[0][0].toFixed(3)+', '+dL_dW1[0][1].toFixed(3)+']', step>=6?C.green:C.dim);
    out+='</div>';
    if(step===6){
      out+='<div class="card" style="margin:0;">';
      out+=div('font-size:10px;color:'+C.green+';margin-bottom:8px;font-weight:700;','WEIGHT UPDATE (\u03b1='+lr+')');
      out+=statRow('W\u00b2[0]: '+W2[0].toFixed(3)+' \u2192', W2new[0].toFixed(3), C.green);
      out+=statRow('W\u00b2[1]: '+W2[1].toFixed(3)+' \u2192', W2new[1].toFixed(3), C.green);
      out+=statRow('W\u00b9[0,0]: '+W1[0][0]+' \u2192', W1new[0][0].toFixed(3), C.green);
      out+='</div>';
    }
  }

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','CHAIN RULE');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:8.5px;color:'+C.muted+';line-height:1.9;font-family:monospace;',
    '\u2202L/\u2202W\u00b9 = <span style="color:'+C.red+';">\u2202L/\u2202\u0177</span> \u00d7 <span style="color:'+C.orange+';">\u2202\u0177/\u2202z\u00b2</span><br>'
    +'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\u00d7 <span style="color:'+C.yellow+';">\u2202z\u00b2/\u2202a\u00b9</span> \u00d7 <span style="color:'+C.accent+';">\u2202a\u00b9/\u2202z\u00b9</span><br>'
    +'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\u00d7 <span style="color:'+C.blue+';">\u2202z\u00b9/\u2202W\u00b9</span><br>'
    +'<br>'
    +'Each term is easy to compute.<br>'
    +'Backprop: compute once per layer,<br>'
    +'reuse \u2014 <span style="color:'+C.green+';">O(2\u00d7forward cost)</span>.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128257;','Backprop is Just the Chain Rule, Applied Cleverly',
    'Every operation in the forward pass (matrix multiply, activation) has a simple derivative. '
    +'Backprop chains them together right-to-left, reusing cached intermediate values. '
    +'The cost is <span style="color:'+C.green+';font-weight:700;">~2\u00d7 forward pass</span> \u2014 remarkably cheap for computing all gradients simultaneously. '
    +'The weight update rule <span style="color:'+C.accent+';font-family:monospace;">W \u2190 W \u2212 \u03b1\u00b7\u2202L/\u2202W</span> '
    +'nudges every weight in the direction that reduces the loss. '
    +'After thousands of such steps, the network converges.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 3 — LOSS CURVES & TRAINING DYNAMICS
══════════════════════════════════════════════════════════ */
var LOSS_MODES=[
  {label:'Healthy',      col:C.green},
  {label:'Overfit',      col:C.red},
  {label:'Underfit',     col:C.orange},
  {label:'Early Stop',   col:C.yellow},
];

function trainLoss(epoch, mode){
  if(mode===0) return 0.08+0.70*Math.exp(-0.06*epoch);
  if(mode===1) return 0.02+0.70*Math.exp(-0.10*epoch);
  if(mode===2) return 0.45+0.20*Math.exp(-0.02*epoch);
  return 0.05+0.70*Math.exp(-0.09*epoch);
}
function valLoss(epoch, mode){
  if(mode===0) return 0.12+0.65*Math.exp(-0.05*epoch)+0.002*epoch*Math.exp(-0.02*epoch);
  if(mode===1) return 0.15+0.60*Math.exp(-0.06*epoch)+0.003*Math.pow(epoch,1.1)*Math.exp(-0.015*epoch);
  if(mode===2) return 0.48+0.18*Math.exp(-0.015*epoch);
  return valLoss(epoch,0);
}
function bestValEpoch(mode){
  var best=0, bestVal=999;
  for(var e=1;e<=100;e++){
    var v=valLoss(e,mode);
    if(v<bestVal){bestVal=v;best=e;}
  }
  return best;
}

function renderLossCurves(){
  var mode=S.lossMode;
  var ep=S.epoch;
  var trE=trainLoss(ep,mode), vaE=valLoss(ep,mode);
  var gap=vaE-trE;
  var bestEp=bestValEpoch(mode);
  var bestVaE=valLoss(bestEp,mode);

  var sv=plotAxes('Epoch','Loss',0,100,0,1.0,
    [0,20,40,60,80,100],[0,0.2,0.4,0.6,0.8,1.0]);

  /* fill between curves (overfit gap) */
  if(mode===1||mode===0){
    var fillPts='';
    var fillPts2='';
    for(var e2=0;e2<=100;e2+=2){
      var tl=trainLoss(e2,mode), vl=valLoss(e2,mode);
      fillPts+=(e2===0?'M':'L')+sx(e2,0,100).toFixed(1)+','+sy(tl,0,1.0).toFixed(1)+' ';
      fillPts2='L'+sx(e2,0,100).toFixed(1)+','+sy(vl,0,1.0).toFixed(1)+' '+fillPts2;
    }
    sv+='<path d="'+fillPts+fillPts2+'Z" fill="'+hex(C.red,0.06)+'"/>';
  }

  /* train and val curves */
  var trPts=[], vaPts=[];
  for(var e2=0;e2<=100;e2+=1){
    trPts.push([sx(e2,0,100),sy(trainLoss(e2,mode),0,1.0)]);
    vaPts.push([sx(e2,0,100),sy(valLoss(e2,mode),0,1.0)]);
  }
  sv+=polyline(trPts,C.green,2);
  sv+=polyline(vaPts,C.orange,2,'6,3');

  /* current epoch cursor */
  var cvx=sx(ep,0,100);
  sv+='<line x1="'+cvx.toFixed(1)+'" y1="'+PT+'" x2="'+cvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.accent+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(trE,0,1.0).toFixed(1)+'" r="4" fill="'+C.green+'" stroke="#0a0a0f" stroke-width="1.5"/>';
  sv+='<circle cx="'+cvx.toFixed(1)+'" cy="'+sy(vaE,0,1.0).toFixed(1)+'" r="4" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';

  /* best val epoch marker */
  if(mode===1||mode===3){
    var bvx=sx(bestEp,0,100);
    sv+='<line x1="'+bvx.toFixed(1)+'" y1="'+PT+'" x2="'+bvx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.yellow+'" stroke-width="1.5" stroke-dasharray="3,2"/>';
    sv+='<text x="'+bvx.toFixed(1)+'" y="'+(PT+10)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8">\u2605 best (ep='+bestEp+')</text>';
    sv+='<circle cx="'+bvx.toFixed(1)+'" cy="'+sy(bestVaE,0,1.0).toFixed(1)+'" r="5" fill="'+C.yellow+'" stroke="#0a0a0f" stroke-width="2"/>';
  }

  /* legend */
  sv+='<rect x="'+(PL+4)+'" y="'+(PT+3)+'" width="100" height="30" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<line x1="'+(PL+8)+'" y1="'+(PT+12)+'" x2="'+(PL+24)+'" y2="'+(PT+12)+'" stroke="'+C.green+'" stroke-width="2"/>';
  sv+='<text x="'+(PL+28)+'" y="'+(PT+16)+'" fill="'+C.green+'" font-size="8.5" font-family="monospace">Train</text>';
  sv+='<line x1="'+(PL+8)+'" y1="'+(PT+24)+'" x2="'+(PL+24)+'" y2="'+(PT+24)+'" stroke="'+C.orange+'" stroke-width="2" stroke-dasharray="6,3"/>';
  sv+='<text x="'+(PL+28)+'" y="'+(PT+28)+'" fill="'+C.orange+'" font-size="8.5" font-family="monospace">Val</text>';

  var regime='';
  var regimeCol=C.muted;
  if(mode===0){regime=gap<0.05?'Healthy — generalising well':'Slight overfitting';regimeCol=gap<0.05?C.green:C.yellow;}
  if(mode===1){regime=ep<20?'Converging':'Overfitting \u2014 val loss rising';regimeCol=ep<20?C.yellow:C.red;}
  if(mode===2){regime='Underfitting \u2014 both losses high';regimeCol=C.orange;}
  if(mode===3){regime=ep<=bestEp?'Healthy zone':'Past optimal \u2014 should have stopped at ep='+bestEp;regimeCol=ep<=bestEp?C.green:C.red;}

  var out=sectionTitle('Loss Curves & Training Dynamics','Train/val curves reveal everything: underfitting, overfitting, and the optimal stopping point');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  LOSS_MODES.forEach(function(lm,i){
    out+=btnSel(i,mode,lm.col,lm.label,'lossMode');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;',LOSS_MODES[mode].label+' Training Pattern')
    +svgBox(sv,VW,VH)
    +sliderRow('epoch',ep,1,100,1,'epoch',0)
    +'<div style="margin-top:8px;padding:8px 10px;background:#08080d;border-radius:6px;border:1px solid '+C.border+';">'
    +div('font-size:9px;color:'+regimeCol+';font-weight:700;',regime)
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','AT EPOCH '+ep);
  out+=statRow('Train loss',trE.toFixed(4),C.green);
  out+=statRow('Val loss',vaE.toFixed(4),C.orange);
  out+=statRow('Train\u2013val gap',gap.toFixed(4),gap>0.15?C.red:gap>0.06?C.yellow:C.green);
  if(mode===1||mode===3) out+=statRow('Best val epoch',bestEp,C.yellow);
  out+=statRow('Diagnosis',regime,regimeCol);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','DIAGNOSIS GUIDE');
  [
    {pattern:'Both losses high & flat',  diag:'Underfitting',   fix:'More capacity / epochs',     c:C.orange},
    {pattern:'Train \u2193, Val \u2193 together',diag:'Healthy',  fix:'Keep training!',           c:C.green},
    {pattern:'Train \u2193, Val \u2191 rises', diag:'Overfitting', fix:'Dropout / L2 / less capacity', c:C.red},
    {pattern:'Val stops improving',       diag:'Early stop here', fix:'Save weights at \u2605',   c:C.yellow},
  ].forEach(function(r){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.c+';">'
      +'<div style="font-size:8.5px;font-weight:700;color:'+r.c+';">'+r.diag+'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:1px;">Pattern: '+r.pattern+'</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">Fix: '+r.fix+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','TRAIN / VAL / TEST SPLIT');
  [
    {name:'Training set',   pct:'60\u201370%', role:'Gradient descent runs here — model sees this',    col:C.blue},
    {name:'Validation set', pct:'10\u201320%', role:'Hyperparameter tuning, early stopping, no grads', col:C.yellow},
    {name:'Test set',       pct:'10\u201320%', role:'Evaluated ONCE at the end — never for decisions', col:C.accent},
  ].forEach(function(r){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+r.col+';">'
      +'<div style="display:flex;justify-content:space-between;">'
      +'<span style="font-size:9px;font-weight:700;color:'+r.col+';">'+r.name+'</span>'
      +'<span style="font-size:9px;color:'+C.dim+';">'+r.pct+'</span></div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.role+'</div></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128274;','The Test Set is a Locked Vault',
    'If you pick the model with the best test accuracy from 30 experiments, '
    +'you have implicitly optimised over the test set. '
    +'The reported accuracy becomes the <span style="color:'+C.red+';font-weight:700;">best of 30 draws</span> \u2014 '
    +'optimistically biased by several percentage points. '
    +'<span style="color:'+C.accent+';font-weight:700;">Correct pattern</span>: '
    +'use validation to pick models and hyperparameters. '
    +'Open the test set <em>exactly once</em> to report the final number.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 4 — REGULARISATION & WHEN TO USE NNs
══════════════════════════════════════════════════════════ */
function renderRegularisation(){
  var mode=S.regMode;

  /* ── dropout illustration SVG ── */
  function dropoutSVG(dropRate){
    var DW=440, DH=180;
    var dv='';
    /* 3 layers of neurons */
    var layers=[
      {x:80,  count:4},
      {x:220, count:6},
      {x:360, count:4}
    ];
    var nodeR=12;
    /* connections */
    layers.forEach(function(layer,li){
      if(li===layers.length-1) return;
      var next=layers[li+1];
      var ly=DH/2-(layer.count-1)*20;
      var ny=DH/2-(next.count-1)*20;
      for(var ni=0;ni<layer.count;ni++){
        for(var nj=0;nj<next.count;nj++){
          var isDead=(dropRate>0&&((ni*7+nj*3+li*5)%10)/10<dropRate);
          dv+='<line x1="'+layer.x+'" y1="'+(ly+ni*40)+'" x2="'+next.x+'" y2="'+(ny+nj*40)+'"'
            +' stroke="'+(isDead?C.dim:C.border)+'" stroke-width="'+(isDead?0.5:0.8)+'" opacity="'+(isDead?0.2:0.6)+'"/>';
        }
      }
    });
    /* nodes */
    layers.forEach(function(layer,li){
      var count=layer.count;
      var startY=DH/2-(count-1)*20;
      for(var ni=0;ni<count;ni++){
        var ny=startY+ni*40;
        var isDead=(dropRate>0&&li===1&&((ni*7+li*5)%10)/10<dropRate);
        dv+='<circle cx="'+layer.x+'" cy="'+ny+'" r="'+nodeR+'"'
          +' fill="'+hex(isDead?C.red:C.accent,isDead?0.1:0.2)+'"'
          +' stroke="'+(isDead?C.red:C.accent)+'" stroke-width="'+(isDead?1.5:1)+'"'
          +' stroke-dasharray="'+(isDead?'4,3':'')+'" opacity="'+(isDead?0.4:1)+'"/>';
        if(isDead){
          dv+='<line x1="'+(layer.x-7)+'" y1="'+(ny-7)+'" x2="'+(layer.x+7)+'" y2="'+(ny+7)+'" stroke="'+C.red+'" stroke-width="1.5" opacity="0.5"/>';
          dv+='<line x1="'+(layer.x+7)+'" y1="'+(ny-7)+'" x2="'+(layer.x-7)+'" y2="'+(ny+7)+'" stroke="'+C.red+'" stroke-width="1.5" opacity="0.5"/>';
        }
      }
    });
    /* labels */
    dv+='<text x="80" y="'+( DH-8)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9">Input</text>';
    dv+='<text x="220" y="'+(DH-8)+'" text-anchor="middle" fill="'+(dropRate>0?C.red:C.muted)+'" font-size="9">Hidden '+(dropRate>0?'(dropout='+dropRate+')':'(no dropout)')+'</text>';
    dv+='<text x="360" y="'+(DH-8)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9">Output</text>';
    if(dropRate>0){
      dv+='<rect x="140" y="4" width="160" height="18" rx="4" fill="'+hex(C.red,0.1)+'" stroke="'+C.red+'" stroke-width="0.8"/>';
      dv+='<text x="220" y="16" text-anchor="middle" fill="'+C.red+'" font-size="8.5">Training: '+Math.round(dropRate*100)+'% neurons zeroed</text>';
    } else {
      dv+='<rect x="140" y="4" width="160" height="18" rx="4" fill="'+hex(C.green,0.1)+'" stroke="'+C.green+'" stroke-width="0.8"/>';
      dv+='<text x="220" y="16" text-anchor="middle" fill="'+C.green+'" font-size="8.5">Inference: all neurons active (\u00d7scale)</text>';
    }
    return svgBox(dv,DW,DH);
  }

  /* ── L2 weight distribution SVG ── */
  function l2SVG(lambda){
    var LW=440,LH=160;
    var lv='';
    /* show weight distribution with and without L2 */
    lv+=plotAxes('Weight value','Count','-3','3','0','50',
      [-3,-2,-1,0,1,2,3],[0,10,20,30,40,50]);
    /* without L2: wider spread */
    var noL2Pts=[],l2Pts=[];
    function gaussian2(x,mu,sig){return 50*Math.exp(-0.5*Math.pow((x-mu)/sig,2));}
    var sigNoL2=1.4, sigL2=Math.max(0.3,1.4-lambda*1.2);
    for(var ww=-3;ww<=3;ww+=0.05){
      noL2Pts.push([sx(ww,-3,3),sy(gaussian2(ww,0,sigNoL2),0,50)]);
      l2Pts.push([sx(ww,-3,3),sy(gaussian2(ww,0,sigL2),0,50)]);
    }
    lv+=polyline(noL2Pts,C.red,1.5,'5,3');
    lv+=polyline(l2Pts,C.green,2.5);
    /* peak labels */
    lv+='<text x="'+sx(0,-3,3)+'" y="'+(sy(gaussian2(0,0,sigNoL2),0,50)-5)+'" text-anchor="middle" fill="'+C.red+'" font-size="8.5">No L2 (\u03c3='+sigNoL2+')</text>';
    lv+='<text x="'+sx(0.5,-3,3)+'" y="'+(sy(gaussian2(0,0,sigL2),0,50)-5)+'" text-anchor="start" fill="'+C.green+'" font-size="8.5">L2 \u03bb='+lambda.toFixed(1)+' (\u03c3='+sigL2.toFixed(1)+')</text>';
    return svgBox(lv,LW,VH);
  }

  /* ── model comparison SVG ── */
  function compareRows(){
    var models=[
      {name:'Linear / LR',  tabular:5,images:1,text:2,speed:5,interp:5},
      {name:'Random Forest', tabular:4,images:1,text:2,speed:4,interp:3},
      {name:'Gradient Boost',tabular:5,images:2,text:2,speed:3,interp:2},
      {name:'Neural Net',    tabular:3,images:5,text:5,speed:2,interp:1},
    ];
    var cols_=['Tabular','Images','Text/Audio','Train Speed','Interpret.'];
    var colCols=[C.accent,C.blue,C.purple,C.green,C.yellow];
    var out2='<div style="overflow-x:auto;">'
      +'<table style="width:100%;border-collapse:collapse;font-size:8.5px;font-family:monospace;">'
      +'<tr><th style="padding:4px 6px;text-align:left;color:'+C.muted+';border-bottom:1px solid '+C.border+';">Model</th>';
    cols_.forEach(function(c2,ci){
      out2+='<th style="padding:4px 5px;text-align:center;color:'+colCols[ci]+';border-bottom:1px solid '+C.border+';">'+c2+'</th>';
    });
    out2+='</tr>';
    models.forEach(function(m){
      out2+='<tr>';
      out2+='<td style="padding:4px 6px;color:'+C.text+';border-bottom:1px solid '+C.border+';font-weight:700;">'+m.name+'</td>';
      [m.tabular,m.images,m.text,m.speed,m.interp].forEach(function(v,vi){
        var bars='';
        for(var bi=0;bi<5;bi++) bars+='<div style="width:8px;height:8px;border-radius:2px;margin:1px;display:inline-block;background:'+(bi<v?colCols[vi]:C.dim)+';opacity:'+(bi<v?0.8:0.2)+'"></div>';
        out2+='<td style="padding:4px 5px;text-align:center;border-bottom:1px solid '+C.border+';">'+bars+'</td>';
      });
      out2+='</tr>';
    });
    out2+='</table></div>';
    return out2;
  }

  var dropRate=mode===0?0.4:0;
  var l2Lambda=mode===1?0.7:0;

  var out=sectionTitle('Regularisation & When to Use NNs','Dropout, L2, and batch norm prevent memorisation — knowing when to use NNs is as important as how');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ['&#128257; Dropout','&#127793; L2 Weight Decay','&#127942; Model Comparison'].forEach(function(lbl,i){
    out+=btnSel(i,mode,[C.orange,C.green,C.accent][i],lbl,'regMode');
  });
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';

  if(mode===0){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Dropout (p=0.4) — Training vs Inference')
      +dropoutSVG(0.4)
      +'<div style="margin-top:8px;font-size:8.5px;color:'+C.muted+';padding:8px;background:#08080d;border-radius:6px;">'
      +'During inference: scale all activations by (1\u22120.4)=0.6 to match expected activation during training.'
      +'</div>'
    );
    out+=card(
      div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','Inference Mode (all neurons active)')
      +dropoutSVG(0)
    );
  } else if(mode===1){
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','L2 Regularisation — Weight Distribution')
      +l2SVG(0.7)
      +'<div style="margin-top:8px;font-size:8.5px;color:'+C.muted+';padding:8px;background:#08080d;border-radius:6px;">'
      +'L2 adds \u03bb/2\u00b7||W||&#178; to the loss. Gradient: \u2202L/\u2202W += \u03bbW. '
      +'Weights shrink toward 0 each step: W \u2190 W\u00b7(1\u2212\u03b1\u03bb) \u2212 \u03b1\u00b7\u2202L/\u2202W.'
      +'</div>'
    );
  } else {
    out+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Which Model for Which Task?')
      +compareRows()
    );
  }
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  if(mode===0){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','DROPOUT');
    out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;',
      '<span style="color:'+C.accent+';font-weight:700;">Training</span>: zero each neuron with prob p.<br>'
      +'<span style="color:'+C.accent+';font-weight:700;">Inference</span>: all neurons, scale by (1\u2212p).<br><br>'
      +'Effect: forces redundancy \u2014 no single neuron can be relied upon. '
      +'<span style="color:'+C.yellow+';font-weight:700;">Approximates training an ensemble of 2\u207f sub-networks.</span>'
    );
    out+=statRow('Typical p (hidden)',  '0.2 \u2013 0.5',  C.accent);
    out+=statRow('Typical p (input)',   '0.1 \u2013 0.2',  C.accent);
    out+=statRow('Train time impact',   '~2\u00d7 slower',  C.yellow);
    out+=statRow('Inference impact',    'None (p=0)',        C.green);
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','OTHER REGULARISERS');
    [
      {n:'L2 weight decay', d:'\u03bb||W||&#178; penalty \u2014 shrinks large weights', c:C.green},
      {n:'Batch Norm',      d:'Normalise each mini-batch; acts as regulariser',       c:C.blue},
      {n:'Early stopping',  d:'Stop at val minimum; simplest and very effective',     c:C.yellow},
      {n:'Data augment.',   d:'More diverse training data; best regulariser for NNs', c:C.purple},
    ].forEach(function(r){
      out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:2px solid '+r.c+';">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.n+'</div>'
        +'<div style="font-size:8px;color:'+C.muted+';margin-top:1px;">'+r.d+'</div></div>';
    });
    out+='</div>';
  } else if(mode===1){
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','L2 WEIGHT DECAY');
    out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
      'Loss = Data loss + <span style="color:'+C.yellow+';">\u03bb</span>/2\u00b7||W||&#178;<br>'
      +'\u2202Loss/\u2202W = \u2202DataLoss/\u2202W + <span style="color:'+C.yellow+';">\u03bb</span>W<br>'
      +'Update: W \u2190 W\u00b7(1\u2212\u03b1<span style="color:'+C.yellow+';">\u03bb</span>) \u2212 \u03b1\u00b7\u2202L/\u2202W'
    );
    out+=div('font-size:9px;color:'+C.muted+';margin-top:8px;line-height:1.8;',
      '<span style="color:'+C.green+';font-weight:700;">Effect</span>: each step, weights shrink by factor (1\u2212\u03b1\u03bb). '
      +'Large weights that do not reduce loss are penalised. '
      +'Prevents any single pathway from dominating. '
      +'Typical \u03bb = 1e-4 to 1e-2.'
    );
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','ADAM vs SGD');
    [
      {n:'Vanilla SGD',   p:'Simple, predictable',         c2:'Slow; sensitive to LR',   c:C.dim},
      {n:'SGD+Momentum',  p:'Faster; escapes ravines',     c2:'One extra hyper (\u03b2)',    c:C.yellow},
      {n:'Adam',          p:'Adaptive LR; fast convergence',c2:'Can generalise worse',   c:C.blue},
      {n:'AdamW',         p:'Adam + proper weight decay',  c2:'Standard for Transformers',c:C.accent},
    ].forEach(function(r){
      out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;border-left:2px solid '+r.c+';">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.c+';">'+r.n+'</div>'
        +'<div style="font-size:8px;color:'+C.green+';margin-top:1px;">\u2714 '+r.p+'</div>'
        +'<div style="font-size:8px;color:'+C.red+';">\u2718 '+r.c2+'</div></div>';
    });
    out+='</div>';
  } else {
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','USE NEURAL NETWORKS WHEN');
    [
      {cond:'Images, text, audio, video',     ans:true},
      {cond:'n > 10k\u2013100k examples',     ans:true},
      {cond:'Feature engineering impractical',ans:true},
      {cond:'Transfer learning available',    ans:true},
      {cond:'End-to-end raw\u2192output',     ans:true},
      {cond:'Small tabular dataset',          ans:false},
      {cond:'Interpretability required',      ans:false},
      {cond:'Max accuracy on tabular',        ans:false},
    ].forEach(function(r){
      out+='<div style="display:flex;gap:6px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
        +'<span style="color:'+(r.ans?C.green:C.red)+';font-size:10px;font-weight:700;">'+(r.ans?'\u2714':'\u2718')+'</span>'
        +'<span style="font-size:9px;color:'+C.muted+';">'+r.cond+'</span></div>';
    });
    out+='</div>';
    out+='<div class="card" style="margin:0;">';
    out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','NN IN THE ERM PICTURE');
    [
      {m:'Linear Reg.',    l:'MSE',    o:'GD',          cvx:true},
      {m:'Logistic Reg.',  l:'BCE',    o:'GD',          cvx:true},
      {m:'SVM',            l:'Hinge',  o:'QP',          cvx:true},
      {m:'Naive Bayes',    l:'NLL',    o:'Counting',    cvx:true},
      {m:'Neural Nets',    l:'Any',    o:'Backprop GD', cvx:false},
    ].forEach(function(r){
      out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
        +'<div style="flex:1;color:'+(r.m==='Neural Nets'?C.accent:C.muted)+';font-weight:'+(r.m==='Neural Nets'?'700':'400')+';">'+r.m+'</div>'
        +'<div style="flex:0.6;color:'+C.yellow+';text-align:center;">'+r.l+'</div>'
        +'<div style="flex:0.7;color:'+C.blue+';text-align:center;">'+r.o+'</div>'
        +'<div style="flex:0.6;color:'+(r.cvx?C.green:C.red)+';text-align:right;">'+(r.cvx?'Convex':'Non-cvx')+'</div>'
        +'</div>';
    });
    out+='</div>';
  }
  out+='</div></div>';

  out+=insight('&#129504;','NNs Win on Unstructured Data; GBM Wins on Tabular',
    'On images, text, and audio, neural networks are unmatched because they learn <span style="color:'+C.accent+';font-weight:700;">hierarchical representations automatically</span> '
    +'(edges \u2192 shapes \u2192 objects). Feature engineering is impractical at this scale. '
    +'On <span style="color:'+C.yellow+';font-weight:700;">tabular data</span>, gradient-boosted trees (XGBoost, LightGBM) '
    +'typically win: they handle mixed feature types, missing values, and non-monotonic relationships natively. '
    +'The non-convex loss surface of NNs means gradient descent finds good \u2014 but not guaranteed optimal \u2014 solutions. '
    +'In practice, <span style="color:'+C.green+';font-weight:700;">flat minima generalise better</span> than sharp ones.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════════ */
var TABS=[
  '&#129504; Neuron &amp; Forward Pass',
  '&#9889; Activation Functions',
  '&#128257; Backpropagation',
  '&#128200; Loss Curves',
  '&#127942; Regularisation'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.purple+','+C.accent+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Neural Networks</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive walkthrough \u2014 from the single neuron and forward pass to backprop, loss curves and regularisation')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderForwardPass();
  else if(S.tab===1) html+=renderActivations();
  else if(S.tab===2) html+=renderBackprop();
  else if(S.tab===3) html+=renderLossCurves();
  else if(S.tab===4) html+=renderRegularisation();
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
        if(action==='tab')          {S.tab=idx;          render();}
        else if(action==='actFn')   {S.actFn=idx;        render();}
        else if(action==='actTab')  {S.actTab=idx;       render();}
        else if(action==='lossMode'){S.lossMode=idx;     render();}
        else if(action==='regMode') {S.regMode=idx;      render();}
        else if(action==='toggleLinear'){S.showLinear=!S.showLinear; render();}
        else if(action==='bpNext')  {if(S.bpStep<6){S.bpStep++;render();}}
        else if(action==='bpPrev')  {if(S.bpStep>0){S.bpStep--;render();}}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='w1')         {S.w1=val;                  render();}
        else if(action==='w2')    {S.w2=val;                  render();}
        else if(action==='b')     {S.b=val;                   render();}
        else if(action==='zVal')  {S.zVal=val;                render();}
        else if(action==='bpLR')  {S.bpLR=val;               render();}
        else if(action==='epoch') {S.epoch=Math.round(val);   render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

NN_VISUAL_HEIGHT = 1100