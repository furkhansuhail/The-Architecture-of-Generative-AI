"""
Self-contained HTML visual for the Reinforcement Learning path.
5 interactive tabs: RL Framework, Grid World (Value Iteration),
Algorithm Families, Key Algorithms, RLHF & Alignment.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(RL_PATH_VISUAL_HTML, height=RL_PATH_VISUAL_HEIGHT, scrolling=True)
"""

RL_PATH_VISUAL_HTML = r"""<!DOCTYPE html>
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

/* ─── STATE ─── */
var S={tab:0, gamma:0.92, gIter:25, taxSel:0, algSel:0, rlhfSel:0};

/* ─── HELPERS ─── */
function hex(c,a){
  var r=parseInt(c.slice(1,3),16),g=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
  return 'rgba('+r+','+g+','+b+','+a+')';}
function div(st,inner){return '<div style="'+st+'">'+inner+'</div>';}
function card(inner,extra){
  return '<div class="card" style="max-width:750px;margin:0 auto 14px;'+(extra||'')+'">'+inner+'</div>';}
function sectionTitle(t,s){
  return '<div class="section-title"><h2>'+t+'</h2><p>'+s+'</p></div>';}
function insight(icon,title,body){
  return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>';}
function btnSel(idx,cur,color,label,action){
  var on=idx===cur;
  return '<button data-action="'+action+'" data-idx="'+idx
    +'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+(on?hex(color,.15):C.card)+';border:1.5px solid '+(on?color:C.border)+';'
    +'color:'+(on?color:C.muted)+';cursor:pointer;transition:all .2s;margin:3px;">'+label+'</button>';}
function sliderRow(action,val,mn,mx,step,label,dec){
  var dv=(dec!==undefined)?val.toFixed(dec):val;
  return '<div style="display:flex;align-items:center;gap:12px;margin-top:10px;">'
    +'<div style="font-size:10px;color:'+C.muted+';width:72px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+mn+'" max="'+mx+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:56px;font-weight:700;">'+dv+'</div>'
    +'</div>';}
function statRow(label,val,color){
  return '<div style="display:flex;justify-content:space-between;font-size:10px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<span style="color:'+C.muted+';">'+label+'</span>'
    +'<span style="color:'+color+';font-weight:700;">'+val+'</span></div>';}
function svgBox(inner,w,h){
  var W=w||520,H=h||340;
  return '<svg width="100%" viewBox="0 0 '+W+' '+H+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+inner+'</svg>';}

/* ─── GRID WORLD: VALUE ITERATION ─── */
/* 5x5 grid. s = row*5 + col.
   Goal s24=(4,4)+10  Trap s11=(2,1)-5  Walls s6,s8,s17
   State layout:
     [ 0][ 1][ 2][ 3][ 4]
     [ 5][W6][ 7][W8][ 9]
     [10][T ][12][13][14]
     [15][16][W ][18][19]
     [20][21][22][23][ G]           */
var GOAL_S=24, TRAP_S=11, WALLS_S=[6,8,17];
var MOVES=[[-1,0],[1,0],[0,-1],[0,1]]; /* up down left right */
var ARROW_CH=['\u2191','\u2193','\u2190','\u2192'];

function gsNext(s,a){
  if(WALLS_S.indexOf(s)>=0||s===GOAL_S||s===TRAP_S) return s;
  var row=Math.floor(s/5), col=s%5;
  var nr=row+MOVES[a][0], nc=col+MOVES[a][1];
  if(nr<0||nr>=5||nc<0||nc>=5) return s;
  var ns=nr*5+nc;
  return WALLS_S.indexOf(ns)>=0 ? s : ns;}
function gsReward(ns){
  return ns===GOAL_S?10 : ns===TRAP_S?-5 : -0.1;}

function runVI(gamma,nIter){
  var V=new Array(25).fill(0);
  V[GOAL_S]=10; V[TRAP_S]=-5;
  for(var it=0;it<nIter;it++){
    var nV=V.slice();
    for(var s=0;s<25;s++){
      if(s===GOAL_S||s===TRAP_S||WALLS_S.indexOf(s)>=0) continue;
      var best=-1e9;
      for(var a=0;a<4;a++){
        var ns=gsNext(s,a);
        var q=gsReward(ns)+gamma*V[ns];
        if(q>best) best=q;}
      nV[s]=best;}
    V=nV;}
  return V;}

function bestAct(s,V){
  var best=-1e9,ba=0;
  for(var a=0;a<4;a++){
    var ns=gsNext(s,a);
    var q=gsReward(ns)+S.gamma*V[ns];
    if(q>best){best=q;ba=a;}}
  return ba;}

function valColor(v,vmin,vmax){
  var t=Math.max(0,Math.min(1,(v-vmin)/(vmax-vmin||1)));
  if(t<0.5){
    var tt=t*2;
    return 'rgb('+Math.round(239*(1-tt)+30*tt)+','+Math.round(68*(1-tt)+30*tt)+','+Math.round(68*(1-tt)+46*tt)+')';}
  else{
    var tt=(t-0.5)*2;
    return 'rgb('+Math.round(30*(1-tt)+74*tt)+','+Math.round(30*(1-tt)+222*tt)+','+Math.round(46*(1-tt)+128*tt)+')';}}

/* ═══════════════════════════════════════════════════
   TAB 0 — RL FRAMEWORK
═══════════════════════════════════════════════════ */
function renderLoop(){
  var g=S.gamma;
  var rewards=[5,-1,3,2,-2,4,1,-1,2,8];
  var G=0, terms=[];
  for(var k=0;k<rewards.length;k++){
    var gk=Math.pow(g,k), c=gk*rewards[k];
    G+=c; terms.push({r:rewards[k],gk:gk.toFixed(3),c:c.toFixed(2)});}

  var sv='';
  /* defs: arrowheads */
  sv+='<defs>'
    +'<marker id="oa" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><polygon points="0,0 7,3.5 0,7" fill="'+C.orange+'"/></marker>'
    +'<marker id="ga" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><polygon points="0,0 7,3.5 0,7" fill="'+C.green+'"/></marker>'
    +'</defs>';

  /* Environment box */
  sv+='<rect x="318" y="84" width="154" height="76" rx="10" fill="'+hex(C.blue,.1)+'" stroke="'+C.blue+'" stroke-width="1.5"/>';
  sv+='<text x="395" y="112" text-anchor="middle" fill="'+C.blue+'" font-size="12" font-weight="700" font-family="monospace">ENVIRONMENT</text>';
  sv+='<text x="395" y="128" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">P(s\'|s,a) — transitions</text>';
  sv+='<text x="395" y="143" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">R(s,a,s\') — rewards</text>';

  /* Agent box */
  sv+='<rect x="48" y="84" width="154" height="76" rx="10" fill="'+hex(C.accent,.1)+'" stroke="'+C.accent+'" stroke-width="1.5"/>';
  sv+='<text x="125" y="112" text-anchor="middle" fill="'+C.accent+'" font-size="12" font-weight="700" font-family="monospace">AGENT</text>';
  sv+='<text x="125" y="128" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">π(a|s) — policy</text>';
  sv+='<text x="125" y="143" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">V(s), Q(s,a) — values</text>';

  /* top arrow: action aₜ  Agent → Env */
  sv+='<path d="M 202 99 C 240 60 280 60 318 99" fill="none" stroke="'+C.orange+'" stroke-width="2" marker-end="url(#oa)"/>';
  sv+='<rect x="218" y="48" width="84" height="28" rx="5" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="260" y="62" text-anchor="middle" fill="'+C.orange+'" font-size="10" font-family="monospace" font-weight="700">action  aₜ</text>';
  sv+='<text x="260" y="73" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">aₜ ~ π(·|sₜ)</text>';

  /* bottom arrow: sₜ₊₁, rₜ  Env → Agent */
  sv+='<path d="M 318 145 C 280 185 240 185 202 145" fill="none" stroke="'+C.green+'" stroke-width="2" marker-end="url(#ga)"/>';
  sv+='<rect x="182" y="172" width="156" height="28" rx="5" fill="#0a0a0f" opacity="0.92"/>';
  sv+='<text x="260" y="186" text-anchor="middle" fill="'+C.green+'" font-size="10" font-family="monospace" font-weight="700">state sₜ₊₁,  reward rₜ</text>';
  sv+='<text x="260" y="198" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">next observation + scalar signal</text>';

  /* return formula strip */
  sv+='<rect x="64" y="228" width="392" height="38" rx="8" fill="'+hex(C.purple,.08)+'" stroke="'+hex(C.purple,.3)+'" stroke-width="1.2"/>';
  sv+='<text x="260" y="244" text-anchor="middle" fill="'+C.purple+'" font-size="10" font-family="monospace" font-weight="700">Return:  Gₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ··· = Σₖ γᵏ rₜ₊ₖ</text>';
  sv+='<text x="260" y="258" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Objective:  max  E_π [ Gₜ ]  from every starting state</text>';

  /* timestep labels */
  sv+='<text x="48" y="80" fill="'+C.dim+'" font-size="8.5" font-family="monospace">time step  t</text>';
  sv+='<text x="318" y="80" fill="'+C.dim+'" font-size="8.5" font-family="monospace">time step  t+1</text>';

  var out=sectionTitle('The RL Agent-Environment Loop','At each step: observe → decide → act → receive reward → update');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Core Interaction Loop')+svgBox(sv,520,280));
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','KEY QUANTITIES');
  [{lbl:'State  sₜ',val:'full observation',col:C.blue},
   {lbl:'Action  aₜ',val:'chosen from π(a|s)',col:C.orange},
   {lbl:'Reward  rₜ',val:'scalar env signal',col:C.green},
   {lbl:'Return  Gₜ',val:'Σ γᵏ rₜ₊ₖ',col:C.purple},
   {lbl:'Policy  π(a|s)',val:'maps states → actions',col:C.accent},
   {lbl:'Value  V(s)',val:'E[Gₜ | sₜ=s]',col:C.yellow},
   {lbl:'Q(s,a)',val:'E[Gₜ | sₜ=s, aₜ=a]',col:C.blue}
  ].forEach(function(r){out+=statRow(r.lbl,r.val,r.col);});
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','DISCOUNTED RETURN  (γ='+g.toFixed(2)+')');
  out+=sliderRow('rlGamma',g,0,0.99,0.01,'γ',2);
  out+='<div style="margin-top:12px;">';
  var absMax=terms.reduce(function(m,t){return Math.max(m,Math.abs(parseFloat(t.c)));},0.001);
  terms.slice(0,6).forEach(function(t,k){
    var frac=Math.abs(parseFloat(t.c))/absMax;
    var pos=parseFloat(t.c)>=0;
    out+='<div style="display:flex;align-items:center;gap:5px;margin:3px 0;font-size:9px;">'
      +'<span style="color:'+C.dim+';width:22px;">k='+k+'</span>'
      +'<span style="color:'+C.yellow+';font-family:monospace;width:38px;">'+t.gk+'</span>'
      +'<span style="color:'+C.dim+';width:8px;">×</span>'
      +'<span style="color:'+C.green+';font-family:monospace;width:24px;">'+(t.r>0?'+':'')+t.r+'</span>'
      +'<div style="flex:1;background:'+C.border+';border-radius:3px;height:12px;">'
      +'<div style="height:100%;width:'+(frac*100).toFixed(0)+'%;background:'+(pos?C.green:C.red)+';border-radius:3px;min-width:2px;"></div>'
      +'</div>'
      +'<span style="color:'+(pos?C.green:C.red)+';font-family:monospace;width:40px;text-align:right;">'+(pos?'+':'')+t.c+'</span>'
      +'</div>';});
  out+='<div style="border-top:1px solid '+C.border+';margin-top:6px;padding-top:5px;display:flex;justify-content:space-between;font-size:10px;">'
    +'<span style="color:'+C.muted+';">Total Gₜ (10 steps)</span>'
    +'<span style="color:'+C.accent+';font-weight:700;">'+(G>=0?'+':'')+G.toFixed(2)+'</span></div>';
  out+='</div></div></div></div>';

  out+=insight('🎯','The Fundamental Goal',
    'Find a policy π* that <span style="color:'+C.yellow+';font-weight:700;">maximises expected discounted return</span> from every state. '
    +'γ=0 → myopic (only next reward). γ→1 → far-sighted. '
    +'The return bars above show how γ discounts each future reward — '
    +'<span style="color:'+C.accent+';font-weight:700;">drag the slider</span> to see how future contributions decay.');
  return out;}

/* ═══════════════════════════════════════════════════
   TAB 1 — GRID WORLD & VALUE FUNCTIONS
═══════════════════════════════════════════════════ */
function renderGrid(){
  var V=runVI(S.gamma,S.gIter);
  var active=[];
  for(var i=0;i<25;i++){ if(WALLS_S.indexOf(i)<0) active.push(V[i]); }
  var vmin=Math.min.apply(null,active), vmax=Math.max.apply(null,active);

  var CS=52, GX=28, GY=14;
  var sv='';
  sv+='<defs><marker id="pa" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">'
    +'<polygon points="0,0 5,2.5 0,5" fill="'+C.yellow+'"/></marker></defs>';

  for(var s=0;s<25;s++){
    var row=Math.floor(s/5), col=s%5;
    var cx=GX+col*CS+2, cy=GY+row*CS+2;
    var cw=CS-4, ch=CS-4;
    var fill, strokeCol, sw=1.5;

    if(WALLS_S.indexOf(s)>=0){
      fill='#1a1a2e'; strokeCol=C.dim; sw=1;}
    else if(s===GOAL_S){
      fill=hex(C.green,.22); strokeCol=C.green; sw=2.5;}
    else if(s===TRAP_S){
      fill=hex(C.red,.22); strokeCol=C.red; sw=2.5;}
    else{
      fill=valColor(V[s],vmin,vmax); strokeCol='#1e1e2e';}

    sv+='<rect x="'+cx+'" y="'+cy+'" width="'+cw+'" height="'+ch+'" rx="5" fill="'+fill+'" stroke="'+strokeCol+'" stroke-width="'+sw+'"/>';

    var mx=cx+cw/2, my=cy+ch/2;
    if(WALLS_S.indexOf(s)>=0){
      sv+='<text x="'+mx+'" y="'+(my+4)+'" text-anchor="middle" fill="'+C.dim+'" font-size="14">■</text>';}
    else if(s===GOAL_S){
      sv+='<text x="'+mx+'" y="'+(my-4)+'" text-anchor="middle" fill="'+C.green+'" font-size="9.5" font-weight="700" font-family="monospace">GOAL</text>';
      sv+='<text x="'+mx+'" y="'+(my+9)+'" text-anchor="middle" fill="'+C.green+'" font-size="9" font-family="monospace">+10</text>';}
    else if(s===TRAP_S){
      sv+='<text x="'+mx+'" y="'+(my-4)+'" text-anchor="middle" fill="'+C.red+'" font-size="9.5" font-weight="700" font-family="monospace">TRAP</text>';
      sv+='<text x="'+mx+'" y="'+(my+9)+'" text-anchor="middle" fill="'+C.red+'" font-size="9" font-family="monospace">-5</text>';}
    else{
      sv+='<text x="'+mx+'" y="'+(my-5)+'" text-anchor="middle" fill="'+C.text+'" font-size="9" font-weight="700" font-family="monospace">'+V[s].toFixed(1)+'</text>';
      var ba=bestAct(s,V);
      sv+='<text x="'+mx+'" y="'+(my+9)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="14" font-family="monospace">'+ARROW_CH[ba]+'</text>';
      sv+='<text x="'+(cx+4)+'" y="'+(cy+10)+'" fill="'+C.dim+'" font-size="7" font-family="monospace">s'+s+'</text>';}
  }

  /* START marker */
  sv+='<rect x="'+(GX+2)+'" y="'+(GY+2)+'" width="'+(CS-4)+'" height="'+(CS-4)+'" rx="5" fill="none" stroke="'+C.accent+'" stroke-width="2.5" stroke-dasharray="4,3"/>';
  sv+='<text x="'+(GX+CS/2)+'" y="'+(GY-3)+'" text-anchor="middle" fill="'+C.accent+'" font-size="8" font-family="monospace" font-weight="700">START</text>';

  /* Color legend bar */
  var bx=GX+5*CS+10, by=GY+2, bh=5*CS-8, bw=10;
  for(var i=0;i<bh;i++){
    var t2=i/bh, vc=vmin+t2*(vmax-vmin);
    sv+='<rect x="'+bx+'" y="'+(by+bh-i)+'" width="'+bw+'" height="1" fill="'+valColor(vc,vmin,vmax)+'"/>';}
  sv+='<text x="'+(bx+bw+4)+'" y="'+(by+10)+'" fill="'+C.green+'" font-size="8" font-family="monospace">'+vmax.toFixed(1)+'</text>';
  sv+='<text x="'+(bx+bw+4)+'" y="'+(by+bh/2+4)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">0</text>';
  sv+='<text x="'+(bx+bw+4)+'" y="'+(by+bh+4)+'" fill="'+C.red+'" font-size="8" font-family="monospace">'+vmin.toFixed(1)+'</text>';
  sv+='<text x="'+bx+'" y="'+(by-4)+'" fill="'+C.muted+'" font-size="8" font-family="monospace">V(s)</text>';

  /* special cell legend */
  var lx=bx, ly=by+bh+18;
  [[C.green,'Goal'],[C.red,'Trap'],[C.dim,'Wall'],[C.yellow,'Arrow=π*']].forEach(function(p,i){
    sv+='<rect x="'+lx+'" y="'+(ly+i*16)+'" width="9" height="9" rx="2" fill="'+hex(p[0],.22)+'" stroke="'+p[0]+'" stroke-width="1"/>';
    sv+='<text x="'+(lx+12)+'" y="'+(ly+i*16+8)+'" fill="'+p[0]+'" font-size="7.5" font-family="monospace">'+p[1]+'</text>';});

  var out=sectionTitle('Grid World — Value Iteration','Cells coloured by V(s); yellow arrows show the greedy optimal policy π*');
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','5×5 Grid World')
    +svgBox(sv,520,290)
    +'<div style="margin-top:8px;">'
    +sliderRow('rlGamma',S.gamma,0.05,0.99,0.01,'γ discount',2)
    +sliderRow('rlIter',S.gIter,1,80,1,'iterations',0)
    +'</div>');
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','VALUE ITERATION STATS');
  out+=statRow('γ (discount)',S.gamma.toFixed(2),C.accent);
  out+=statRow('Iterations',S.gIter,C.yellow);
  out+=statRow('V(start s0)',V[0].toFixed(3),C.blue);
  out+=statRow('V(goal s24)',V[24].toFixed(1),C.green);
  out+=statRow('V(trap s11)',V[11].toFixed(1),C.red);
  out+=statRow('V range','['+vmin.toFixed(1)+', '+vmax.toFixed(1)+']',C.muted);
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','BELLMAN OPTIMALITY');
  [{eq:'V*(s) = max_a Q*(s,a)',col:C.accent},
   {eq:'Q*(s,a)= R + γΣ P V*(s\')',col:C.yellow},
   {eq:'π*(s) = argmax_a Q*(s,a)',col:C.purple}
  ].forEach(function(e){
    out+='<div style="padding:4px 8px;margin:3px 0;border-radius:5px;background:#08080d;border-left:3px solid '+e.col+';">'
      +'<div style="font-size:9px;font-family:monospace;color:'+e.col+';">'+e.eq+'</div></div>';});
  out+='<div style="margin-top:10px;font-size:9px;color:'+C.muted+';line-height:1.7;">'
    +'Try γ=0 → only immediate reward visible. '
    +'Increase γ to watch <span style="color:'+C.yellow+';">long-range planning emerge</span>. '
    +'Low iterations shows convergence in progress.</div>';
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GRID LAYOUT');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;',
    '5×5 = 25 states, actions: ↑↓←→<br>'
    +'<span style="color:'+C.green+';">Goal s24</span>: reward +10 (terminal)<br>'
    +'<span style="color:'+C.red+';">Trap s11</span>: reward −5 (terminal)<br>'
    +'<span style="color:'+C.dim+';">Walls s6,s8,s17</span>: impassable<br>'
    +'All other moves: step cost −0.1');
  out+='</div></div></div>';

  out+=insight('💡','Why Value Functions Are the Core of RL',
    'Every RL algorithm is essentially trying to estimate V(s) or Q(s,a). '
    +'Notice values <span style="color:'+C.green+';font-weight:700;">radiate outward from the goal</span> like heat diffusion. '
    +'States near the trap have negative values — the agent learns to avoid them even before reaching them. '
    +'This long-range credit assignment is what separates RL from reactive control.');
  return out;}

/* ═══════════════════════════════════════════════════
   TAB 2 — ALGORITHM TAXONOMY
═══════════════════════════════════════════════════ */
function renderTaxonomy(){
  var sel=S.taxSel;
  var allOn=(sel===0);
  var mfOn=(sel===0||sel===1), mbOn=(sel===0||sel===2);
  var offOn=(sel===0||sel===3), metaOn=(sel===0||sel===4), safeOn=(sel===0||sel===5);
  var sv='';

  function nd(x,y,w,h,col,text,sub,on){
    var fill=on?hex(col,.18):hex(col,.05);
    var border=on?col:hex(col,.2); var sw=on?2:1;
    sv+='<rect x="'+(x-w/2)+'" y="'+(y-h/2)+'" width="'+w+'" height="'+h+'" rx="7" fill="'+fill+'" stroke="'+border+'" stroke-width="'+sw+'"/>';
    sv+='<text x="'+x+'" y="'+(sub?(y-4):(y+4))+'" text-anchor="middle" fill="'+(on?col:C.dim)+'" font-size="9.5" font-weight="700" font-family="monospace">'+text+'</text>';
    if(sub) sv+='<text x="'+x+'" y="'+(y+9)+'" text-anchor="middle" fill="'+(on?hex(col,1):C.dim)+'" font-size="7.5" font-family="monospace">'+sub+'</text>';}
  function ed(x1,y1,x2,y2,col,on){
    sv+='<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+(on?col:C.dim)+'" stroke-width="'+(on?1.8:.7)+'" opacity="'+(on?1:.4)+'"/>';}

  /* ── Row 0: root ── */
  nd(260,24,170,26,C.accent,'REINFORCEMENT LEARNING',null,allOn);

  /* ── Row 1: Model-Free | Model-Based ── */
  ed(260,37,140,60,C.blue,mfOn); ed(260,37,380,60,C.purple,mbOn);
  nd(140,72,128,26,C.blue,'MODEL-FREE','no env model',mfOn);
  nd(380,72,128,26,C.purple,'MODEL-BASED','uses/learns model',mbOn);

  /* ── Row 2: branches ── */
  ed(140,85,82,108,C.blue,mfOn);  ed(140,85,198,108,C.orange,mfOn);
  nd(82,120,106,26,C.blue,'VALUE-BASED','learn Q(s,a)',mfOn);
  nd(198,120,106,26,C.orange,'POLICY GRADIENT','optimise π_θ',mfOn);

  ed(380,85,330,108,C.green,mbOn); ed(380,85,430,108,C.purple,mbOn);
  nd(330,120,90,26,C.green,'KNOWN MODEL','DP/Planning',mbOn);
  nd(430,120,90,26,C.purple,'LEARNED MODEL','neural model',mbOn);

  /* ── Row 3: algorithm nodes ── */
  ed(82,133,52,155,C.blue,mfOn);   ed(82,133,112,155,C.blue,mfOn);
  nd(52,165,80,22,C.blue,'Q-Learning',null,mfOn);
  nd(117,165,76,22,C.blue,'DQN/Rainbow',null,mfOn);

  ed(198,133,162,155,C.orange,mfOn); ed(198,133,234,155,C.orange,mfOn);
  nd(162,165,68,22,C.orange,'REINFORCE',null,mfOn);
  nd(238,165,76,22,C.orange,'PPO / TRPO',null,mfOn);

  /* actor-critic bridge */
  ed(82,175,160,198,C.yellow,(mfOn));
  ed(198,175,160,198,C.yellow,(mfOn));
  nd(160,208,100,24,C.yellow,'ACTOR-CRITIC','A2C · SAC · TD3',mfOn);

  ed(330,133,312,155,C.green,mbOn); ed(430,133,448,155,C.purple,mbOn);
  nd(312,165,82,22,C.green,'Val. Iteration',null,mbOn);
  nd(448,165,82,22,C.purple,'Dreamer / PlaNet',null,mbOn);
  ed(380,133,380,155,C.accent,mbOn);
  nd(380,165,78,22,C.accent,'MCTS/MuZero',null,mbOn);

  /* ── Special families row ── */
  sv+='<line x1="20" y1="240" x2="500" y2="240" stroke="'+C.dim+'" stroke-width=".8" stroke-dasharray="3,3"/>';
  sv+='<text x="260" y="252" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">SPECIAL FAMILIES</text>';

  [[64,270,96,26,C.red,'OFFLINE RL','CQL/IQL/DT',offOn],
   [168,270,94,26,C.purple,'META-RL','MAML/RL²',metaOn],
   [268,270,96,26,C.yellow,'HIERARCHICAL','Options/HIRO',allOn],
   [366,270,90,26,C.orange,'MULTI-AGENT','QMIX/MADDPG',allOn],
   [460,270,80,26,C.blue,'SAFE RL','CPO/Lagr.',safeOn]
  ].forEach(function(p){nd(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]);});

  /* ── IRL/RLHF row ── */
  sv+='<line x1="20" y1="302" x2="500" y2="302" stroke="'+C.dim+'" stroke-width=".8" stroke-dasharray="3,3"/>';
  sv+='<text x="260" y="313" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">LEARNING FROM DEMONSTRATIONS & HUMANS</text>';
  nd(142,330,148,26,C.green,'IMITATION / IRL','BC · DAgger · GAIL · AIRL',allOn);
  nd(352,330,148,26,C.accent,'RLHF & ALIGNMENT','RLHF · DPO · CAI · PRM',allOn);

  var out=sectionTitle('RL Algorithm Taxonomy','The complete family tree — click a filter to highlight sub-families');
  out+='<div style="display:flex;gap:4px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  out+=btnSel(0,sel,C.accent,'All','taxSel');
  out+=btnSel(1,sel,C.blue,'Model-Free','taxSel');
  out+=btnSel(2,sel,C.purple,'Model-Based','taxSel');
  out+=btnSel(3,sel,C.red,'Offline RL','taxSel');
  out+=btnSel(4,sel,C.purple,'Meta-RL','taxSel');
  out+=btnSel(5,sel,C.blue,'Safe RL','taxSel');
  out+='</div>';
  out+=card(svgBox(sv,520,354),'max-width:750px;');
  out+=insight('🌳','How to Read the Taxonomy',
    '<span style="color:'+C.blue+';font-weight:700;">Model-free</span>: learns directly from experience, no model needed (DQN, PPO, SAC). '
    +'<span style="color:'+C.purple+';font-weight:700;">Model-based</span>: uses a dynamics model to plan (Dreamer, MuZero) — far more sample efficient but risks model bias. '
    +'The <span style="color:'+C.yellow+';font-weight:700;">actor-critic</span> bridge combines value estimation with direct policy optimisation. '
    +'Every modern LLM (GPT-4, Claude) uses <span style="color:'+C.accent+';font-weight:700;">RLHF</span> from the bottom row.');
  return out;}

/* ═══════════════════════════════════════════════════
   TAB 3 — KEY ALGORITHMS
═══════════════════════════════════════════════════ */
var ALGS=[
  {name:'DQN',yr:'\'15',full:'Deep Q-Network',family:'Value-Based',policy:'Off-policy',
   action:'Discrete',se:4,st:3,sc:3,col:C.blue,
   innov:'Q-learning + CNN + experience replay + target network. First deep RL to achieve human-level Atari.',
   pros:'Handles raw pixels; replay buffer reuses data; stable with target net; widely studied',
   cons:'Discrete actions only; overestimates Q-values; replay buffer cannot be used with on-policy methods',
   eq:'Q(s,a) ← r + γ max_{a\'} Q_target(s\',a\')',paper:'Mnih et al., Nature 2015'},
  {name:'PPO',yr:'\'17',full:'Proximal Policy Optimisation',family:'Actor-Critic',policy:'On-policy',
   action:'Both',se:3,st:5,sc:5,col:C.orange,
   innov:'Clips the probability ratio r_θ = π_new/π_old to [1−ε, 1+ε] — simple, robust trust-region update.',
   pros:'Stable; simple; discrete + continuous; default for RLHF; excellent scaling to large distributed setups',
   cons:'On-policy → less sample efficient than SAC; requires many parallel envs; sensitive to reward normalisation',
   eq:'L = E[min(r_θ A, clip(r_θ, 1±ε) A)]',paper:'Schulman et al., arXiv 2017'},
  {name:'SAC',yr:'\'18',full:'Soft Actor-Critic',family:'Actor-Critic',policy:'Off-policy',
   action:'Continuous',se:5,st:4,sc:4,col:C.accent,
   innov:'Entropy-augmented reward: maximise E[r + α H(π)]. Automatic temperature α tuning. Off-policy.',
   pros:'State-of-the-art continuous control; sample efficient; robust hyperparameters; handles multi-modal policies',
   cons:'Continuous actions only; more complex than PPO; α tuning adds complexity; less used for RLHF',
   eq:'J(π) = E[ Σ γᵏ(r + α H(π(·|sₖ))) ]',paper:'Haarnoja et al., ICML 2018'},
  {name:'TD3',yr:'\'18',full:'Twin Delayed DDPG',family:'Actor-Critic',policy:'Off-policy',
   action:'Continuous',se:4,st:4,sc:4,col:C.purple,
   innov:'3 fixes to DDPG: (1) twin critics take min, (2) delayed actor updates, (3) target policy smoothing.',
   pros:'More stable than DDPG; strong benchmark; deterministic policy means repeatable behaviour',
   cons:'Deterministic → less exploration diversity; continuous only; may need more tuning than SAC',
   eq:'y = r + γ min(Q₁,Q₂)(s\', π(s\')+ε)',paper:'Fujimoto et al., ICML 2018'},
  {name:'A3C',yr:'\'16',full:'Async Advantage Actor-Critic',family:'Actor-Critic',policy:'On-policy',
   action:'Both',se:3,st:3,sc:5,col:C.green,
   innov:'Asynchronous workers each explore with different ε. Gradient accumulation via parameter server.',
   pros:'Fast wall-clock time on CPU; natural exploration diversity; no replay buffer needed',
   cons:'Asynchronous updates add noise; mostly superseded by synchronous A2C + vectorised environments',
   eq:'∇θ J ≈ ∇θ log π(a|s) · A,  A = Σγᵏr − V(s)',paper:'Mnih et al., ICML 2016'},
  {name:'Dreamer',yr:'\'23',full:'DreamerV3 (World Model)',family:'Model-Based',policy:'Off-policy (latent)',
   action:'Both',se:5,st:3,sc:4,col:C.yellow,
   innov:'Learns an RSSM latent world model. Actor-critic trained entirely in imagination via latent rollouts.',
   pros:'Extremely sample efficient; learns long-horizon tasks; DreamerV3 works across domains with fixed HPs',
   cons:'Model bias from learned dynamics; complex implementation; latent space may miss critical features',
   eq:'J = E_model[ Σ γᵏ r(zₖ,aₖ) ]  (imagined)',paper:'Hafner et al., ICLR 2021 / 2023'},
  {name:'CQL',yr:'\'20',full:'Conservative Q-Learning (Offline)',family:'Offline RL',policy:'Fixed dataset',
   action:'Both',se:5,st:4,sc:3,col:C.red,
   innov:'Adds a penalty to Q-training that minimises Q for OOD actions, preventing unsafe extrapolation.',
   pros:'No env interaction; strong on D4RL; principled OOD penalty; widely adopted offline RL baseline',
   cons:'Conservative → can underperform when dataset is suboptimal; hyperparameter sensitivity',
   eq:'min_Q  α·E_policy[Q] − E_data[Q] + TD-error',paper:'Kumar et al., NeurIPS 2020'}
];

function renderAlgorithms(){
  var ai=S.algSel, a=ALGS[ai];

  function bar(v,col,lbl){
    return '<div style="margin:5px 0;">'
      +'<div style="display:flex;justify-content:space-between;font-size:8.5px;margin-bottom:2px;">'
      +'<span style="color:'+C.muted+';">'+lbl+'</span>'
      +'<span style="color:'+col+';font-weight:700;">'+'★'.repeat(v)+'☆'.repeat(5-v)+'</span></div>'
      +'<div style="background:'+C.border+';border-radius:3px;height:7px;">'
      +'<div style="height:100%;width:'+(v/5*100)+'%;background:'+col+';border-radius:3px;transition:width .4s;"></div></div></div>';}

  var out=sectionTitle('Key Algorithms — Deep Dive','Select an algorithm to explore its mechanics, strengths, and tradeoffs');
  out+='<div style="display:flex;gap:3px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ALGS.forEach(function(alg,i){
    out+=btnSel(i,ai,alg.col,'▶ '+alg.name+' '+alg.yr,'algSel');});
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  /* left column */
  out+='<div style="flex:1 1 340px;">';
  out+='<div class="card" style="margin:0 0 12px;">';
  out+='<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:12px;">'
    +'<span style="font-size:20px;font-weight:800;color:'+a.col+';">'+a.name+'</span>'
    +'<span style="font-size:9.5px;color:'+C.muted+';">'+a.full+'</span></div>';
  out+=statRow('Family',a.family,a.col);
  out+=statRow('Policy type',a.policy,a.policy.startsWith('Off')?C.green:C.orange);
  out+=statRow('Action space',a.action,C.blue);
  out+=statRow('Paper',a.paper,C.dim);
  out+='<div style="margin-top:14px;">';
  out+=bar(a.se,C.green,'Sample Efficiency');
  out+=bar(a.st,C.blue,'Stability / Reliability');
  out+=bar(a.sc,C.purple,'Scalability');
  out+='</div></div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY UPDATE EQUATION');
  out+='<div style="padding:8px 10px;background:#08080d;border-radius:6px;border-left:3px solid '+a.col+';">'
    +'<div style="font-size:8.5px;font-family:monospace;color:'+a.col+';line-height:1.7;">'+a.eq+'</div></div>';
  out+='</div></div>';

  /* right column */
  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','CORE INNOVATION');
  out+='<div style="font-size:9.5px;color:'+C.text+';line-height:1.8;border-left:3px solid '+a.col+';padding-left:10px;">'+a.innov+'</div>';
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:6px;','STRENGTHS');
  out+='<div style="font-size:9px;color:'+C.green+';line-height:1.8;">✓ '+a.pros+'</div>';
  out+=div('font-size:10px;color:'+C.muted+';margin:8px 0 6px;','LIMITATIONS');
  out+='<div style="font-size:9px;color:'+C.red+';line-height:1.8;">✗ '+a.cons+'</div>';
  out+='</div>';

  /* mini comparison */
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','vs. OTHER ALGORITHMS');
  out+='<div style="display:flex;font-size:8.5px;font-weight:700;padding:3px 0;border-bottom:1px solid '+C.border+';">'
    +'<div style="flex:1;color:'+C.muted+';">Algo</div>'
    +'<div style="flex:1.2;color:'+C.green+';text-align:center;">SampleEff</div>'
    +'<div style="flex:1;color:'+C.blue+';text-align:center;">Stable</div>'
    +'<div style="flex:1.2;color:'+C.muted+';text-align:center;">Action</div>'
    +'</div>';
  ALGS.slice(0,5).filter(function(o){return o.name!==a.name;}).slice(0,4).forEach(function(o){
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';align-items:center;">'
      +'<div style="flex:1;color:'+o.col+';font-weight:700;">'+o.name+'</div>'
      +'<div style="flex:1.2;text-align:center;color:'+C.green+';">'+'★'.repeat(o.se)+'</div>'
      +'<div style="flex:1;text-align:center;color:'+C.blue+';">'+'★'.repeat(o.st)+'</div>'
      +'<div style="flex:1.2;text-align:center;color:'+C.muted+';font-size:7px;">'+o.action+'</div>'
      +'</div>';});
  out+='</div></div></div>';

  out+=insight('⚡','When to Use Which Algorithm',
    '<span style="color:'+C.blue+';font-weight:700;">DQN</span>: discrete actions, Atari-style games. '
    +'<span style="color:'+C.orange+';font-weight:700;">PPO</span>: general purpose, RLHF, discrete or continuous, production default. '
    +'<span style="color:'+C.accent+';font-weight:700;">SAC</span>: continuous control needing sample efficiency. '
    +'<span style="color:'+C.yellow+';font-weight:700;">Dreamer</span>: expensive simulators. '
    +'<span style="color:'+C.red+';font-weight:700;">CQL/IQL</span>: fixed dataset, no environment access.');
  return out;}

/* ═══════════════════════════════════════════════════
   TAB 4 — RLHF & LLM ALIGNMENT
═══════════════════════════════════════════════════ */
function renderRLHF(){
  var mode=S.rlhfSel;
  var sv='';
  var W=520, H=270;

  sv+='<defs>'
    +'<marker id="ra1" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><polygon points="0,0 7,3.5 0,7" fill="'+C.muted+'"/></marker>'
    +'<marker id="ra2" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><polygon points="0,0 7,3.5 0,7" fill="'+C.orange+'"/></marker>'
    +'</defs>';

  function pbox(x,y,w,h,col,label,sub,note){
    sv+='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="9" fill="'+hex(col,.12)+'" stroke="'+col+'" stroke-width="1.8"/>';
    var cx=x+w/2;
    sv+='<text x="'+cx+'" y="'+(y+h/2-(sub?8:3))+'" text-anchor="middle" fill="'+col+'" font-size="10" font-weight="700" font-family="monospace">'+label+'</text>';
    if(sub) sv+='<text x="'+cx+'" y="'+(y+h/2+5)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+sub+'</text>';
    if(note) sv+='<text x="'+cx+'" y="'+(y+h/2+17)+'" text-anchor="middle" fill="'+C.dim+'" font-size="7.5" font-family="monospace">'+note+'</text>';}
  function arr(x1,y1,x2,y2,col,lbl){
    var mid=col||C.muted;
    sv+='<line x1="'+x1+'" y1="'+y1+'" x2="'+(x2-6)+'" y2="'+y2+'" stroke="'+mid+'" stroke-width="1.5" marker-end="url(#ra1)"/>';
    if(lbl) sv+='<text x="'+((x1+x2)/2)+'" y="'+(y1-6)+'" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">'+lbl+'</text>';}

  if(mode===0){
    /* RLHF 3-stage pipeline */
    pbox(16, 28,136,68,C.blue,   '① SFT',    'Supervised FT',   'expert demos');
    arr(152,62,166,62,null,'base LLM');
    pbox(166,28,148,68,C.yellow, '② Reward Model','Human pref labels','pairwise ranking');
    arr(314,62,330,62,null,'scores');
    pbox(330,28,174,68,C.accent, '③ PPO Fine-tune','RL with RM score','+ KL penalty');

    sv+='<text x="260" y="116" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">──────────────  data flows  ──────────────</text>';

    /* Human labels */
    pbox(16,128,148,50,C.orange,'Human Labels','prefer A over B','pairwise ranking');
    sv+='<line x1="164" y1="153" x2="163" y2="153" stroke="'+C.orange+'" stroke-width="0"/>';
    arr(164,153,166,96,C.orange);

    /* reward model detail box */
    sv+='<rect x="166" y="124" width="148" height="58" rx="8" fill="'+hex(C.yellow,.05)+'" stroke="'+hex(C.yellow,.2)+'" stroke-width="1" stroke-dasharray="4,3"/>';
    sv+='<text x="240" y="140" text-anchor="middle" fill="'+C.yellow+'" font-size="9" font-family="monospace" font-weight="700">R_θ(prompt, response)</text>';
    sv+='<text x="240" y="153" text-anchor="middle" fill="'+C.muted+'" font-size="7.5" font-family="monospace">Bradley-Terry preference model</text>';
    sv+='<text x="240" y="164" text-anchor="middle" fill="'+C.dim+'" font-size="7.5" font-family="monospace">P(A≻B) = σ( R(A) − R(B) )</text>';
    sv+='<text x="240" y="175" text-anchor="middle" fill="'+C.dim+'" font-size="7.5" font-family="monospace">trained via cross-entropy on pairs</text>';

    /* PPO objective */
    pbox(16,200,488,58,C.accent,'PPO Objective',
      'max  E[ R_θ(y) − β · KL(π_RL || π_SFT) ]',
      'reward model score  minus  KL divergence from SFT reference — prevents reward hacking');

  } else {
    /* DPO pipeline */
    pbox(16,28,136,68,C.blue,'① SFT','Same as RLHF','expert demos');
    arr(152,62,182,62);

    /* crossed-out reward model */
    sv+='<rect x="182" y="28" width="148" height="68" rx="9" fill="'+hex(C.dim,.05)+'" stroke="'+C.dim+'" stroke-width="1" stroke-dasharray="5,4"/>';
    sv+='<text x="256" y="54" text-anchor="middle" fill="'+C.dim+'" font-size="10" font-family="monospace" font-weight="700">❌ No Reward Model</text>';
    sv+='<text x="256" y="68" text-anchor="middle" fill="'+C.dim+'" font-size="8" font-family="monospace">skipped entirely</text>';
    sv+='<text x="256" y="81" text-anchor="middle" fill="'+C.dim+'" font-size="7.5" font-family="monospace">implicit in the policy</text>';
    arr(330,62,350,62);

    pbox(350,28,154,68,C.purple,'② DPO Train','Direct on pref pairs','no RL loop needed');

    sv+='<text x="260" y="116" text-anchor="middle" fill="'+C.dim+'" font-size="8.5" font-family="monospace">──────────────  DPO objective  ──────────────</text>';

    sv+='<rect x="16" y="124" width="488" height="52" rx="8" fill="'+hex(C.purple,.07)+'" stroke="'+hex(C.purple,.3)+'" stroke-width="1.5"/>';
    sv+='<text x="260" y="142" text-anchor="middle" fill="'+C.purple+'" font-size="9.5" font-family="monospace" font-weight="700">L = −E[ log σ(β log π_θ(y_w)/π_ref(y_w) − β log π_θ(y_l)/π_ref(y_l)) ]</text>';
    sv+='<text x="260" y="157" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">y_w = preferred response,  y_l = rejected,  β = KL coefficient</text>';
    sv+='<text x="260" y="169" text-anchor="middle" fill="'+C.dim+'" font-size="7.5" font-family="monospace">implicit reward: r*(y) = β log π*(y)/π_ref(y) + β log Z</text>';

    pbox(16,192,488,56,C.purple,'Key Insight: The Reward is Implicit in the Policy',
      'No separate RM needed — reward is encoded in the ratio π_θ/π_ref.',
      'DPO derives a closed-form supervised objective from the optimal RLHF solution.');
  }

  var out=sectionTitle('RLHF & LLM Alignment','Reinforcement learning is what makes language models helpful, harmless, and honest');
  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;">';
  out+=btnSel(0,mode,C.accent,'🧠 RLHF (3-stage PPO)','rlhfSel');
  out+=btnSel(1,mode,C.purple,'⚡ DPO (Direct)','rlhfSel');
  out+='</div>';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',
        mode===0?'RLHF — InstructGPT/Claude Pipeline':'DPO — Direct Preference Optimisation  (Rafailov et al., 2023)')
    +svgBox(sv,W,H),'max-width:750px;');

  /* comparison table */
  out+='<div class="card" style="max-width:750px;margin:0 auto 14px;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','ALIGNMENT METHODS AT A GLANCE');
  out+='<div style="display:flex;font-size:9px;font-weight:700;padding:4px 0;border-bottom:1px solid '+C.border+';">'
    +'<div style="flex:2;color:'+C.muted+';">Method</div>'
    +'<div style="flex:1.5;color:'+C.muted+';text-align:center;">Reward Model</div>'
    +'<div style="flex:1.2;color:'+C.muted+';text-align:center;">RL Step</div>'
    +'<div style="flex:3;color:'+C.muted+';text-align:center;">Key Advantage</div>'
    +'</div>';
  [{m:'RLHF (PPO)',rm:'✓ Explicit',rl:'✓ Full PPO',adv:'Flexible; handles complex reward landscapes',cm:C.accent},
   {m:'DPO',rm:'✗ Implicit',rl:'✗ None',adv:'Simpler; stable; no reward hacking',cm:C.purple},
   {m:'RLAIF (CAI)',rm:'✓ AI Judge',rl:'✓ PPO',adv:'Scalable; no human labellers needed',cm:C.blue},
   {m:'Process RM',rm:'✓ Step-level',rl:'✓ PPO',adv:'Combats final-answer hacking',cm:C.yellow},
   {m:'RLHF + KL',rm:'✓ + penalty',rl:'✓ PPO',adv:'Prevents policy collapse / drift',cm:C.green}
  ].forEach(function(r){
    out+='<div style="display:flex;font-size:9px;padding:4px 0;border-bottom:1px solid '+C.border+';align-items:center;">'
      +'<div style="flex:2;color:'+r.cm+';font-weight:700;">'+r.m+'</div>'
      +'<div style="flex:1.5;text-align:center;color:'+(r.rm[0]==='✓'?C.green:r.rm[0]==='✗'?C.dim:C.yellow)+';">'+r.rm+'</div>'
      +'<div style="flex:1.2;text-align:center;color:'+(r.rl[0]==='✓'?C.orange:C.dim)+';">'+r.rl+'</div>'
      +'<div style="flex:3;color:'+C.muted+';font-size:8px;">'+r.adv+'</div>'
      +'</div>';});
  out+='</div>';

  out+=insight('🤖','Why RLHF Changed Everything',
    'RLHF is why ChatGPT, Claude, and Gemini feel helpful. '
    +'The core challenge: the reward model can be <span style="color:'+C.red+';font-weight:700;">hacked</span> — the policy finds high-reward outputs that aren\'t actually good. '
    +'The <span style="color:'+C.yellow+';font-weight:700;">KL penalty</span> prevents the LLM from drifting too far from the original SFT model. '
    +'<span style="color:'+C.purple+';font-weight:700;">DPO</span> elegantly eliminates the RL loop by showing the optimal RLHF policy has a '
    +'<span style="color:'+C.purple+';font-weight:700;">closed-form</span> supervised objective — making alignment as simple as fine-tuning on preference pairs.');
  return out;}

/* ═══════════════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════════════ */
var TABS=[
  '&#127863; RL Framework',
  '&#128506; Grid World',
  '&#127795; Algorithm Map',
  '&#9889; Key Algorithms',
  '&#129302; RLHF'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.accent+','+C.purple+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">'
    +'Reinforcement Learning</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive visual walkthrough &#8212; from the RL loop &amp; value functions to modern RLHF')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';});
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0) html+=renderLoop();
  else if(S.tab===1) html+=renderGrid();
  else if(S.tab===2) html+=renderTaxonomy();
  else if(S.tab===3) html+=renderAlgorithms();
  else if(S.tab===4) html+=renderRLHF();
  html+='</div></div>';
  return html;}

function render(){
  document.getElementById('app').innerHTML=renderApp();
  bindEvents();}

function bindEvents(){
  document.querySelectorAll('[data-action]').forEach(function(el){
    var action=el.getAttribute('data-action');
    var idx=parseInt(el.getAttribute('data-idx'));
    var tag=el.tagName.toLowerCase();
    if(tag==='button'){
      el.addEventListener('click',function(){
        if(action==='tab')          {S.tab=idx;    render();}
        else if(action==='taxSel')  {S.taxSel=idx; render();}
        else if(action==='algSel')  {S.algSel=idx; render();}
        else if(action==='rlhfSel') {S.rlhfSel=idx;render();}});
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='rlGamma')     {S.gamma=val; render();}
        else if(action==='rlIter') {S.gIter=val; render();}});
    }});}

render();

/* ─── CTRL+SCROLL ZOOM ─── */
var ZOOM=1.0, zoomToast=null;
function applyZoom(){
  document.body.style.zoom=ZOOM;
  clearTimeout(zoomToast);
  var t=document.getElementById('zoom-toast');
  if(!t){
    t=document.createElement('div');
    t.id='zoom-toast';
    t.style.cssText='position:fixed;bottom:20px;right:20px;background:#12121a;border:1px solid #4ecdc4;'
      +'color:#4ecdc4;font-family:monospace;font-size:12px;font-weight:700;padding:8px 14px;'
      +'border-radius:8px;z-index:9999;pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(t);}
  t.textContent='zoom '+Math.round(ZOOM*100)+'%';
  t.style.opacity='1';
  zoomToast=setTimeout(function(){t.style.opacity='0';},1200);}
document.addEventListener('wheel',function(e){
  if(!e.ctrlKey&&!e.metaKey) return;
  e.preventDefault();
  ZOOM=Math.min(3.0,Math.max(0.4,Math.round((ZOOM+(e.deltaY>0?-0.05:0.05))*100)/100));
  applyZoom();},{passive:false});

</script>
</body>
</html>"""

RL_PATH_VISUAL_HEIGHT = 1350