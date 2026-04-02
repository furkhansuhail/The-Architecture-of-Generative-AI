"""
Self-contained HTML visual for Naive Bayes.
5 interactive tabs: Bayes' Theorem & Belief Update, Three NB Variants,
Training via MLE & Laplace Smoothing, Spam Classification Step-by-Step,
NB vs Logistic Regression.
Pure vanilla HTML/JS — zero CDN dependencies.
Embed via: st.components.v1.html(NB_VISUAL_HTML, height=NB_VISUAL_HEIGHT, scrolling=True)
"""

NB_VISUAL_HTML = r"""<!DOCTYPE html>
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
    +'<div style="font-size:10px;color:'+C.muted+';width:90px;text-align:right;">'+label+'</div>'
    +'<input type="range" data-action="'+action+'" min="'+min+'" max="'+max+'" step="'+step+'" value="'+val+'" style="flex:1;">'
    +'<div style="font-size:10px;color:'+C.accent+';width:52px;font-weight:700;">'+dv+'</div>'
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
function mono(s,col){return '<span style="font-family:monospace;color:'+(col||C.accent)+';">'+s+'</span>';}

/* ─── SVG SCAFFOLD ─── */
var VW=440,VH=280,PL=46,PR=16,PT=16,PB=38;
var PW=VW-PL-PR, PH=VH-PT-PB;
function sx(x,xmax){return PL+((x)/(xmax||10))*PW;}
function sy(y,ymax){return PT+PH-((y)/(ymax||10))*PH;}
function plotAxes(xl,yl,xmax,ymax,xticks,yticks){
  var xm=xmax||10,ym=ymax||10;
  var xts=xticks||[0,2,4,6,8,10],yts=yticks||[0,2,4,6,8,10];
  var o='';
  xts.forEach(function(v){o+='<line x1="'+sx(v,xm).toFixed(1)+'" y1="'+PT+'" x2="'+sx(v,xm).toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  yts.forEach(function(v){o+='<line x1="'+PL+'" y1="'+sy(v,ym).toFixed(1)+'" x2="'+(PL+PW)+'" y2="'+sy(v,ym).toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';});
  o+='<line x1="'+PL+'" y1="'+PT+'" x2="'+PL+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  o+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+(PT+PH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  xts.forEach(function(v){o+='<text x="'+sx(v,xm).toFixed(1)+'" y="'+(PT+PH+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
  yts.forEach(function(v){o+='<text x="'+(PL-6)+'" y="'+(sy(v,ym)+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v+'</text>';});
  o+='<text x="'+(PL+PW/2)+'" y="'+(VH-4)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+xl+'</text>';
  o+='<text x="10" y="'+(PT+PH/2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace" transform="rotate(-90,10,'+(PT+PH/2)+')">'+yl+'</text>';
  return o;
}

/* ─── STATE ─── */
var S={
  tab:0,
  priorSpam:0.3,    /* Bayes tab: P(spam) prior */
  pFreeSpam:0.8,    /* Bayes tab: P(FREE|spam) */
  pFreeHam:0.1,     /* Bayes tab: P(FREE|ham) */
  showMeeting:false,/* Bayes tab: add second word */
  variant:0,        /* Variants tab: 0=Gaussian,1=Multinomial,2=Bernoulli */
  gnbFeature:0,     /* Variants tab: which GNB feature to show */
  alpha:1,          /* Training tab: Laplace smoothing */
  spamStep:0,       /* Spam tab: 0..5 words added */
  nbVsLR:0          /* Comparison tab: 0=small data,1=large data */
};

/* ══════════════════════════════════════════════════════════
   TAB 0 — BAYES' THEOREM & BELIEF UPDATE
══════════════════════════════════════════════════════════ */
function renderBayes(){
  var ps=S.priorSpam;
  var ph=1-ps;
  var pfs=S.pFreeSpam;
  var pfh=S.pFreeHam;
  var showM=S.showMeeting;

  /* Bayes update for "FREE" */
  var unnormSpam=ps*pfs;
  var unnormHam=ph*pfh;
  var norm=unnormSpam+unnormHam;
  var postSpam1=unnormSpam/norm;
  var postHam1=unnormHam/norm;

  /* Update again for "Meeting" (P(Mtg|spam)=0.1, P(Mtg|ham)=0.4) */
  var pmts=0.1, pmth=0.4;
  var unnormSpam2=unnormSpam*pmts;
  var unnormHam2=unnormHam*pmth;
  var norm2=unnormSpam2+unnormHam2;
  var postSpam2=unnormSpam2/norm2;
  var postHam2=unnormHam2/norm2;

  var finalSpam=showM?postSpam2:postSpam1;
  var finalHam=showM?postHam2:postHam1;
  var verdict=finalSpam>finalHam?'SPAM':'NOT SPAM';
  var verdictCol=finalSpam>finalHam?C.red:C.green;

  /* ─── belief bar chart SVG ─── */
  var BW=440, BH=200;
  var bPL=90, bPT=15, bPB=20, bPR=16;
  var bPW=BW-bPL-bPR;
  var barH=28, barGap=12;
  var sv='';

  /* axes */
  sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(BH-bPB)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  [0,0.25,0.5,0.75,1.0].forEach(function(v){
    var bx=bPL+v*bPW;
    sv+='<line x1="'+bx.toFixed(1)+'" y1="'+bPT+'" x2="'+bx.toFixed(1)+'" y2="'+(BH-bPB)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+bx.toFixed(1)+'" y="'+(BH-bPB+12)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8.5">'+(v*100).toFixed(0)+'%</text>';
  });

  var stages=[
    {label:'Prior',    spam:ps,         ham:ph},
    {label:'After\nFREE', spam:postSpam1, ham:postHam1},
  ];
  if(showM) stages.push({label:'After\nMeeting',spam:postSpam2,ham:postHam2});

  var rowH=barH*2+barGap+8;
  stages.forEach(function(stage,si){
    var baseY=bPT+8+si*rowH;
    var labelY=baseY+barH/2+4;

    /* row label */
    var lparts=stage.label.split('\n');
    lparts.forEach(function(lp,li){
      sv+='<text x="'+(bPL-5)+'" y="'+(baseY+barH/2+(li-lparts.length/2+0.5)*11).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="9" font-family="monospace">'+lp+'</text>';
    });

    /* spam bar */
    var sw=stage.spam*bPW;
    sv+='<rect x="'+bPL+'" y="'+baseY+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(C.red,0.08)+'"/>';
    sv+='<rect x="'+bPL+'" y="'+baseY+'" width="'+sw.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(C.red,0.75)+'"/>';
    if(sw>24) sv+='<text x="'+(bPL+sw-4).toFixed(1)+'" y="'+(baseY+barH/2+4)+'" text-anchor="end" fill="#0a0a0f" font-size="9" font-weight="700">'+(stage.spam*100).toFixed(0)+'%</text>';
    sv+='<text x="'+(bPL+bPW+4)+'" y="'+(baseY+barH/2+4)+'" fill="'+C.red+'" font-size="8.5" font-weight="700">spam</text>';

    /* ham bar */
    var by2=baseY+barH+3;
    var hw=stage.ham*bPW;
    sv+='<rect x="'+bPL+'" y="'+by2+'" width="'+bPW+'" height="'+barH+'" rx="3" fill="'+hex(C.green,0.08)+'"/>';
    sv+='<rect x="'+bPL+'" y="'+by2+'" width="'+hw.toFixed(1)+'" height="'+barH+'" rx="3" fill="'+hex(C.green,0.75)+'"/>';
    if(hw>24) sv+='<text x="'+(bPL+hw-4).toFixed(1)+'" y="'+(by2+barH/2+4)+'" text-anchor="end" fill="#0a0a0f" font-size="9" font-weight="700">'+(stage.ham*100).toFixed(0)+'%</text>';
    sv+='<text x="'+(bPL+bPW+4)+'" y="'+(by2+barH/2+4)+'" fill="'+C.green+'" font-size="8.5" font-weight="700">ham</text>';

    /* arrow to next */
    if(si<stages.length-1){
      var ay=baseY+rowH-4;
      sv+='<text x="'+(bPL+bPW/2)+'" y="'+ay+'" text-anchor="middle" fill="'+C.yellow+'" font-size="11">\u00d7 evidence</text>';
    }
  });

  var out=sectionTitle("Bayes' Theorem & Belief Update","Prior \u00d7 Likelihood \u2192 Posterior — each word of evidence shifts our belief");

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  var chartH=showM?190:145;
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Belief About an Email')
    +svgBox(sv,BW,chartH)
    +'<div style="margin-top:10px;">'
    +sliderRow('priorSpam',ps,0.05,0.95,0.01,'P(spam)',2)
    +sliderRow('pFreeSpam',pfs,0.01,0.99,0.01,'P(FREE|spam)',2)
    +sliderRow('pFreeHam',pfh,0.01,0.99,0.01,'P(FREE|ham)',2)
    +'</div>'
    +'<div style="margin-top:12px;display:flex;align-items:center;gap:10px;">'
    +'<button data-action="toggleMeeting" style="padding:7px 14px;border-radius:7px;font-size:9.5px;font-weight:700;'
    +'font-family:inherit;background:'+hex(C.yellow,showM?0.15:0.05)+';border:1.5px solid '+(showM?C.yellow:C.border)+';'
    +'color:'+(showM?C.yellow:C.muted)+';cursor:pointer;">'+(showM?'\u2714 ':'+ ')+'Add word: "Meeting"</button>'
    +div('font-size:8.5px;color:'+C.dim+';','P(Mtg|spam)=0.10, P(Mtg|ham)=0.40')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:10px;','COMPUTATION');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    'P(C|x) = <span style="color:'+C.blue+';">P(x|C)</span> \u00d7 <span style="color:'+C.orange+';">P(C)</span> / P(x)<br>'
    +'<br>'
    +'<span style="color:'+C.orange+';">P(spam)</span> = '+ps.toFixed(2)+'<br>'
    +'<span style="color:'+C.blue+';">P(FREE|spam)</span> = '+pfs.toFixed(2)+'<br>'
    +'Unnorm: '+ps.toFixed(2)+' \u00d7 '+pfs.toFixed(2)+' = <span style="color:'+C.red+';">'+unnormSpam.toFixed(4)+'</span><br>'
    +'<br>'
    +'<span style="color:'+C.orange+';">P(ham)</span> = '+ph.toFixed(2)+'<br>'
    +'<span style="color:'+C.blue+';">P(FREE|ham)</span> = '+pfh.toFixed(2)+'<br>'
    +'Unnorm: '+ph.toFixed(2)+' \u00d7 '+pfh.toFixed(2)+' = <span style="color:'+C.green+';">'+unnormHam.toFixed(4)+'</span><br>'
    +'<br>'
    +'P(spam|FREE) = <span style="color:'+C.red+';font-weight:700;">'+(postSpam1*100).toFixed(1)+'%</span><br>'
    +(showM?'P(spam|FREE,Mtg) = <span style="color:'+(postSpam2>0.5?C.red:C.green)+';font-weight:700;">'+(postSpam2*100).toFixed(1)+'%</span>':'')
  );
  out+=div('font-size:11px;font-weight:700;margin-top:10px;padding:10px 12px;border-radius:8px;'
    +'background:'+hex(verdictCol,0.12)+';border:1.5px solid '+verdictCol+';text-align:center;color:'+verdictCol+';',
    '&#128231; Verdict: '+verdict+' ('+( finalSpam*100).toFixed(1)+'%)'
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;',"BAYES' THEOREM ANATOMY");
  [
    {t:'Posterior',sym:'P(C|x)',   desc:'What we want: class prob. after seeing x',       col:C.accent},
    {t:'Likelihood',sym:'P(x|C)',  desc:'How probable is this evidence given class C?',   col:C.blue},
    {t:'Prior',     sym:'P(C)',    desc:'Class frequency before seeing any evidence',     col:C.orange},
    {t:'Evidence',  sym:'P(x)',    desc:'Normalising constant — same for all classes',    col:C.dim},
  ].forEach(function(r){
    out+='<div style="display:flex;gap:8px;padding:4px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="font-family:monospace;font-size:9px;color:'+r.col+';width:68px;flex-shrink:0;font-weight:700;">'+r.sym+'</div>'
      +'<div style="font-size:8.5px;color:'+C.muted+';">'+r.desc+'</div></div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHY DROP P(x)?');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'We compare P(spam|x) vs P(ham|x). Both share the same denominator P(x). '
    +'So we just compare unnormalised scores and pick the largest \u2014 this is the '
    +'<span style="color:'+C.accent+';font-weight:700;">MAP decision rule</span>.'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#127922;','Multiplying Probabilities = Adding Log-Probabilities',
    'With many features, multiplying tiny probabilities causes numerical underflow to zero. '
    +'Work in log-space: '
    +'<span style="color:'+C.accent+';font-family:monospace;">log P(C|x) = log P(C) + \u03a3 log P(x\u2c7c|C)</span>. '
    +'Products become sums \u2014 <span style="color:'+C.green+';font-weight:700;">numerically stable</span> for thousands of features. '
    +'Drag the sliders to see how each word flips the verdict.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 1 — THREE VARIANTS
══════════════════════════════════════════════════════════ */
/* Gaussian NB: Iris-inspired class-conditional distributions */
var GNB_DATA={
  features:['Sepal Length','Petal Length','Petal Width'],
  classes:['Setosa','Versicolor','Virginica'],
  cols:[C.blue,C.orange,C.purple],
  /* [class][feature] = {mu, sigma} */
  params:[
    [{mu:5.0,s:0.35},{mu:1.46,s:0.17},{mu:0.24,s:0.11}],
    [{mu:5.94,s:0.52},{mu:4.26,s:0.47},{mu:1.33,s:0.20}],
    [{mu:6.59,s:0.64},{mu:5.55,s:0.55},{mu:2.03,s:0.27}]
  ]
};
function gaussian(x,mu,sigma){
  return Math.exp(-0.5*Math.pow((x-mu)/sigma,2))/(sigma*Math.sqrt(2*Math.PI));
}

/* Multinomial NB: word count example */
var MNB_DATA={
  vocab:['FREE','offer','urgent','meeting','report','dear'],
  /* [class][word] = P(word|class) */
  theta:[
    [0.24,0.18,0.20,0.03,0.05,0.14], /* spam */
    [0.02,0.03,0.01,0.18,0.22,0.08]  /* ham */
  ],
  cols:[C.red,C.green],
  classes:['Spam','Ham']
};

/* Bernoulli NB: same but binary */
var BNB_DATA={
  vocab:['FREE','offer','urgent','meeting','report','dear'],
  theta:[
    [0.78,0.65,0.72,0.12,0.15,0.55],
    [0.08,0.12,0.04,0.68,0.75,0.30]
  ],
  cols:[C.red,C.green],
  classes:['Spam','Ham']
};

function renderVariants(){
  var vi=S.variant;
  var fi=S.gnbFeature;

  var out=sectionTitle('Three Naive Bayes Variants','Choose the variant that matches your data type');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  ['&#127381; Gaussian NB','&#128203; Multinomial NB','&#9681; Bernoulli NB'].forEach(function(lbl,i){
    out+=btnSel(i,vi,[C.blue,C.orange,C.purple][i],lbl,'variant');
  });
  out+='</div>';

  var sv='';

  if(vi===0){
    /* Gaussian: show PDF curves for selected feature */
    var xmin=3.5, xmax=8.5, xrange=xmax-xmin;
    sv=plotAxes('Feature value','P(x\u2c7c | class)',xmax-xmin,0.25,
      [0,1,2,3,4,5],[0,0.05,0.10,0.15,0.20,0.25]);

    GNB_DATA.classes.forEach(function(cls,ci){
      var p=GNB_DATA.params[ci][fi];
      var path='';
      for(var xv=xmin;xv<=xmax;xv+=0.05){
        var g=gaussian(xv,p.mu,p.s);
        var px=sx(xv-xmin,xrange), py=sy(Math.min(g,0.25),0.25);
        path+=(xv===xmin||path===''?'M':'L')+px.toFixed(1)+','+py.toFixed(1)+' ';
      }
      /* fill area */
      var firstPx=sx(0,xrange), lastPx=sx(xrange,xrange), baseY=(PT+PH);
      sv+='<path d="'+path+'L'+lastPx.toFixed(1)+','+baseY+' L'+firstPx.toFixed(1)+','+baseY+' Z" fill="'+hex(GNB_DATA.cols[ci],0.12)+'"/>';
      sv+='<path d="'+path+'" fill="none" stroke="'+GNB_DATA.cols[ci]+'" stroke-width="2.5"/>';
      /* mean line */
      var mupx=sx(p.mu-xmin,xrange);
      sv+='<line x1="'+mupx.toFixed(1)+'" y1="'+PT+'" x2="'+mupx.toFixed(1)+'" y2="'+(PT+PH)+'" stroke="'+GNB_DATA.cols[ci]+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>';
      sv+='<text x="'+mupx.toFixed(1)+'" y="'+(PT+12)+'" text-anchor="middle" fill="'+GNB_DATA.cols[ci]+'" font-size="8">\u03bc='+p.mu+'</text>';
    });

  } else {
    /* Multinomial or Bernoulli bar charts */
    var data=vi===1?MNB_DATA:BNB_DATA;
    var BW2=440,BH2=180;
    var nbars=data.vocab.length, grpW=(BW2-60)/(nbars), barW2=grpW*0.38;
    var bPL2=50, bPT2=15, bPH2=130;
    sv='';
    /* grid */
    [0,0.2,0.4,0.6,0.8].forEach(function(v){
      var py=bPT2+bPH2*(1-v/(vi===1?0.3:0.9));
      sv+='<line x1="'+bPL2+'" y1="'+py.toFixed(1)+'" x2="'+(BW2-10)+'" y2="'+py.toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    });
    sv+='<line x1="'+bPL2+'" y1="'+bPT2+'" x2="'+bPL2+'" y2="'+(bPT2+bPH2)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
    sv+='<line x1="'+bPL2+'" y1="'+(bPT2+bPH2)+'" x2="'+(BW2-10)+'" y2="'+(bPT2+bPH2)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';

    data.vocab.forEach(function(word,wi){
      var gx=bPL2+4+wi*grpW;
      data.theta.forEach(function(cls,ci){
        var v=cls[wi];
        var maxV=vi===1?0.3:0.9;
        var bh=(v/maxV)*bPH2;
        var bx=gx+ci*(barW2+3);
        var by=bPT2+bPH2-bh;
        sv+='<rect x="'+bx.toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+barW2.toFixed(1)+'" height="'+bh.toFixed(1)+'" rx="2" fill="'+hex(data.cols[ci],0.75)+'"/>';
      });
      sv+='<text x="'+(gx+barW2).toFixed(1)+'" y="'+(bPT2+bPH2+13)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8" font-family="monospace">'+word+'</text>';
    });
    /* y-axis labels */
    [0,0.1,0.2,vi===1?0.3:0.8].forEach(function(v){
      var py=bPT2+bPH2*(1-v/(vi===1?0.3:0.9));
      sv+='<text x="'+(bPL2-4)+'" y="'+(py+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8" font-family="monospace">'+v.toFixed(1)+'</text>';
    });
    sv+='<text x="'+(bPL2+((BW2-bPL2-10)/2))+'" y="'+(bPT2+bPH2+26)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">'+(vi===1?'P(word | class)':'P(word present | class)')+'</text>';
    var H2=BH2; /* use this below */
    var tmpSVG=sv;
    sv='<svg width="100%" viewBox="0 0 '+BW2+' '+H2+'" style="background:#08080d;border-radius:8px;border:1px solid '+C.border+';display:block;">'+tmpSVG+'</svg>';
    /* legend */
    sv+=div('display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;',
      data.classes.map(function(cls,ci){
        return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
          +'<div style="width:10px;height:10px;border-radius:2px;background:'+data.cols[ci]+'"></div>'+cls+'</div>';
      }).join('')
    );
    var out2=sectionTitle('Three Naive Bayes Variants','Choose the variant that matches your data type');
    out2+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
    ['&#127381; Gaussian NB','&#128203; Multinomial NB','&#9681; Bernoulli NB'].forEach(function(lbl,i){
      out2+=btnSel(i,vi,[C.blue,C.orange,C.purple][i],lbl,'variant');
    });
    out2+='</div>';
    out2+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
    out2+='<div style="flex:1 1 380px;">';
    out2+=card(
      div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',(vi===1?'Multinomial':'Bernoulli')+' NB \u2014 Word Probabilities per Class')
      +sv
    );
    out2+='</div>';

    out2+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
    if(vi===1){
      out2+='<div class="card" style="margin:0;">';
      out2+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MULTINOMIAL NB');
      out2+=div('font-size:9px;color:'+C.muted+';line-height:1.9;',
        '<span style="color:'+C.orange+';">Features</span>: word <em>counts</em> per document<br>'
        +'<span style="color:'+C.blue+';">P(x|C)</span>: word frequency in class<br>'
        +'Training: <span style="color:'+C.accent+';font-family:monospace;">\u03b8\u2c7c\u1d9c = (count(j,c)+\u03b1) / (count(c)+\u03b1\u00b7p)</span><br>'
        +'<br>Best for: <span style="color:'+C.green+';">email spam, news categories, NLP</span>'
      );
      out2+='</div>';
      out2+='<div class="card" style="margin:0;">';
      out2+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','KEY OBSERVATION');
      out2+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
        '"FREE" \u2014 <span style="color:'+C.red+';">spam=24%</span> vs <span style="color:'+C.green+';">ham=2%</span><br>'
        +'"meeting" \u2014 <span style="color:'+C.red+';">spam=3%</span> vs <span style="color:'+C.green+';">ham=18%</span><br><br>'
        +'MNB counts how many times each word appears. '
        +'If "FREE" appears 3 times, its probability is <em>cubed</em>.'
      );
      out2+='</div>';
    } else {
      out2+='<div class="card" style="margin:0;">';
      out2+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','BERNOULLI NB');
      out2+=div('font-size:9px;color:'+C.muted+';line-height:1.9;',
        '<span style="color:'+C.orange+';">Features</span>: word <em>present/absent</em> (0/1)<br>'
        +'<span style="color:'+C.blue+';">P(x\u2c7c|C)</span>: probability feature j is 1<br>'
        +'<span style="color:'+C.red+';">P(x\u2c7c=0|C)</span>: 1\u2212\u03b8\u2c7c\u1d9c also contributes!<br>'
        +'<br>Best for: <span style="color:'+C.green+';">binary features, short text, presence/absence</span>'
      );
      out2+='</div>';
      out2+='<div class="card" style="margin:0;">';
      out2+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','MNB vs BNB KEY DIFF');
      out2+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
        'MNB ignores absent words.<br>'
        +'BNB explicitly uses: if "report" is <em>absent</em>, that is <span style="color:'+C.red+';">spam evidence</span> '
        +'(report absent in spam=85%). '
        +'BNB captures this; MNB does not.'
      );
      out2+='</div>';
    }
    out2+='<div class="card" style="margin:0;">';
    out2+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHICH VARIANT?');
    [
      {t:'Continuous data',    v:'Gaussian NB',     c:C.blue},
      {t:'Word count vectors', v:'Multinomial NB',  c:C.orange},
      {t:'Binary features',    v:'Bernoulli NB',    c:C.purple},
      {t:'Categorical',        v:'CategoricalNB',   c:C.green},
    ].forEach(function(r){
      out2+=statRow(r.t,r.v,r.c);
    });
    out2+='</div>';
    out2+='</div></div>';
    out2+=insight('&#9681;','Bernoulli NB Explicitly Punishes Absences',
      'In Bernoulli NB, every feature in the vocabulary contributes to the score, whether present or absent. '
      +'If "report" is absent from an email, '
      +'<span style="color:'+C.red+';font-weight:700;">P(report=0|spam)</span> is applied — a spam signal because legitimate emails often contain "report". '
      +'Multinomial NB only multiplies likelihoods for <em>present</em> words. '
      +'For short texts, BNB often outperforms MNB.'
    );
    return out2;
  }

  /* Gaussian path continues here */
  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;','Gaussian NB \u2014 Class-Conditional Distributions')
    +svgBox(sv)
    +'<div style="display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap;">'
    +GNB_DATA.features.map(function(f,i){return btnSel(i,fi,GNB_DATA.cols[0],f,'gnbFeat');}).join('')
    +'</div>'
    +'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    +GNB_DATA.classes.map(function(cls,ci){
      return '<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
        +'<div style="width:10px;height:2px;background:'+GNB_DATA.cols[ci]+'"></div>'+cls+'</div>';
    }).join('')
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GAUSSIAN NB');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;',
    '<span style="color:'+C.blue+';">P(x\u2c7c|C=c)</span> = N(x; \u03bc\u2c7c\u1d9c, \u03c3\u00b2\u2c7c\u1d9c)<br>'
    +'Training: compute class-conditional<br>'
    +'<span style="color:'+C.accent+';font-family:monospace;">\u03bc\u2c7c\u1d9c</span> = mean of feature j in class c<br>'
    +'<span style="color:'+C.accent+';font-family:monospace;">\u03c3\u00b2\u2c7c\u1d9c</span> = variance of feature j in class c<br><br>'
    +'No gradient descent \u2014 just mean and variance from data.'
  );
  out+='</div>';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;',GNB_DATA.features[fi].toUpperCase()+' PARAMS');
  GNB_DATA.classes.forEach(function(cls,ci){
    var p=GNB_DATA.params[ci][fi];
    out+=div('padding:4px 8px;margin:3px 0;border-radius:5px;border-left:3px solid '+GNB_DATA.cols[ci]+';',
      '<span style="font-size:9px;font-weight:700;color:'+GNB_DATA.cols[ci]+';">'+cls+'</span>'
      +'<div style="font-size:8.5px;color:'+C.muted+';margin-top:2px;">'
      +'\u03bc = '+p.mu.toFixed(2)+' &nbsp; \u03c3 = '+p.s.toFixed(2)+'</div>'
    );
  });
  out+='</div>';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','INTUITION');
  out+=div('font-size:9px;color:'+C.muted+';line-height:1.8;',
    'Overlapping distributions \u2192 ambiguous region. '
    +'Well-separated peaks \u2192 high confidence. '
    +'The wider the Gaussian (\u03c3), the more the class-conditional evidence spreads and overlaps.'
  );
  out+='</div>';
  out+='</div></div>';
  out+=insight('&#127381;','Gaussian NB = Parametric Density Estimation per Class',
    'For each feature j and each class c, Gaussian NB fits a 1D Gaussian N(\u03bc\u2c7c\u1d9c, \u03c3\u00b2\u2c7c\u1d9c). '
    +'This is equivalent to <span style="color:'+C.yellow+';font-weight:700;">Linear Discriminant Analysis (LDA)</span> when all class variances are equal, '
    +'and to <span style="color:'+C.purple+';font-weight:700;">Quadratic Discriminant Analysis (QDA)</span> when they differ. '
    +'The key assumption: each feature is modelled independently, '
    +'even though in reality height and weight are correlated.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 2 — TRAINING VIA MLE & LAPLACE SMOOTHING
══════════════════════════════════════════════════════════ */
/* Small training dataset: 8 emails */
var TRAIN_EMAILS=[
  {words:['FREE','offer','urgent'],         label:1, id:'E1'},
  {words:['FREE','FREE','urgent'],          label:1, id:'E2'},
  {words:['FREE','offer','dear'],           label:1, id:'E3'},
  {words:['dear','offer'],                  label:1, id:'E4'},
  {words:['meeting','report','dear'],       label:0, id:'E5'},
  {words:['report','meeting'],              label:0, id:'E6'},
  {words:['meeting','dear','report'],       label:0, id:'E7'},
  {words:['report','dear'],                 label:0, id:'E8'},
];
var VOCAB=['FREE','offer','urgent','meeting','report','dear'];
var ALPHA_DEF=1;

function getMNBParams(alpha){
  var counts=[{},{}];
  var totals=[0,0];
  VOCAB.forEach(function(w){counts[0][w]=0;counts[1][w]=0;});
  TRAIN_EMAILS.forEach(function(e){
    e.words.forEach(function(w){
      if(counts[e.label][w]!==undefined){counts[e.label][w]++;totals[e.label]++;}
    });
  });
  var p=VOCAB.length;
  var theta=VOCAB.map(function(w){
    return [
      (counts[0][w]+alpha)/(totals[0]+alpha*p),
      (counts[1][w]+alpha)/(totals[1]+alpha*p)
    ];
  });
  return {counts:counts,totals:totals,theta:theta};
}

function renderTraining(){
  var alpha=S.alpha;
  var params=getMNBParams(alpha);

  /* ─── bar chart of learned theta ─── */
  var BW=440,BH=170;
  var bPL=50, bPR=10, bPT=15, bPB=28;
  var bPW=BW-bPL-bPR, bPH=BH-bPT-bPB;
  var nv=VOCAB.length, grpW=bPW/nv, barW=grpW*0.37;
  var sv='';
  /* grid */
  [0,0.1,0.2,0.3].forEach(function(v){
    var py=bPT+bPH-v/0.35*bPH;
    sv+='<line x1="'+bPL+'" y1="'+py.toFixed(1)+'" x2="'+(bPL+bPW)+'" y2="'+py.toFixed(1)+'" stroke="'+C.border+'" stroke-width="0.5"/>';
    sv+='<text x="'+(bPL-4)+'" y="'+(py+3).toFixed(1)+'" text-anchor="end" fill="'+C.muted+'" font-size="8">'+(v*100).toFixed(0)+'</text>';
  });
  sv+='<line x1="'+bPL+'" y1="'+bPT+'" x2="'+bPL+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  sv+='<line x1="'+bPL+'" y1="'+(bPT+bPH)+'" x2="'+(bPL+bPW)+'" y2="'+(bPT+bPH)+'" stroke="'+C.dim+'" stroke-width="1.5"/>';
  VOCAB.forEach(function(w,wi){
    var gx=bPL+wi*grpW+4;
    params.theta[wi].forEach(function(th,ci){
      var bh=(th/0.35)*bPH;
      var bx=gx+ci*(barW+3);
      var by=bPT+bPH-bh;
      sv+='<rect x="'+bx.toFixed(1)+'" y="'+by.toFixed(1)+'" width="'+barW.toFixed(1)+'" height="'+bh.toFixed(1)+'" rx="2" fill="'+hex([C.green,C.red][ci],0.75)+'"/>';
    });
    sv+='<text x="'+(gx+barW).toFixed(1)+'" y="'+(bPT+bPH+12)+'" text-anchor="middle" fill="'+C.muted+'" font-size="8">'+w+'</text>';
  });
  sv+='<text x="'+(bPL+bPW/2)+'" y="'+(BH-2)+'" text-anchor="middle" fill="'+C.muted+'" font-size="9" font-family="monospace">Learned \u03b8\u2c7c\u1d9c (Laplace \u03b1='+alpha+')</text>';

  /* zero-freq highlight: urgent for ham */
  var urgentHam=params.counts[0]['urgent'];
  var noSmooth=urgentHam===0&&alpha===0;

  var out=sectionTitle('Training: MLE & Laplace Smoothing','Training = counting \u2014 one pass through data, no gradient descent');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';

  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Learned Word Probabilities (8 training emails)')
    +svgBox(sv,BW,BH)
    +'<div style="display:flex;gap:14px;margin-top:8px;">'
    +'<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
    +'<div style="width:10px;height:10px;border-radius:2px;background:'+C.green+'"></div>Ham</div>'
    +'<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
    +'<div style="width:10px;height:10px;border-radius:2px;background:'+C.red+'"></div>Spam</div>'
    +'</div>'
    +sliderRow('alpha',alpha,0,3,0.5,'alpha (\u03b1)',1)
    +'<div style="display:flex;justify-content:space-between;font-size:8.5px;color:'+C.muted+';margin-top:4px;padding:0 4px;">'
    +'<span style="color:'+C.red+';">\u2190 0 (zero-freq risk)</span>'
    +'<span style="color:'+C.green+';">3 (heavy smooth) \u2192</span></div>'
    +(noSmooth?'<div style="margin-top:8px;padding:7px 10px;border-radius:6px;background:'+hex(C.red,0.1)+';border:1px solid '+C.red+';">'
      +'<div style="font-size:9px;color:'+C.red+';font-weight:700;">\u26a0 Zero-frequency catastrophe!</div>'
      +'<div style="font-size:8px;color:'+C.muted+';margin-top:2px;">"urgent" never appears in ham training data. P(urgent|ham)=0. Any email with "urgent" gets P(spam)\u221d0.</div>'
      +'</div>':'')
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WORD COUNTS (training)');
  out+=div('font-size:9px;color:'+C.muted+';margin-bottom:6px;','4 ham + 4 spam emails');
  var header='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
    +'<div style="flex:1.2;color:'+C.muted+';">word</div>'
    +'<div style="flex:0.6;color:'+C.green+';text-align:center;">ham</div>'
    +'<div style="flex:0.6;color:'+C.red+';text-align:center;">spam</div>'
    +'<div style="flex:1;color:'+C.accent+';text-align:right;">\u03b8 ham (\u03b1='+alpha+')</div>'
    +'</div>';
  out+=header;
  VOCAB.forEach(function(w,wi){
    var hc=params.counts[0][w], sc=params.counts[1][w];
    var th=params.theta[wi][0];
    var zeroFlag=hc===0&&alpha===0;
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:1.2;color:'+(zeroFlag?C.red:C.muted)+';font-family:monospace;">'+w+(zeroFlag?' \u26a0':'')+'</div>'
      +'<div style="flex:0.6;color:'+C.green+';text-align:center;">'+hc+'</div>'
      +'<div style="flex:0.6;color:'+C.red+';text-align:center;">'+sc+'</div>'
      +'<div style="flex:1;color:'+(zeroFlag?C.red:C.accent)+';text-align:right;font-family:monospace;">'+(zeroFlag?'0 \u26a0':th.toFixed(3))+'</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','LAPLACE FORMULA');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:9px;color:'+C.muted+';line-height:2;font-family:monospace;',
    '\u03b8\u2c7c\u1d9c = (count(j,c) + <span style="color:'+C.yellow+';">\u03b1</span>) / (count(c) + <span style="color:'+C.yellow+';">\u03b1</span>\u00b7p)<br>'
    +'<span style="color:'+C.muted+';font-size:8px;">\u03b1=0: MLE (zero-freq problem)</span><br>'
    +'<span style="color:'+C.muted+';font-size:8px;">\u03b1=1: Laplace (add-one)</span><br>'
    +'<span style="color:'+C.muted+';font-size:8px;">\u03b1&gt;1: heavier regularisation</span>'
  );
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9881;&#65039;','One Pass = Full Training',
    'Multinomial NB training is just <span style="color:'+C.accent+';font-weight:700;">counting</span>. '
    +'Scan all training emails once, increment word counters per class, then divide by totals. '
    +'Training complexity: <span style="color:'+C.green+';font-family:monospace;">O(n\u00b7p)</span>. '
    +'No matrix inversions, no gradient descent, no hyperparameter to tune besides \u03b1. '
    +'This makes NB the <span style="color:'+C.yellow+';font-weight:700;">fastest-training model</span> in the supervised learning family. '
    +'Set \u03b1=0 to observe the zero-frequency catastrophe.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 3 — SPAM STEP-BY-STEP
══════════════════════════════════════════════════════════ */
/* Classify "FREE urgent meeting report" step by step */
var SPAM_WORDS=['FREE','urgent','meeting','report'];
var SPAM_PSPAM=0.4; /* prior */
/* log-likelihoods from learned params (alpha=1) */
var LL={
  spam:{FREE:Math.log(0.31),urgent:Math.log(0.22),meeting:Math.log(0.04),report:Math.log(0.06)},
  ham: {FREE:Math.log(0.04),urgent:Math.log(0.03),meeting:Math.log(0.28),report:Math.log(0.30)}
};

function renderSpam(){
  var step=S.spamStep; /* 0..4 = number of words included */
  var pspam=SPAM_PSPAM, pham=1-pspam;
  var logPostSpam=Math.log(pspam);
  var logPostHam=Math.log(pham);
  for(var i=0;i<step;i++){
    var w=SPAM_WORDS[i];
    logPostSpam+=LL.spam[w];
    logPostHam+=LL.ham[w];
  }
  /* softmax to get probabilities */
  var maxL=Math.max(logPostSpam,logPostHam);
  var eSpam=Math.exp(logPostSpam-maxL);
  var eHam=Math.exp(logPostHam-maxL);
  var pSpamFinal=eSpam/(eSpam+eHam);
  var pHamFinal=eHam/(eSpam+eHam);

  /* ─── progress SVG: probability bar updating word by word ─── */
  var PBW=440, PBH=200;
  var stages=[];
  /* compute probability at each step */
  var lps=Math.log(pspam), lph=Math.log(pham);
  stages.push({label:'Prior',ps:pspam,ph:pham});
  SPAM_WORDS.forEach(function(w,i){
    lps+=LL.spam[w]; lph+=LL.ham[w];
    var mx=Math.max(lps,lph);
    var es=Math.exp(lps-mx), eh=Math.exp(lph-mx);
    stages.push({label:w,ps:es/(es+eh),ph:eh/(es+eh)});
  });

  /* stacked horizontal bar chart */
  var sv='';
  var barH2=24, barGap2=6;
  var bPL2=60, bPT2=15, bPR2=50;
  var bPW2=PBW-bPL2-bPR2;
  stages.forEach(function(stg,si){
    var active=(si<=step);
    var by=bPT2+si*(barH2+barGap2);
    var spW=stg.ps*bPW2, hmW=stg.ph*bPW2;
    /* background bar */
    sv+='<rect x="'+bPL2+'" y="'+by+'" width="'+bPW2+'" height="'+barH2+'" rx="4" fill="'+hex(C.border,0.5)+'"/>';
    if(active){
      /* spam portion */
      sv+='<rect x="'+bPL2+'" y="'+by+'" width="'+spW.toFixed(1)+'" height="'+barH2+'" rx="4" fill="'+hex(C.red,0.75)+'"/>';
      /* ham portion */
      sv+='<rect x="'+(bPL2+spW).toFixed(1)+'" y="'+by+'" width="'+hmW.toFixed(1)+'" height="'+barH2+'" rx="4" fill="'+hex(C.green,0.75)+'"/>';
      if(spW>24) sv+='<text x="'+(bPL2+spW/2).toFixed(1)+'" y="'+(by+barH2/2+4)+'" text-anchor="middle" fill="#0a0a0f" font-size="9" font-weight="700">'+(stg.ps*100).toFixed(0)+'%</text>';
      if(hmW>24) sv+='<text x="'+(bPL2+spW+hmW/2).toFixed(1)+'" y="'+(by+barH2/2+4)+'" text-anchor="middle" fill="#0a0a0f" font-size="9" font-weight="700">'+(stg.ph*100).toFixed(0)+'%</text>';
    }
    /* stage label */
    var lColor=active?(si===step?C.text:C.muted):C.dim;
    sv+='<text x="'+(bPL2-4)+'" y="'+(by+barH2/2+4)+'" text-anchor="end" fill="'+lColor+'" font-size="9" font-family="monospace">'+stg.label+'</text>';
    /* current indicator */
    if(si===step){
      sv+='<rect x="'+bPL2+'" y="'+by+'" width="'+bPW2+'" height="'+barH2+'" rx="4" fill="none" stroke="'+C.accent+'" stroke-width="1.5"/>';
    }
  });
  /* legend */
  sv+='<rect x="'+(bPL2+bPW2+4)+'" y="'+bPT2+'" width="44" height="44" rx="4" fill="#0a0a0f" opacity="0.9"/>';
  sv+='<rect x="'+(bPL2+bPW2+8)+'" y="'+(bPT2+5)+'" width="10" height="10" rx="2" fill="'+C.red+'"/>';
  sv+='<text x="'+(bPL2+bPW2+21)+'" y="'+(bPT2+14)+'" fill="'+C.red+'" font-size="8">sp</text>';
  sv+='<rect x="'+(bPL2+bPW2+8)+'" y="'+(bPT2+20)+'" width="10" height="10" rx="2" fill="'+C.green+'"/>';
  sv+='<text x="'+(bPL2+bPW2+21)+'" y="'+(bPT2+29)+'" fill="'+C.green+'" font-size="8">hm</text>';

  var verdict2=pSpamFinal>0.5?'SPAM':'NOT SPAM';
  var verdictCol2=pSpamFinal>0.5?C.red:C.green;

  var out=sectionTitle('Spam Classification: Step by Step','Each word updates the posterior multiplicatively (= additively in log-space)');

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;','Classifying: "FREE urgent meeting report"')
    +svgBox(sv,PBW,PBH)
    +'<div style="margin-top:10px;display:flex;justify-content:center;gap:8px;">'
    +'<button data-action="spamPrev" style="padding:8px 18px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step>0?0.12:0.04)+';border:1.5px solid '+(step>0?C.accent:C.border)+';'
    +'color:'+(step>0?C.accent:C.dim)+';cursor:'+(step>0?'pointer':'default')+';">\u2190 Remove word</button>'
    +'<button data-action="spamNext" style="padding:8px 18px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;'
    +'background:'+hex(C.accent,step<4?0.12:0.04)+';border:1.5px solid '+(step<4?C.accent:C.border)+';'
    +'color:'+(step<4?C.accent:C.dim)+';cursor:'+(step<4?'pointer':'default')+';">Add word \u2192</button>'
    +'</div>'
    +'<div style="margin-top:10px;padding:10px 14px;border-radius:8px;background:'+hex(verdictCol2,0.1)+';border:1.5px solid '+verdictCol2+';text-align:center;">'
    +'<div style="font-size:11px;font-weight:800;color:'+verdictCol2+';">'+verdict2+'</div>'
    +'<div style="font-size:9px;color:'+C.muted+';margin-top:2px;">P(spam)='+(pSpamFinal*100).toFixed(1)+'%  |  P(ham)='+(pHamFinal*100).toFixed(1)+'%</div>'
    +'</div>'
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';
  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','LOG-SPACE COMPUTATION');
  out+=div('background:#08080d;border-radius:6px;padding:8px 10px;border:1px solid '+C.border+';font-size:8.5px;color:'+C.muted+';line-height:1.9;font-family:monospace;',
    'log P(spam|x) =<br>'
    +'&nbsp; log P(spam)&nbsp;&nbsp;= '+Math.log(pspam).toFixed(3)+'<br>'
    +SPAM_WORDS.slice(0,step).map(function(w){
      return '&nbsp; + log P('+w+'|sp) = '+LL.spam[w].toFixed(3);
    }).join('<br>')+'<br>'
    +(step>0?'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:'+C.red+';">'+logPostSpam.toFixed(3)+'</span>':'')
    +'<br><br>'
    +'log P(ham|x)&nbsp;&nbsp;=<br>'
    +'&nbsp; log P(ham)&nbsp;&nbsp;&nbsp;= '+Math.log(pham).toFixed(3)+'<br>'
    +SPAM_WORDS.slice(0,step).map(function(w){
      return '&nbsp; + log P('+w+'|hm) = '+LL.ham[w].toFixed(3);
    }).join('<br>')+'<br>'
    +(step>0?'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style="color:'+C.green+';">'+logPostHam.toFixed(3)+'</span>':'')
  );
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WORD CONTRIBUTIONS');
  out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
    +'<div style="flex:1.2;color:'+C.muted+';">word</div>'
    +'<div style="flex:1;color:'+C.red+';text-align:right;">log P|spam</div>'
    +'<div style="flex:1;color:'+C.green+';text-align:right;">log P|ham</div>'
    +'<div style="flex:0.7;color:'+C.muted+';text-align:right;">signal</div>'
    +'</div>';
  SPAM_WORDS.forEach(function(w,i){
    var ls=LL.spam[w], lh=LL.ham[w];
    var sig=ls-lh;
    var sigCol=sig>0?C.red:C.green;
    var active=i<step;
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';opacity:'+(active?1:0.35)+';">'
      +'<div style="flex:1.2;color:'+(active?C.text:C.muted)+';font-family:monospace;">'+w+'</div>'
      +'<div style="flex:1;color:'+C.red+';text-align:right;font-family:monospace;">'+ls.toFixed(2)+'</div>'
      +'<div style="flex:1;color:'+C.green+';text-align:right;font-family:monospace;">'+lh.toFixed(2)+'</div>'
      +'<div style="flex:0.7;color:'+sigCol+';text-align:right;font-family:monospace;font-size:8px;">'+(sig>0?'spam':'ham')+'</div>'
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#128231;','Why "meeting" Saves the Email',
    '"FREE" and "urgent" are strong spam signals. But each subsequent word updates the belief multiplicatively. '
    +'<span style="color:'+C.green+';font-weight:700;">"meeting"</span> is 7x more likely in ham than spam — adding it swings the posterior strongly toward ham. '
    +'<span style="color:'+C.green+';font-weight:700;">"report"</span> compounds this. '
    +'This is the power of Naive Bayes: each feature is a vote, and the votes are simply accumulated in log-space. '
    +'Press "Add word" to watch the belief flip in real time.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   TAB 4 — NB vs LOGISTIC REGRESSION
══════════════════════════════════════════════════════════ */
function renderComparison(){
  var mode=S.nbVsLR; /* 0=accuracy vs data size, 1=calibration */

  /* ─── accuracy vs training size curves ─── */
  function nbAcc(n){return 0.88-0.22*Math.exp(-0.015*n);}
  function lrAcc(n){return 0.94-0.55*Math.exp(-0.02*n);}
  function crossover(){
    /* find n where lrAcc(n) > nbAcc(n) meaningfully */
    for(var n=10;n<=500;n+=5){
      if(lrAcc(n)>nbAcc(n)+0.01) return n;
    }
    return 120;
  }
  var cx=crossover(); /* ~55-80 */

  /* ─── calibration comparison ─── */
  /* predicted probs vs actual frequency (reliability diagram) */
  var bins=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95];
  var nbActual= [0.22,0.18,0.28,0.30,0.44,0.60,0.71,0.79,0.91,0.97]; /* overconfident at extremes */
  var lrActual= [0.05,0.14,0.24,0.35,0.46,0.55,0.66,0.76,0.86,0.96]; /* near-diagonal */

  var sv='';
  if(mode===0){
    sv=plotAxes('Training set size','Accuracy',500,1.0,
      [0,100,200,300,400,500],[0,0.2,0.4,0.6,0.8,1.0]);
    /* crossover band */
    var cxpx=sx(cx,500);
    sv+='<rect x="'+sx(cx-15,500).toFixed(1)+'" y="'+PT+'" width="'+(sx(cx+15,500)-sx(cx-15,500)).toFixed(1)+'" height="'+PH+'" fill="'+hex(C.yellow,0.05)+'"/>';
    sv+='<text x="'+cxpx.toFixed(1)+'" y="'+(PT+12)+'" text-anchor="middle" fill="'+C.yellow+'" font-size="8">crossover~n='+cx+'</text>';
    /* NB curve */
    var nbPath='';
    for(var n=10;n<=500;n+=5){
      nbPath+=(n===10?'M':'L')+sx(n,500).toFixed(1)+','+sy(nbAcc(n),1.0).toFixed(1)+' ';
    }
    sv+='<path d="'+nbPath+'" fill="none" stroke="'+C.orange+'" stroke-width="2.5"/>';
    /* LR curve */
    var lrPath='';
    for(var n2=10;n2<=500;n2+=5){
      lrPath+=(n2===10?'M':'L')+sx(n2,500).toFixed(1)+','+sy(lrAcc(n2),1.0).toFixed(1)+' ';
    }
    sv+='<path d="'+lrPath+'" fill="none" stroke="'+C.blue+'" stroke-width="2.5"/>';
    /* labels */
    sv+='<text x="'+sx(480,500).toFixed(1)+'" y="'+sy(nbAcc(480),1.0).toFixed(1)+'" fill="'+C.orange+'" font-size="9" text-anchor="end">NB</text>';
    sv+='<text x="'+sx(480,500).toFixed(1)+'" y="'+sy(lrAcc(480)-0.03,1.0).toFixed(1)+'" fill="'+C.blue+'" font-size="9" text-anchor="end">LR</text>';
    /* NB better region */
    sv+='<text x="'+sx(30,500).toFixed(1)+'" y="'+(PT+28)+'" fill="'+C.orange+'" font-size="8">NB wins</text>';
    sv+='<text x="'+sx(220,500).toFixed(1)+'" y="'+(PT+28)+'" fill="'+C.blue+'" font-size="8">LR wins</text>';
  } else {
    /* calibration / reliability diagram */
    sv=plotAxes('Predicted Probability','Actual Frequency',1.0,1.0,
      [0,0.2,0.4,0.6,0.8,1.0],[0,0.2,0.4,0.6,0.8,1.0]);
    /* perfect calibration diagonal */
    sv+='<line x1="'+PL+'" y1="'+(PT+PH)+'" x2="'+(PL+PW)+'" y2="'+PT+'" stroke="'+C.dim+'" stroke-width="1.5" stroke-dasharray="6,4"/>';
    sv+='<text x="'+sx(0.85,1.0).toFixed(1)+'" y="'+sy(0.8,1.0).toFixed(1)+'" fill="'+C.dim+'" font-size="8">Perfect</text>';
    /* NB points */
    var nbPts='', lrPts='';
    bins.forEach(function(bv,bi){
      var nx=sx(bv,1.0), ny=sy(nbActual[bi],1.0);
      var lx=sx(bv,1.0), ly=sy(lrActual[bi],1.0);
      nbPts+=(bi===0?'M':'L')+nx.toFixed(1)+','+ny.toFixed(1)+' ';
      lrPts+=(bi===0?'M':'L')+lx.toFixed(1)+','+ly.toFixed(1)+' ';
      sv+='<circle cx="'+nx.toFixed(1)+'" cy="'+ny.toFixed(1)+'" r="4" fill="'+C.orange+'" stroke="#0a0a0f" stroke-width="1.5"/>';
      sv+='<circle cx="'+lx.toFixed(1)+'" cy="'+ly.toFixed(1)+'" r="4" fill="'+C.blue+'" stroke="#0a0a0f" stroke-width="1.5"/>';
    });
    sv+='<path d="'+nbPts+'" fill="none" stroke="'+C.orange+'" stroke-width="1.5" opacity="0.6"/>';
    sv+='<path d="'+lrPts+'" fill="none" stroke="'+C.blue+'" stroke-width="1.5" opacity="0.6"/>';
    sv+='<text x="'+sx(0.1,1.0).toFixed(1)+'" y="'+sy(nbActual[0]+0.08,1.0).toFixed(1)+'" fill="'+C.orange+'" font-size="9">NB</text>';
    sv+='<text x="'+sx(0.1,1.0).toFixed(1)+'" y="'+sy(lrActual[0]+0.13,1.0).toFixed(1)+'" fill="'+C.blue+'" font-size="9">LR</text>';
  }

  var out=sectionTitle('NB vs Logistic Regression','Generative vs Discriminative \u2014 two routes to the same linear boundary');

  out+='<div style="display:flex;gap:8px;justify-content:center;margin-bottom:16px;flex-wrap:wrap;">';
  out+=btnSel(0,mode,C.accent,'Accuracy vs Training Size','nbVsLR');
  out+=btnSel(1,mode,C.accent,'Calibration (Reliability)','nbVsLR');
  out+='</div>';

  out+='<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;max-width:750px;margin:0 auto 16px;">';
  out+='<div style="flex:1 1 380px;">';
  out+=card(
    div('font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;',
      mode===0?'Accuracy vs Training Set Size':'Reliability Diagram (calibration)')
    +svgBox(sv)
    +'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    +'<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
    +'<div style="width:12px;height:2px;background:'+C.orange+'"></div>Naive Bayes</div>'
    +'<div style="display:flex;align-items:center;gap:4px;font-size:8.5px;color:'+C.muted+';">'
    +'<div style="width:12px;height:2px;background:'+C.blue+'"></div>Logistic Regression</div>'
    +'</div>'
    +(mode===1?'<div style="margin-top:8px;font-size:8.5px;color:'+C.muted+';padding:6px 8px;background:#08080d;border-radius:5px;">'
      +'NB probabilities cluster near 0 and 1 \u2014 overconfident. '
      +'LR stays close to the diagonal (well-calibrated).</div>':'')
  );
  out+='</div>';

  out+='<div style="flex:1 1 280px;display:flex;flex-direction:column;gap:12px;">';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','GENERATIVE vs DISCRIMINATIVE');
  [
    {lbl:'Models',         nb:'P(x, C) = P(x|C)\u00b7P(C)',             lr:'P(C|x) directly'},
    {lbl:'Optimises',      nb:'likelihood P(x,C)',                        lr:'posterior P(C|x)'},
    {lbl:'Training',       nb:'counting (MLE)',                           lr:'gradient descent'},
    {lbl:'Speed',          nb:'very fast O(np)',                          lr:'slower (iterative)'},
    {lbl:'Small data',     nb:'better (independence=reg.)',               lr:'can overfit'},
    {lbl:'Large data',     nb:'worse (wrong model)',                      lr:'converges to truth'},
    {lbl:'Calibration',    nb:'poor (overconfident)',                     lr:'good'},
    {lbl:'Missing feats',  nb:'skip in product',                         lr:'needs imputation'},
  ].forEach(function(r){
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:0.8;color:'+C.dim+';">'+r.lbl+'</div>'
      +'<div style="flex:1;color:'+C.orange+';">\uD83C\uDFB2 '+r.nb+'</div>'
      +'</div>';
    out+='<div style="display:flex;font-size:8.5px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<div style="flex:0.8;color:'+C.dim+';">&nbsp;</div>'
      +'<div style="flex:1;color:'+C.blue+';"><span style="font-family:monospace;font-size:7px;">\u03c3</span> '+r.lr+'</div>'
      +'</div>';
  });
  out+='</div>';

  out+='<div class="card" style="margin:0;">';
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:8px;','WHEN TO USE NB');
  [
    {cond:'Very small data (n<100)',  ans:'NB',  c:C.orange},
    {cond:'High-dim sparse text',     ans:'NB',  c:C.orange},
    {cond:'Need fast online learning',ans:'NB',  c:C.orange},
    {cond:'Calibrated probabilities', ans:'LR',  c:C.blue},
    {cond:'Correlated features',      ans:'LR',  c:C.blue},
    {cond:'Large data available',     ans:'LR',  c:C.blue},
    {cond:'Correlated feats, sparse', ans:'Maybe NB',c:C.yellow},
  ].forEach(function(r){
    out+='<div style="display:flex;justify-content:space-between;font-size:9px;padding:3px 0;border-bottom:1px solid '+C.border+';">'
      +'<span style="color:'+C.muted+';">'+r.cond+'</span>'
      +'<span style="color:'+r.c+';font-weight:700;">'+r.ans+'</span></div>';
  });
  out+='</div>';
  out+='</div></div>';

  out+=insight('&#9889;','NB and LR Both Learn Linear Boundaries',
    'In binary classification, taking the log-odds ratio of NB gives: '
    +'<span style="color:'+C.accent+';font-family:monospace;">log P(C=1|x) / P(C=0|x) = w\u1d40x + b</span>. '
    +'This is exactly Logistic Regression\'s decision function \u2014 '
    +'both models are <span style="color:'+C.yellow+';font-weight:700;">linear classifiers</span> in disguise. '
    +'The difference: NB estimates weights <em>generatively</em> via counting; LR estimates them '
    +'<em>discriminatively</em> via gradient descent. '
    +'As n\u2192\u221e, LR converges to the true boundary; NB converges only if independence holds.'
  );
  return out;
}

/* ══════════════════════════════════════════════════════════
   ROOT RENDER
══════════════════════════════════════════════════════════ */
var TABS=[
  "&#127922; Bayes' Theorem",
  '&#9681; Three Variants',
  '&#9881;&#65039; Training & Smoothing',
  '&#128231; Spam Step-by-Step',
  '&#9889; NB vs LR'
];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;">';
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.purple+','+C.orange+');'
    +'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Naive Bayes</div>'
    +div('font-size:11px;color:'+C.muted+';margin-top:4px;',
      'Interactive walkthrough \u2014 from Bayes\' theorem and belief updates to MLE training, spam classification and the NB\u2013LR duality')
    +'</div>';
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  html+='<div class="fade">';
  if(S.tab===0)      html+=renderBayes();
  else if(S.tab===1) html+=renderVariants();
  else if(S.tab===2) html+=renderTraining();
  else if(S.tab===3) html+=renderSpam();
  else if(S.tab===4) html+=renderComparison();
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
        if(action==='tab')            {S.tab=idx;            render();}
        else if(action==='toggleMeeting'){S.showMeeting=!S.showMeeting; render();}
        else if(action==='variant')   {S.variant=idx;        render();}
        else if(action==='gnbFeat')   {S.gnbFeature=idx;     render();}
        else if(action==='nbVsLR')    {S.nbVsLR=idx;         render();}
        else if(action==='spamNext')  {if(S.spamStep<4){S.spamStep++;render();}}
        else if(action==='spamPrev')  {if(S.spamStep>0){S.spamStep--;render();}}
      });
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseFloat(this.value);
        if(action==='priorSpam')     {S.priorSpam=val;            render();}
        else if(action==='pFreeSpam'){S.pFreeSpam=val;            render();}
        else if(action==='pFreeHam') {S.pFreeHam=val;             render();}
        else if(action==='alpha')    {S.alpha=val;                 render();}
      });
    }
  });
}

render();
</script>
</body>
</html>"""

NB_VISUAL_HEIGHT = 1100