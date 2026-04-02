"""
Self-contained HTML for the Tokenization & Embeddings interactive walkthrough.
Covers: Pipeline, Tokenizers, Embedding Space, Positional Encoding,
Tokenizer Zoo, and Quirks & Gotchas.

Rewritten as pure vanilla HTML/CSS/JS — zero CDN dependencies, zero React,
zero Babel. Renders reliably in any Streamlit sandboxed iframe.

Embed in Streamlit via:
    st.components.v1.html(TOK_EMBED_HTML, height=TOK_EMBED_HEIGHT, scrolling=True)
"""

TOK_EMBED_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
button{cursor:pointer;font-family:inherit;}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#a78bfa;cursor:pointer;}
table{border-collapse:separate;border-spacing:0 4px;width:100%;}
th{text-align:left;padding:6px 10px;color:#3f3f46;font-size:8px;font-weight:700;}
td{padding:7px 10px;font-size:9px;}
@keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.fade{animation:fadeIn .3s ease both;}
.card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
.card.hl{border-color:#a78bfa;}
.section-title{text-align:center;margin-bottom:20px;}
.section-title h2{font-size:18px;font-weight:800;color:#e4e4e7;margin-bottom:4px;}
.section-title p{font-size:12px;color:#71717a;}
.insight{max-width:1100px;margin:16px auto 0;padding:16px 22px;background:rgba(167,139,250,.06);border-radius:10px;border:1px solid rgba(167,139,250,.2);}
.insight .ins-title{font-size:11px;font-weight:700;color:#a78bfa;margin-bottom:6px;}
.insight .ins-body{font-size:11px;color:#71717a;line-height:1.8;}
.tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
.tab-btn{padding:12px 18px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:11px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
.tab-btn.active{border-bottom-color:#a78bfa;color:#a78bfa;}
.tok-chip{display:inline-block;padding:3px 8px;border-radius:4px;font-size:9px;font-family:inherit;font-weight:700;}
.btn-sel{padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;transition:all .2s;border:1.5px solid #1e1e2e;background:#12121a;color:#71717a;}
.btn-sel.on{background:rgba(167,139,250,.12);}
.stat-box{text-align:center;min-width:90px;}
.stat-box .stat-lbl{font-size:8px;color:#71717a;margin-bottom:4px;letter-spacing:1px;}
.stat-box .stat-val{font-size:22px;font-weight:800;color:#a78bfa;}
.stat-box .stat-sub{font-size:8px;color:#3f3f46;margin-top:2px;}
.wrap{display:flex;flex-wrap:wrap;gap:10px;}
</style>
</head>
<body>
<div id="app" style="max-width:1400px;margin:0 auto;padding:24px 16px;"></div>

<script>
/* ═══════════════════════════════════════════════════════════
   COLOUR PALETTE
═══════════════════════════════════════════════════════════ */
var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",accent:"#a78bfa",blue:"#4ecdc4",purple:"#c084fc",yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",dim:"#3f3f46",red:"#ef4444",green:"#4ade80",cyan:"#38bdf8",pink:"#f472b6",orange:"#fb923c"};

/* ═══════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════ */
var S={
  tab:0,
  pipeStep:-1,
  tokSel:2, bpeStep:0,
  embVocab:50257, embDim:768,
  posSel:0, posSlider:64,
  zooSel:0,
  quirkSel:0,
};

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
function h(tag,attrs,inner){
  var s='<'+tag;
  for(var k in attrs) s+=' '+k+'="'+attrs[k]+'"';
  s+='>'+inner+'</'+tag+'>';
  return s;
}
function div(style,inner){ return '<div style="'+style+'">'+inner+'</div>'; }
function span(style,inner){ return '<span style="'+style+'">'+inner+'</span>'; }
function hex(color,alpha){ // color like "#a78bfa", alpha 0-1
  var r=parseInt(color.slice(1,3),16);
  var g=parseInt(color.slice(3,5),16);
  var b=parseInt(color.slice(5,7),16);
  return 'rgba('+r+','+g+','+b+','+alpha+')';
}
function card(inner,extra){ return '<div class="card'+(extra?' '+extra:'')+'" style="max-width:1100px;margin:0 auto 14px;">'+inner+'</div>'; }
function sectionTitle(title,sub){ return '<div class="section-title"><h2>'+title+'</h2><p>'+sub+'</p></div>'; }
function insight(icon,title,body){ return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>'; }
function btnSel(idx,current,color,label,action){
  var on=idx===current;
  var bg=on?hex(color,.12):'#12121a';
  var bc=on?color:'#1e1e2e';
  var tc=on?color:'#71717a';
  return '<button data-action="'+action+'" data-idx="'+idx+'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+bg+';border:1.5px solid '+bc+';color:'+tc+';cursor:pointer;transition:all .2s;">'+label+'</button>';
}
function chip(text,color){
  return '<span style="padding:3px 8px;border-radius:4px;font-size:9px;font-family:inherit;font-weight:700;background:'+hex(color,.15)+';border:1px solid '+hex(color,.5)+';color:'+color+';">'+text+'</span>';
}
function statBox(label,value,color,sub){
  return '<div class="stat-box"><div class="stat-lbl">'+label+'</div><div class="stat-val" style="color:'+color+';">'+value+'</div>'+(sub?'<div class="stat-sub">'+sub+'</div>':'')+'</div>';
}

/* ═══════════════════════════════════════════════════════════
   TAB 1 — PIPELINE
═══════════════════════════════════════════════════════════ */
function renderPipeline(){
  var stages=[
    {id:0,label:"Raw Text",    sub:'"The cat sat"',      color:C.muted},
    {id:1,label:"Tokenize",    sub:"BPE split",          color:C.blue},
    {id:2,label:"Token IDs",   sub:"[791,8415,7482]",    color:C.cyan},
    {id:3,label:"Add BOS",     sub:"[1,791,8415,7482]",  color:C.yellow},
    {id:4,label:"Embed Lookup",sub:"[3, 4096] floats",   color:C.accent},
    {id:5,label:"+ Positional",sub:"RoPE/sinusoidal",    color:C.purple},
    {id:6,label:"Transformer", sub:"32 layers → output", color:C.green},
  ];
  var details=[
    {title:'Raw string — the model cannot process this directly',body:'"The cat sat on the mat" — Neural networks only understand numbers. Text must be converted.',color:C.muted},
    {title:'BPE tokenizer splits text into subword pieces',body:'Tokens: "The" | "_cat" | "_sat" — Leading space is part of the token in GPT-style BPE.',color:C.blue},
    {title:'Each token mapped to a unique integer ID (vocabulary index)',body:'"The" → 791 &nbsp; "_cat" → 8415 &nbsp; "_sat" → 7482 — IDs are just indices into the vocab table, not meaningful numbers.',color:C.cyan},
    {title:'BOS token prepended; instruction markers added for chat models',body:'[BOS=1, "The"=791, "_cat"=8415, "_sat"=7482] — LLaMA uses BOS=1; BERT uses [CLS]=101.',color:C.yellow},
    {title:'Each ID indexes a row in the embedding table [vocab × d_model]',body:'IDs 1, 791, 8415, 7482 → each maps to 4096 floats. Result tensor: [batch=1, seq=4, d_model=4096].',color:C.accent},
    {title:'Position information injected — no inherent order in Transformers',body:'Methods: Sinusoidal | Learned | RoPE | ALiBi — LLaMA uses RoPE on Q,K inside each attention head.',color:C.purple},
    {title:'32 transformer layers: static embeddings → contextual representations',body:'"cat" vector now knows about "sat" — context-aware at every layer. Output: logits over vocabulary.',color:C.green},
  ];

  var step=S.pipeStep;
  var out=sectionTitle('The Complete Pipeline','Raw text → token IDs → dense vectors → positional encoding → transformer — every step, every time');

  // Pipeline boxes
  out+='<div class="card" style="max-width:1100px;margin:0 auto 14px;overflow-x:auto;">';
  out+='<div style="display:flex;gap:0;align-items:center;min-width:700px;padding:10px 0;">';
  stages.forEach(function(s,i){
    var active=step===i;
    var bg=active?hex(s.color,.13):'#0d0d14';
    var bc=active?s.color:hex(s.color,.3);
    var tc=active?s.color:hex(s.color,.7);
    out+='<button data-action="pipeStep" data-idx="'+i+'" style="flex:1;min-width:90px;padding:10px 6px;border-radius:8px;border:1.5px solid '+bc+';background:'+bg+';color:'+tc+';font-size:9px;font-weight:700;font-family:inherit;cursor:pointer;transition:all .3s;line-height:1.5;'+(active?'box-shadow:0 0 10px '+hex(s.color,.3)+';':'')+'">'+s.label+'<br><span style="font-size:7px;color:'+C.dim+';font-weight:400;">'+s.sub+'</span></button>';
    if(i<stages.length-1) out+='<div style="color:'+C.dim+';font-size:14px;padding:0 2px;flex-shrink:0;">›</div>';
  });
  out+='</div>';

  // Detail panel
  if(step===-1){
    out+=div('text-align:center;padding:20px 0;color:'+C.dim+';font-size:9px;','Click any stage above to inspect it');
  } else {
    var d=details[step];
    out+='<div class="fade" style="margin-top:14px;padding:14px 16px;border-radius:8px;background:'+hex(d.color,.06)+';border:1px solid '+hex(d.color,.25)+';">';
    out+=div('font-size:11px;font-weight:700;color:'+d.color+';margin-bottom:6px;',d.title);
    out+=div('font-size:10px;color:'+C.muted+';line-height:1.7;',d.body);
    out+='</div>';
  }
  out+='</div>';

  // Stats row
  out+=card('<div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">'
    +statBox('INPUT FORMAT','String',C.muted,'raw unicode text')
    +statBox('AFTER TOKENIZE','Integer[]',C.blue,'token IDs')
    +statBox('AFTER EMBED','Float32[][]',C.accent,'[seq × d_model]')
    +statBox('AFTER PE','Float32[][]',C.purple,'position-aware')
    +statBox('AFTER TRANSFORMER','Float32[][]',C.green,'contextual reps')
    +'</div>');

  out+=insight('💡','The Two-Stage Trick','Text cannot be fed directly into a neural network. Stage 1 (tokenization) converts text to integer IDs — a lookup index, not a meaningful number. Stage 2 (embedding) converts each integer ID to a dense float vector. These are fundamentally different operations: tokenization is <strong style="color:'+C.blue+'">deterministic and fixed</strong> after training the tokenizer; the embedding table is a <strong style="color:'+C.accent+'">learned model weight</strong> updated by backpropagation.');

  return out;
}

/* ═══════════════════════════════════════════════════════════
   TAB 2 — TOKENIZERS
═══════════════════════════════════════════════════════════ */
function renderTokenizers(){
  var strategies=[
    {name:'Character',color:C.muted,vocab:'~256',seqLen:'Very long',oov:'None',used:'Early char-RNNs',
     tokens:[chip('H',C.muted),chip('e',C.muted),chip('l',C.muted),chip('l',C.muted),chip('o',C.muted)],ids:['72','69','76','76','79'],
     pros:['No OOV problem','Tiny vocabulary (~256)','Works any language'],cons:['Very long sequences','Model learns from raw chars','Quadratic attention cost']},
    {name:'Word',color:C.orange,vocab:'100K–1M+',seqLen:'Short',oov:'[UNK] token',used:'Legacy NLP, spaCy',
     tokens:[chip('"The"',C.orange),chip('"quick"',C.orange),chip('"fox"',C.orange)],ids:['427','890','88'],
     pros:['Short sequences','Each token semantically whole'],cons:['Vocabulary explosion','OOV — information lost','Morphology ignored']},
    {name:'Subword (BPE)',color:C.accent,vocab:'30K–128K',seqLen:'Moderate',oov:'Always decomposable',used:'GPT-2/3/4, LLaMA, Mistral',
     tokens:[chip('"token"',C.accent),chip('"ization"',C.accent)],ids:['3993','1634'],
     pros:['No OOV ever','Manageable vocab','Morpheme-aware splits'],cons:['Suboptimal for numbers','Context-sensitive splits']},
  ];

  var bpeSteps=[
    {label:'Step 0: Characters',desc:'Every character is its own token',
     rows:[['l','o','w','</w>','×2'],['l','o','w','e','r','</w>','×1'],['n','e','w','e','r','</w>','×1'],['n','e','w','e','s','t','</w>','×1'],['w','i','d','e','s','t','</w>','×1']],merge:null},
    {label:'Step 1: Merge (e,s)→es',desc:'Most frequent adjacent pair',
     rows:[['l','o','w','</w>'],['l','o','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','es','t','</w>'],['w','i','d','es','t','</w>']],merge:'es'},
    {label:'Step 2: Merge (es,t)→est',desc:'Second most frequent pair',
     rows:[['l','o','w','</w>'],['l','o','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'est'},
    {label:'Step 3: Merge (l,o)→lo',desc:'Build up common prefixes',
     rows:[['lo','w','</w>'],['lo','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'lo'},
    {label:'Step 4: Merge (lo,w)→low',desc:'Common words emerge as single tokens',
     rows:[['low','</w>'],['low','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'low'},
    {label:'Step 5: Merge (n,e)→ne',desc:"Building 'new' prefix",
     rows:[['low','</w>'],['low','e','r','</w>'],['ne','w','e','r','</w>'],['ne','w','est','</w>'],['w','i','d','est','</w>']],merge:'ne'},
    {label:'Step 6: Merge (ne,w)→new',desc:'"new" is now a single token',
     rows:[['low','</w>'],['low','e','r','</w>'],['new','e','r','</w>'],['new','est','</w>'],['w','i','d','est','</w>']],merge:'new'},
    {label:'Step 7: Merge (new,est)→newest',desc:'Whole word merged — frequently co-occurring',
     rows:[['low','</w>'],['low','e','r','</w>'],['new','e','r','</w>'],['newest','</w>'],['w','i','d','est','</w>']],merge:'newest'},
  ];

  var sel=S.tokSel; var v=strategies[sel];
  var bs=bpeSteps[S.bpeStep];

  var out=sectionTitle('Tokenization Strategies','Three approaches to splitting text — only subword is used by modern LLMs');

  // Strategy selector
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
  strategies.forEach(function(s,i){ out+=btnSel(i,sel,s.color,s.name,'tokSel'); });
  out+='</div>';

  // Detail card
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
  out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
  // Left
  out+='<div style="flex:1;min-width:260px;">';
  out+=div('font-size:16px;font-weight:800;color:'+v.color+';margin-bottom:8px;',v.name+' Tokenization');
  out+='<div style="display:flex;gap:14px;margin-bottom:10px;flex-wrap:wrap;">';
  out+=div('font-size:8px;color:'+C.dim+';','VOCAB: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.vocab));
  out+=div('font-size:8px;color:'+C.dim+';','OOV: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.oov));
  out+=div('font-size:8px;color:'+C.dim+';','SEQ: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.seqLen));
  out+='</div>';
  // Token example
  out+='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;">';
  v.tokens.forEach(function(t){ out+=t; });
  out+='</div>';
  out+=div('font-size:8px;color:'+C.dim+';margin-bottom:10px;','Used by: '+span('color:'+v.color+';',v.used));
  out+='</div>';
  // Right — pros/cons
  out+='<div style="min-width:220px;">';
  out+=div('font-size:9px;color:'+C.green+';font-weight:700;margin-bottom:6px;','&#10003; Advantages');
  v.pros.forEach(function(p){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#10003; '+p); });
  out+=div('font-size:9px;color:'+C.red+';font-weight:700;margin-top:10px;margin-bottom:6px;','&#9888; Trade-offs');
  v.cons.forEach(function(p){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#9888; '+p); });
  out+='</div>';
  out+='</div></div>';

  // BPE slider
  out+=card('<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:6px;">BPE Algorithm — Step by Step</div>'
    +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:14px;">Corpus: ["low"×2, "lower", "newer", "newest", "widest"] — drag the slider to trace merges</div>'
    +'<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
    +'<div style="font-size:9px;color:'+C.dim+';min-width:40px;">Step:</div>'
    +'<input type="range" min="0" max="7" value="'+S.bpeStep+'" data-action="bpeStep" style="flex:1;accent-color:'+C.accent+';">'
    +'<div style="font-size:9px;color:'+C.accent+';font-weight:700;min-width:15px;">'+S.bpeStep+'</div>'
    +'</div>'
    +'<div style="padding:4px 12px;border-radius:6px;background:'+hex(C.accent,.13)+';border:1px solid '+hex(C.accent,.35)+';font-size:9px;font-weight:700;color:'+C.accent+';display:inline-block;margin-bottom:8px;">'+bs.label+'</div>'
    +div('font-size:9px;color:'+C.muted+';margin-bottom:12px;',bs.desc)
    // token rows
    +(function(){
      var r='<div style="display:flex;flex-direction:column;gap:5px;">';
      bs.rows.forEach(function(row,ri){
        r+='<div style="display:flex;gap:4px;align-items:center;">';
        r+='<div style="font-size:8px;color:'+C.dim+';width:20px;text-align:right;flex-shrink:0;">'+(ri+1)+'</div>';
        row.forEach(function(tok){
          var isMerged=bs.merge&&tok===bs.merge;
          r+='<div style="padding:3px 7px;border-radius:4px;font-family:inherit;font-size:9px;background:'+(isMerged?hex(C.accent,.25):hex(C.dim,.25))+';border:1px solid '+(isMerged?C.accent:hex(C.dim,.5))+';color:'+(isMerged?C.accent:C.muted)+';font-weight:'+(isMerged?'800':'400')+';transition:all .4s;">'+tok+'</div>';
        });
        r+='</div>';
      });
      r+='</div>';
      if(bs.merge) r+='<div style="margin-top:10px;padding:8px 12px;border-radius:6px;background:'+hex(C.accent,.08)+';border:1px solid '+hex(C.accent,.25)+';font-size:9px;color:'+C.muted+';">Merged token: <span style="color:'+C.accent+';font-weight:700;">'+bs.merge+'</span></div>';
      return r;
    })()
  );

  // Algorithm comparison
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">BPE vs WordPiece vs Unigram</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {name:'BPE',color:C.accent,dir:'Bottom-up',crit:'Most frequent pair',note:'No prefix',models:'GPT-2/3/4, LLaMA, Mistral'},
      {name:'WordPiece',color:C.blue,dir:'Bottom-up',crit:'Max likelihood ratio',note:'## continuation prefix',models:'BERT, DistilBERT'},
      {name:'Unigram LM',color:C.purple,dir:'Top-down',crit:'Prune by likelihood impact',note:'No prefix (SentencePiece)',models:'T5, mT5, LLaMA-1/2'},
    ].map(function(alg){
      return '<div style="flex:1;min-width:200px;padding:12px 14px;border-radius:8px;background:'+hex(alg.color,.06)+';border:1px solid '+hex(alg.color,.25)+'">'
        +'<div style="font-size:13px;font-weight:800;color:'+alg.color+';margin-bottom:6px;">'+alg.name+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';line-height:1.9;">Direction: <span style="color:'+alg.color+';">'+alg.dir+'</span><br>Criterion: <span style="color:'+C.muted+';">'+alg.crit+'</span><br>Notation: <span style="color:'+C.muted+';font-family:inherit;">'+alg.note+'</span></div>'
        +'<div style="font-size:8px;color:'+alg.color+';font-style:italic;margin-top:8px;">'+alg.models+'</div>'
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('&#11088;','Why Subword Won','Character tokenization produces sequences <strong style="color:'+C.red+'">10× too long</strong> — attention scales O(n²). Word tokenization causes <strong style="color:'+C.orange+'">vocabulary explosion and OOV collapse</strong>. Subword is the sweet spot: common words get their own token, rare words decompose into recognizable pieces, <strong style="color:'+C.accent+'">and OOV never occurs</strong>. Real models run thousands of BPE merges to reach their 32K–128K vocab.');

  return out;
}

/* ═══════════════════════════════════════════════════════════
   TAB 3 — EMBEDDING SPACE
═══════════════════════════════════════════════════════════ */
function renderEmbedding(){
  var vocab=S.embVocab, dim=S.embDim;
  var bytes=vocab*dim*2;
  var mb=(bytes/1e6).toFixed(0);
  var gb=(bytes/1e9).toFixed(2);
  var pct=(vocab*dim/7e9*100).toFixed(1);
  var sizeStr=parseFloat(gb)<1?mb+' MB':gb+' GB';

  var out=sectionTitle('The Embedding Space','IDs → dense vectors: a learned geometry where meaning is distance');

  // Calculator
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:14px;">Embedding Table Explorer — Shape: [vocab_size × d_model]</div>'
    +'<div style="display:flex;gap:30px;flex-wrap:wrap;margin-bottom:16px;">'
    +'<div style="flex:1;min-width:220px;">'
    +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Vocabulary size: '+vocab.toLocaleString()+'</div>'
    +'<input type="range" min="10000" max="200000" step="1000" value="'+vocab+'" data-action="embVocab" style="accent-color:'+C.blue+';">'
    +'<div style="display:flex;justify-content:space-between;font-size:8px;color:'+C.dim+';"><span>10K</span><span>50K</span><span>100K</span><span>200K</span></div>'
    +'</div>'
    +'<div style="flex:1;min-width:220px;">'
    +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Embedding dim (d_model): '+dim+'</div>'
    +'<input type="range" min="256" max="8192" step="128" value="'+dim+'" data-action="embDim" style="accent-color:'+C.accent+';">'
    +'<div style="display:flex;justify-content:space-between;font-size:8px;color:'+C.dim+';"><span>256</span><span>768</span><span>4096</span><span>8192</span></div>'
    +'</div>'
    +'</div>'
    +'<div style="display:flex;justify-content:center;gap:30px;flex-wrap:wrap;margin-bottom:14px;">'
    +statBox('TABLE SHAPE',Math.round(vocab/1000)+'K \xd7 '+dim,C.blue,'')
    +statBox('SIZE (BF16)',sizeStr,C.accent,'')
    +statBox('PARAMETERS',(vocab*dim/1e6).toFixed(0)+'M',C.purple,'')
    +statBox('% OF 7B MODEL',pct+'%',C.yellow,'embedding only')
    +'</div>'
    // Mini grid visualisation
    +'<div style="font-size:9px;color:'+C.dim+';margin-bottom:8px;">Token ID → Embedding Row (each cell = one float):</div>'
    +'<div style="overflow-x:auto;">'
    +(function(){
      var rows=['[PAD]','[BOS]','[EOS]','"the"','"cat"','"sat"'];
      var r='<div style="display:flex;flex-direction:column;gap:3px;">';
      rows.forEach(function(lbl,ri){
        r+='<div style="display:flex;gap:3px;align-items:center;">';
        r+='<div style="font-size:8px;color:'+C.dim+';width:50px;flex-shrink:0;font-family:inherit;">'+lbl+'</div>';
        for(var ci=0;ci<16;ci++){
          var v=Math.sin(ri*3.7+ci*1.3);
          var pos=v>0;
          var intensity=Math.abs(v);
          var col=pos?C.blue:C.purple;
          r+='<div style="width:24px;height:18px;border-radius:3px;background:'+hex(col,intensity*.6+.1)+';flex-shrink:0;"></div>';
        }
        r+='<div style="font-size:7px;color:'+C.dim+';margin-left:4px;">... '+dim+' dims</div>';
        r+='</div>';
      });
      r+='</div>';
      return r;
    })()
    +'</div>'
  );

  // Word geometry
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Semantic Geometry — Words as Points in Space</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">'
    +[
      {label:'Royalty',color:C.yellow,words:['king','queen','prince','throne']},
      {label:'Animals',color:C.green, words:['cat','dog','lion','tiger']},
      {label:'Places', color:C.blue,  words:['Paris','Berlin','Tokyo','Rome']},
      {label:'Verbs',  color:C.purple,words:['run','walk','jump','swim']},
    ].map(function(g){
      return '<div style="flex:1;min-width:140px;padding:10px 14px;border-radius:8px;background:'+hex(g.color,.07)+';border:1px solid '+hex(g.color,.25)+'">'
        +'<div style="font-size:10px;font-weight:700;color:'+g.color+';margin-bottom:8px;">'+g.label+'</div>'
        +g.words.map(function(w){return '<div style="font-size:9px;color:'+C.muted+';line-height:1.9;">'+w+'</div>';}).join('')
        +'</div>';
    }).join('')
    +'</div>'
    +'<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;">Vector Analogies (Arithmetic in Embedding Space)</div>'
    +'<div style="display:flex;flex-direction:column;gap:8px;">'
    +[
      {eq:'king &minus; man + woman',result:'queen',color:C.yellow},
      {eq:'Paris &minus; France + Italy',result:'Rome',color:C.blue},
      {eq:'swim &minus; water + air',result:'fly',color:C.cyan},
    ].map(function(a){
      return '<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;background:'+hex(a.color,.06)+';border:1px solid '+hex(a.color,.2)+'">'
        +'<div style="font-size:10px;font-family:inherit;color:'+C.muted+';">'+a.eq+' &asymp;</div>'
        +'<div style="font-size:12px;font-weight:800;color:'+a.color+';">'+a.result+'</div>'
        +'</div>';
    }).join('')
    +'</div>'
  );

  // Static vs Contextual
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Static vs Contextual Embeddings</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {title:'Static (Word2Vec, GloVe)',color:C.orange,
       desc:'One fixed vector per word — context ignored.',
       examples:[
         {sent:'"I went to the bank to deposit money"',vec:'"bank" → [0.23, -0.71, ...]  (financial)'},
         {sent:'"She sat by the river bank"',vec:'"bank" → [0.23, -0.71, ...]  (SAME vector!)'},
       ],note:'Problem: "bank" has identical representation in both sentences.'},
      {title:'Contextual (BERT, GPT, LLaMA)',color:C.green,
       desc:'Vector changes based on surrounding context — from transformer layers.',
       examples:[
         {sent:'"I went to the bank to deposit money"',vec:'"bank" → [0.91, 0.12, ...]  (financial context)'},
         {sent:'"She sat by the river bank"',vec:'"bank" → [-0.34, 0.88, ...]  (geographic context)'},
       ],note:'The Transformer produces different representations for the same token in different contexts.'},
    ].map(function(col){
      return '<div style="flex:1;min-width:260px;padding:12px 14px;border-radius:8px;background:'+hex(col.color,.06)+';border:1px solid '+hex(col.color,.25)+'">'
        +'<div style="font-size:11px;font-weight:800;color:'+col.color+';margin-bottom:6px;">'+col.title+'</div>'
        +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:8px;">'+col.desc+'</div>'
        +col.examples.map(function(ex){
          return '<div style="margin-bottom:5px;padding:5px 8px;border-radius:4px;background:'+hex(C.border,.4)+';font-family:inherit;">'
            +'<div style="font-size:8px;color:'+col.color+';">'+ex.sent+'</div>'
            +'<div style="font-size:7px;color:'+C.dim+';">'+ex.vec+'</div>'
            +'</div>';
        }).join('')
        +'<div style="margin-top:8px;font-size:8px;color:'+C.dim+';font-style:italic;">'+col.note+'</div>'
        +'</div>';
    }).join('')
    +'</div>'
  );

  out+=insight('💡','Why Embeddings Learn Meaning Without Being Taught','The model is never told "king and queen should be similar." It learns this because they appear in <strong style="color:'+C.accent+'">identical contexts</strong>: "the ___ wore a crown", "the ___ ruled the kingdom." The training loss rewards correct context prediction — placing co-occurring words nearby in vector space is the <strong style="color:'+C.purple+'">most efficient way to achieve that</strong>. Semantic similarity is a side effect of learning to predict well, not a design choice.');

  return out;
}

/* ═══════════════════════════════════════════════════════════
   TAB 4 — POSITIONAL ENCODING
═══════════════════════════════════════════════════════════ */
function renderPositional(){
  var methods=[
    {name:'Sinusoidal',color:C.blue,year:'2017 (Vaswani)',learnable:'No',relative:'Partial',extrapolates:'Yes (degrades)',appliedTo:'Embeddings (added)',usedBy:'Original Transformer',
     formula:'PE(pos,2i)   = sin(pos / 10000^(2i/d))\\nPE(pos,2i+1) = cos(pos / 10000^(2i/d))',
     desc:'Fixed math formula. Each dimension oscillates at a different frequency. No parameters to learn. Can extend beyond training length.'},
    {name:'Learned',color:C.yellow,year:'2018 (BERT, GPT-2)',learnable:'Yes',relative:'No',extrapolates:'No (hard limit)',appliedTo:'Embeddings (added)',usedBy:'BERT (max 512), GPT-2 (max 1024)',
     formula:'pos_embed = EmbeddingTable[pos]   (trained)',
     desc:'Position vectors trained alongside the model. Flexible but hard-limited to max_seq_length seen during training.'},
    {name:'RoPE',color:C.accent,year:'2021 (Su et al.)',learnable:'No',relative:'Yes (exact)',extrapolates:'Yes (w/ NTK scaling)',appliedTo:'Q, K in each attention head',usedBy:'LLaMA (all), Mistral, Falcon, Qwen',
     formula:'[q_out, k_out] = Rotate(m · θ_i) · [q_in, k_in]<br>θ_i = 10000^(-2i/d)',
     desc:'Encodes position as a rotation in 2D embedding subspaces. Q[m]\xb7K[n] depends only on (m-n), making attention naturally relative.'},
    {name:'ALiBi',color:C.cyan,year:'2021 (Press et al.)',learnable:'No',relative:'Yes (linear)',extrapolates:'Yes (best)',appliedTo:'Attention scores (bias)',usedBy:'BLOOM (176B), MPT, OpenLLM',
     formula:'score(q_i, k_j) = (q_i\xb7k_j)/\u221ad  \u2212  m\xb7|i\u2212j|',
     desc:'Adds a linear distance penalty to attention scores. No position vectors. Excellent length extrapolation. Slope m is head-specific.'},
  ];

  var sel=S.posSel; var v=methods[sel];
  var pos=S.posSlider;

  // sinusoidal bars
  var peVals=[];
  for(var di=0;di<20;di++){
    var freq=Math.pow(10000,(2*di)/512);
    peVals.push(di%2===0?Math.sin(pos/freq):Math.cos(pos/freq));
  }

  var out=sectionTitle('Positional Encoding','Transformers process tokens in parallel — position must be injected explicitly');

  // Why needed
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;">Why Positional Encoding Is Necessary</div>'
    +'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
    +'<div style="flex:1;min-width:250px;padding:10px 14px;border-radius:8px;background:'+hex(C.red,.08)+';border:1px solid '+hex(C.red,.25)+'">'
    +'<div style="font-size:10px;font-weight:700;color:'+C.red+';margin-bottom:6px;">Without PE</div>'
    +'<div style="font-size:9px;font-family:inherit;color:'+C.muted+';line-height:2;">"The cat sat on the mat"<br>"The mat sat on the cat"</div>'
    +'<div style="font-size:8px;color:'+C.red+';margin-top:6px;">&rarr; Identical representations. Same tokens, same attention.</div>'
    +'</div>'
    +'<div style="flex:1;min-width:250px;padding:10px 14px;border-radius:8px;background:'+hex(C.green,.08)+';border:1px solid '+hex(C.green,.25)+'">'
    +'<div style="font-size:10px;font-weight:700;color:'+C.green+';margin-bottom:6px;">With PE</div>'
    +'<div style="font-size:9px;color:'+C.muted+';line-height:2;">final[pos] = embed[token_id] + pos_encoding[pos]<br>Same token, different position &rarr; different vector</div>'
    +'<div style="font-size:8px;color:'+C.green+';margin-top:6px;">&rarr; "cat" at pos 1 &ne; "cat" at pos 5. Order is preserved.</div>'
    +'</div>'
    +'</div>'
  );

  // Method selector
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
  methods.forEach(function(m,i){ out+=btnSel(i,sel,m.color,m.name,'posSel'); });
  out+='</div>';

  // Detail card
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
  out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
  out+='<div style="flex:1;min-width:260px;">';
  out+=div('font-size:16px;font-weight:800;color:'+v.color+';',v.name);
  out+=div('font-size:9px;color:'+C.dim+';margin-bottom:8px;',v.year);
  out+=div('font-size:11px;color:'+C.muted+';line-height:1.7;margin-bottom:8px;',v.desc);
  out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(v.color,.1)+';border:1px solid '+hex(v.color,.3)+';font-family:inherit;font-size:9px;color:'+v.color+';white-space:pre-line;">'+v.formula+'</div>';
  out+=div('margin-top:8px;font-size:8px;color:'+C.dim+';','Applied to: '+span('color:'+v.color+';',v.appliedTo)+'<br>Used by: '+span('color:'+v.color+';',v.usedBy));
  out+='</div>';
  out+='<div style="min-width:180px;display:flex;flex-direction:column;gap:10px;">';
  [{l:'Learnable params',val:v.learnable,good:v.learnable==='Yes'},{l:'Relative positions',val:v.relative,good:v.relative!=='No'},{l:'Extrapolates',val:v.extrapolates,good:v.extrapolates.startsWith('Yes')}].forEach(function(r){
    out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(C.border,.15)+';border:1px solid '+C.border+';">'
      +div('font-size:8px;color:'+C.dim+';margin-bottom:3px;',r.l)
      +div('font-size:10px;font-weight:700;color:'+(r.good?C.green:C.orange)+';',r.val)
      +'</div>';
  });
  out+='</div>';
  out+='</div></div>';

  // Sinusoidal visualiser
  if(sel===0){
    out+=card(
      '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;">Sinusoidal Pattern Visualizer</div>'
      +'<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
      +'<div style="font-size:9px;color:'+C.muted+';min-width:80px;">Position: '+pos+'</div>'
      +'<input type="range" min="0" max="511" value="'+pos+'" data-action="posSlider" style="flex:1;accent-color:'+C.blue+';">'
      +'</div>'
      +'<div style="display:flex;gap:3px;align-items:flex-end;height:90px;">'
      +peVals.map(function(val,i){
        var h=Math.max(Math.abs(val)*44+4,2);
        var col=val>0?C.blue:C.purple;
        return '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">'
          +'<div style="height:'+h+'px;border-radius:3px 3px 0 0;background:'+hex(col,Math.abs(val)*.7+.3)+';width:100%;transition:height .3s,background .3s;"></div>'
          +(i%4===0?'<div style="font-size:6px;color:'+C.dim+';margin-top:2px;">'+i+'</div>':'')
          +'</div>';
      }).join('')
      +'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';margin-top:8px;">Each bar = one dimension of the PE vector at position '+pos+'. Low dims oscillate slowly (coarse position), high dims oscillate fast (fine position).</div>'
    );
  }

  // Comparison table
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Comparison: All Positional Encoding Methods</div>'
    +'<div style="overflow-x:auto;">'
    +'<table><thead><tr>'
    +['Method','Learnable','Relative','Extrapolates','Applied To','Used By'].map(function(h){return '<th>'+h+'</th>';}).join('')
    +'</tr></thead><tbody>'
    +methods.map(function(m,i){
      var bg=sel===i?hex(m.color,.1):'transparent';
      return '<tr style="background:'+bg+';">'
        +'<td style="color:'+m.color+';font-weight:700;border-radius:6px 0 0 6px;">'+m.name+'</td>'
        +'<td style="color:'+(m.learnable==='Yes'?C.green:C.orange)+';">'+m.learnable+'</td>'
        +'<td style="color:'+(m.relative==='No'?C.orange:C.green)+';">'+m.relative+'</td>'
        +'<td style="color:'+(m.extrapolates.startsWith('Yes')?C.green:C.red)+';">'+m.extrapolates+'</td>'
        +'<td style="color:'+C.muted+';">'+m.appliedTo+'</td>'
        +'<td style="color:'+C.dim+';border-radius:0 6px 6px 0;">'+m.usedBy+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table>'
    +'</div>'
  );

  out+=insight('&#9881;','Why RoPE Dominates Modern LLMs','Learned positional embeddings (BERT, GPT-2) have a <strong style="color:'+C.red+'">hard sequence length limit</strong> — trained on 1024 tokens, they cannot handle 1025. Sinusoidal degrades beyond training length. <strong style="color:'+C.accent+'">RoPE</strong> encodes position as rotation, making attention inherently relative and enabling context window extension through NTK-aware scaling. This is why every modern open-source LLM (LLaMA, Mistral, Qwen, Falcon) uses RoPE.');

  return out;
}

/* ═══════════════════════════════════════════════════════════
   TAB 5 — TOKENIZER ZOO
═══════════════════════════════════════════════════════════ */
function renderZoo(){
  var models=[
    {name:'GPT-4 / GPT-3.5',algo:'BPE (tiktoken cl100k)',vocab:100256,dim:12288,contextK:'128K',released:'2022/23',color:C.green,
     notes:['Byte-level BPE — vocab covers all UTF-8 bytes','Numbers tokenized digit-by-digit for most cases','cl100k has better multilingual coverage than GPT-2','Used by ChatGPT, GPT-4 API']},
    {name:'LLaMA-3',algo:'BPE (tiktoken)',vocab:128256,dim:4096,contextK:'128K',released:'2024',color:C.accent,
     notes:['Largest vocab of any major open-source model','128K vocab reduces sequence length vs LLaMA-2 (was 32K)','Shares tokenizer format with GPT-4 (cl100k-based)','4\xd7 vocab increase over LLaMA-2']},
    {name:'LLaMA-2',algo:'BPE (SentencePiece)',vocab:32000,dim:4096,contextK:'4K',released:'2023',color:C.blue,
     notes:['Smaller 32K vocab — longer sequences than LLaMA-3','SentencePiece treats raw bytes — language agnostic','BOS=1, EOS=2, [INST]/[/INST] for chat','Default choice for open-source fine-tuning in 2023']},
    {name:'BERT',algo:'WordPiece',vocab:30522,dim:768,contextK:'512 tok',released:'2018',color:C.yellow,
     notes:['WordPiece marks continuations with ## prefix','[CLS] prepended, [SEP] separates segments','Max 512 tokens (learned positional embeddings)','Encoder-only — no generation, embeddings focused']},
    {name:'T5',algo:'Unigram (SentencePiece)',vocab:32100,dim:1024,contextK:'512 tok',released:'2019',color:C.purple,
     notes:['Unigram: starts large, prunes by likelihood','SentencePiece framework — treats text as byte stream','Excellent multilingual support out of the box','No language-specific whitespace assumptions']},
    {name:'Mistral-7B',algo:'BPE (SentencePiece)',vocab:32000,dim:4096,contextK:'32K',released:'2023',color:C.cyan,
     notes:['Same tokenizer as LLaMA-2 (compatible)','Sliding window attention extends effective context','&lt;s&gt; BOS, &lt;/s&gt; EOS, [INST]/[/INST] chat tokens','32K vocab — efficient, widely compatible']},
  ];

  var sel=S.zooSel; var v=models[sel];
  var out=sectionTitle('Tokenizer Zoo','Every major model has its own tokenizer — different algorithm, vocab size, and quirks');

  // Model selector
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
  models.forEach(function(m,i){ out+=btnSel(i,sel,m.color,m.name,'zooSel'); });
  out+='</div>';

  // Detail card
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
  out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
  out+='<div style="flex:1;min-width:260px;">';
  out+=div('font-size:18px;font-weight:800;color:'+v.color+';margin-bottom:4px;',v.name);
  out+=div('font-size:10px;color:'+C.dim+';margin-bottom:12px;','Algorithm: <span style="color:'+v.color+';">'+v.algo+'</span> | Released: '+v.released);
  v.notes.forEach(function(n){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#10003; '+n); });
  out+='</div>';
  out+='<div style="display:flex;flex-direction:column;gap:10px;min-width:120px;">';
  out+=statBox('VOCAB SIZE',v.vocab.toLocaleString(),v.color,'');
  out+=statBox('d_model',v.dim.toLocaleString(),v.color,'');
  out+=statBox('CONTEXT',v.contextK,v.color,'');
  out+='</div>';
  out+='</div></div>';

  // Comparison table
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Side-by-Side Comparison</div>'
    +'<div style="overflow-x:auto;">'
    +'<table><thead><tr>'
    +['Model','Algorithm','Vocab Size','d_model','Context','Released'].map(function(h){return '<th>'+h+'</th>';}).join('')
    +'</tr></thead><tbody>'
    +models.map(function(m,i){
      var bg=sel===i?hex(m.color,.1):'transparent';
      return '<tr style="background:'+bg+';cursor:pointer;" data-action="zooSel" data-idx="'+i+'">'
        +'<td style="color:'+m.color+';font-weight:700;border-radius:6px 0 0 6px;">'+m.name+'</td>'
        +'<td style="color:'+C.muted+';">'+m.algo+'</td>'
        +'<td style="color:'+C.accent+';font-weight:700;">'+m.vocab.toLocaleString()+'</td>'
        +'<td style="color:'+C.blue+';">'+m.dim+'</td>'
        +'<td style="color:'+C.cyan+';">'+m.contextK+'</td>'
        +'<td style="color:'+C.dim+';border-radius:0 6px 6px 0;">'+m.released+'</td>'
        +'</tr>';
    }).join('')
    +'</tbody></table>'
    +'</div>'
  );

  out+=insight('&#10024;','Why Vocabulary Size Is an Engineering Tradeoff','Larger vocab (LLaMA-3: 128K) means <strong style="color:'+C.green+'">shorter sequences</strong> — same text uses fewer tokens, fitting more content in the context window. But the embedding table grows proportionally: LLaMA-3 128K × 4096 table is <strong style="color:'+C.accent+'">~1 GB</strong> of model weights (12% of total). Smaller vocab (LLaMA-2: 32K) saves memory but produces longer sequences that fill the context window faster.');

  return out;
}

/* ═══════════════════════════════════════════════════════════
   TAB 6 — QUIRKS & GOTCHAS
═══════════════════════════════════════════════════════════ */
function renderQuirks(){
  var quirks=[
    {title:'Leading Space Changes the Token',color:C.red,icon:'&#9888;',
     desc:'In GPT-style BPE, a word with a leading space is a completely different token from the same word without one.',
     examples:[
       {input:'"dog"',tokens:['dog'],note:'→ ID 18031 (no space)'},
       {input:'" dog"',tokens:['\u2581dog'],note:'→ ID 5679  (space included — totally different token!)'},
       {input:'"The dog sat"',tokens:['The','\u2581dog','\u2581sat'],note:'→ "dog" uses the space version'},
     ],
     insight:'This is why prompt phrasing matters more than you would think. "Translate: cat" and "Translate cat" tokenize differently. GPT-style models mark leading-space tokens with \u2581.'},
    {title:'Capitalization Creates New Tokens',color:C.yellow,icon:'&#9888;',
     desc:'Case changes often produce completely different token IDs — not just re-weighted.',
     examples:[
       {input:'"Hello"',tokens:['Hello'],note:'→ 1 token, ID 15043'},
       {input:'"hello"',tokens:['h','ello'],note:'→ 2 tokens! Lowercase split differently'},
       {input:'"HELLO"',tokens:['H','E','L','L','O'],note:'→ 5 tokens (all caps tokenizes letter by letter)'},
     ],
     insight:'Capitalization is not "free" for LLMs. Uppercase text can explode your token count. For code: indentation, variable name casing, and symbol placement all affect tokenization.'},
    {title:'Non-English Is Token-Expensive',color:C.purple,icon:'&#11088;',
     desc:'Vocabularies are built on mostly English text. Non-English languages get fewer dedicated tokens.',
     examples:[
       {input:'"Hello"',tokens:['Hello'],note:'→ 1 token (English)'},
       {input:'"Hola"',tokens:['Hola'],note:'→ 1 token (Spanish, common word)'},
       {input:'"&#1055;&#1088;&#1080;&#1074;&#1077;&#1090;"',tokens:['\u041f\u0440','\u0438','\u0432\u0435\u0442'],note:'→ 3 tokens (Russian)'},
       {input:'"\u3053\u3093\u306b\u3061\u306f"',tokens:['\u3053','\u3093','\u306b','\u3061','\u306f'],note:'→ 5 tokens (Japanese — 1 char each)'},
     ],
     insight:'Chinese, Japanese, and Korean characters often tokenize 1-per-character. Non-English text uses 3-5\xd7 more tokens for the same semantic content — directly inflating cost.'},
    {title:'Code Tokenizes Differently',color:C.cyan,icon:'&#9881;',
     desc:'Indentation, operators, and special characters each cost tokens.',
     examples:[
       {input:'def foo():',tokens:['def',' foo','():'],note:'→ 3 tokens'},
       {input:'    return x+1',tokens:['    ','return',' x','+','1'],note:'→ 5 tokens (indentation is its own token!)'},
       {input:'x = [1,2,3]',tokens:['x',' =',' [','1',',','2',',','3',']'],note:'→ 9 tokens'},
     ],
     insight:'Code tokenization explains why LLMs are sensitive to indentation style and whitespace. 4-space vs 2-space indent can produce different token counts and different model behaviour.'},
  ];

  var sel=S.quirkSel; var v=quirks[sel];
  var out=sectionTitle('Quirks &amp; Gotchas','Real tokenizer behaviour that surprises every beginner — not bugs, just consequences of the algorithm');

  // Selector
  out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
  quirks.forEach(function(q,i){ out+=btnSel(i,sel,q.color,q.icon+' '+q.title.split(' ').slice(0,2).join(' ')+'...','quirkSel'); });
  out+='</div>';

  // Detail card
  out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
  out+=div('font-size:14px;font-weight:800;color:'+v.color+';margin-bottom:6px;',v.icon+' '+v.title);
  out+=div('font-size:10px;color:'+C.muted+';margin-bottom:14px;line-height:1.7;',v.desc);
  out+='<div style="display:flex;flex-direction:column;gap:8px;">';
  v.examples.forEach(function(ex){
    out+='<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;padding:10px 14px;border-radius:8px;background:'+hex(v.color,.05)+';border:1px solid '+hex(v.color,.2)+'">'
      +'<div style="min-width:160px;font-family:inherit;font-size:9px;color:'+v.color+';font-weight:700;">'+ex.input+'</div>'
      +'<div style="display:flex;gap:4px;flex-wrap:wrap;flex:1;">'
      +ex.tokens.map(function(tok){return chip(tok,v.color);}).join(' ')
      +'</div>'
      +'<div style="font-size:8px;color:'+C.dim+';font-family:inherit;min-width:200px;">'+ex.note+'</div>'
      +'</div>';
  });
  out+='</div></div>';

  out+=insight(v.icon,v.title,v.insight);

  // Rule of thumb
  out+=card(
    '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Practical Token Counting Rule of Thumb</div>'
    +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
    +[
      {label:'English prose',ratio:'~1.3–1.5 tok/word',color:C.green,note:'Well-covered in training data'},
      {label:'Code',ratio:'~1.5–2.5 tok/word',color:C.cyan,note:'Operators, spaces, symbols'},
      {label:'Non-Latin script',ratio:'~2–4 tok/char',color:C.orange,note:'Fewer dedicated vocab entries'},
      {label:'Numbers / math',ratio:'Highly variable',color:C.red,note:'1 to N digits = 1 to N tokens'},
    ].map(function(r){
      return '<div style="flex:1;min-width:160px;padding:10px 14px;border-radius:8px;background:'+hex(r.color,.06)+';border:1px solid '+hex(r.color,.25)+'">'
        +'<div style="font-size:9px;font-weight:700;color:'+r.color+';margin-bottom:4px;">'+r.label+'</div>'
        +'<div style="font-size:11px;font-weight:800;color:'+r.color+';margin-bottom:4px;">'+r.ratio+'</div>'
        +'<div style="font-size:8px;color:'+C.dim+';">'+r.note+'</div>'
        +'</div>';
    }).join('')
    +'</div>'
    +'<div style="margin-top:12px;font-size:9px;color:'+C.dim+';text-align:center;">Token count = context window cost. Always count tokens, never estimate by word count alone.</div>'
  );

  return out;
}

/* ═══════════════════════════════════════════════════════════
   ROOT RENDER
═══════════════════════════════════════════════════════════ */
var TABS=['&#128260; Pipeline','&#9986;&#65039; Tokenizers','&#128202; Embedding Space','&#128205; Positional Encoding','&#129513; Tokenizer Zoo','&#9888;&#65039; Quirks &amp; Gotchas'];

function renderApp(){
  var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;color:'+C.text+';max-width:1400px;margin:0 auto;">';
  // Header
  html+='<div style="text-align:center;margin-bottom:16px;">'
    +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.accent+','+C.purple+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Tokenization &amp; Embeddings</div>'
    +'<div style="font-size:11px;color:'+C.muted+';margin-top:4px;">Interactive visual walkthrough — from raw text to transformer-ready vectors</div>'
    +'</div>';
  // Tab bar
  html+='<div class="tab-bar">';
  TABS.forEach(function(t,i){
    html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
  });
  html+='</div>';
  // Content
  if(S.tab===0) html+=renderPipeline();
  else if(S.tab===1) html+=renderTokenizers();
  else if(S.tab===2) html+=renderEmbedding();
  else if(S.tab===3) html+=renderPositional();
  else if(S.tab===4) html+=renderZoo();
  else if(S.tab===5) html+=renderQuirks();
  html+='</div>';
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
        if(action==='tab') S.tab=idx;
        else if(action==='pipeStep') S.pipeStep=(S.pipeStep===idx?-1:idx);
        else if(action==='tokSel') S.tokSel=idx;
        else if(action==='posSel') S.posSel=idx;
        else if(action==='zooSel') S.zooSel=idx;
        else if(action==='quirkSel') S.quirkSel=idx;
        render();
      });
    } else if(tag==='tr'){
      el.addEventListener('click',function(){
        if(action==='zooSel') S.zooSel=idx;
        render();
      });
      el.style.cursor='pointer';
    } else if(tag==='input'){
      el.addEventListener('input',function(){
        var val=parseInt(this.value);
        if(action==='bpeStep') S.bpeStep=val;
        else if(action==='embVocab') S.embVocab=val;
        else if(action==='embDim') S.embDim=val;
        else if(action==='posSlider') S.posSlider=val;
        render();
      });
    }
  });
}

// Initial render
render();
</script>
</body>
</html>"""

TOK_EMBED_HEIGHT = 1700

# """
# Self-contained HTML for the Tokenization & Embeddings interactive walkthrough.
# Covers: Pipeline, Tokenizers, Embedding Space, Positional Encoding,
# Tokenizer Zoo, and Quirks & Gotchas.
#
# Rewritten as pure vanilla HTML/CSS/JS — zero CDN dependencies, zero React,
# zero Babel. Renders reliably in any Streamlit sandboxed iframe.
#
# Embed in Streamlit via:
#     st.components.v1.html(TOK_EMBED_HTML, height=TOK_EMBED_HEIGHT, scrolling=True)
# """
#
# TOK_EMBED_HTML = """<!DOCTYPE html>
# <html>
# <head>
# <meta charset="utf-8"/>
# <style>
# *{margin:0;padding:0;box-sizing:border-box;}
# body{background:#0a0a0f;color:#e4e4e7;font-family:'JetBrains Mono','SF Mono',Consolas,monospace;overflow-x:hidden;}
# button{cursor:pointer;font-family:inherit;}
# input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:3px;background:#1e1e2e;outline:none;width:100%;}
# input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;border-radius:50%;background:#a78bfa;cursor:pointer;}
# table{border-collapse:separate;border-spacing:0 4px;width:100%;}
# th{text-align:left;padding:6px 10px;color:#3f3f46;font-size:8px;font-weight:700;}
# td{padding:7px 10px;font-size:9px;}
# @keyframes pulse{0%,100%{opacity:.6}50%{opacity:1}}
# @keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
# .fade{animation:fadeIn .3s ease both;}
# .card{background:#12121a;border-radius:10px;padding:18px 22px;border:1px solid #1e1e2e;margin-bottom:14px;}
# .card.hl{border-color:#a78bfa;}
# .section-title{text-align:center;margin-bottom:20px;}
# .section-title h2{font-size:18px;font-weight:800;color:#e4e4e7;margin-bottom:4px;}
# .section-title p{font-size:12px;color:#71717a;}
# .insight{max-width:1100px;margin:16px auto 0;padding:16px 22px;background:rgba(167,139,250,.06);border-radius:10px;border:1px solid rgba(167,139,250,.2);}
# .insight .ins-title{font-size:11px;font-weight:700;color:#a78bfa;margin-bottom:6px;}
# .insight .ins-body{font-size:11px;color:#71717a;line-height:1.8;}
# .tab-bar{display:flex;gap:0;border-bottom:2px solid #1e1e2e;margin-bottom:24px;overflow-x:auto;}
# .tab-btn{padding:12px 18px;background:none;border:none;border-bottom:2px solid transparent;color:#71717a;font-size:11px;font-weight:700;font-family:inherit;white-space:nowrap;margin-bottom:-2px;transition:all .2s;}
# .tab-btn.active{border-bottom-color:#a78bfa;color:#a78bfa;}
# .tok-chip{display:inline-block;padding:3px 8px;border-radius:4px;font-size:9px;font-family:inherit;font-weight:700;}
# .btn-sel{padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;transition:all .2s;border:1.5px solid #1e1e2e;background:#12121a;color:#71717a;}
# .btn-sel.on{background:rgba(167,139,250,.12);}
# .stat-box{text-align:center;min-width:90px;}
# .stat-box .stat-lbl{font-size:8px;color:#71717a;margin-bottom:4px;letter-spacing:1px;}
# .stat-box .stat-val{font-size:22px;font-weight:800;color:#a78bfa;}
# .stat-box .stat-sub{font-size:8px;color:#3f3f46;margin-top:2px;}
# .wrap{display:flex;flex-wrap:wrap;gap:10px;}
# </style>
# </head>
# <body>
# <div id="app" style="max-width:1400px;margin:0 auto;padding:24px 16px;"></div>
#
# <script>
# /* ═══════════════════════════════════════════════════════════
#    COLOUR PALETTE
# ═══════════════════════════════════════════════════════════ */
# var C={bg:"#0a0a0f",card:"#12121a",border:"#1e1e2e",accent:"#a78bfa",blue:"#4ecdc4",purple:"#c084fc",yellow:"#fbbf24",text:"#e4e4e7",muted:"#71717a",dim:"#3f3f46",red:"#ef4444",green:"#4ade80",cyan:"#38bdf8",pink:"#f472b6",orange:"#fb923c"};
#
# /* ═══════════════════════════════════════════════════════════
#    STATE
# ═══════════════════════════════════════════════════════════ */
# var S={
#   tab:0,
#   pipeStep:-1,
#   tokSel:2, bpeStep:0,
#   embVocab:50257, embDim:768,
#   posSel:0, posSlider:64,
#   zooSel:0,
#   quirkSel:0,
# };
#
# /* ═══════════════════════════════════════════════════════════
#    HELPERS
# ═══════════════════════════════════════════════════════════ */
# function h(tag,attrs,inner){
#   var s='<'+tag;
#   for(var k in attrs) s+=' '+k+'="'+attrs[k]+'"';
#   s+='>'+inner+'</'+tag+'>';
#   return s;
# }
# function div(style,inner){ return '<div style="'+style+'">'+inner+'</div>'; }
# function span(style,inner){ return '<span style="'+style+'">'+inner+'</span>'; }
# function hex(color,alpha){ // color like "#a78bfa", alpha 0-1
#   var r=parseInt(color.slice(1,3),16);
#   var g=parseInt(color.slice(3,5),16);
#   var b=parseInt(color.slice(5,7),16);
#   return 'rgba('+r+','+g+','+b+','+alpha+')';
# }
# function card(inner,extra){ return '<div class="card'+(extra?' '+extra:'')+'" style="max-width:1100px;margin:0 auto 14px;">'+inner+'</div>'; }
# function sectionTitle(title,sub){ return '<div class="section-title"><h2>'+title+'</h2><p>'+sub+'</p></div>'; }
# function insight(icon,title,body){ return '<div class="insight"><div class="ins-title">'+icon+' '+title+'</div><div class="ins-body">'+body+'</div></div>'; }
# function btnSel(idx,current,color,label,action){
#   var on=idx===current;
#   var bg=on?hex(color,.12):'#12121a';
#   var bc=on?color:'#1e1e2e';
#   var tc=on?color:'#71717a';
#   return '<button data-action="'+action+'" data-idx="'+idx+'" style="padding:8px 16px;border-radius:8px;font-size:10px;font-weight:700;font-family:inherit;background:'+bg+';border:1.5px solid '+bc+';color:'+tc+';cursor:pointer;transition:all .2s;">'+label+'</button>';
# }
# function chip(text,color){
#   return '<span style="padding:3px 8px;border-radius:4px;font-size:9px;font-family:inherit;font-weight:700;background:'+hex(color,.15)+';border:1px solid '+hex(color,.5)+';color:'+color+';">'+text+'</span>';
# }
# function statBox(label,value,color,sub){
#   return '<div class="stat-box"><div class="stat-lbl">'+label+'</div><div class="stat-val" style="color:'+color+';">'+value+'</div>'+(sub?'<div class="stat-sub">'+sub+'</div>':'')+'</div>';
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 1 — PIPELINE
# ═══════════════════════════════════════════════════════════ */
# function renderPipeline(){
#   var stages=[
#     {id:0,label:"Raw Text",    sub:'"The cat sat"',      color:C.muted},
#     {id:1,label:"Tokenize",    sub:"BPE split",          color:C.blue},
#     {id:2,label:"Token IDs",   sub:"[791,8415,7482]",    color:C.cyan},
#     {id:3,label:"Add BOS",     sub:"[1,791,8415,7482]",  color:C.yellow},
#     {id:4,label:"Embed Lookup",sub:"[3, 4096] floats",   color:C.accent},
#     {id:5,label:"+ Positional",sub:"RoPE/sinusoidal",    color:C.purple},
#     {id:6,label:"Transformer", sub:"32 layers → output", color:C.green},
#   ];
#   var details=[
#     {title:'Raw string — the model cannot process this directly',body:'"The cat sat on the mat" — Neural networks only understand numbers. Text must be converted.',color:C.muted},
#     {title:'BPE tokenizer splits text into subword pieces',body:'Tokens: "The" | "_cat" | "_sat" — Leading space is part of the token in GPT-style BPE.',color:C.blue},
#     {title:'Each token mapped to a unique integer ID (vocabulary index)',body:'"The" → 791 &nbsp; "_cat" → 8415 &nbsp; "_sat" → 7482 — IDs are just indices into the vocab table, not meaningful numbers.',color:C.cyan},
#     {title:'BOS token prepended; instruction markers added for chat models',body:'[BOS=1, "The"=791, "_cat"=8415, "_sat"=7482] — LLaMA uses BOS=1; BERT uses [CLS]=101.',color:C.yellow},
#     {title:'Each ID indexes a row in the embedding table [vocab × d_model]',body:'IDs 1, 791, 8415, 7482 → each maps to 4096 floats. Result tensor: [batch=1, seq=4, d_model=4096].',color:C.accent},
#     {title:'Position information injected — no inherent order in Transformers',body:'Methods: Sinusoidal | Learned | RoPE | ALiBi — LLaMA uses RoPE on Q,K inside each attention head.',color:C.purple},
#     {title:'32 transformer layers: static embeddings → contextual representations',body:'"cat" vector now knows about "sat" — context-aware at every layer. Output: logits over vocabulary.',color:C.green},
#   ];
#
#   var step=S.pipeStep;
#   var out=sectionTitle('The Complete Pipeline','Raw text → token IDs → dense vectors → positional encoding → transformer — every step, every time');
#
#   // Pipeline boxes
#   out+='<div class="card" style="max-width:1100px;margin:0 auto 14px;overflow-x:auto;">';
#   out+='<div style="display:flex;gap:0;align-items:center;min-width:700px;padding:10px 0;">';
#   stages.forEach(function(s,i){
#     var active=step===i;
#     var bg=active?hex(s.color,.13):'#0d0d14';
#     var bc=active?s.color:hex(s.color,.3);
#     var tc=active?s.color:hex(s.color,.7);
#     out+='<button data-action="pipeStep" data-idx="'+i+'" style="flex:1;min-width:90px;padding:10px 6px;border-radius:8px;border:1.5px solid '+bc+';background:'+bg+';color:'+tc+';font-size:9px;font-weight:700;font-family:inherit;cursor:pointer;transition:all .3s;line-height:1.5;'+(active?'box-shadow:0 0 10px '+hex(s.color,.3)+';':'')+'">'+s.label+'<br><span style="font-size:7px;color:'+C.dim+';font-weight:400;">'+s.sub+'</span></button>';
#     if(i<stages.length-1) out+='<div style="color:'+C.dim+';font-size:14px;padding:0 2px;flex-shrink:0;">›</div>';
#   });
#   out+='</div>';
#
#   // Detail panel
#   if(step===-1){
#     out+=div('text-align:center;padding:20px 0;color:'+C.dim+';font-size:9px;','Click any stage above to inspect it');
#   } else {
#     var d=details[step];
#     out+='<div class="fade" style="margin-top:14px;padding:14px 16px;border-radius:8px;background:'+hex(d.color,.06)+';border:1px solid '+hex(d.color,.25)+';">';
#     out+=div('font-size:11px;font-weight:700;color:'+d.color+';margin-bottom:6px;',d.title);
#     out+=div('font-size:10px;color:'+C.muted+';line-height:1.7;',d.body);
#     out+='</div>';
#   }
#   out+='</div>';
#
#   // Stats row
#   out+=card('<div style="display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">'
#     +statBox('INPUT FORMAT','String',C.muted,'raw unicode text')
#     +statBox('AFTER TOKENIZE','Integer[]',C.blue,'token IDs')
#     +statBox('AFTER EMBED','Float32[][]',C.accent,'[seq × d_model]')
#     +statBox('AFTER PE','Float32[][]',C.purple,'position-aware')
#     +statBox('AFTER TRANSFORMER','Float32[][]',C.green,'contextual reps')
#     +'</div>');
#
#   out+=insight('💡','The Two-Stage Trick','Text cannot be fed directly into a neural network. Stage 1 (tokenization) converts text to integer IDs — a lookup index, not a meaningful number. Stage 2 (embedding) converts each integer ID to a dense float vector. These are fundamentally different operations: tokenization is <strong style="color:'+C.blue+'">deterministic and fixed</strong> after training the tokenizer; the embedding table is a <strong style="color:'+C.accent+'">learned model weight</strong> updated by backpropagation.');
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 2 — TOKENIZERS
# ═══════════════════════════════════════════════════════════ */
# function renderTokenizers(){
#   var strategies=[
#     {name:'Character',color:C.muted,vocab:'~256',seqLen:'Very long',oov:'None',used:'Early char-RNNs',
#      tokens:[chip('H',C.muted),chip('e',C.muted),chip('l',C.muted),chip('l',C.muted),chip('o',C.muted)],ids:['72','69','76','76','79'],
#      pros:['No OOV problem','Tiny vocabulary (~256)','Works any language'],cons:['Very long sequences','Model learns from raw chars','Quadratic attention cost']},
#     {name:'Word',color:C.orange,vocab:'100K–1M+',seqLen:'Short',oov:'[UNK] token',used:'Legacy NLP, spaCy',
#      tokens:[chip('"The"',C.orange),chip('"quick"',C.orange),chip('"fox"',C.orange)],ids:['427','890','88'],
#      pros:['Short sequences','Each token semantically whole'],cons:['Vocabulary explosion','OOV — information lost','Morphology ignored']},
#     {name:'Subword (BPE)',color:C.accent,vocab:'30K–128K',seqLen:'Moderate',oov:'Always decomposable',used:'GPT-2/3/4, LLaMA, Mistral',
#      tokens:[chip('"token"',C.accent),chip('"ization"',C.accent)],ids:['3993','1634'],
#      pros:['No OOV ever','Manageable vocab','Morpheme-aware splits'],cons:['Suboptimal for numbers','Context-sensitive splits']},
#   ];
#
#   var bpeSteps=[
#     {label:'Step 0: Characters',desc:'Every character is its own token',
#      rows:[['l','o','w','</w>','×2'],['l','o','w','e','r','</w>','×1'],['n','e','w','e','r','</w>','×1'],['n','e','w','e','s','t','</w>','×1'],['w','i','d','e','s','t','</w>','×1']],merge:null},
#     {label:'Step 1: Merge (e,s)→es',desc:'Most frequent adjacent pair',
#      rows:[['l','o','w','</w>'],['l','o','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','es','t','</w>'],['w','i','d','es','t','</w>']],merge:'es'},
#     {label:'Step 2: Merge (es,t)→est',desc:'Second most frequent pair',
#      rows:[['l','o','w','</w>'],['l','o','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'est'},
#     {label:'Step 3: Merge (l,o)→lo',desc:'Build up common prefixes',
#      rows:[['lo','w','</w>'],['lo','w','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'lo'},
#     {label:'Step 4: Merge (lo,w)→low',desc:'Common words emerge as single tokens',
#      rows:[['low','</w>'],['low','e','r','</w>'],['n','e','w','e','r','</w>'],['n','e','w','est','</w>'],['w','i','d','est','</w>']],merge:'low'},
#     {label:'Step 5: Merge (n,e)→ne',desc:"Building 'new' prefix",
#      rows:[['low','</w>'],['low','e','r','</w>'],['ne','w','e','r','</w>'],['ne','w','est','</w>'],['w','i','d','est','</w>']],merge:'ne'},
#     {label:'Step 6: Merge (ne,w)→new',desc:'"new" is now a single token',
#      rows:[['low','</w>'],['low','e','r','</w>'],['new','e','r','</w>'],['new','est','</w>'],['w','i','d','est','</w>']],merge:'new'},
#     {label:'Step 7: Merge (new,est)→newest',desc:'Whole word merged — frequently co-occurring',
#      rows:[['low','</w>'],['low','e','r','</w>'],['new','e','r','</w>'],['newest','</w>'],['w','i','d','est','</w>']],merge:'newest'},
#   ];
#
#   var sel=S.tokSel; var v=strategies[sel];
#   var bs=bpeSteps[S.bpeStep];
#
#   var out=sectionTitle('Tokenization Strategies','Three approaches to splitting text — only subword is used by modern LLMs');
#
#   // Strategy selector
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
#   strategies.forEach(function(s,i){ out+=btnSel(i,sel,s.color,s.name,'tokSel'); });
#   out+='</div>';
#
#   // Detail card
#   out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
#   out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
#   // Left
#   out+='<div style="flex:1;min-width:260px;">';
#   out+=div('font-size:16px;font-weight:800;color:'+v.color+';margin-bottom:8px;',v.name+' Tokenization');
#   out+='<div style="display:flex;gap:14px;margin-bottom:10px;flex-wrap:wrap;">';
#   out+=div('font-size:8px;color:'+C.dim+';','VOCAB: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.vocab));
#   out+=div('font-size:8px;color:'+C.dim+';','OOV: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.oov));
#   out+=div('font-size:8px;color:'+C.dim+';','SEQ: '+span('color:'+v.color+';font-weight:700;font-size:9px;',v.seqLen));
#   out+='</div>';
#   // Token example
#   out+='<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;">';
#   v.tokens.forEach(function(t){ out+=t; });
#   out+='</div>';
#   out+=div('font-size:8px;color:'+C.dim+';margin-bottom:10px;','Used by: '+span('color:'+v.color+';',v.used));
#   out+='</div>';
#   // Right — pros/cons
#   out+='<div style="min-width:220px;">';
#   out+=div('font-size:9px;color:'+C.green+';font-weight:700;margin-bottom:6px;','&#10003; Advantages');
#   v.pros.forEach(function(p){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#10003; '+p); });
#   out+=div('font-size:9px;color:'+C.red+';font-weight:700;margin-top:10px;margin-bottom:6px;','&#9888; Trade-offs');
#   v.cons.forEach(function(p){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#9888; '+p); });
#   out+='</div>';
#   out+='</div></div>';
#
#   // BPE slider
#   out+=card('<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:6px;">BPE Algorithm — Step by Step</div>'
#     +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:14px;">Corpus: ["low"×2, "lower", "newer", "newest", "widest"] — drag the slider to trace merges</div>'
#     +'<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
#     +'<div style="font-size:9px;color:'+C.dim+';min-width:40px;">Step:</div>'
#     +'<input type="range" min="0" max="7" value="'+S.bpeStep+'" data-action="bpeStep" style="flex:1;accent-color:'+C.accent+';">'
#     +'<div style="font-size:9px;color:'+C.accent+';font-weight:700;min-width:15px;">'+S.bpeStep+'</div>'
#     +'</div>'
#     +'<div style="padding:4px 12px;border-radius:6px;background:'+hex(C.accent,.13)+';border:1px solid '+hex(C.accent,.35)+';font-size:9px;font-weight:700;color:'+C.accent+';display:inline-block;margin-bottom:8px;">'+bs.label+'</div>'
#     +div('font-size:9px;color:'+C.muted+';margin-bottom:12px;',bs.desc)
#     // token rows
#     +(function(){
#       var r='<div style="display:flex;flex-direction:column;gap:5px;">';
#       bs.rows.forEach(function(row,ri){
#         r+='<div style="display:flex;gap:4px;align-items:center;">';
#         r+='<div style="font-size:8px;color:'+C.dim+';width:20px;text-align:right;flex-shrink:0;">'+(ri+1)+'</div>';
#         row.forEach(function(tok){
#           var isMerged=bs.merge&&tok===bs.merge;
#           r+='<div style="padding:3px 7px;border-radius:4px;font-family:inherit;font-size:9px;background:'+(isMerged?hex(C.accent,.25):hex(C.dim,.25))+';border:1px solid '+(isMerged?C.accent:hex(C.dim,.5))+';color:'+(isMerged?C.accent:C.muted)+';font-weight:'+(isMerged?'800':'400')+';transition:all .4s;">'+tok+'</div>';
#         });
#         r+='</div>';
#       });
#       r+='</div>';
#       if(bs.merge) r+='<div style="margin-top:10px;padding:8px 12px;border-radius:6px;background:'+hex(C.accent,.08)+';border:1px solid '+hex(C.accent,.25)+';font-size:9px;color:'+C.muted+';">Merged token: <span style="color:'+C.accent+';font-weight:700;">'+bs.merge+'</span></div>';
#       return r;
#     })()
#   );
#
#   // Algorithm comparison
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">BPE vs WordPiece vs Unigram</div>'
#     +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
#     +[
#       {name:'BPE',color:C.accent,dir:'Bottom-up',crit:'Most frequent pair',note:'No prefix',models:'GPT-2/3/4, LLaMA, Mistral'},
#       {name:'WordPiece',color:C.blue,dir:'Bottom-up',crit:'Max likelihood ratio',note:'## continuation prefix',models:'BERT, DistilBERT'},
#       {name:'Unigram LM',color:C.purple,dir:'Top-down',crit:'Prune by likelihood impact',note:'No prefix (SentencePiece)',models:'T5, mT5, LLaMA-1/2'},
#     ].map(function(alg){
#       return '<div style="flex:1;min-width:200px;padding:12px 14px;border-radius:8px;background:'+hex(alg.color,.06)+';border:1px solid '+hex(alg.color,.25)+'">'
#         +'<div style="font-size:13px;font-weight:800;color:'+alg.color+';margin-bottom:6px;">'+alg.name+'</div>'
#         +'<div style="font-size:8px;color:'+C.dim+';line-height:1.9;">Direction: <span style="color:'+alg.color+';">'+alg.dir+'</span><br>Criterion: <span style="color:'+C.muted+';">'+alg.crit+'</span><br>Notation: <span style="color:'+C.muted+';font-family:inherit;">'+alg.note+'</span></div>'
#         +'<div style="font-size:8px;color:'+alg.color+';font-style:italic;margin-top:8px;">'+alg.models+'</div>'
#         +'</div>';
#     }).join('')
#     +'</div>'
#   );
#
#   out+=insight('&#11088;','Why Subword Won','Character tokenization produces sequences <strong style="color:'+C.red+'">10× too long</strong> — attention scales O(n²). Word tokenization causes <strong style="color:'+C.orange+'">vocabulary explosion and OOV collapse</strong>. Subword is the sweet spot: common words get their own token, rare words decompose into recognizable pieces, <strong style="color:'+C.accent+'">and OOV never occurs</strong>. Real models run thousands of BPE merges to reach their 32K–128K vocab.');
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 3 — EMBEDDING SPACE
# ═══════════════════════════════════════════════════════════ */
# function renderEmbedding(){
#   var vocab=S.embVocab, dim=S.embDim;
#   var bytes=vocab*dim*2;
#   var mb=(bytes/1e6).toFixed(0);
#   var gb=(bytes/1e9).toFixed(2);
#   var pct=(vocab*dim/7e9*100).toFixed(1);
#   var sizeStr=parseFloat(gb)<1?mb+' MB':gb+' GB';
#
#   var out=sectionTitle('The Embedding Space','IDs → dense vectors: a learned geometry where meaning is distance');
#
#   // Calculator
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:14px;">Embedding Table Explorer — Shape: [vocab_size × d_model]</div>'
#     +'<div style="display:flex;gap:30px;flex-wrap:wrap;margin-bottom:16px;">'
#     +'<div style="flex:1;min-width:220px;">'
#     +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Vocabulary size: '+vocab.toLocaleString()+'</div>'
#     +'<input type="range" min="10000" max="200000" step="1000" value="'+vocab+'" data-action="embVocab" style="accent-color:'+C.blue+';">'
#     +'<div style="display:flex;justify-content:space-between;font-size:8px;color:'+C.dim+';"><span>10K</span><span>50K</span><span>100K</span><span>200K</span></div>'
#     +'</div>'
#     +'<div style="flex:1;min-width:220px;">'
#     +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:6px;">Embedding dim (d_model): '+dim+'</div>'
#     +'<input type="range" min="256" max="8192" step="128" value="'+dim+'" data-action="embDim" style="accent-color:'+C.accent+';">'
#     +'<div style="display:flex;justify-content:space-between;font-size:8px;color:'+C.dim+';"><span>256</span><span>768</span><span>4096</span><span>8192</span></div>'
#     +'</div>'
#     +'</div>'
#     +'<div style="display:flex;justify-content:center;gap:30px;flex-wrap:wrap;margin-bottom:14px;">'
#     +statBox('TABLE SHAPE',Math.round(vocab/1000)+'K \xd7 '+dim,C.blue,'')
#     +statBox('SIZE (BF16)',sizeStr,C.accent,'')
#     +statBox('PARAMETERS',(vocab*dim/1e6).toFixed(0)+'M',C.purple,'')
#     +statBox('% OF 7B MODEL',pct+'%',C.yellow,'embedding only')
#     +'</div>'
#     // Mini grid visualisation
#     +'<div style="font-size:9px;color:'+C.dim+';margin-bottom:8px;">Token ID → Embedding Row (each cell = one float):</div>'
#     +'<div style="overflow-x:auto;">'
#     +(function(){
#       var rows=['[PAD]','[BOS]','[EOS]','"the"','"cat"','"sat"'];
#       var r='<div style="display:flex;flex-direction:column;gap:3px;">';
#       rows.forEach(function(lbl,ri){
#         r+='<div style="display:flex;gap:3px;align-items:center;">';
#         r+='<div style="font-size:8px;color:'+C.dim+';width:50px;flex-shrink:0;font-family:inherit;">'+lbl+'</div>';
#         for(var ci=0;ci<16;ci++){
#           var v=Math.sin(ri*3.7+ci*1.3);
#           var pos=v>0;
#           var intensity=Math.abs(v);
#           var col=pos?C.blue:C.purple;
#           r+='<div style="width:24px;height:18px;border-radius:3px;background:'+hex(col,intensity*.6+.1)+';flex-shrink:0;"></div>';
#         }
#         r+='<div style="font-size:7px;color:'+C.dim+';margin-left:4px;">... '+dim+' dims</div>';
#         r+='</div>';
#       });
#       r+='</div>';
#       return r;
#     })()
#     +'</div>'
#   );
#
#   // Word geometry
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Semantic Geometry — Words as Points in Space</div>'
#     +'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;">'
#     +[
#       {label:'Royalty',color:C.yellow,words:['king','queen','prince','throne']},
#       {label:'Animals',color:C.green, words:['cat','dog','lion','tiger']},
#       {label:'Places', color:C.blue,  words:['Paris','Berlin','Tokyo','Rome']},
#       {label:'Verbs',  color:C.purple,words:['run','walk','jump','swim']},
#     ].map(function(g){
#       return '<div style="flex:1;min-width:140px;padding:10px 14px;border-radius:8px;background:'+hex(g.color,.07)+';border:1px solid '+hex(g.color,.25)+'">'
#         +'<div style="font-size:10px;font-weight:700;color:'+g.color+';margin-bottom:8px;">'+g.label+'</div>'
#         +g.words.map(function(w){return '<div style="font-size:9px;color:'+C.muted+';line-height:1.9;">'+w+'</div>';}).join('')
#         +'</div>';
#     }).join('')
#     +'</div>'
#     +'<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;">Vector Analogies (Arithmetic in Embedding Space)</div>'
#     +'<div style="display:flex;flex-direction:column;gap:8px;">'
#     +[
#       {eq:'king &minus; man + woman',result:'queen',color:C.yellow},
#       {eq:'Paris &minus; France + Italy',result:'Rome',color:C.blue},
#       {eq:'swim &minus; water + air',result:'fly',color:C.cyan},
#     ].map(function(a){
#       return '<div style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;background:'+hex(a.color,.06)+';border:1px solid '+hex(a.color,.2)+'">'
#         +'<div style="font-size:10px;font-family:inherit;color:'+C.muted+';">'+a.eq+' &asymp;</div>'
#         +'<div style="font-size:12px;font-weight:800;color:'+a.color+';">'+a.result+'</div>'
#         +'</div>';
#     }).join('')
#     +'</div>'
#   );
#
#   // Static vs Contextual
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Static vs Contextual Embeddings</div>'
#     +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
#     +[
#       {title:'Static (Word2Vec, GloVe)',color:C.orange,
#        desc:'One fixed vector per word — context ignored.',
#        examples:[
#          {sent:'"I went to the bank to deposit money"',vec:'"bank" → [0.23, -0.71, ...]  (financial)'},
#          {sent:'"She sat by the river bank"',vec:'"bank" → [0.23, -0.71, ...]  (SAME vector!)'},
#        ],note:'Problem: "bank" has identical representation in both sentences.'},
#       {title:'Contextual (BERT, GPT, LLaMA)',color:C.green,
#        desc:'Vector changes based on surrounding context — from transformer layers.',
#        examples:[
#          {sent:'"I went to the bank to deposit money"',vec:'"bank" → [0.91, 0.12, ...]  (financial context)'},
#          {sent:'"She sat by the river bank"',vec:'"bank" → [-0.34, 0.88, ...]  (geographic context)'},
#        ],note:'The Transformer produces different representations for the same token in different contexts.'},
#     ].map(function(col){
#       return '<div style="flex:1;min-width:260px;padding:12px 14px;border-radius:8px;background:'+hex(col.color,.06)+';border:1px solid '+hex(col.color,.25)+'">'
#         +'<div style="font-size:11px;font-weight:800;color:'+col.color+';margin-bottom:6px;">'+col.title+'</div>'
#         +'<div style="font-size:9px;color:'+C.muted+';margin-bottom:8px;">'+col.desc+'</div>'
#         +col.examples.map(function(ex){
#           return '<div style="margin-bottom:5px;padding:5px 8px;border-radius:4px;background:'+hex(C.border,.4)+';font-family:inherit;">'
#             +'<div style="font-size:8px;color:'+col.color+';">'+ex.sent+'</div>'
#             +'<div style="font-size:7px;color:'+C.dim+';">'+ex.vec+'</div>'
#             +'</div>';
#         }).join('')
#         +'<div style="margin-top:8px;font-size:8px;color:'+C.dim+';font-style:italic;">'+col.note+'</div>'
#         +'</div>';
#     }).join('')
#     +'</div>'
#   );
#
#   out+=insight('💡','Why Embeddings Learn Meaning Without Being Taught','The model is never told "king and queen should be similar." It learns this because they appear in <strong style="color:'+C.accent+'">identical contexts</strong>: "the ___ wore a crown", "the ___ ruled the kingdom." The training loss rewards correct context prediction — placing co-occurring words nearby in vector space is the <strong style="color:'+C.purple+'">most efficient way to achieve that</strong>. Semantic similarity is a side effect of learning to predict well, not a design choice.');
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 4 — POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════ */
# function renderPositional(){
#   var methods=[
#     {name:'Sinusoidal',color:C.blue,year:'2017 (Vaswani)',learnable:'No',relative:'Partial',extrapolates:'Yes (degrades)',appliedTo:'Embeddings (added)',usedBy:'Original Transformer',
#      formula:'PE(pos,2i)   = sin(pos / 10000^(2i/d))\nPE(pos,2i+1) = cos(pos / 10000^(2i/d))',
#      desc:'Fixed math formula. Each dimension oscillates at a different frequency. No parameters to learn. Can extend beyond training length.'},
#     {name:'Learned',color:C.yellow,year:'2018 (BERT, GPT-2)',learnable:'Yes',relative:'No',extrapolates:'No (hard limit)',appliedTo:'Embeddings (added)',usedBy:'BERT (max 512), GPT-2 (max 1024)',
#      formula:'pos_embed = EmbeddingTable[pos]   (trained)',
#      desc:'Position vectors trained alongside the model. Flexible but hard-limited to max_seq_length seen during training.'},
#     {name:'RoPE',color:C.accent,year:'2021 (Su et al.)',learnable:'No',relative:'Yes (exact)',extrapolates:'Yes (w/ NTK scaling)',appliedTo:'Q, K in each attention head',usedBy:'LLaMA (all), Mistral, Falcon, Qwen',
#      formula:'[v2i\', v2i+1\'] = R(m\xb7\u03b8i) \xb7 [v2i, v2i+1]\n\u03b8i = 10000^(-2i/d)',
#      desc:'Encodes position as a rotation in 2D embedding subspaces. Q[m]\xb7K[n] depends only on (m-n), making attention naturally relative.'},
#     {name:'ALiBi',color:C.cyan,year:'2021 (Press et al.)',learnable:'No',relative:'Yes (linear)',extrapolates:'Yes (best)',appliedTo:'Attention scores (bias)',usedBy:'BLOOM (176B), MPT, OpenLLM',
#      formula:'score(q_i, k_j) = (q_i\xb7k_j)/\u221ad  \u2212  m\xb7|i\u2212j|',
#      desc:'Adds a linear distance penalty to attention scores. No position vectors. Excellent length extrapolation. Slope m is head-specific.'},
#   ];
#
#   var sel=S.posSel; var v=methods[sel];
#   var pos=S.posSlider;
#
#   // sinusoidal bars
#   var peVals=[];
#   for(var di=0;di<20;di++){
#     var freq=Math.pow(10000,(2*di)/512);
#     peVals.push(di%2===0?Math.sin(pos/freq):Math.cos(pos/freq));
#   }
#
#   var out=sectionTitle('Positional Encoding','Transformers process tokens in parallel — position must be injected explicitly');
#
#   // Why needed
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:10px;">Why Positional Encoding Is Necessary</div>'
#     +'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
#     +'<div style="flex:1;min-width:250px;padding:10px 14px;border-radius:8px;background:'+hex(C.red,.08)+';border:1px solid '+hex(C.red,.25)+'">'
#     +'<div style="font-size:10px;font-weight:700;color:'+C.red+';margin-bottom:6px;">Without PE</div>'
#     +'<div style="font-size:9px;font-family:inherit;color:'+C.muted+';line-height:2;">"The cat sat on the mat"<br>"The mat sat on the cat"</div>'
#     +'<div style="font-size:8px;color:'+C.red+';margin-top:6px;">&rarr; Identical representations. Same tokens, same attention.</div>'
#     +'</div>'
#     +'<div style="flex:1;min-width:250px;padding:10px 14px;border-radius:8px;background:'+hex(C.green,.08)+';border:1px solid '+hex(C.green,.25)+'">'
#     +'<div style="font-size:10px;font-weight:700;color:'+C.green+';margin-bottom:6px;">With PE</div>'
#     +'<div style="font-size:9px;color:'+C.muted+';line-height:2;">final[pos] = embed[token_id] + pos_encoding[pos]<br>Same token, different position &rarr; different vector</div>'
#     +'<div style="font-size:8px;color:'+C.green+';margin-top:6px;">&rarr; "cat" at pos 1 &ne; "cat" at pos 5. Order is preserved.</div>'
#     +'</div>'
#     +'</div>'
#   );
#
#   // Method selector
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
#   methods.forEach(function(m,i){ out+=btnSel(i,sel,m.color,m.name,'posSel'); });
#   out+='</div>';
#
#   // Detail card
#   out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
#   out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
#   out+='<div style="flex:1;min-width:260px;">';
#   out+=div('font-size:16px;font-weight:800;color:'+v.color+';',v.name);
#   out+=div('font-size:9px;color:'+C.dim+';margin-bottom:8px;',v.year);
#   out+=div('font-size:11px;color:'+C.muted+';line-height:1.7;margin-bottom:8px;',v.desc);
#   out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(v.color,.1)+';border:1px solid '+hex(v.color,.3)+';font-family:inherit;font-size:9px;color:'+v.color+';white-space:pre-line;">'+v.formula+'</div>';
#   out+=div('margin-top:8px;font-size:8px;color:'+C.dim+';','Applied to: '+span('color:'+v.color+';',v.appliedTo)+'<br>Used by: '+span('color:'+v.color+';',v.usedBy));
#   out+='</div>';
#   out+='<div style="min-width:180px;display:flex;flex-direction:column;gap:10px;">';
#   [{l:'Learnable params',val:v.learnable,good:v.learnable==='Yes'},{l:'Relative positions',val:v.relative,good:v.relative!=='No'},{l:'Extrapolates',val:v.extrapolates,good:v.extrapolates.startsWith('Yes')}].forEach(function(r){
#     out+='<div style="padding:8px 12px;border-radius:6px;background:'+hex(C.border,.15)+';border:1px solid '+C.border+';">'
#       +div('font-size:8px;color:'+C.dim+';margin-bottom:3px;',r.l)
#       +div('font-size:10px;font-weight:700;color:'+(r.good?C.green:C.orange)+';',r.val)
#       +'</div>';
#   });
#   out+='</div>';
#   out+='</div></div>';
#
#   // Sinusoidal visualiser
#   if(sel===0){
#     out+=card(
#       '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:8px;">Sinusoidal Pattern Visualizer</div>'
#       +'<div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">'
#       +'<div style="font-size:9px;color:'+C.muted+';min-width:80px;">Position: '+pos+'</div>'
#       +'<input type="range" min="0" max="511" value="'+pos+'" data-action="posSlider" style="flex:1;accent-color:'+C.blue+';">'
#       +'</div>'
#       +'<div style="display:flex;gap:3px;align-items:flex-end;height:90px;">'
#       +peVals.map(function(val,i){
#         var h=Math.max(Math.abs(val)*44+4,2);
#         var col=val>0?C.blue:C.purple;
#         return '<div style="flex:1;display:flex;flex-direction:column;align-items:center;">'
#           +'<div style="height:'+h+'px;border-radius:3px 3px 0 0;background:'+hex(col,Math.abs(val)*.7+.3)+';width:100%;transition:height .3s,background .3s;"></div>'
#           +(i%4===0?'<div style="font-size:6px;color:'+C.dim+';margin-top:2px;">'+i+'</div>':'')
#           +'</div>';
#       }).join('')
#       +'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';margin-top:8px;">Each bar = one dimension of the PE vector at position '+pos+'. Low dims oscillate slowly (coarse position), high dims oscillate fast (fine position).</div>'
#     );
#   }
#
#   // Comparison table
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Comparison: All Positional Encoding Methods</div>'
#     +'<div style="overflow-x:auto;">'
#     +'<table><thead><tr>'
#     +['Method','Learnable','Relative','Extrapolates','Applied To','Used By'].map(function(h){return '<th>'+h+'</th>';}).join('')
#     +'</tr></thead><tbody>'
#     +methods.map(function(m,i){
#       var bg=sel===i?hex(m.color,.1):'transparent';
#       return '<tr style="background:'+bg+';">'
#         +'<td style="color:'+m.color+';font-weight:700;border-radius:6px 0 0 6px;">'+m.name+'</td>'
#         +'<td style="color:'+(m.learnable==='Yes'?C.green:C.orange)+';">'+m.learnable+'</td>'
#         +'<td style="color:'+(m.relative==='No'?C.orange:C.green)+';">'+m.relative+'</td>'
#         +'<td style="color:'+(m.extrapolates.startsWith('Yes')?C.green:C.red)+';">'+m.extrapolates+'</td>'
#         +'<td style="color:'+C.muted+';">'+m.appliedTo+'</td>'
#         +'<td style="color:'+C.dim+';border-radius:0 6px 6px 0;">'+m.usedBy+'</td>'
#         +'</tr>';
#     }).join('')
#     +'</tbody></table>'
#     +'</div>'
#   );
#
#   out+=insight('&#9881;','Why RoPE Dominates Modern LLMs','Learned positional embeddings (BERT, GPT-2) have a <strong style="color:'+C.red+'">hard sequence length limit</strong> — trained on 1024 tokens, they cannot handle 1025. Sinusoidal degrades beyond training length. <strong style="color:'+C.accent+'">RoPE</strong> encodes position as rotation, making attention inherently relative and enabling context window extension through NTK-aware scaling. This is why every modern open-source LLM (LLaMA, Mistral, Qwen, Falcon) uses RoPE.');
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 5 — TOKENIZER ZOO
# ═══════════════════════════════════════════════════════════ */
# function renderZoo(){
#   var models=[
#     {name:'GPT-4 / GPT-3.5',algo:'BPE (tiktoken cl100k)',vocab:100256,dim:12288,contextK:'128K',released:'2022/23',color:C.green,
#      notes:['Byte-level BPE — vocab covers all UTF-8 bytes','Numbers tokenized digit-by-digit for most cases','cl100k has better multilingual coverage than GPT-2','Used by ChatGPT, GPT-4 API']},
#     {name:'LLaMA-3',algo:'BPE (tiktoken)',vocab:128256,dim:4096,contextK:'128K',released:'2024',color:C.accent,
#      notes:['Largest vocab of any major open-source model','128K vocab reduces sequence length vs LLaMA-2\'s 32K','Shares tokenizer format with GPT-4 (cl100k-based)','4\xd7 vocab increase over LLaMA-2']},
#     {name:'LLaMA-2',algo:'BPE (SentencePiece)',vocab:32000,dim:4096,contextK:'4K',released:'2023',color:C.blue,
#      notes:['Smaller 32K vocab — longer sequences than LLaMA-3','SentencePiece treats raw bytes — language agnostic','BOS=1, EOS=2, [INST]/[/INST] for chat','Default choice for open-source fine-tuning in 2023']},
#     {name:'BERT',algo:'WordPiece',vocab:30522,dim:768,contextK:'512 tok',released:'2018',color:C.yellow,
#      notes:['WordPiece marks continuations with ## prefix','[CLS] prepended, [SEP] separates segments','Max 512 tokens (learned positional embeddings)','Encoder-only — no generation, embeddings focused']},
#     {name:'T5',algo:'Unigram (SentencePiece)',vocab:32100,dim:1024,contextK:'512 tok',released:'2019',color:C.purple,
#      notes:['Unigram: starts large, prunes by likelihood','SentencePiece framework — treats text as byte stream','Excellent multilingual support out of the box','No language-specific whitespace assumptions']},
#     {name:'Mistral-7B',algo:'BPE (SentencePiece)',vocab:32000,dim:4096,contextK:'32K',released:'2023',color:C.cyan,
#      notes:['Same tokenizer as LLaMA-2 (compatible)','Sliding window attention extends effective context','&lt;s&gt; BOS, &lt;/s&gt; EOS, [INST]/[/INST] chat tokens','32K vocab — efficient, widely compatible']},
#   ];
#
#   var sel=S.zooSel; var v=models[sel];
#   var out=sectionTitle('Tokenizer Zoo','Every major model has its own tokenizer — different algorithm, vocab size, and quirks');
#
#   // Model selector
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
#   models.forEach(function(m,i){ out+=btnSel(i,sel,m.color,m.name,'zooSel'); });
#   out+='</div>';
#
#   // Detail card
#   out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
#   out+='<div style="display:flex;gap:20px;flex-wrap:wrap;">';
#   out+='<div style="flex:1;min-width:260px;">';
#   out+=div('font-size:18px;font-weight:800;color:'+v.color+';margin-bottom:4px;',v.name);
#   out+=div('font-size:10px;color:'+C.dim+';margin-bottom:12px;','Algorithm: <span style="color:'+v.color+';">'+v.algo+'</span> | Released: '+v.released);
#   v.notes.forEach(function(n){ out+=div('font-size:9px;color:'+C.muted+';line-height:1.9;','&#10003; '+n); });
#   out+='</div>';
#   out+='<div style="display:flex;flex-direction:column;gap:10px;min-width:120px;">';
#   out+=statBox('VOCAB SIZE',v.vocab.toLocaleString(),v.color,'');
#   out+=statBox('d_model',v.dim.toLocaleString(),v.color,'');
#   out+=statBox('CONTEXT',v.contextK,v.color,'');
#   out+='</div>';
#   out+='</div></div>';
#
#   // Comparison table
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Side-by-Side Comparison</div>'
#     +'<div style="overflow-x:auto;">'
#     +'<table><thead><tr>'
#     +['Model','Algorithm','Vocab Size','d_model','Context','Released'].map(function(h){return '<th>'+h+'</th>';}).join('')
#     +'</tr></thead><tbody>'
#     +models.map(function(m,i){
#       var bg=sel===i?hex(m.color,.1):'transparent';
#       return '<tr style="background:'+bg+';cursor:pointer;" data-action="zooSel" data-idx="'+i+'">'
#         +'<td style="color:'+m.color+';font-weight:700;border-radius:6px 0 0 6px;">'+m.name+'</td>'
#         +'<td style="color:'+C.muted+';">'+m.algo+'</td>'
#         +'<td style="color:'+C.accent+';font-weight:700;">'+m.vocab.toLocaleString()+'</td>'
#         +'<td style="color:'+C.blue+';">'+m.dim+'</td>'
#         +'<td style="color:'+C.cyan+';">'+m.contextK+'</td>'
#         +'<td style="color:'+C.dim+';border-radius:0 6px 6px 0;">'+m.released+'</td>'
#         +'</tr>';
#     }).join('')
#     +'</tbody></table>'
#     +'</div>'
#   );
#
#   out+=insight('&#10024;','Why Vocabulary Size Is an Engineering Tradeoff','Larger vocab (LLaMA-3: 128K) means <strong style="color:'+C.green+'">shorter sequences</strong> — same text uses fewer tokens, fitting more content in the context window. But the embedding table grows proportionally: LLaMA-3\'s 128K × 4096 table is <strong style="color:'+C.accent+'">~1 GB</strong> of model weights (12% of total). Smaller vocab (LLaMA-2: 32K) saves memory but produces longer sequences that fill the context window faster.');
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    TAB 6 — QUIRKS & GOTCHAS
# ═══════════════════════════════════════════════════════════ */
# function renderQuirks(){
#   var quirks=[
#     {title:'Leading Space Changes the Token',color:C.red,icon:'&#9888;',
#      desc:'In GPT-style BPE, a word with a leading space is a completely different token from the same word without one.',
#      examples:[
#        {input:'"dog"',tokens:['dog'],note:'→ ID 18031 (no space)'},
#        {input:'" dog"',tokens:['\u2581dog'],note:'→ ID 5679  (space included — totally different token!)'},
#        {input:'"The dog sat"',tokens:['The','\u2581dog','\u2581sat'],note:'→ "dog" uses the space version'},
#      ],
#      insight:'This is why prompt phrasing matters more than you\'d think. "Translate: cat" and "Translate cat" tokenize differently. GPT-style models mark leading-space tokens with \u2581.'},
#     {title:'Capitalization Creates New Tokens',color:C.yellow,icon:'&#9888;',
#      desc:'Case changes often produce completely different token IDs — not just re-weighted.',
#      examples:[
#        {input:'"Hello"',tokens:['Hello'],note:'→ 1 token, ID 15043'},
#        {input:'"hello"',tokens:['h','ello'],note:'→ 2 tokens! Lowercase split differently'},
#        {input:'"HELLO"',tokens:['H','E','L','L','O'],note:'→ 5 tokens (all caps tokenizes letter by letter)'},
#      ],
#      insight:'Capitalization is not "free" for LLMs. Uppercase text can explode your token count. For code: indentation, variable name casing, and symbol placement all affect tokenization.'},
#     {title:'Non-English Is Token-Expensive',color:C.purple,icon:'&#11088;',
#      desc:'Vocabularies are built on mostly English text. Non-English languages get fewer dedicated tokens.',
#      examples:[
#        {input:'"Hello"',tokens:['Hello'],note:'→ 1 token (English)'},
#        {input:'"Hola"',tokens:['Hola'],note:'→ 1 token (Spanish, common word)'},
#        {input:'"&#1055;&#1088;&#1080;&#1074;&#1077;&#1090;"',tokens:['\u041f\u0440','\u0438','\u0432\u0435\u0442'],note:'→ 3 tokens (Russian)'},
#        {input:'"\u3053\u3093\u306b\u3061\u306f"',tokens:['\u3053','\u3093','\u306b','\u3061','\u306f'],note:'→ 5 tokens (Japanese — 1 char each)'},
#      ],
#      insight:'Chinese, Japanese, and Korean characters often tokenize 1-per-character. Non-English text uses 3-5\xd7 more tokens for the same semantic content — directly inflating cost.'},
#     {title:'Code Tokenizes Differently',color:C.cyan,icon:'&#9881;',
#      desc:'Indentation, operators, and special characters each cost tokens.',
#      examples:[
#        {input:'def foo():',tokens:['def',' foo','():'],note:'→ 3 tokens'},
#        {input:'    return x+1',tokens:['    ','return',' x','+','1'],note:'→ 5 tokens (indentation is its own token!)'},
#        {input:'x = [1,2,3]',tokens:['x',' =',' [','1',',','2',',','3',']'],note:'→ 9 tokens'},
#      ],
#      insight:'Code tokenization explains why LLMs are sensitive to indentation style and whitespace. 4-space vs 2-space indent can produce different token counts and different model behaviour.'},
#   ];
#
#   var sel=S.quirkSel; var v=quirks[sel];
#   var out=sectionTitle('Quirks &amp; Gotchas','Real tokenizer behaviour that surprises every beginner — not bugs, just consequences of the algorithm');
#
#   // Selector
#   out+='<div style="display:flex;gap:6px;justify-content:center;margin-bottom:20px;flex-wrap:wrap;">';
#   quirks.forEach(function(q,i){ out+=btnSel(i,sel,q.color,q.icon+' '+q.title.split(' ').slice(0,2).join(' ')+'...','quirkSel'); });
#   out+='</div>';
#
#   // Detail card
#   out+='<div class="card fade" style="max-width:1100px;margin:0 auto 14px;border-color:'+v.color+';">';
#   out+=div('font-size:14px;font-weight:800;color:'+v.color+';margin-bottom:6px;',v.icon+' '+v.title);
#   out+=div('font-size:10px;color:'+C.muted+';margin-bottom:14px;line-height:1.7;',v.desc);
#   out+='<div style="display:flex;flex-direction:column;gap:8px;">';
#   v.examples.forEach(function(ex){
#     out+='<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap;padding:10px 14px;border-radius:8px;background:'+hex(v.color,.05)+';border:1px solid '+hex(v.color,.2)+'">'
#       +'<div style="min-width:160px;font-family:inherit;font-size:9px;color:'+v.color+';font-weight:700;">'+ex.input+'</div>'
#       +'<div style="display:flex;gap:4px;flex-wrap:wrap;flex:1;">'
#       +ex.tokens.map(function(tok){return chip(tok,v.color);}).join(' ')
#       +'</div>'
#       +'<div style="font-size:8px;color:'+C.dim+';font-family:inherit;min-width:200px;">'+ex.note+'</div>'
#       +'</div>';
#   });
#   out+='</div></div>';
#
#   out+=insight(v.icon,v.title,v.insight);
#
#   // Rule of thumb
#   out+=card(
#     '<div style="font-size:11px;font-weight:700;color:'+C.text+';margin-bottom:12px;">Practical Token Counting Rule of Thumb</div>'
#     +'<div style="display:flex;gap:10px;flex-wrap:wrap;">'
#     +[
#       {label:'English prose',ratio:'~1.3–1.5 tok/word',color:C.green,note:'Well-covered in training data'},
#       {label:'Code',ratio:'~1.5–2.5 tok/word',color:C.cyan,note:'Operators, spaces, symbols'},
#       {label:'Non-Latin script',ratio:'~2–4 tok/char',color:C.orange,note:'Fewer dedicated vocab entries'},
#       {label:'Numbers / math',ratio:'Highly variable',color:C.red,note:'1 to N digits = 1 to N tokens'},
#     ].map(function(r){
#       return '<div style="flex:1;min-width:160px;padding:10px 14px;border-radius:8px;background:'+hex(r.color,.06)+';border:1px solid '+hex(r.color,.25)+'">'
#         +'<div style="font-size:9px;font-weight:700;color:'+r.color+';margin-bottom:4px;">'+r.label+'</div>'
#         +'<div style="font-size:11px;font-weight:800;color:'+r.color+';margin-bottom:4px;">'+r.ratio+'</div>'
#         +'<div style="font-size:8px;color:'+C.dim+';">'+r.note+'</div>'
#         +'</div>';
#     }).join('')
#     +'</div>'
#     +'<div style="margin-top:12px;font-size:9px;color:'+C.dim+';text-align:center;">Token count = context window cost. Always count tokens, never estimate by word count alone.</div>'
#   );
#
#   return out;
# }
#
# /* ═══════════════════════════════════════════════════════════
#    ROOT RENDER
# ═══════════════════════════════════════════════════════════ */
# var TABS=['&#128260; Pipeline','&#9986;&#65039; Tokenizers','&#128202; Embedding Space','&#128205; Positional Encoding','&#129513; Tokenizer Zoo','&#9888;&#65039; Quirks &amp; Gotchas'];
#
# function renderApp(){
#   var html='<div style="background:'+C.bg+';min-height:100vh;padding:24px 16px;color:'+C.text+';max-width:1400px;margin:0 auto;">';
#   // Header
#   html+='<div style="text-align:center;margin-bottom:16px;">'
#     +'<div style="font-size:22px;font-weight:800;background:linear-gradient(135deg,'+C.blue+','+C.accent+','+C.purple+');-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block;">Tokenization &amp; Embeddings</div>'
#     +'<div style="font-size:11px;color:'+C.muted+';margin-top:4px;">Interactive visual walkthrough — from raw text to transformer-ready vectors</div>'
#     +'</div>';
#   // Tab bar
#   html+='<div class="tab-bar">';
#   TABS.forEach(function(t,i){
#     html+='<button class="tab-btn'+(S.tab===i?' active':'')+'" data-action="tab" data-idx="'+i+'">'+t+'</button>';
#   });
#   html+='</div>';
#   // Content
#   if(S.tab===0) html+=renderPipeline();
#   else if(S.tab===1) html+=renderTokenizers();
#   else if(S.tab===2) html+=renderEmbedding();
#   else if(S.tab===3) html+=renderPositional();
#   else if(S.tab===4) html+=renderZoo();
#   else if(S.tab===5) html+=renderQuirks();
#   html+='</div>';
#   return html;
# }
#
# function render(){
#   document.getElementById('app').innerHTML=renderApp();
#   bindEvents();
# }
#
# function bindEvents(){
#   document.querySelectorAll('[data-action]').forEach(function(el){
#     var action=el.getAttribute('data-action');
#     var idx=parseInt(el.getAttribute('data-idx'));
#     var tag=el.tagName.toLowerCase();
#
#     if(tag==='button'){
#       el.addEventListener('click',function(){
#         if(action==='tab') S.tab=idx;
#         else if(action==='pipeStep') S.pipeStep=(S.pipeStep===idx?-1:idx);
#         else if(action==='tokSel') S.tokSel=idx;
#         else if(action==='posSel') S.posSel=idx;
#         else if(action==='zooSel') S.zooSel=idx;
#         else if(action==='quirkSel') S.quirkSel=idx;
#         render();
#       });
#     } else if(tag==='tr'){
#       el.addEventListener('click',function(){
#         if(action==='zooSel') S.zooSel=idx;
#         render();
#       });
#       el.style.cursor='pointer';
#     } else if(tag==='input'){
#       el.addEventListener('input',function(){
#         var val=parseInt(this.value);
#         if(action==='bpeStep') S.bpeStep=val;
#         else if(action==='embVocab') S.embVocab=val;
#         else if(action==='embDim') S.embDim=val;
#         else if(action==='posSlider') S.posSlider=val;
#         render();
#       });
#     }
#   });
# }
#
# // Initial render
# render();
# </script>
# </body>
# </html>"""
#
# TOK_EMBED_HEIGHT = 1700
#
# # """
# # Self-contained HTML for the Tokenization & Embeddings interactive walkthrough.
# # Covers: Pipeline, Tokenizers, Embedding Space, Positional Encoding,
# # Tokenizer Zoo, and Quirks & Gotchas.
# # Embed in Streamlit via st.components.v1.html(TOK_EMBED_HTML, height=TOK_EMBED_HEIGHT).
# # """
# #
# # TOK_EMBED_HTML = """
# # <!DOCTYPE html>
# # <html>
# # <head>
# # <meta charset="utf-8"/>
# # <script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
# # <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
# # <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
# # <style>
# #   * { margin: 0; padding: 0; box-sizing: border-box; }
# #   body { background: #0a0a0f; overflow-x: hidden; }
# #   input[type="range"] { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 3px; background: #1e1e2e; outline: none; }
# #   input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 18px; height: 18px; border-radius: 50%; cursor: pointer; background: #a78bfa; }
# #   @keyframes pulse { 0%,100%{opacity:0.6} 50%{opacity:1} }
# #   @keyframes flowRight { 0%{transform:translateX(-8px);opacity:0} 50%{opacity:1} 100%{transform:translateX(8px);opacity:0} }
# #   @keyframes glow { 0%,100%{box-shadow:0 0 6px rgba(167,139,250,0.3)} 50%{box-shadow:0 0 16px rgba(167,139,250,0.7)} }
# #   @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
# # </style>
# # </head>
# # <body>
# # <div id="root"></div>
# # <script type="text/babel">
# #
# # var useState   = React.useState;
# # var useEffect  = React.useEffect;
# # var useMemo    = React.useMemo;
# #
# # /* ─── colour palette (same as PEFT visual) ─── */
# # var C = {
# #   bg: "#0a0a0f", card: "#12121a", border: "#1e1e2e",
# #   accent: "#a78bfa", blue: "#4ecdc4", purple: "#c084fc",
# #   yellow: "#fbbf24", text: "#e4e4e7", muted: "#71717a",
# #   dim: "#3f3f46", red: "#ef4444", green: "#4ade80",
# #   cyan: "#38bdf8", pink: "#f472b6", orange: "#fb923c",
# # };
# #
# # /* ─── unicode symbols ─── */
# # var ARR   = "\u2192";
# # var DASH  = "\u2014";
# # var CHK   = "\u2713";
# # var WARN  = "\u26A0";
# # var BULB  = "\uD83D\uDCA1";
# # var LOCK  = "\uD83D\uDD12";
# # var BRAIN = "\uD83E\uDDE0";
# # var GEAR  = "\u2699";
# # var STAR  = "\u2605";
# # var TEXT  = "\uD83D\uDCDD";
# # var NUM   = "\uD83D\uDD22";
# # var MAG   = "\uD83D\uDD0D";
# # var TARG  = "\uD83C\uDFAF";
# # var WARN2 = "\u26A0\uFE0F";
# # var PLUS  = "+";
# # var MUL   = "\u00D7";
# #
# # /* ─── Shared layout components ─── */
# # function TabBar(props) {
# #   var tabs = props.tabs, active = props.active, onChange = props.onChange;
# #   return (
# #     <div style={{ display:"flex", gap:0, borderBottom:"2px solid "+C.border, marginBottom:24, overflowX:"auto" }}>
# #       {tabs.map(function(t,i) {
# #         return (
# #           <button key={i} onClick={function(){ onChange(i); }} style={{
# #             padding:"12px 18px", background:"none", border:"none",
# #             borderBottom: active===i ? "2px solid "+C.accent : "2px solid transparent",
# #             color: active===i ? C.accent : C.muted, cursor:"pointer",
# #             fontSize:11, fontWeight:700, fontFamily:"'JetBrains Mono',monospace",
# #             transition:"all 0.2s", whiteSpace:"nowrap", marginBottom:-2,
# #           }}>{t}</button>
# #         );
# #       })}
# #     </div>
# #   );
# # }
# #
# # function Card(props) {
# #   return (
# #     <div style={Object.assign({
# #       background:C.card, borderRadius:10, padding:"18px 22px",
# #       border:"1px solid "+(props.highlight ? C.accent : C.border),
# #       transition:"border 0.3s",
# #     }, props.style||{})}>
# #       {props.children}
# #     </div>
# #   );
# # }
# #
# # function Insight(props) {
# #   return (
# #     <div style={Object.assign({
# #       maxWidth:1100, margin:"16px auto 0",
# #       padding:"16px 22px", background:"rgba(167,139,250,0.06)",
# #       borderRadius:10, border:"1px solid rgba(167,139,250,0.2)",
# #     }, props.style||{})}>
# #       <div style={{ fontSize:11, fontWeight:700, color:C.accent, marginBottom:6 }}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
# #       <div style={{ fontSize:11, color:C.muted, lineHeight:1.8 }}>{props.children}</div>
# #     </div>
# #   );
# # }
# #
# # function SectionTitle(props) {
# #   return (
# #     <div style={{ textAlign:"center", marginBottom:20 }}>
# #       <div style={{ fontSize:18, fontWeight:800, color:C.text, marginBottom:4 }}>{props.title}</div>
# #       <div style={{ fontSize:12, color:C.muted }}>{props.subtitle}</div>
# #     </div>
# #   );
# # }
# #
# # function StatBox(props) {
# #   return (
# #     <div style={{ textAlign:"center", minWidth:props.minW||90 }}>
# #       <div style={{ fontSize:8, color:C.muted, marginBottom:4, letterSpacing:1 }}>{props.label}</div>
# #       <div style={{ fontSize:props.bigFont||22, fontWeight:800, color:props.color||C.accent }}>{props.value}</div>
# #       {props.sub && <div style={{ fontSize:8, color:C.dim, marginTop:2 }}>{props.sub}</div>}
# #     </div>
# #   );
# # }
# #
# # /* ===============================================================
# #    TAB 1 — THE PIPELINE
# #    =============================================================== */
# # function TabPipeline() {
# #   var _a = useState(false); var anim = _a[0], setAnim = _a[1];
# #   var _s = useState(-1); var step = _s[0], setStep = _s[1];
# #
# #   useEffect(function(){ var t=setTimeout(function(){ setAnim(true); },400); return function(){ clearTimeout(t); }; },[]);
# #
# #   var stages = [
# #     { id:0, label:"Raw Text",       sub:'"The cat sat"',        color:C.muted,  x:30  },
# #     { id:1, label:"Tokenize",       sub:"BPE split",            color:C.blue,   x:165 },
# #     { id:2, label:"Token IDs",      sub:"[791, 8415, 7482]",    color:C.cyan,   x:300 },
# #     { id:3, label:"Add BOS",        sub:"[1, 791, 8415, 7482]", color:C.yellow, x:435 },
# #     { id:4, label:"Embed Lookup",   sub:"[3, 4096] floats",     color:C.accent, x:570 },
# #     { id:5, label:"+ Positional",   sub:"RoPE / sinusoidal",    color:C.purple, x:705 },
# #     { id:6, label:"Transformer",    sub:"32 layers → output",   color:C.green,  x:840 },
# #   ];
# #
# #   return (
# #     <div>
# #       <SectionTitle
# #         title="The Complete Pipeline"
# #         subtitle={"Raw text \u2192 token IDs \u2192 dense vectors \u2192 positional encoding \u2192 transformer "+DASH+" every step, every time"}
# #       />
# #
# #       {/* Main pipeline SVG */}
# #       <div style={{ display:"flex", justifyContent:"center", marginBottom:20 }}>
# #         <svg width={1050} height={310} viewBox="0 0 980 310"
# #           style={{ background:"#08080d", borderRadius:10, border:"1px solid "+C.border }}>
# #
# #           {/* Stage boxes */}
# #           {stages.map(function(s,i){
# #             var active = step===i;
# #             var gone   = anim && step===-1;
# #             return (
# #               <g key={i} style={{ cursor:"pointer" }} onClick={function(){ setStep(step===i ? -1 : i); }}>
# #                 <rect x={s.x} y={80} width={115} height={60} rx={8}
# #                   fill={active ? s.color+"22" : "#0d0d14"}
# #                   stroke={active ? s.color : s.color+"50"} strokeWidth={active?2:1.2}
# #                   style={{ transition:"all 0.3s", filter: active ? "drop-shadow(0 0 8px "+s.color+"60)" : "none" }}/>
# #                 <text x={s.x+57} y={107} textAnchor="middle" fill={active ? s.color : s.color+"bb"}
# #                   fontSize={9} fontWeight={700} fontFamily="monospace">{s.label}</text>
# #                 <text x={s.x+57} y={122} textAnchor="middle" fill={active ? s.color+"cc" : C.dim}
# #                   fontSize={7} fontFamily="monospace">{s.sub}</text>
# #                 {/* connector arrow */}
# #                 {i<stages.length-1 && (
# #                   <g>
# #                     <line x1={s.x+115} y1={110} x2={s.x+130} y2={110}
# #                       stroke={anim ? s.color+"80" : C.dim+"30"} strokeWidth={1.5}
# #                       style={{ transition:"stroke 1s", transitionDelay:(i*0.12)+"s" }}/>
# #                     <polygon points={(s.x+132)+",110 "+(s.x+126)+",106 "+(s.x+126)+",114"}
# #                       fill={anim ? s.color+"80" : C.dim+"30"}
# #                       style={{ transition:"fill 1s", transitionDelay:(i*0.12)+"s" }}/>
# #                   </g>
# #                 )}
# #               </g>
# #             );
# #           })}
# #
# #           {/* Detail panel per step */}
# #           {step===-1 && (
# #             <text x={490} y={200} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">
# #               {"Click any stage above to inspect it"}
# #             </text>
# #           )}
# #           {step===0 && (
# #             <g>
# #               <text x={490} y={185} textAnchor="middle" fill={C.muted} fontSize={10} fontWeight={700} fontFamily="monospace">{"Raw string — the model cannot process this directly"}</text>
# #               <text x={490} y={205} textAnchor="middle" fill={C.blue} fontSize={11} fontFamily="monospace">{'"The cat sat on the mat"'}</text>
# #               <text x={490} y={225} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Neural networks only understand numbers. Text must be converted."}</text>
# #             </g>
# #           )}
# #           {step===1 && (
# #             <g>
# #               <text x={490} y={180} textAnchor="middle" fill={C.blue} fontSize={10} fontWeight={700} fontFamily="monospace">{"BPE tokenizer splits text into subword pieces"}</text>
# #               {['"The"','"_cat"','"_sat"'].map(function(tok,i){
# #                 return (
# #                   <g key={i}>
# #                     <rect x={340+i*120} y={190} width={90} height={26} rx={5}
# #                       fill={C.blue+"18"} stroke={C.blue+"60"} strokeWidth={1}/>
# #                     <text x={385+i*120} y={207} textAnchor="middle"
# #                       fill={C.blue} fontSize={9} fontFamily="monospace">{tok}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={240} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Leading space is part of the token in GPT-style BPE"}</text>
# #             </g>
# #           )}
# #           {step===2 && (
# #             <g>
# #               <text x={490} y={180} textAnchor="middle" fill={C.cyan} fontSize={10} fontWeight={700} fontFamily="monospace">{"Each token mapped to a unique integer ID (vocabulary index)"}</text>
# #               {[["\"The\"","791"],["\"_cat\"","8415"],["\"_sat\"","7482"]].map(function(p,i){
# #                 return (
# #                   <g key={i}>
# #                     <text x={340+i*120} y={205} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{p[0]}</text>
# #                     <text x={390+i*120} y={205} textAnchor="middle" fill={C.cyan} fontSize={9} fontFamily="monospace">{ARR}</text>
# #                     <text x={440+i*120} y={205} textAnchor="middle" fill={C.cyan} fontSize={11} fontWeight={800} fontFamily="monospace">{p[1]}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={240} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"IDs are just indices into the vocab table — not meaningful numbers"}</text>
# #             </g>
# #           )}
# #           {step===3 && (
# #             <g>
# #               <text x={490} y={185} textAnchor="middle" fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">{"BOS token prepended; instruction markers added for chat models"}</text>
# #               {[["BOS","1"],["\"The\"","791"],["\"_cat\"","8415"],["\"_sat\"","7482"]].map(function(p,i){
# #                 return (
# #                   <g key={i}>
# #                     <rect x={270+i*110} y={195} width={85} height={26} rx={5}
# #                       fill={i===0 ? C.yellow+"22" : C.dim+"15"}
# #                       stroke={i===0 ? C.yellow : C.dim+"40"} strokeWidth={i===0?1.5:1}/>
# #                     <text x={313+i*110} y={212} textAnchor="middle"
# #                       fill={i===0 ? C.yellow : C.muted} fontSize={9} fontFamily="monospace">{p[1]}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={242} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"LLaMA uses BOS=1; BERT uses [CLS]=101; GPT-2 uses <|endoftext|>"}</text>
# #             </g>
# #           )}
# #           {step===4 && (
# #             <g>
# #               <text x={490} y={180} textAnchor="middle" fill={C.accent} fontSize={10} fontWeight={700} fontFamily="monospace">{"Each ID indexes a row in the embedding table [vocab \u00D7 d_model]"}</text>
# #               <rect x={220} y={192} width={520} height={50} rx={6} fill={C.accent+"08"} stroke={C.accent+"30"} strokeWidth={1}/>
# #               {["ID 1","ID 791","ID 8415","ID 7482"].map(function(lbl,i){
# #                 return (
# #                   <g key={i}>
# #                     <text x={260+i*120} y={210} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{lbl}</text>
# #                     <text x={260+i*120} y={228} textAnchor="middle" fill={C.accent} fontSize={7} fontFamily="monospace">{"[4096 floats]"}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={258} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"Result tensor: [batch=1, seq=4, d_model=4096]"}</text>
# #             </g>
# #           )}
# #           {step===5 && (
# #             <g>
# #               <text x={490} y={182} textAnchor="middle" fill={C.purple} fontSize={10} fontWeight={700} fontFamily="monospace">{"Position information injected — no inherent order in Transformers"}</text>
# #               {["Sinusoidal","Learned","RoPE","ALiBi"].map(function(name,i){
# #                 var cols=[C.blue,C.yellow,C.accent,C.cyan];
# #                 return (
# #                   <g key={i}>
# #                     <rect x={230+i*130} y={194} width={110} height={28} rx={5}
# #                       fill={cols[i]+"12"} stroke={cols[i]+"50"} strokeWidth={1}/>
# #                     <text x={285+i*130} y={212} textAnchor="middle"
# #                       fill={cols[i]} fontSize={8} fontWeight={700} fontFamily="monospace">{name}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={242} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{"LLaMA uses RoPE on Q,K inside each attention head"}</text>
# #             </g>
# #           )}
# #           {step===6 && (
# #             <g>
# #               <text x={490} y={183} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"32 transformer layers: static embeddings \u2192 contextual representations"}</text>
# #               {[0,1,2,3].map(function(i){
# #                 return (
# #                   <g key={i}>
# #                     <rect x={260+i*120} y={193} width={100} height={28} rx={5}
# #                       fill={C.green+"0"+(8+i*4)} stroke={C.green+"40"} strokeWidth={1}
# #                       style={{ animation:"pulse 1.5s infinite", animationDelay:(i*0.2)+"s" }}/>
# #                     <text x={310+i*120} y={211} textAnchor="middle"
# #                       fill={C.green} fontSize={8} fontFamily="monospace">{"Layer "+(i===3?"...32":i+1)}</text>
# #                   </g>
# #                 );
# #               })}
# #               <text x={490} y={240} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{'"cat" vector now knows about "sat" — context-aware at every layer'}</text>
# #             </g>
# #           )}
# #
# #           {/* stage labels row */}
# #           {stages.map(function(s,i){
# #             return (
# #               <text key={"lbl"+i} x={s.x+57} y={160} textAnchor="middle"
# #                 fill={step===i ? s.color : C.dim} fontSize={7} fontFamily="monospace">
# #                 {"STEP "+(i+1)}
# #               </text>
# #             );
# #           })}
# #         </svg>
# #       </div>
# #
# #       {/* Why two stages */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>
# #           {"Why Two Separate Stages? "+DASH+" Tokenization vs Embedding"}
# #         </div>
# #         <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
# #           {[
# #             { title:"Tokenization", sub:"CPU, before model",
# #               points:["Deterministic lookup — same input always same output","Fixed vocabulary built once at training time","Runs on CPU, no neural network needed","Output: integers (vocabulary indices)"],
# #               color:C.blue },
# #             { title:"Embedding", sub:"GPU, inside model",
# #               points:["Learned weight matrix — updated by backprop","Each ID mapped to a dense floating-point vector","The model learns what each token 'means'","Output: d_model floats per token"],
# #               color:C.accent },
# #           ].map(function(col,i){
# #             return (
# #               <div key={i} style={{ flex:1, minWidth:280, padding:"12px 16px", borderRadius:8,
# #                 background:col.color+"06", border:"1px solid "+col.color+"25" }}>
# #                 <div style={{ fontSize:13, fontWeight:800, color:col.color, marginBottom:2 }}>{col.title}</div>
# #                 <div style={{ fontSize:9, color:C.dim, marginBottom:8 }}>{col.sub}</div>
# #                 {col.points.map(function(p,j){
# #                   return <div key={j} style={{ fontSize:9, color:C.muted, lineHeight:1.9 }}>{CHK+" "+p}</div>;
# #                 })}
# #               </div>
# #             );
# #           })}
# #         </div>
# #       </Card>
# #
# #       {/* Stats row */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ display:"flex", justifyContent:"center", gap:30, flexWrap:"wrap" }}>
# #           <StatBox label="TYPICAL VOCAB SIZE"    value="32K–128K"  color={C.blue}   />
# #           <StatBox label="TOKENS PER WORD (EN)"  value="~1.3–1.5" color={C.cyan}   />
# #           <StatBox label="EMBED DIM (LLaMA-3)"   value="4,096"    color={C.accent} />
# #           <StatBox label="EMBED TABLE (LLaMA-3)" value="~1 GB"    color={C.purple} sub="128K × 4096 × bf16"/>
# #         </div>
# #       </Card>
# #
# #       <Insight icon={BRAIN} title="The Core Problem">
# #         Neural networks only process numbers. But you can't just assign <span style={{color:C.red,fontWeight:700}}>word → number</span> directly — that would impose a false numeric ordering ("cat"=3 means nothing). The solution: tokenization assigns <span style={{color:C.blue,fontWeight:700}}>arbitrary integer IDs</span> (indices), then the embedding layer converts those indices into <span style={{color:C.accent,fontWeight:700}}>learned dense vectors</span> where position in space carries real meaning.
# #       </Insight>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    TAB 2 — TOKENIZERS
# #    =============================================================== */
# # function TabTokenizers() {
# #   var _sel = useState(2); var sel = _sel[0], setSel = _sel[1];
# #   var _step = useState(0); var bpeStep = _step[0], setBpeStep = _step[1];
# #   var _anim = useState(false); var anim = _anim[0], setAnim = _anim[1];
# #
# #   useEffect(function(){ var t=setTimeout(function(){ setAnim(true); },300); return function(){ clearTimeout(t); }; },[]);
# #
# #   var strategies = [
# #     {
# #       name:"Character", short:"CHAR", color:C.blue,
# #       vocab:"~256", seqLen:"Very long", oov:"None",
# #       used:"Early char-RNNs",
# #       example:{ input:'"Hello"', tokens:['H','e','l','l','o'], ids:['72','69','76','76','79'] },
# #       pros:["No OOV problem","Tiny vocabulary","Works any language"],
# #       cons:["Very long sequences","Model must learn from characters","Quadratic attention cost"],
# #     },
# #     {
# #       name:"Word", short:"WORD", color:C.orange,
# #       vocab:"100K–1M+", seqLen:"Short", oov:"[UNK] token",
# #       used:"Legacy NLP, spaCy",
# #       example:{ input:'"The quick fox"', tokens:['"The"','"quick"','"fox"'], ids:['427','890','88'] },
# #       pros:["Short sequences","Each token semantically whole"],
# #       cons:["Vocabulary explosion","OOV problem — info lost","Morphology ignored"],
# #     },
# #     {
# #       name:"Subword (BPE)", short:"SUBWORD", color:C.accent,
# #       vocab:"30K–128K", seqLen:"Moderate", oov:"Always decomposable",
# #       used:"GPT-2/3/4, LLaMA, Mistral",
# #       example:{ input:'"tokenization"', tokens:['"token"','"ization"'], ids:['3993','1634'] },
# #       pros:["No OOV ever","Manageable vocab","Morpheme-aware splits"],
# #       cons:["Suboptimal for numbers/code","Context-sensitive splits"],
# #     },
# #   ];
# #
# #   var bpeSteps = [
# #     { label:"Step 0: Characters",     desc:"Every character is its own token",
# #       tokens:[["l","o","w","</w>","×2"],["l","o","w","e","r","</w>","×1"],["n","e","w","e","r","</w>","×1"],["n","e","w","e","s","t","</w>","×1"],["w","i","d","e","s","t","</w>","×1"]], merge:null },
# #     { label:"Step 1: Merge (e,s)→es",  desc:"Most frequent adjacent pair in corpus",
# #       tokens:[["l","o","w","</w>"],["l","o","w","e","r","</w>"],["n","e","w","e","r","</w>"],["n","e","w","es","t","</w>"],["w","i","d","es","t","</w>"]], merge:["e","s"] },
# #     { label:"Step 2: Merge (es,t)→est", desc:"Second most frequent pair",
# #       tokens:[["l","o","w","</w>"],["l","o","w","e","r","</w>"],["n","e","w","e","r","</w>"],["n","e","w","est","</w>"],["w","i","d","est","</w>"]], merge:["es","t"] },
# #     { label:"Step 3: Merge (l,o)→lo",   desc:"Build up common prefixes",
# #       tokens:[["lo","w","</w>"],["lo","w","e","r","</w>"],["n","e","w","e","r","</w>"],["n","e","w","est","</w>"],["w","i","d","est","</w>"]], merge:["l","o"] },
# #     { label:"Step 4: Merge (lo,w)→low", desc:"Common words emerge as single tokens",
# #       tokens:[["low","</w>"],["low","e","r","</w>"],["n","e","w","e","r","</w>"],["n","e","w","est","</w>"],["w","i","d","est","</w>"]], merge:["lo","w"] },
# #     { label:"Step 5: Merge (n,e)→ne",  desc:"Building 'new' prefix",
# #       tokens:[["low","</w>"],["low","e","r","</w>"],["ne","w","e","r","</w>"],["ne","w","est","</w>"],["w","i","d","est","</w>"]], merge:["n","e"] },
# #     { label:"Step 6: Merge (ne,w)→new", desc:'"new" is now a single token',
# #       tokens:[["low","</w>"],["low","e","r","</w>"],["new","e","r","</w>"],["new","est","</w>"],["w","i","d","est","</w>"]], merge:["ne","w"] },
# #     { label:"Step 7: Merge (new,est)→newest", desc:"Whole word merged — frequently co-occurring",
# #       tokens:[["low","</w>"],["low","e","r","</w>"],["new","e","r","</w>"],["newest","</w>"],["w","i","d","est","</w>"]], merge:["new","est"] },
# #   ];
# #
# #   var v = strategies[sel];
# #   var bs = bpeSteps[bpeStep];
# #   var mergedTok = bs.merge ? bs.merge[0]+bs.merge[1] : null;
# #
# #   return (
# #     <div>
# #       <SectionTitle title="Tokenization Strategies" subtitle={"Three approaches to splitting text "+DASH+" only subword is used by modern LLMs"} />
# #
# #       {/* Strategy selector */}
# #       <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
# #         {strategies.map(function(s,i){
# #           var on=sel===i;
# #           return (
# #             <button key={i} onClick={function(){ setSel(i); }} style={{
# #               padding:"8px 20px", borderRadius:8,
# #               border:"1.5px solid "+(on ? s.color : C.border),
# #               background: on ? s.color+"20" : C.card, color: on ? s.color : C.muted,
# #               cursor:"pointer", fontSize:11, fontWeight:700, fontFamily:"monospace",
# #               transition:"all 0.2s"
# #             }}>{s.name}</button>
# #           );
# #         })}
# #       </div>
# #
# #       {/* Detail card */}
# #       <Card highlight={true} style={{ maxWidth:1100, margin:"0 auto 16px", borderColor:v.color }}>
# #         <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
# #           <div style={{ flex:1, minWidth:260 }}>
# #             <div style={{ fontSize:16, fontWeight:800, color:v.color, marginBottom:6 }}>{v.name} Tokenization</div>
# #             <div style={{ display:"flex", gap:16, marginBottom:10, flexWrap:"wrap" }}>
# #               <div><span style={{ fontSize:8, color:C.dim }}>VOCAB: </span><span style={{ fontSize:9, color:v.color, fontWeight:700 }}>{v.vocab}</span></div>
# #               <div><span style={{ fontSize:8, color:C.dim }}>SEQ LEN: </span><span style={{ fontSize:9, color:v.color, fontWeight:700 }}>{v.seqLen}</span></div>
# #               <div><span style={{ fontSize:8, color:C.dim }}>OOV: </span><span style={{ fontSize:9, color:v.color, fontWeight:700 }}>{v.oov}</span></div>
# #             </div>
# #             <div style={{ marginBottom:10 }}>
# #               <div style={{ fontSize:9, color:C.dim, marginBottom:4 }}>Example: {v.example.input}</div>
# #               <div style={{ display:"flex", gap:4, flexWrap:"wrap" }}>
# #                 {v.example.tokens.map(function(tok,i){
# #                   return (
# #                     <div key={i} style={{ padding:"3px 8px", borderRadius:4, background:v.color+"18",
# #                       border:"1px solid "+v.color+"50", fontSize:9, color:v.color, fontFamily:"monospace" }}>
# #                       {tok}
# #                       <span style={{ fontSize:7, color:C.dim, display:"block", textAlign:"center" }}>{v.example.ids[i]}</span>
# #                     </div>
# #                   );
# #                 })}
# #               </div>
# #             </div>
# #             <div style={{ fontSize:8, color:C.dim, marginBottom:4 }}>Used by: <span style={{color:v.color}}>{v.used}</span></div>
# #           </div>
# #           <div style={{ minWidth:220 }}>
# #             <div style={{ fontSize:9, color:C.green, fontWeight:700, marginBottom:6 }}>{CHK+" Advantages"}</div>
# #             {v.pros.map(function(p,i){ return (<div key={i} style={{ fontSize:9, color:C.muted, lineHeight:1.9 }}>{CHK+" "+p}</div>); })}
# #             <div style={{ fontSize:9, color:C.red, fontWeight:700, marginBottom:6, marginTop:10 }}>{WARN+" Trade-offs"}</div>
# #             {v.cons.map(function(p,i){ return (<div key={i} style={{ fontSize:9, color:C.muted, lineHeight:1.9 }}>{WARN+" "+p}</div>); })}
# #           </div>
# #         </div>
# #       </Card>
# #
# #       {/* BPE step-by-step */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:6 }}>
# #           {"BPE Algorithm "+DASH+" Step by Step"}
# #         </div>
# #         <div style={{ fontSize:9, color:C.muted, marginBottom:14 }}>
# #           {"Corpus: [\"low\"×2, \"lower\", \"newer\", \"newest\", \"widest\"] — drag the slider to trace merges"}
# #         </div>
# #
# #         <div style={{ display:"flex", alignItems:"center", gap:14, marginBottom:14 }}>
# #           <div style={{ fontSize:9, color:C.dim, minWidth:40 }}>{"Step:"}</div>
# #           <input type="range" min={0} max={bpeSteps.length-1} value={bpeStep}
# #             onChange={function(e){ setBpeStep(parseInt(e.target.value)); }}
# #             style={{ flex:1, accentColor:C.accent }}/>
# #           <div style={{ fontSize:9, color:C.accent, fontWeight:700, minWidth:15 }}>{bpeStep}</div>
# #         </div>
# #
# #         {/* Step label */}
# #         <div style={{ display:"flex", gap:10, alignItems:"center", marginBottom:12 }}>
# #           <div style={{ padding:"4px 12px", borderRadius:6, background:C.accent+"20",
# #             border:"1px solid "+C.accent+"50", fontSize:9, fontWeight:700, color:C.accent }}>
# #             {bs.label}
# #           </div>
# #           <div style={{ fontSize:9, color:C.muted }}>{bs.desc}</div>
# #         </div>
# #
# #         {/* Token rows */}
# #         <div style={{ display:"flex", flexDirection:"column", gap:5 }}>
# #           {bs.tokens.map(function(row,ri){
# #             return (
# #               <div key={ri} style={{ display:"flex", gap:4, alignItems:"center" }}>
# #                 <div style={{ fontSize:8, color:C.dim, width:24, textAlign:"right", flexShrink:0 }}>{ri+1}</div>
# #                 {row.map(function(tok,ti){
# #                   var isMerged = mergedTok && tok===mergedTok;
# #                   return (
# #                     <div key={ti} style={{
# #                       padding:"3px 7px", borderRadius:4, fontFamily:"monospace", fontSize:9,
# #                       background: isMerged ? C.accent+"30" : C.dim+"20",
# #                       border:"1px solid "+(isMerged ? C.accent : C.dim+"40"),
# #                       color: isMerged ? C.accent : C.muted,
# #                       fontWeight: isMerged ? 800 : 400,
# #                       transition:"all 0.4s"
# #                     }}>{tok}</div>
# #                   );
# #                 })}
# #               </div>
# #             );
# #           })}
# #         </div>
# #
# #         {bs.merge && (
# #           <div style={{ marginTop:10, padding:"8px 12px", borderRadius:6,
# #             background:C.accent+"08", border:"1px solid "+C.accent+"25" }}>
# #             <span style={{ fontSize:9, color:C.dim }}>Merge: </span>
# #             <span style={{ fontSize:9, fontFamily:"monospace", color:C.purple }}>({bs.merge[0]}, {bs.merge[1]})</span>
# #             <span style={{ fontSize:9, color:C.dim }}> {ARR} </span>
# #             <span style={{ fontSize:9, fontFamily:"monospace", color:C.accent, fontWeight:700 }}>{mergedTok}</span>
# #           </div>
# #         )}
# #       </Card>
# #
# #       {/* WordPiece vs Unigram */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>{"BPE vs WordPiece vs Unigram — The Three Subword Algorithms"}</div>
# #         <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
# #           {[
# #             { name:"BPE", color:C.accent, direction:"Bottom-up",
# #               criterion:"Most frequent pair", notation:"No prefix",
# #               models:"GPT-2/3/4, LLaMA, Mistral" },
# #             { name:"WordPiece", color:C.blue, direction:"Bottom-up",
# #               criterion:"Highest likelihood ratio freq(AB)/(freq(A)×freq(B))",
# #               notation:"## continuation prefix", models:"BERT, DistilBERT" },
# #             { name:"Unigram LM", color:C.purple, direction:"Top-down",
# #               criterion:"Prune tokens with least likelihood impact",
# #               notation:"No prefix (SentencePiece)", models:"T5, mT5, LLaMA-1/2" },
# #           ].map(function(alg,i){
# #             return (
# #               <div key={i} style={{ flex:1, minWidth:200, padding:"12px 14px", borderRadius:8,
# #                 background:alg.color+"06", border:"1px solid "+alg.color+"25" }}>
# #                 <div style={{ fontSize:13, fontWeight:800, color:alg.color, marginBottom:6 }}>{alg.name}</div>
# #                 <div style={{ fontSize:8, color:C.dim, marginBottom:2 }}>Direction: <span style={{color:alg.color}}>{alg.direction}</span></div>
# #                 <div style={{ fontSize:8, color:C.dim, marginBottom:2 }}>Criterion: <span style={{color:C.muted}}>{alg.criterion}</span></div>
# #                 <div style={{ fontSize:8, color:C.dim, marginBottom:8 }}>Notation: <span style={{color:C.muted,fontFamily:"monospace"}}>{alg.notation}</span></div>
# #                 <div style={{ fontSize:8, color:alg.color, fontStyle:"italic" }}>{alg.models}</div>
# #               </div>
# #             );
# #           })}
# #         </div>
# #       </Card>
# #
# #       <Insight icon={STAR} title="Why Subword Won">
# #         Character tokenization produces <span style={{color:C.red,fontWeight:700}}>sequences 10× too long</span> — attention scales O(n²). Word tokenization causes <span style={{color:C.orange,fontWeight:700}}>vocabulary explosion and OOV collapse</span>. Subword is the sweet spot: common words get their own token, rare words decompose into recognizable pieces, <span style={{color:C.accent,fontWeight:700}}>and OOV never occurs</span>. Real models run thousands of BPE merges to reach their 32K–128K vocab.
# #       </Insight>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    TAB 3 — EMBEDDING SPACE
# #    =============================================================== */
# # function TabEmbedding() {
# #   var _v = useState(50257); var vocab = _v[0], setVocab = _v[1];
# #   var _d = useState(768);  var dim   = _d[0], setDim   = _d[1];
# #   var _anim = useState(false); var anim = _anim[0], setAnim = _anim[1];
# #   var _hov  = useState(-1);   var hov  = _hov[0],  setHov  = _hov[1];
# #
# #   useEffect(function(){ var t=setTimeout(function(){ setAnim(true); },400); return function(){ clearTimeout(t); }; },[]);
# #
# #   var tableBytes = vocab * dim * 2; // bfloat16
# #   var tableMB    = (tableBytes / 1e6).toFixed(0);
# #   var tableGB    = (tableBytes / 1e9).toFixed(2);
# #   var totalPct   = (vocab * dim / (7e9)).toFixed(2); // fraction of 7B model
# #
# #   var wordGroups = [
# #     { label:"Royalty",  color:C.yellow,  words:["king","queen","prince","throne"] },
# #     { label:"Animals",  color:C.green,   words:["cat","dog","lion","tiger"]       },
# #     { label:"Places",   color:C.blue,    words:["Paris","Berlin","Tokyo","Rome"]  },
# #     { label:"Verbs",    color:C.purple,  words:["run","walk","jump","swim"]       },
# #   ];
# #
# #   var analogies = [
# #     { eq:"king - man + woman",    result:"queen",  color:C.yellow  },
# #     { eq:"Paris - France + Italy",result:"Rome",   color:C.blue    },
# #     { eq:"swim - water + air",    result:"fly",    color:C.cyan    },
# #   ];
# #
# #   return (
# #     <div>
# #       <SectionTitle title="The Embedding Space" subtitle={"IDs \u2192 dense vectors: a learned geometry where meaning is distance"} />
# #
# #       {/* Interactive table size calculator */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:14 }}>
# #           {"Embedding Table Explorer "+DASH+" Shape: [vocab_size \u00D7 d_model]"}
# #         </div>
# #         <div style={{ display:"flex", gap:30, flexWrap:"wrap", alignItems:"center", marginBottom:16 }}>
# #           <div style={{ flex:1, minWidth:220 }}>
# #             <div style={{ fontSize:9, color:C.muted, marginBottom:6 }}>{"Vocabulary size: "+vocab.toLocaleString()}</div>
# #             <input type="range" min={10000} max={200000} step={1000} value={vocab}
# #               onChange={function(e){ setVocab(parseInt(e.target.value)); }}
# #               style={{ width:"100%", accentColor:C.blue }}/>
# #             <div style={{ display:"flex", justifyContent:"space-between", fontSize:8, color:C.dim }}>
# #               <span>10K</span><span>50K</span><span>100K</span><span>200K</span>
# #             </div>
# #           </div>
# #           <div style={{ flex:1, minWidth:220 }}>
# #             <div style={{ fontSize:9, color:C.muted, marginBottom:6 }}>{"Embedding dim (d_model): "+dim}</div>
# #             <input type="range" min={256} max={8192} step={128} value={dim}
# #               onChange={function(e){ setDim(parseInt(e.target.value)); }}
# #               style={{ width:"100%", accentColor:C.accent }}/>
# #             <div style={{ display:"flex", justifyContent:"space-between", fontSize:8, color:C.dim }}>
# #               <span>256</span><span>768</span><span>4096</span><span>8192</span>
# #             </div>
# #           </div>
# #         </div>
# #         <div style={{ display:"flex", justifyContent:"center", gap:30, flexWrap:"wrap", marginBottom:14 }}>
# #           <StatBox label="TABLE SHAPE"   value={vocab<1000?vocab:Math.round(vocab/1000)+"K"+" \u00D7 "+dim} color={C.blue}   bigFont={14}/>
# #           <StatBox label="SIZE (BF16)"   value={parseFloat(tableGB)<1 ? tableMB+" MB" : tableGB+" GB"} color={C.accent}/>
# #           <StatBox label="PARAMETERS"    value={(vocab*dim/1e6).toFixed(0)+"M"}      color={C.purple}/>
# #           <StatBox label="% OF 7B MODEL" value={totalPct+"%"}                         color={C.yellow} sub="embedding table only"/>
# #         </div>
# #         {/* Visual grid */}
# #         <div style={{ overflowX:"auto" }}>
# #           <div style={{ display:"flex", alignItems:"flex-start", gap:6, minWidth:500 }}>
# #             <div style={{ flexShrink:0 }}>
# #               <div style={{ fontSize:8, color:C.dim, marginBottom:4, height:18 }}>Token ID</div>
# #               {["[PAD]","[BOS]","[EOS]",'"the"','"cat"','"sat"'].map(function(id,i){
# #                 return (
# #                   <div key={i} onMouseEnter={function(){ setHov(i); }} onMouseLeave={function(){ setHov(-1); }}
# #                     style={{ height:24, fontSize:8, color:hov===i ? C.accent : C.dim,
# #                       fontFamily:"monospace", display:"flex", alignItems:"center",
# #                       marginBottom:2, cursor:"default", transition:"color 0.2s" }}>
# #                     {i+" ("+id+")"}
# #                   </div>
# #                 );
# #               })}
# #             </div>
# #             <div style={{ flex:1 }}>
# #               <div style={{ fontSize:8, color:C.dim, marginBottom:4, height:18 }}>
# #                 {"dim_0  dim_1  dim_2  dim_3  ... dim_"+(dim-1)+" ("+dim+" floats total)"}
# #               </div>
# #               {[
# #                 [0.000,0.000,0.000,0.000],
# #                 [0.234,-0.871,0.512,0.009],
# #                 [-0.341,0.222,-0.711,0.444],
# #                 [0.543,-0.211,0.089,-0.432],
# #                 [-0.671,0.123,0.556,0.338],
# #                 [0.021,0.788,-0.234,-0.109],
# #               ].map(function(row,ri){
# #                 var isH=hov===ri;
# #                 return (
# #                   <div key={ri} onMouseEnter={function(){ setHov(ri); }} onMouseLeave={function(){ setHov(-1); }}
# #                     style={{ display:"flex", gap:2, marginBottom:2, cursor:"default", padding:"1px 0" }}>
# #                     {row.map(function(v,ci){
# #                       var intensity = Math.abs(v);
# #                       var col = v>0 ? C.cyan : C.pink;
# #                       return (
# #                         <div key={ci} style={{ width:52, height:22, borderRadius:3, display:"flex",
# #                           alignItems:"center", justifyContent:"center",
# #                           background: isH ? col+"22" : col+(Math.round(intensity*20+5)).toString(16).padStart(2,"0"),
# #                           border:"1px solid "+(isH ? col+"80" : col+"20"),
# #                           transition:"all 0.3s" }}>
# #                           <span style={{ fontSize:7, fontFamily:"monospace", color:isH ? col : C.dim }}>
# #                             {v.toFixed(3)}
# #                           </span>
# #                         </div>
# #                       );
# #                     })}
# #                     <div style={{ fontSize:7, color:C.dim, display:"flex", alignItems:"center",
# #                       paddingLeft:6, fontFamily:"monospace" }}>{"... "+dim+" values"}</div>
# #                   </div>
# #                 );
# #               })}
# #             </div>
# #           </div>
# #         </div>
# #       </Card>
# #
# #       {/* Semantic geometry */}
# #       <div style={{ display:"flex", gap:12, maxWidth:1100, margin:"0 auto 16px", flexWrap:"wrap" }}>
# #         <Card style={{ flex:1, minWidth:280 }}>
# #           <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>
# #             {"Semantic Clustering "+DASH+" Similar Words Nearby"}
# #           </div>
# #           <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
# #             {wordGroups.map(function(g,i){
# #               return (
# #                 <div key={i}>
# #                   <div style={{ fontSize:8, color:g.color, fontWeight:700, marginBottom:4 }}>{g.label}</div>
# #                   <div style={{ display:"flex", gap:4 }}>
# #                     {g.words.map(function(w,j){
# #                       return (
# #                         <div key={j} style={{ padding:"3px 8px", borderRadius:4, fontSize:8,
# #                           fontFamily:"monospace", background:g.color+"12",
# #                           border:"1px solid "+g.color+"40", color:g.color }}>{w}</div>
# #                       );
# #                     })}
# #                   </div>
# #                 </div>
# #               );
# #             })}
# #           </div>
# #           <div style={{ marginTop:10, fontSize:8, color:C.dim }}>
# #             {"Similarity = cosine distance between vectors. Emerges from training, not programmed."}
# #           </div>
# #         </Card>
# #         <Card style={{ flex:1, minWidth:280 }}>
# #           <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>
# #             {"Vector Arithmetic "+DASH+" Analogy Relationships"}
# #           </div>
# #           {analogies.map(function(a,i){
# #             return (
# #               <div key={i} style={{ marginBottom:12, padding:"10px 12px", borderRadius:6,
# #                 background:a.color+"08", border:"1px solid "+a.color+"25" }}>
# #                 <div style={{ fontSize:9, fontFamily:"monospace", color:a.color, fontWeight:700 }}>{a.eq}</div>
# #                 <div style={{ fontSize:9, color:C.dim, marginTop:3 }}>
# #                   {"\u2248 "}<span style={{ color:a.color, fontWeight:800 }}>{a.result}</span>
# #                 </div>
# #               </div>
# #             );
# #           })}
# #           <div style={{ fontSize:8, color:C.dim }}>
# #             {"Directions in embedding space encode concepts: gender, royalty, geography."}
# #           </div>
# #         </Card>
# #       </div>
# #
# #       {/* Static vs Contextual */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>
# #           {"Static vs Contextual Embeddings"}
# #         </div>
# #         <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
# #           {[
# #             { title:"Static (Word2Vec, GloVe, FastText)", color:C.orange,
# #               desc:"Each word has ONE vector regardless of context",
# #               examples:[
# #                 { sent:'"river bank"', vec:"bank → [0.23, -0.41, ...]  ← same vector"},
# #                 { sent:'"money bank"', vec:"bank → [0.23, -0.41, ...]  ← always identical"},
# #               ],
# #               note:"Cannot distinguish 'bank' (river) from 'bank' (finance)" },
# #             { title:"Contextual (BERT, GPT, LLaMA)", color:C.green,
# #               desc:"Vector depends on surrounding context — different per sentence",
# #               examples:[
# #                 { sent:'"river bank"', vec:"bank → [-0.32, 0.71, ...]  ← river sense"},
# #                 { sent:'"money bank"', vec:"bank → [0.88, -0.21, ...]  ← finance sense"},
# #               ],
# #               note:"Full Transformer needed — covered in attention module" },
# #           ].map(function(col,i){
# #             return (
# #               <div key={i} style={{ flex:1, minWidth:260, padding:"12px 14px", borderRadius:8,
# #                 background:col.color+"06", border:"1px solid "+col.color+"25" }}>
# #                 <div style={{ fontSize:11, fontWeight:800, color:col.color, marginBottom:6 }}>{col.title}</div>
# #                 <div style={{ fontSize:9, color:C.muted, marginBottom:8 }}>{col.desc}</div>
# #                 {col.examples.map(function(ex,j){
# #                   return (
# #                     <div key={j} style={{ marginBottom:5, padding:"5px 8px", borderRadius:4,
# #                       background:C.border+"10", fontFamily:"monospace" }}>
# #                       <div style={{ fontSize:8, color:col.color }}>{ex.sent}</div>
# #                       <div style={{ fontSize:7, color:C.dim }}>{ex.vec}</div>
# #                     </div>
# #                   );
# #                 })}
# #                 <div style={{ marginTop:8, fontSize:8, color:C.dim, fontStyle:"italic" }}>{col.note}</div>
# #               </div>
# #             );
# #           })}
# #         </div>
# #       </Card>
# #
# #       <Insight icon={BULB} title="Why Embeddings Learn Meaning Without Being Taught">
# #         The model is never told "king and queen should be similar." It learns this because they appear in <span style={{color:C.accent,fontWeight:700}}>identical contexts</span>: "the ___ wore a crown", "the ___ ruled the kingdom." The training loss rewards correct context prediction — and placing co-occurring words nearby in vector space is the <span style={{color:C.purple,fontWeight:700}}>most efficient way to do that</span>. Semantic similarity is a <em>side effect</em> of learning to predict well, not a design choice.
# #       </Insight>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    TAB 4 — POSITIONAL ENCODING
# #    =============================================================== */
# # function TabPositional() {
# #   var _sel   = useState(0); var sel   = _sel[0],   setSel   = _sel[1];
# #   var _pos   = useState(0); var pos   = _pos[0],   setPos   = _pos[1];
# #   var _anim  = useState(false); var anim = _anim[0], setAnim  = _anim[1];
# #
# #   useEffect(function(){ var t=setTimeout(function(){ setAnim(true); },300); return function(){ clearTimeout(t); }; },[]);
# #
# #   var methods = [
# #     {
# #       name:"Sinusoidal", color:C.blue, year:"2017 (Vaswani)",
# #       learnable:"No", relative:"Partial", extrapolates:"Yes (degrades)",
# #       appliedTo:"Embeddings (added)",
# #       usedBy:"Original Transformer, some encoder models",
# #       formula:"PE(pos,2i) = sin(pos / 10000^(2i/d))\nPE(pos,2i+1) = cos(pos / 10000^(2i/d))",
# #       desc:"Fixed math formula. Each dimension oscillates at a different frequency. No parameters to learn. Can extend beyond training length.",
# #     },
# #     {
# #       name:"Learned", color:C.yellow, year:"2018 (BERT, GPT-2)",
# #       learnable:"Yes", relative:"No", extrapolates:"No (hard limit)",
# #       appliedTo:"Embeddings (added)",
# #       usedBy:"BERT (max 512), GPT-2 (max 1024)",
# #       formula:"pos_embed = EmbeddingTable[pos]  (trained)",
# #       desc:"Position vectors trained alongside the model. Flexible but hard-limited to max_seq_length seen during training.",
# #     },
# #     {
# #       name:"RoPE", color:C.accent, year:"2021 (Su et al.)",
# #       learnable:"No", relative:"Yes (exact)", extrapolates:"Yes (w/ NTK scaling)",
# #       appliedTo:"Q, K in each attention head",
# #       usedBy:"LLaMA (all), Mistral, Falcon, Qwen, GPT-NeoX",
# #       formula:"[v_2i', v_2i+1'] = R(m\u00B7\u03B8_i) \u00B7 [v_2i, v_2i+1]\n\u03B8_i = 10000^(-2i/d)",
# #       desc:"Encodes position as a rotation in 2D embedding subspaces. Key property: Q[m]·K[n] depends only on (m-n), making attention naturally relative.",
# #     },
# #     {
# #       name:"ALiBi", color:C.cyan, year:"2021 (Press et al.)",
# #       learnable:"No", relative:"Yes (linear)", extrapolates:"Yes (best)",
# #       appliedTo:"Attention scores (bias added)",
# #       usedBy:"BLOOM (176B), MPT, OpenLLM",
# #       formula:"score(q_i, k_j) = (q_i\u00B7k_j)/\u221Ad  \u2212  m\u00B7|i\u2212j|",
# #       desc:"Adds a linear distance penalty to attention scores. No position vectors. Excellent length extrapolation. Slope m is head-specific.",
# #     },
# #   ];
# #
# #   var v = methods[sel];
# #
# #   /* sinusoidal demo: compute PE for selected position */
# #   function sineVal(posArg, dim_i, d) {
# #     var freq = Math.pow(10000, (2*dim_i)/d);
# #     return (dim_i % 2 === 0) ? Math.sin(posArg/freq) : Math.cos(posArg/freq);
# #   }
# #   var NUM_DIMS = 20;
# #   var peVals = [];
# #   for (var di=0; di<NUM_DIMS; di++) { peVals.push(sineVal(pos, di, 512)); }
# #
# #   return (
# #     <div>
# #       <SectionTitle title="Positional Encoding" subtitle={"Transformers process tokens in parallel \u2014 position must be injected explicitly"} />
# #
# #       {/* Why needed */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>{"Why Positional Encoding Is Necessary"}</div>
# #         <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
# #           <div style={{ flex:1, minWidth:250, padding:"10px 14px", borderRadius:8,
# #             background:C.red+"08", border:"1px solid "+C.red+"25" }}>
# #             <div style={{ fontSize:10, fontWeight:700, color:C.red, marginBottom:6 }}>{"Without PE"}</div>
# #             <div style={{ fontSize:9, fontFamily:"monospace", color:C.muted, lineHeight:1.8 }}>
# #               {'"The cat sat on the mat"'}<br/>
# #               {'"The mat sat on the cat"'}<br/>
# #             </div>
# #             <div style={{ fontSize:8, color:C.red, marginTop:6 }}>
# #               {"→ Identical representations. Same tokens, same attention."}
# #             </div>
# #           </div>
# #           <div style={{ flex:1, minWidth:250, padding:"10px 14px", borderRadius:8,
# #             background:C.green+"08", border:"1px solid "+C.green+"25" }}>
# #             <div style={{ fontSize:10, fontWeight:700, color:C.green, marginBottom:6 }}>{"With PE"}</div>
# #             <div style={{ fontSize:9, color:C.muted, lineHeight:1.8 }}>
# #               {"final[pos] = embed[token_id] + pos_encoding[pos]"}<br/>
# #               {"Same token, different position → different vector"}
# #             </div>
# #             <div style={{ fontSize:8, color:C.green, marginTop:6 }}>
# #               {'→ "cat" at pos 1 ≠ "cat" at pos 5. Order is preserved.'}
# #             </div>
# #           </div>
# #         </div>
# #       </Card>
# #
# #       {/* Method selector */}
# #       <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
# #         {methods.map(function(m,i){
# #           var on=sel===i;
# #           return (
# #             <button key={i} onClick={function(){ setSel(i); }} style={{
# #               padding:"8px 16px", borderRadius:8,
# #               border:"1.5px solid "+(on ? m.color : C.border),
# #               background: on ? m.color+"20" : C.card, color: on ? m.color : C.muted,
# #               cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace",
# #               transition:"all 0.2s"
# #             }}>{m.name}</button>
# #           );
# #         })}
# #       </div>
# #
# #       {/* Detail card */}
# #       <Card highlight={true} style={{ maxWidth:1100, margin:"0 auto 16px", borderColor:v.color }}>
# #         <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
# #           <div style={{ flex:1, minWidth:260 }}>
# #             <div style={{ fontSize:16, fontWeight:800, color:v.color }}>{v.name}</div>
# #             <div style={{ fontSize:9, color:C.dim, marginBottom:8 }}>{v.year}</div>
# #             <div style={{ fontSize:11, color:C.muted, lineHeight:1.7, marginBottom:8 }}>{v.desc}</div>
# #             <div style={{ padding:"8px 12px", borderRadius:6, background:v.color+"10",
# #               border:"1px solid "+v.color+"30", fontFamily:"monospace", fontSize:9, color:v.color,
# #               whiteSpace:"pre-line" }}>{v.formula}</div>
# #             <div style={{ marginTop:8, fontSize:8, color:C.dim }}>Applied to: <span style={{color:v.color}}>{v.appliedTo}</span></div>
# #             <div style={{ fontSize:8, color:C.dim }}>Used by: <span style={{color:v.color}}>{v.usedBy}</span></div>
# #           </div>
# #           <div style={{ minWidth:180 }}>
# #             {[
# #               { l:"Learnable params", v:v.learnable, good:v.learnable==="Yes" },
# #               { l:"Relative positions", v:v.relative,  good:v.relative!=="No" },
# #               { l:"Extrapolates",       v:v.extrapolates, good:v.extrapolates.startsWith("Yes") },
# #             ].map(function(row,i){
# #               var isGood = row.good;
# #               return (
# #                 <div key={i} style={{ marginBottom:10, padding:"8px 12px", borderRadius:6,
# #                   background:C.border+"10", border:"1px solid "+C.border }}>
# #                   <div style={{ fontSize:8, color:C.dim, marginBottom:3 }}>{row.l}</div>
# #                   <div style={{ fontSize:10, fontWeight:700, color:isGood ? C.green : C.orange }}>{row.v}</div>
# #                 </div>
# #               );
# #             })}
# #           </div>
# #         </div>
# #       </Card>
# #
# #       {/* Sinusoidal visualizer */}
# #       {sel===0 && (
# #         <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #           <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:8 }}>{"Sinusoidal Pattern Visualizer"}</div>
# #           <div style={{ display:"flex", alignItems:"center", gap:14, marginBottom:14 }}>
# #             <div style={{ fontSize:9, color:C.muted, minWidth:80 }}>{"Position: "+pos}</div>
# #             <input type="range" min={0} max={511} value={pos}
# #               onChange={function(e){ setPos(parseInt(e.target.value)); }}
# #               style={{ flex:1, accentColor:C.blue }}/>
# #           </div>
# #           <div style={{ display:"flex", gap:3, alignItems:"flex-end", height:80 }}>
# #             {peVals.map(function(val,i){
# #               var h = Math.abs(val) * 36 + 4;
# #               var col = val>0 ? C.blue : C.purple;
# #               return (
# #                 <div key={i} style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center" }}>
# #                   <div style={{ height:h+"px", borderRadius:"3px 3px 0 0",
# #                     background:col+(val>0?"cc":"88"),
# #                     transition:"height 0.3s, background 0.3s", width:"100%" }}/>
# #                   {i%4===0 && <div style={{ fontSize:6, color:C.dim, marginTop:2 }}>{i}</div>}
# #                 </div>
# #               );
# #             })}
# #           </div>
# #           <div style={{ fontSize:8, color:C.dim, marginTop:6 }}>
# #             {"Each bar = one dimension of the PE vector at position "+pos+". Low dims oscillate slowly (coarse position), high dims oscillate fast (fine position)."}
# #           </div>
# #         </Card>
# #       )}
# #
# #       {/* RoPE explanation */}
# #       {sel===2 && (
# #         <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #           <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:10 }}>{"RoPE: Why Rotation Gives Relative Position"}</div>
# #           <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
# #             <div style={{ flex:1, minWidth:260 }}>
# #               <div style={{ fontSize:9, color:C.muted, lineHeight:1.9 }}>
# #                 {"For each pair of dimensions (2i, 2i+1), RoPE rotates the vector by angle m\u00B7\u03B8_i where m is the position."}<br/><br/>
# #                 {"The key result: Q[m] \u00B7 K[n] depends only on (m\u2212n)."}<br/>
# #                 {"Same relative distance = same attention pattern, regardless of absolute position."}
# #               </div>
# #             </div>
# #             <div style={{ flex:1, minWidth:200 }}>
# #               <div style={{ padding:"10px 14px", borderRadius:8, background:C.accent+"08",
# #                 border:"1px solid "+C.accent+"25", fontFamily:"monospace", fontSize:9, lineHeight:1.8 }}>
# #                 <span style={{color:C.purple}}>Q[m]</span>{" \u00B7 "}<span style={{color:C.cyan}}>K[n]</span>
# #                 {" = f(m\u2212n)"}<br/>
# #                 <span style={{color:C.dim}}>{"not f(m) and f(n) separately"}</span><br/><br/>
# #                 <span style={{color:C.accent}}>{"NTK-aware scaling"}</span><br/>
# #                 <span style={{color:C.dim}}>{"→ extends 4K context to 32K+"}</span>
# #               </div>
# #             </div>
# #           </div>
# #         </Card>
# #       )}
# #
# #       {/* Comparison table */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>{"Comparison: All Positional Encoding Methods"}</div>
# #         <div style={{ overflowX:"auto" }}>
# #           <table style={{ width:"100%", borderCollapse:"separate", borderSpacing:"0 4px", fontSize:9, fontFamily:"monospace" }}>
# #             <thead>
# #               <tr>
# #                 {["Method","Learnable","Relative","Extrapolates","Applied To","Used By"].map(function(h,i){
# #                   return <th key={i} style={{ padding:"6px 10px", textAlign:"left", color:C.dim, fontWeight:700, fontSize:8 }}>{h}</th>;
# #                 })}
# #               </tr>
# #             </thead>
# #             <tbody>
# #               {methods.map(function(m,i){
# #                 return (
# #                   <tr key={i} style={{ background: sel===i ? m.color+"12" : C.border+"08" }}>
# #                     <td style={{ padding:"7px 10px", borderRadius:"6px 0 0 6px", color:m.color, fontWeight:700 }}>{m.name}</td>
# #                     <td style={{ padding:"7px 10px", color:m.learnable==="Yes" ? C.green : C.orange }}>{m.learnable}</td>
# #                     <td style={{ padding:"7px 10px", color:m.relative==="No" ? C.orange : C.green }}>{m.relative}</td>
# #                     <td style={{ padding:"7px 10px", color:m.extrapolates.startsWith("Yes") ? C.green : C.red }}>{m.extrapolates}</td>
# #                     <td style={{ padding:"7px 10px", color:C.muted }}>{m.appliedTo}</td>
# #                     <td style={{ padding:"7px 10px", borderRadius:"0 6px 6px 0", color:C.dim }}>{m.usedBy}</td>
# #                   </tr>
# #                 );
# #               })}
# #             </tbody>
# #           </table>
# #         </div>
# #       </Card>
# #
# #       <Insight icon={GEAR} title="Why RoPE Dominates Modern LLMs">
# #         Learned positional embeddings (BERT, GPT-2) have a <span style={{color:C.red,fontWeight:700}}>hard sequence length limit</span> — if trained on 1024 tokens, they cannot handle 1025. Sinusoidal degrades beyond training length. <span style={{color:C.accent,fontWeight:700}}>RoPE</span> encodes position as rotation, making attention inherently relative and enabling context window extension through NTK-aware scaling. This is why every modern open-source LLM (LLaMA, Mistral, Qwen, Falcon) uses RoPE.
# #       </Insight>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    TAB 5 — TOKENIZER ZOO
# #    =============================================================== */
# # function TabZoo() {
# #   var _sel = useState(0); var sel = _sel[0], setSel = _sel[1];
# #
# #   var models = [
# #     {
# #       name:"GPT-4 / GPT-3.5",   algo:"BPE (tiktoken cl100k)",
# #       vocab:100256, dim:12288, contextK:128, released:"2022/23",
# #       color:C.green, icon:BRAIN,
# #       notes:["Byte-level BPE — vocab covers all UTF-8 bytes","Numbers tokenized digit-by-digit for most cases","cl100k has better multilingual coverage than GPT-2","Used by ChatGPT, GPT-4 API"],
# #     },
# #     {
# #       name:"LLaMA-3",  algo:"BPE (tiktoken)",
# #       vocab:128256, dim:4096, contextK:128, released:"2024",
# #       color:C.accent, icon:FIRE,
# #       notes:["Largest vocab of any major open-source model","128K vocab reduces sequence length vs LLaMA-2's 32K","Shares tokenizer format with GPT-4 (cl100k-based)","4× vocab increase over LLaMA-2"],
# #     },
# #     {
# #       name:"LLaMA-2",  algo:"BPE (SentencePiece)",
# #       vocab:32000, dim:4096, contextK:4, released:"2023",
# #       color:C.blue, icon:GEAR,
# #       notes:["Smaller 32K vocab means longer sequences","SentencePiece treats raw bytes — language agnostic","BOS=1, EOS=2, [INST]/[/INST] for chat","Default choice for open-source fine-tuning in 2023"],
# #     },
# #     {
# #       name:"BERT",     algo:"WordPiece",
# #       vocab:30522, dim:768, contextK:0.512, released:"2018",
# #       color:C.yellow, icon:MAG,
# #       notes:["WordPiece marks continuations with ## prefix","[CLS] prepended, [SEP] separates segments","Max 512 tokens (learned positional embeddings)","Encoder-only — no generation, embeddings focused"],
# #     },
# #     {
# #       name:"T5",       algo:"Unigram (SentencePiece)",
# #       vocab:32100, dim:1024, contextK:512, released:"2019",
# #       color:C.purple, icon:TEXT,
# #       notes:["Unigram: starts large, prunes by likelihood","SentencePiece framework — treats text as byte stream","Excellent multilingual support out of the box","No language-specific whitespace assumptions"],
# #     },
# #     {
# #       name:"Mistral-7B", algo:"BPE (SentencePiece)",
# #       vocab:32000, dim:4096, contextK:32, released:"2023",
# #       color:C.cyan, icon:STAR,
# #       notes:["Same tokenizer as LLaMA-2 (compatible)","Sliding window attention extends effective context","<s> BOS, </s> EOS, [INST]/[/INST] chat tokens","32K vocab — efficient, widely compatible"],
# #     },
# #   ];
# #
# #   var v = models[sel];
# #
# #   return (
# #     <div>
# #       <SectionTitle title="Tokenizer Zoo" subtitle={"Every major model has its own tokenizer — different algorithm, vocab size, and quirks"} />
# #
# #       {/* Model selector */}
# #       <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
# #         {models.map(function(m,i){
# #           var on=sel===i;
# #           return (
# #             <button key={i} onClick={function(){ setSel(i); }} style={{
# #               padding:"7px 14px", borderRadius:8,
# #               border:"1.5px solid "+(on ? m.color : C.border),
# #               background: on ? m.color+"20" : C.card, color: on ? m.color : C.muted,
# #               cursor:"pointer", fontSize:10, fontWeight:700, fontFamily:"monospace",
# #               transition:"all 0.2s"
# #             }}>{m.icon+" "+m.name}</button>
# #           );
# #         })}
# #       </div>
# #
# #       {/* Detail card */}
# #       <Card highlight={true} style={{ maxWidth:1100, margin:"0 auto 16px", borderColor:v.color }}>
# #         <div style={{ display:"flex", gap:20, flexWrap:"wrap" }}>
# #           <div style={{ flex:1, minWidth:260 }}>
# #             <div style={{ fontSize:18, fontWeight:800, color:v.color, marginBottom:2 }}>{v.name}</div>
# #             <div style={{ fontSize:10, color:C.dim, marginBottom:10 }}>{"Algorithm: "}<span style={{color:v.color}}>{v.algo}</span>{" | Released: "+v.released}</div>
# #             {v.notes.map(function(n,i){
# #               return <div key={i} style={{ fontSize:9, color:C.muted, lineHeight:1.9 }}>{CHK+" "+n}</div>;
# #             })}
# #           </div>
# #           <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
# #             <StatBox label="VOCAB SIZE"   value={v.vocab.toLocaleString()} color={v.color}/>
# #             <StatBox label="d_model"      value={v.dim.toLocaleString()}   color={v.color}/>
# #             <StatBox label="CONTEXT (K)"  value={v.contextK<1 ? v.contextK*1000+" tok" : v.contextK+"K"}  color={v.color}/>
# #           </div>
# #         </div>
# #       </Card>
# #
# #       {/* Vocab size comparison chart */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:14 }}>{"Vocabulary Size Comparison"}</div>
# #         {models.map(function(m,i){
# #           var pct = (m.vocab / 128256 * 100);
# #           var isS = sel===i;
# #           return (
# #             <div key={i} onClick={function(){ setSel(i); }}
# #               style={{ display:"flex", alignItems:"center", gap:10, marginBottom:7,
# #                 cursor:"pointer", padding:"4px 8px", borderRadius:6,
# #                 background: isS ? m.color+"10" : "transparent", transition:"background 0.2s" }}>
# #               <div style={{ width:120, fontSize:9, fontFamily:"monospace",
# #                 color: isS ? m.color : C.muted, fontWeight: isS ? 700 : 400 }}>{m.name}</div>
# #               <div style={{ flex:1, position:"relative", height:14 }}>
# #                 <div style={{ width:"100%", height:"100%", borderRadius:3, background:C.border }}/>
# #                 <div style={{ position:"absolute", top:0, left:0, width:pct+"%", height:"100%",
# #                   borderRadius:3, background:m.color+(isS?"80":"40"),
# #                   border:"1px solid "+m.color+(isS?"90":"50"), transition:"all 0.3s" }}/>
# #               </div>
# #               <div style={{ width:70, fontSize:9, fontFamily:"monospace", color: isS ? m.color : C.dim, textAlign:"right" }}>
# #                 {m.vocab.toLocaleString()}
# #               </div>
# #               <div style={{ width:55, fontSize:8, color:C.dim }}>{m.algo.split(" ")[0]}</div>
# #             </div>
# #           );
# #         })}
# #       </Card>
# #
# #       {/* Special tokens comparison */}
# #       <Card style={{ maxWidth:1100, margin:"0 auto 16px" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>{"Special Tokens by Model Family"}</div>
# #         <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
# #           {[
# #             { family:"BERT / Encoder", color:C.yellow,
# #               tokens:["[CLS]=101 (sentence start)","[SEP]=102 (separator)","[PAD]=0","[MASK]=103 (MLM training)","[UNK]=100"] },
# #             { family:"GPT / Decoder", color:C.green,
# #               tokens:["<|endoftext|>=50256 (GPT-2)","BOS=1, EOS=2 (LLaMA)","[INST]/[/INST] (LLaMA-2 chat)","<|im_start|>/<|im_end|> (ChatML)","No [CLS] or [SEP]"] },
# #             { family:"T5 / Seq2Seq", color:C.purple,
# #               tokens:["<pad>=0","</s>=1 (EOS)","<unk>=2","<extra_id_0...99> (sentinel tokens)","<s>=unused (no BOS in T5)"] },
# #           ].map(function(col,i){
# #             return (
# #               <div key={i} style={{ flex:1, minWidth:200, padding:"12px 14px", borderRadius:8,
# #                 background:col.color+"06", border:"1px solid "+col.color+"25" }}>
# #                 <div style={{ fontSize:10, fontWeight:800, color:col.color, marginBottom:8 }}>{col.family}</div>
# #                 {col.tokens.map(function(t,j){
# #                   return <div key={j} style={{ fontSize:8, fontFamily:"monospace", color:C.muted, lineHeight:1.9 }}>{t}</div>;
# #                 })}
# #               </div>
# #             );
# #           })}
# #         </div>
# #       </Card>
# #
# #       <Insight icon={WARN2} title="Tokenizer Compatibility Matters">
# #         You cannot use a BERT tokenizer with a LLaMA model or vice versa. Each model's weights were trained with a specific tokenizer — <span style={{color:C.red,fontWeight:700}}>swapping them produces gibberish</span>. LLaMA-2 and Mistral share a tokenizer (compatible). LLaMA-3 changed to a 128K vocab — LLaMA-2 adapters are <span style={{color:C.orange,fontWeight:700}}>not transferable</span> across this boundary. Always verify tokenizer and model are matched.
# #       </Insight>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    TAB 6 — QUIRKS & GOTCHAS
# #    =============================================================== */
# # function TabQuirks() {
# #   var _sel  = useState(0); var sel   = _sel[0],  setSel  = _sel[1];
# #   var _text = useState("Hello"); var text = _text[0], setText = _text[1];
# #
# #   var quirks = [
# #     {
# #       title:"Numbers Are Unpredictable", icon:NUM, color:C.orange,
# #       desc:"LLMs struggle with arithmetic partly because numbers rarely get single tokens.",
# #       examples:[
# #         { input:'"100"',     tokens:["100"],          note:"→ 1 token (common)" },
# #         { input:'"999"',     tokens:["9","9","9"],     note:"→ 3 tokens (digit-by-digit!)" },
# #         { input:'"2024"',    tokens:["20","24"],       note:"→ 2 tokens (depends on model)" },
# #         { input:'"3.14159"', tokens:["3",".","141","59"], note:"→ 4 tokens" },
# #       ],
# #       insight:"The tokenizer has no concept of numeric value. '999' tokenizes character-by-character in some models, meaning the 'number' 999 is three separate unrelated tokens.",
# #     },
# #     {
# #       title:"Leading Spaces Change IDs", icon:TEXT, color:C.blue,
# #       desc:"In GPT-style BPE, a space is part of the token. The same word has different IDs depending on whether it begins a sentence.",
# #       examples:[
# #         { input:'"dog"',        tokens:["dog"],          note:"→ ID 18031 (no space)" },
# #         { input:'" dog"',       tokens:["\u2581dog"],    note:"→ ID 5679  (space included, totally different token!)" },
# #         { input:'"The dog sat"',tokens:["The","\u2581dog","\u2581sat"], note:"→ 'dog' uses the space version" },
# #       ],
# #       insight:"This is why prompt phrasing matters more than you'd think. 'Translate: cat' and 'Translate cat' tokenize differently. GPT-style models use \u2581 (underscore) to mark leading-space tokens.",
# #     },
# #     {
# #       title:"Capitalization Creates New Tokens", icon:WARN, color:C.yellow,
# #       desc:"Case changes often produce completely different token IDs — not just re-weighted.",
# #       examples:[
# #         { input:'"Hello"', tokens:["Hello"],            note:"→ 1 token, ID 15043" },
# #         { input:'"hello"', tokens:["h","ello"],         note:"→ 2 tokens! Lowercase version split differently" },
# #         { input:'"HELLO"', tokens:["H","E","L","L","O"],note:"→ 5 tokens (all caps often tokenizes letter by letter)" },
# #       ],
# #       insight:"Capitalization is not 'free' for LLMs. Uppercase text can explode your token count and context usage. For code: indentation, variable name casing, and symbol placement all affect tokenization.",
# #     },
# #     {
# #       title:"Non-English Is Token-Expensive", icon:STAR, color:C.purple,
# #       desc:"Vocabularies are built on mostly English text. Non-English languages get fewer dedicated tokens.",
# #       examples:[
# #         { input:'"Hello"',    tokens:["Hello"],           note:"→ 1 token (English)" },
# #         { input:'"Hola"',     tokens:["Hola"],            note:"→ 1 token (Spanish, common word)" },
# #         { input:'"Привет"',   tokens:["\u041F\u0440","\u0438","\u0432\u0435\u0442"], note:"→ 3 tokens (Russian)" },
# #         { input:'"こんにちは"', tokens:["\u3053","\u3093","\u306B","\u3061","\u306F"], note:"→ 5 tokens (Japanese — 1 char each)" },
# #       ],
# #       insight:"Chinese, Japanese, and Korean characters often tokenize 1-per-character because the vocab has limited CJK coverage. This means non-English text uses 3-5× more tokens for the same semantic content — directly inflating cost.",
# #     },
# #     {
# #       title:"Code Tokenizes Differently", icon:GEAR, color:C.cyan,
# #       desc:"Indentation, operators, and special characters each cost tokens. Compact code can save context.",
# #       examples:[
# #         { input:"def foo():",      tokens:["def"," foo","():"],     note:"→ 3 tokens" },
# #         { input:"    return x+1",  tokens:["    ","return"," x","+","1"], note:"→ 5 tokens (indentation is its own token!)" },
# #         { input:"x = [1,2,3]",     tokens:["x"," ="," [","1",",","2",",","3","]"], note:"→ 9 tokens" },
# #       ],
# #       insight:"Code tokenization explains why LLMs are sensitive to indentation style and whitespace. 4-space indent vs 2-space indent can produce different token counts and different model behaviour.",
# #     },
# #   ];
# #
# #   var v = quirks[sel];
# #
# #   return (
# #     <div>
# #       <SectionTitle title="Quirks & Gotchas" subtitle={"Real tokenizer behaviour that surprises every beginner "+DASH+" not bugs, just consequences of the algorithm"} />
# #
# #       {/* Quirk selector */}
# #       <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:20, flexWrap:"wrap" }}>
# #         {quirks.map(function(q,i){
# #           var on=sel===i;
# #           return (
# #             <button key={i} onClick={function(){ setSel(i); }} style={{
# #               padding:"7px 12px", borderRadius:8,
# #               border:"1.5px solid "+(on ? q.color : C.border),
# #               background: on ? q.color+"20" : C.card, color: on ? q.color : C.muted,
# #               cursor:"pointer", fontSize:9, fontWeight:700, fontFamily:"monospace",
# #               transition:"all 0.2s"
# #             }}>{q.icon+" "+q.title.split(" ")[0]+"..."}</button>
# #           );
# #         })}
# #       </div>
# #
# #       {/* Detail card */}
# #       <Card highlight={true} style={{ maxWidth:1100, margin:"0 auto 16px", borderColor:v.color }}>
# #         <div style={{ fontSize:14, fontWeight:800, color:v.color, marginBottom:4 }}>{v.icon+" "+v.title}</div>
# #         <div style={{ fontSize:10, color:C.muted, marginBottom:14, lineHeight:1.7 }}>{v.desc}</div>
# #         <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
# #           {v.examples.map(function(ex,i){
# #             return (
# #               <div key={i} style={{ display:"flex", gap:12, alignItems:"flex-start", flexWrap:"wrap",
# #                 padding:"10px 14px", borderRadius:8, background:v.color+"06",
# #                 border:"1px solid "+v.color+"20" }}>
# #                 <div style={{ minWidth:160, fontFamily:"monospace", fontSize:9, color:v.color, fontWeight:700 }}>{ex.input}</div>
# #                 <div style={{ display:"flex", gap:4, flexWrap:"wrap", flex:1 }}>
# #                   {ex.tokens.map(function(tok,j){
# #                     return (
# #                       <div key={j} style={{ padding:"2px 8px", borderRadius:4, fontSize:9,
# #                         fontFamily:"monospace", background:v.color+"18",
# #                         border:"1px solid "+v.color+"50", color:v.color }}>
# #                         {tok}
# #                       </div>
# #                     );
# #                   })}
# #                 </div>
# #                 <div style={{ fontSize:8, color:C.dim, fontFamily:"monospace", minWidth:200 }}>{ex.note}</div>
# #               </div>
# #             );
# #           })}
# #         </div>
# #       </Card>
# #
# #       <Insight icon={v.icon} title={v.title}>
# #         {v.insight}
# #       </Insight>
# #
# #       {/* Token counting rule of thumb */}
# #       <Card style={{ maxWidth:1100, margin:"16px auto 0" }}>
# #         <div style={{ fontSize:11, fontWeight:700, color:C.text, marginBottom:12 }}>{"Practical Token Counting Rule of Thumb"}</div>
# #         <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
# #           {[
# #             { label:"English prose",   ratio:"~1.3–1.5 tokens/word", color:C.green, note:"Well-covered in training data" },
# #             { label:"Code",            ratio:"~1.5–2.5 tokens/word", color:C.cyan,  note:"Operators, spaces, symbols" },
# #             { label:"Non-Latin script",ratio:"~2–4 tokens/char",     color:C.orange,note:"Fewer dedicated vocab entries" },
# #             { label:"Numbers/math",    ratio:"Highly variable",       color:C.red,   note:"1 to N digits = 1 to N tokens" },
# #           ].map(function(r,i){
# #             return (
# #               <div key={i} style={{ flex:1, minWidth:160, padding:"10px 14px", borderRadius:8,
# #                 background:r.color+"06", border:"1px solid "+r.color+"25" }}>
# #                 <div style={{ fontSize:9, fontWeight:700, color:r.color, marginBottom:4 }}>{r.label}</div>
# #                 <div style={{ fontSize:11, fontWeight:800, color:r.color, marginBottom:4 }}>{r.ratio}</div>
# #                 <div style={{ fontSize:8, color:C.dim }}>{r.note}</div>
# #               </div>
# #             );
# #           })}
# #         </div>
# #         <div style={{ marginTop:12, fontSize:9, color:C.dim, textAlign:"center" }}>
# #           {"Token count = context window cost. For production prompts: always count tokens, never estimate by word count alone."}
# #         </div>
# #       </Card>
# #     </div>
# #   );
# # }
# #
# #
# # /* ===============================================================
# #    ROOT APP
# #    =============================================================== */
# # function App() {
# #   var _t = useState(0); var tab = _t[0], setTab = _t[1];
# #   var tabs = [
# #     "\uD83D\uDD04 Pipeline",
# #     "\u2702\uFE0F Tokenizers",
# #     "\uD83D\uDCCA Embedding Space",
# #     "\uD83D\uDCCD Positional Encoding",
# #     "\uD83E\uDDE9 Tokenizer Zoo",
# #     "\u26A0\uFE0F Quirks & Gotchas",
# #   ];
# #   return (
# #     <div style={{ background:C.bg, minHeight:"100vh", padding:"24px 16px",
# #       fontFamily:"'JetBrains Mono','SF Mono',monospace", color:C.text,
# #       maxWidth:1400, margin:"0 auto" }}>
# #       <div style={{ textAlign:"center", marginBottom:16 }}>
# #         <div style={{ fontSize:22, fontWeight:800,
# #           background:"linear-gradient(135deg,"+C.blue+","+C.accent+","+C.purple+")",
# #           WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", display:"inline-block" }}>
# #           Tokenization & Embeddings
# #         </div>
# #         <div style={{ fontSize:11, color:C.muted, marginTop:4 }}>
# #           {"Interactive visual walkthrough "+DASH+" from raw text to transformer-ready vectors"}
# #         </div>
# #       </div>
# #       <TabBar tabs={tabs} active={tab} onChange={setTab} />
# #       {tab===0 && <TabPipeline/>}
# #       {tab===1 && <TabTokenizers/>}
# #       {tab===2 && <TabEmbedding/>}
# #       {tab===3 && <TabPositional/>}
# #       {tab===4 && <TabZoo/>}
# #       {tab===5 && <TabQuirks/>}
# #     </div>
# #   );
# # }
# #
# # ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
# #
# # </script>
# # </body>
# # </html>
# # """
# #
# # TOK_EMBED_HEIGHT = 1600