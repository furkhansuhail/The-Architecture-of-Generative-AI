CONTEXT_WINDOW_VISUAL_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', sans-serif;
  background: #08080f;
  color: #e4e4e7;
  padding: 20px;
  min-height: 100vh;
}

h2 { font-family: 'JetBrains Mono', monospace; font-size: 1.15em; color: #ff6b35; margin-bottom: 3px; }
.subtitle { color: #52525b; font-size: 0.8em; margin-bottom: 20px; }

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
button {
  background: #1e1e2e; color: #a1a1aa;
  border: 1px solid #2d2d40; border-radius: 6px;
  padding: 5px 13px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75em; font-weight: 700;
  transition: all 0.15s;
}
button:hover:not(:disabled) { background: #2d2d40; color: #e4e4e7; }
button.active { background: #ff6b35; color: #08080f; border-color: #ff6b35; }
button:disabled { opacity: 0.35; cursor: default; }

input[type=range] {
  flex: 1; height: 4px; border-radius: 2px;
  -webkit-appearance: none; appearance: none;
  background: #1e1e2e; outline: none; cursor: pointer;
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 14px; height: 14px;
  border-radius: 50%; background: #ff6b35; cursor: pointer;
}

.row   { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
.label { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #71717a; min-width: 110px; }
.val   { font-family: 'JetBrains Mono', monospace; font-size: 0.75em; color: #ff6b35; min-width: 42px; text-align: right; }

.info-box {
  background: #050508; border: 1px solid #1e1e2e; border-radius: 8px;
  padding: 8px 12px; font-size: 0.78em; color: #94a3b8;
  line-height: 1.7; margin-top: 10px;
  font-family: 'JetBrains Mono', monospace;
}
.hl  { color: #ff6b35; font-weight: 700; }
.hl2 { color: #4ecdc4; font-weight: 700; }
.hl3 { color: #fbbf24; font-weight: 700; }
.hlg { color: #4ade80; font-weight: 700; }
.hlr { color: #ef4444; font-weight: 700; }
</style>
</head>
<body>

<h2>&#9889; Context Window &#8212; How LLMs Actually Process Your Prompts</h2>
<p class="subtitle">Token budget &middot; Prompt lifecycle &middot; Memory architecture</p>

<div class="btn-row" id="tab-bar"></div>
<div id="tab-content"></div>

<script>
// ── Tab shell (vanilla JS) ──────────────────────────────────────────────────
const TABS = ['① Context Window', '② Prompt Lifecycle', '③ Memory &amp; State'];
let activeTab = 0;

function renderTabBar() {
  document.getElementById('tab-bar').innerHTML = TABS.map((t, i) =>
    `<button class="${i === activeTab ? 'active' : ''}" onclick="switchTab(${i})">${t}</button>`
  ).join('');
}

function switchTab(i) {
  activeTab = i;
  renderTabBar();
  if (i === 0) renderTab1();
  if (i === 1) renderTab2();
  if (i === 2) renderTab3();
}

renderTabBar();
</script>

<!-- ═══════════════════════════════════════════════════════════
     TAB 1 — Context Window Token Budget (vanilla JS + canvas)
     ═══════════════════════════════════════════════════════════ -->
<script>
function renderTab1() {
  document.getElementById('tab-content').innerHTML = `
    <div style="background:#111118;border:1px solid #1e1e2e;border-radius:14px;padding:20px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.78em;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;color:#ff6b35;margin-bottom:14px;">
        ① Context Window — Token Budget Visualizer
      </div>
      <div class="btn-row" id="t1-wsize">
        <span class="label" style="min-width:90px;">Window size</span>
        <button id="w8"   onclick="t1_setW(8)"  >8k</button>
        <button id="w32"  onclick="t1_setW(32)" class="active">32k</button>
        <button id="w128" onclick="t1_setW(128)">128k</button>
        <button id="w200" onclick="t1_setW(200)">200k</button>
        <span class="label" style="margin-left:12px;">Show dropped</span>
        <button id="t1-drop" onclick="t1_toggleDrop()">On</button>
      </div>
      <canvas id="cvWindow" width="600" height="230" style="border-radius:10px;display:block;"></canvas>
      <div class="row">
        <label class="label">Conversation turns</label>
        <input type="range" id="t1-turns" min="1" max="28" value="5" oninput="t1_draw()">
        <span class="val" id="t1-turnsv">5</span>
      </div>
      <div class="info-box" id="t1-info">—</div>
    </div>`;

  window.t1_wk       = 32;
  window.t1_showDrop = true;

  window.t1_setW = function(k) {
    t1_wk = k;
    ['8','32','128','200'].forEach(v => document.getElementById('w'+v).classList.toggle('active', +v === k));
    t1_draw();
  };
  window.t1_toggleDrop = function() {
    t1_showDrop = !t1_showDrop;
    document.getElementById('t1-drop').textContent = t1_showDrop ? 'On' : 'Off';
    document.getElementById('t1-drop').classList.toggle('active', t1_showDrop);
    t1_draw();
  };
  window.t1_draw = function() {
    const turns = +document.getElementById('t1-turns').value;
    document.getElementById('t1-turnsv').textContent = turns;
    const TOTAL = t1_wk * 1000, SYS = 600, UT = 150, AT = 240, TURN = UT + AT, CUR = 160;
    const histTok = turns * TURN, used = SYS + histTok + CUR;
    const over = used > TOTAL;
    const dropped = over ? Math.max(0, Math.ceil((used - TOTAL) / TURN)) : 0;
    const pct = Math.min(100, (used / TOTAL) * 100);
    const free = Math.max(0, TOTAL - used);
    const fmt = n => n >= 1000 ? (n/1000).toFixed(1)+'k' : String(n);

    const cv = document.getElementById('cvWindow');
    const ctx = cv.getContext('2d');
    const W = 600, H = 230;
    const BW = 540, BH = 54, BX = 30, BY = 58;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#08080d';
    ctx.beginPath(); ctx.roundRect(0, 0, W, H, 10); ctx.fill();

    // Header text
    ctx.font = 'bold 10px JetBrains Mono,monospace'; ctx.textAlign = 'left';
    ctx.fillStyle = '#71717a';
    ctx.fillText(`CONTEXT WINDOW — ${t1_wk}k = ${TOTAL.toLocaleString()} tokens total`, BX, 30);
    ctx.textAlign = 'right';
    ctx.fillStyle = over ? '#ef4444' : pct > 75 ? '#fbbf24' : '#4ade80';
    ctx.font = 'bold 10px JetBrains Mono,monospace';
    ctx.fillText(pct.toFixed(1) + '% used', W - 12, 30);

    // Rail
    ctx.fillStyle = '#1e1e2e'; ctx.globalAlpha = 0.35;
    ctx.beginPath(); ctx.roundRect(BX, BY, BW, BH, 7); ctx.fill(); ctx.globalAlpha = 1;

    // Build segments
    const segs = [{ tok: SYS, color: '#4ecdc4', label: 'Sys', dim: false }];
    for (let i = 0; i < turns; i++) {
      const d = t1_showDrop && i < dropped;
      segs.push({ tok: UT, color: d ? '#ef4444' : '#38bdf8', label: `U${i+1}`, dim: d, dropped: d });
      segs.push({ tok: AT, color: d ? '#ef4444' : '#c084fc', label: `A${i+1}`, dim: d, dropped: d });
    }
    segs.push({ tok: CUR, color: '#ff6b35', label: 'New', current: true });

    const totTok = segs.reduce((s, x) => s + x.tok, 0);
    const norm = Math.max(1, totTok / TOTAL);
    let rx = 0;
    const bars = segs.map((s, i) => {
      const w = Math.max(2, (s.tok / (totTok * norm)) * BW);
      const b = { ...s, x: rx, w };
      rx += w; return b;
    });
    const freeW = Math.max(0, BW - rx);

    bars.forEach((b, i) => {
      ctx.globalAlpha = b.dropped ? 0.22 : b.current ? 1 : 0.82;
      ctx.fillStyle = b.color;
      if (i === 0) { ctx.beginPath(); ctx.roundRect(BX + b.x, BY, b.w - 1, BH, [7,0,0,7]); ctx.fill(); }
      else { ctx.fillRect(BX + b.x, BY, Math.max(1, b.w - 1), BH); }
      ctx.globalAlpha = 1;
      if (b.dropped) {
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 0.8; ctx.globalAlpha = 0.45;
        ctx.beginPath(); ctx.moveTo(BX+b.x+3, BY+6); ctx.lineTo(BX+b.x+b.w-4, BY+BH-6); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(BX+b.x+3, BY+BH-6); ctx.lineTo(BX+b.x+b.w-4, BY+6); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // Free space
    if (freeW > 3) {
      ctx.fillStyle = '#2d2d40'; ctx.globalAlpha = 0.28;
      ctx.fillRect(BX + BW - freeW, BY, freeW, BH);
      ctx.globalAlpha = 1;
    }

    // Overflow border
    if (over) {
      ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.8;
      ctx.setLineDash([5, 3]);
      ctx.beginPath(); ctx.roundRect(BX, BY, BW, BH, 7); ctx.stroke();
      ctx.setLineDash([]);
    }

    // Labels under bar
    ctx.font = '8.5px JetBrains Mono,monospace'; ctx.textAlign = 'center';
    bars.forEach(b => {
      if (b.w > 18) {
        ctx.fillStyle = b.dropped ? '#ef4444' : b.color;
        ctx.globalAlpha = b.dropped ? 0.6 : 1;
        ctx.font = b.current ? 'bold 9.5px JetBrains Mono,monospace' : '8.5px JetBrains Mono,monospace';
        ctx.fillText(b.dropped ? '✗' : b.label, BX + b.x + b.w / 2, BY + BH + 15);
        ctx.globalAlpha = 1;
      }
    });

    // Token counts
    ctx.textAlign = 'left'; ctx.font = '9px JetBrains Mono,monospace';
    [[BX, '#4ecdc4', `Sys: ${fmt(SYS)}`],
     [BX+80, '#38bdf8', `History: ${fmt(histTok)}`],
     [BX+210, '#ff6b35', `Current: ${fmt(CUR)}`],
     [BX+330, over ? '#ef4444' : '#4ade80', over ? `⚠ +${fmt(used-TOTAL)} overflow` : `Free: ${fmt(free)}`]
    ].forEach(([x, col, txt]) => { ctx.fillStyle = col; ctx.fillText(txt, x, BY+BH+33); });

    // Progress bar
    ctx.fillStyle = '#1e1e2e'; ctx.beginPath(); ctx.roundRect(BX, BY+BH+46, BW, 7, 3.5); ctx.fill();
    ctx.fillStyle = over ? '#ef4444' : pct > 80 ? '#fbbf24' : '#4ecdc4';
    ctx.beginPath(); ctx.roundRect(BX, BY+BH+46, Math.min(BW, BW*pct/100), 7, 3.5); ctx.fill();

    // Legend
    const leg = [['#4ecdc4','System prompt'],['#38bdf8','User msgs'],['#c084fc','Asst replies'],['#ff6b35','Current msg'],['#3f3f46','Free space'],['#ef4444','Dropped (overflow)']];
    leg.forEach(([col, lbl], i) => {
      ctx.fillStyle = col;
      ctx.fillRect(BX + (i%3)*182, i<3 ? 196 : 212, 9, 9);
      ctx.fillStyle = '#52525b'; ctx.font = '8.5px JetBrains Mono,monospace'; ctx.textAlign = 'left';
      ctx.fillText(lbl, BX + (i%3)*182 + 13, i<3 ? 204 : 220);
    });

    // Info box
    document.getElementById('t1-info').innerHTML =
      `<span class="hl2">Context window = </span>the model's entire working memory for this call.&nbsp;` +
      `<span class="hl">${fmt(used)}</span> of <span class="hl">${fmt(TOTAL)}</span> tokens consumed (${pct.toFixed(1)}%).&nbsp;` +
      (over
        ? `<span class="hlr">⚠ ${dropped} oldest turn${dropped>1?'s':''} dropped — model can no longer see them.</span> Increase window size or summarise history.`
        : pct > 75
          ? `<span class="hl3">⚡ Getting full — consider summarising old turns soon.</span>`
          : `<span class="hlg">✓ All ${turns} turn${turns>1?'s':''} visible to the model.</span>`
      );
  };

  document.getElementById('t1-drop').classList.add('active');
  t1_draw();
}
renderTab1();
</script>

<!-- ═══════════════════════════════════════════════════════════
     TAB 2 — Prompt Lifecycle (React / Babel)
     ═══════════════════════════════════════════════════════════ -->
<div id="tab2-react-root" style="display:none;"></div>

<script type="text/babel" id="tab2-script">
(function(){
  const { useState } = React;
  const mono = "'JetBrains Mono','Courier New',monospace";
  const C = { bg:'#08080f',card:'#111118',card2:'#0d0d18',card3:'#050508',border:'#1e1e2e',border2:'#2d2d40',accent:'#ff6b35',teal:'#4ecdc4',purple:'#c084fc',green:'#4ade80',red:'#ef4444',yellow:'#fbbf24',blue:'#38bdf8',text:'#e4e4e7',muted:'#71717a',dim:'#3f3f46',subtle:'#52525b',slate:'#94a3b8' };
  const btnS = on => ({ background:on?C.accent:C.border, color:on?C.bg:'#a1a1aa', border:`1px solid ${on?C.accent:C.border2}`, borderRadius:6, padding:'5px 13px', cursor:'pointer', fontFamily:mono, fontSize:'0.75em', fontWeight:700, transition:'all 0.15s' });
  const navS = (dis) => ({ ...btnS(false), padding:'7px 18px', borderRadius:8, fontSize:'0.8em', opacity:dis?0.35:1, cursor:dis?'default':'pointer' });

  const STEPS = [
    { id:'user',     title:'Step 1 — You send a message',                    sub:"Input captured by the chat interface",             hl:['user'],               code:['user_message = "Why are C4 plants more efficient?"','# The LLM is not invoked yet'] },
    { id:'retrieve', title:'Step 2 — App retrieves conversation history',     sub:"Database lookup on the app's servers",             hl:['user','app','db'],    code:['history = db.fetch_messages(conversation_id)','# Returns all previous turns from app DB','# Model has no knowledge of this yet'] },
    { id:'assemble', title:'Step 3 — App assembles the full context',         sub:"System prompt + history + new message bundled",    hl:['app','bundle'],       code:['prompt  = system_prompt','for msg in history:','    prompt += format_turn(msg)','prompt += new_message  # appended last'] },
    { id:'send',     title:'Step 4 — Full context sent to the model',         sub:"One HTTP request, potentially 100k+ tokens",       hl:['app','llm'],          code:['response = llm_api.complete(prompt)','# prompt = system + all_history + new_msg','# Could be 100,000+ tokens in one request'] },
    { id:'process',  title:'Step 5 — Model processes ALL tokens at once',     sub:"Self-attention reads the entire context",           hl:['llm'],                code:['# Inside the transformer — attention:','for each token t in prompt:','    weights = softmax(Q[t] \xb7 K.T / \u221ad)','    output[t] = weights \xb7 V'] },
    { id:'generate', title:'Step 6 — Response generated token by token',      sub:"Autoregressive decoding",                          hl:['llm','stream'],       code:['output = ""','while not end_of_sequence:','    token = model.next_token(prompt + output)','    stream_to_client(token)'] },
    { id:'save',     title:'Step 7 — App saves; model forgets everything',    sub:"Model state wiped — app remembers permanently",    hl:['stream','app','db'],  code:['db.save(user_msg, assistant_reply)','# \u2191 App stores this permanently','','# model.state = None  \u2190 everything wiped','# Next call starts completely fresh'] },
  ];

  const nodes = { user:{x:52,y:78,label:'You',color:C.accent}, app:{x:172,y:78,label:'App',color:C.teal}, db:{x:172,y:175,label:'DB',color:C.yellow}, bundle:{x:292,y:78,label:'Prompt',color:C.purple}, llm:{x:422,y:78,label:'LLM',color:C.blue}, stream:{x:422,y:175,label:'Reply',color:C.green} };
  const edges = [
    {a:'user',b:'app',dir:'h',lbl:'sends msg'},{a:'app',b:'db',dir:'v',lbl:'fetch'},
    {a:'app',b:'bundle',dir:'h',lbl:'assemble'},{a:'bundle',b:'llm',dir:'h',lbl:'API call'},
    {a:'llm',b:'stream',dir:'v',lbl:'generate'},{a:'stream',b:'app',dir:'hr',lbl:'stream back'},
  ];
  const R = 22;
  const gc = e => {
    const f=nodes[e.a],t=nodes[e.b];
    if(e.dir==='h')  return {x1:f.x+R,y1:f.y,x2:t.x-R,y2:t.y,lx:(f.x+t.x)/2,ly:f.y-9};
    if(e.dir==='v')  return {x1:f.x,y1:f.y+R,x2:t.x,y2:t.y-R,lx:f.x+14,ly:(f.y+t.y)/2+4};
    return {x1:f.x-R,y1:f.y,x2:t.x+R,y2:t.y,lx:(f.x+t.x)/2,ly:f.y-9};
  };

  function LifecycleSVG({hl}) {
    return (
      <svg width="100%" viewBox="0 0 490 225" style={{display:'block',borderRadius:10}}>
        <rect width={490} height={225} fill="#08080d" rx={10}/>
        <defs><marker id="lcarr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="context-stroke" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></marker></defs>
        {edges.map((e,i) => {
          const active = hl.includes(e.a) && hl.includes(e.b);
          const {x1,y1,x2,y2,lx,ly} = gc(e);
          const col = active ? nodes[e.b].color : C.border2;
          return (<g key={i} opacity={active?1:0.18} style={{transition:'opacity 0.35s'}}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={active?2:1.2} markerEnd="url(#lcarr)"/>
            <text x={lx} y={ly} fill={col} fontSize={8.5} fontFamily={mono} textAnchor="middle">{e.lbl}</text>
          </g>);
        })}
        {Object.entries(nodes).map(([id,n]) => {
          const active = hl.includes(id);
          return (<g key={id} opacity={active?1:0.18} style={{transition:'opacity 0.35s'}}>
            <circle cx={n.x} cy={n.y} r={R} fill={active?n.color+'1a':'#08080d'} stroke={n.color} strokeWidth={active?2.5:1.5}/>
            <text x={n.x} y={n.y+4} fill={active?n.color:C.dim} fontSize={id==='bundle'?10:12} fontFamily={mono} fontWeight={700} textAnchor="middle">{n.label}</text>
          </g>);
        })}
        <text x={422} y={113} fill={C.subtle} fontSize={8} fontFamily={mono} textAnchor="middle" opacity={0.6}>stateless</text>
        {[{x:52,l:'User'},{x:172,l:'App layer'},{x:292,l:'Context'},{x:422,l:'Model'}].map((l,i)=>
          <text key={i} x={l.x} y={215} fill={C.dim} fontSize={8.5} fontFamily={mono} textAnchor="middle">{l.l}</text>
        )}
      </svg>
    );
  }

  function Tab2() {
    const [step, setStep] = useState(0);
    const s = STEPS[step];
    return (
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:20}}>
        <div style={{fontFamily:mono,fontSize:'0.78em',fontWeight:800,textTransform:'uppercase',letterSpacing:'0.1em',color:C.accent,marginBottom:14}}>
          ② Prompt Lifecycle — Step-by-Step Walkthrough
        </div>
        <LifecycleSVG hl={s.hl}/>
        <div style={{background:C.card2,border:`1px solid ${C.border}`,borderRadius:10,padding:12,marginTop:12}}>
          <div style={{fontFamily:mono,fontSize:'0.88em',fontWeight:800,color:C.accent,marginBottom:3}}>{s.title}</div>
          <div style={{fontFamily:mono,fontSize:'0.74em',color:C.subtle,fontStyle:'italic',marginBottom:9}}>{s.sub}</div>
          <div style={{fontFamily:mono,fontSize:'0.78em',color:C.slate,lineHeight:1.65,marginBottom:10}}>{s.desc||''}</div>
          <div style={{background:C.card3,border:`1px solid ${C.border}`,borderRadius:7,padding:'8px 14px',fontFamily:mono,fontSize:'0.72em',lineHeight:1.65}}>
            {s.code.map((line,i)=><div key={i} style={{color:line.startsWith('#')?C.dim:line===''?'transparent':C.teal}}>{line||'\u00a0'}</div>)}
          </div>
        </div>
        <div style={{display:'flex',alignItems:'center',gap:12,marginTop:14,justifyContent:'center'}}>
          <button style={navS(step===0)} onClick={()=>setStep(p=>Math.max(0,p-1))} disabled={step===0}>&#8592; Back</button>
          <div style={{display:'flex',gap:5}}>
            {STEPS.map((_,i)=><div key={i} onClick={()=>setStep(i)} style={{width:i===step?22:7,height:7,borderRadius:3.5,cursor:'pointer',background:i===step?C.accent:i<step?C.accent+'55':C.border2,transition:'all 0.25s'}}/>)}
          </div>
          <button style={navS(step===STEPS.length-1)} onClick={()=>setStep(p=>Math.min(STEPS.length-1,p+1))} disabled={step===STEPS.length-1}>Next &#8594;</button>
        </div>
        <div style={{textAlign:'center',color:C.dim,fontFamily:mono,fontSize:'0.7em',marginTop:6}}>Step {step+1} of {STEPS.length}</div>
      </div>
    );
  }

  window._tab2Rendered = false;
  window._renderTab2 = function() {
    const root = document.getElementById('tab2-react-root');
    if (!root) return;
    if (!window._tab2ReactRoot) window._tab2ReactRoot = ReactDOM.createRoot(root);
    window._tab2ReactRoot.render(<Tab2/>);
  };
})();
</script>

<!-- ═══════════════════════════════════════════════════════════
     TAB 3 — Memory Architecture (React / Babel)
     ═══════════════════════════════════════════════════════════ -->
<div id="tab3-react-root" style="display:none;"></div>

<script type="text/babel" id="tab3-script">
(function(){
  const { useState, useRef } = React;
  const mono = "'JetBrains Mono','Courier New',monospace";
  const C = { bg:'#08080f',card:'#111118',card2:'#0d0d18',card3:'#050508',border:'#1e1e2e',border2:'#2d2d40',accent:'#ff6b35',teal:'#4ecdc4',purple:'#c084fc',green:'#4ade80',red:'#ef4444',yellow:'#fbbf24',blue:'#38bdf8',muted:'#71717a',dim:'#3f3f46',subtle:'#52525b',slate:'#94a3b8' };
  const btnS = on => ({ background:on?C.accent:C.border, color:on?C.bg:'#a1a1aa', border:`1px solid ${on?C.accent:C.border2}`, borderRadius:6, padding:'5px 13px', cursor:'pointer', fontFamily:mono, fontSize:'0.75em', fontWeight:700, transition:'all 0.15s' });

  const CONVOS = [
    { user:'What is a context window?',      asst:'A context window is the total text an LLM can see at once, measured in tokens. Everything outside it is invisible to the model.' },
    { user:'How does it work exactly?',       asst:'Every token is processed simultaneously via self-attention. The model weighs how much each token should influence every other.' },
    { user:'What happens when it fills up?',  asst:"Oldest messages are dropped — the model can no longer see them. Your app stores history, but the model itself forgets after each call." },
  ];

  function MsgBlock({label,text,color,bg}) {
    return <div style={{background:bg,border:`1px solid ${color}28`,borderRadius:5,padding:'4px 9px',marginBottom:3,fontFamily:mono,fontSize:'0.7em',color,lineHeight:1.4}}><span style={{color:C.dim,fontSize:'0.85em'}}>{label} </span>{text.substring(0,44)}&hellip;</div>;
  }

  function Tab3() {
    const [turn,  setTurn]  = useState(0);
    const [phase, setPhase] = useState('idle');
    const [ctx,   setCtx]   = useState([]);
    const timers = useRef([]);
    const clr = () => { timers.current.forEach(clearTimeout); timers.current = []; };

    const simulate = () => {
      if (phase !== 'idle' || turn >= CONVOS.length) return;
      const t = turn; clr(); setPhase('assemble');
      timers.current.push(setTimeout(() => { setPhase('send'); setCtx(CONVOS.slice(0,t+1).map((c,i)=>({user:c.user,asst:i<t?c.asst:null}))); }, 850));
      timers.current.push(setTimeout(() => setPhase('generate'), 1700));
      timers.current.push(setTimeout(() => { setPhase('save'); setTurn(t+1); }, 2550));
      timers.current.push(setTimeout(() => { setPhase('idle'); setCtx([]); }, 3300));
    };
    const reset = () => { clr(); setTurn(0); setPhase('idle'); setCtx([]); };

    const ag = phase==='assemble'||phase==='save', mg = phase==='send'||phase==='generate';
    const ar = phase==='send', al = phase==='generate';

    const PT = {
      idle:     { app:'Holds all conversation history permanently',                                                  model:'No memory between API calls — starts fresh every call' },
      assemble: { app:'⚡ Assembling: system + full history + new message\u2026',                                    model:'Waiting for API call\u2026' },
      send:     { app:'Sending full context bundle to model\u2026',                                                  model:`⚡ Received ${ctx.length>0?ctx.length*2-1:1} messages. Processing all tokens via attention\u2026` },
      generate: { app:'Waiting for streamed response\u2026',                                                         model:'\u2713 Generating reply token by token\u2026' },
      save:     { app:`\u2713 Saved turn ${turn} to database`,                                                       model:'\u2014 state wiped, nothing retained \u2014' },
    };

    return (
      <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:14,padding:20}}>
        <div style={{fontFamily:mono,fontSize:'0.78em',fontWeight:800,textTransform:'uppercase',letterSpacing:'0.1em',color:C.accent,marginBottom:14}}>
          ③ Memory Architecture — Where History Lives
        </div>
        <div style={{display:'grid',gridTemplateColumns:'1fr 72px 1fr',gap:14,alignItems:'start'}}>

          {/* App column */}
          <div style={{background:C.card2,borderRadius:12,padding:14,border:`2px solid ${ag?C.teal:C.border}`,boxShadow:ag?`0 0 22px ${C.teal}18`:'none',transition:'border-color 0.3s,box-shadow 0.3s'}}>
            <div style={{fontFamily:mono,fontSize:'0.75em',fontWeight:800,color:C.teal,textTransform:'uppercase',letterSpacing:'0.08em',marginBottom:10}}>Chat Application</div>
            <div style={{background:C.card3,border:`1px solid ${C.border2}`,borderRadius:8,padding:10,marginBottom:10}}>
              <div style={{fontFamily:mono,fontSize:'0.67em',color:C.yellow,fontWeight:700,textTransform:'uppercase',marginBottom:8}}>Persistent database</div>
              {turn===0
                ? <div style={{fontFamily:mono,fontSize:'0.69em',color:C.dim,fontStyle:'italic'}}>Empty &mdash; no turns yet</div>
                : CONVOS.slice(0,turn).map((c,i)=><div key={i}><MsgBlock label="U:" text={c.user} color={C.blue} bg="#060e1a"/><MsgBlock label="A:" text={c.asst} color={C.purple} bg="#0e0614"/></div>)
              }
            </div>
            <div style={{fontFamily:mono,fontSize:'0.69em',color:ag?C.teal:C.subtle,lineHeight:1.5,transition:'color 0.3s',minHeight:30}}>{PT[phase].app}</div>
          </div>

          {/* Arrow column */}
          <div style={{display:'flex',flexDirection:'column',alignItems:'center',gap:6,paddingTop:40}}>
            <div style={{fontFamily:mono,fontSize:'0.6em',color:C.dim,textAlign:'center',lineHeight:1.3,marginBottom:6}}>one API call<br/>per response</div>
            <svg width={68} height={120} viewBox="0 0 68 120">
              <defs><marker id="ma2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L8 5L1 9" fill="none" stroke="context-stroke" strokeWidth="1.5"/></marker></defs>
              <line x1={6} y1={36} x2={58} y2={36} stroke={ar?C.accent:C.border2} strokeWidth={ar?2.2:1} markerEnd="url(#ma2)" style={{transition:'stroke 0.3s'}}/>
              <text x={34} y={27} fill={ar?C.accent:C.dim} fontSize={8} fontFamily={mono} textAnchor="middle" style={{transition:'fill 0.3s'}}>full context</text>
              <line x1={58} y1={84} x2={6} y2={84} stroke={al?C.green:C.border2} strokeWidth={al?2.2:1} markerEnd="url(#ma2)" style={{transition:'stroke 0.3s'}}/>
              <text x={34} y={75} fill={al?C.green:C.dim} fontSize={8} fontFamily={mono} textAnchor="middle" style={{transition:'fill 0.3s'}}>response</text>
            </svg>
          </div>

          {/* Model column */}
          <div style={{background:C.card2,borderRadius:12,padding:14,border:`2px solid ${mg?C.blue:C.border}`,boxShadow:mg?`0 0 22px ${C.blue}18`:'none',transition:'border-color 0.3s,box-shadow 0.3s'}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:10}}>
              <div style={{fontFamily:mono,fontSize:'0.75em',fontWeight:800,color:C.blue,textTransform:'uppercase',letterSpacing:'0.08em'}}>Language Model</div>
              <div style={{fontFamily:mono,fontSize:'0.6em',background:C.red+'1a',border:`1px solid ${C.red}40`,color:C.red,borderRadius:4,padding:'2px 7px'}}>STATELESS</div>
            </div>
            <div style={{background:C.card3,border:`1px solid ${C.border2}`,borderRadius:8,padding:10,marginBottom:10,minHeight:70}}>
              <div style={{fontFamily:mono,fontSize:'0.67em',color:C.dim,fontWeight:700,textTransform:'uppercase',marginBottom:8}}>Context window (this call only)</div>
              {(phase==='send'||phase==='generate')&&ctx.length>0
                ? ctx.map((c,i)=><div key={i}><MsgBlock label="U:" text={c.user} color={C.blue} bg="#060e1a"/>{c.asst&&<MsgBlock label="A:" text={c.asst} color={C.purple} bg="#0e0614"/>}</div>)
                : <div style={{fontFamily:mono,fontSize:'0.69em',color:C.dim,fontStyle:'italic'}}>{turn===0?'Awaiting first call':'\u2014 wiped after each response \u2014'}</div>
              }
            </div>
            <div style={{fontFamily:mono,fontSize:'0.69em',color:mg?C.blue:C.subtle,lineHeight:1.5,transition:'color 0.3s',minHeight:30}}>{PT[phase].model}</div>
          </div>
        </div>

        <div style={{display:'flex',gap:8,marginTop:14,alignItems:'center'}}>
          <button style={btnS(phase==='idle'&&turn<CONVOS.length)} onClick={simulate} disabled={phase!=='idle'||turn>=CONVOS.length}>
            {turn>=CONVOS.length ? '\u2713 All turns done' : `\u25b6 Simulate turn ${turn+1}`}
          </button>
          <button style={btnS(false)} onClick={reset}>&#8635; Reset</button>
          <span style={{fontFamily:mono,fontSize:'0.7em',color:C.dim}}>{turn} / {CONVOS.length} turns</span>
          {phase!=='idle' && <span style={{fontFamily:mono,fontSize:'0.7em',color:C.accent,marginLeft:4}}>[{phase}]</span>}
        </div>

        <div style={{background:C.card3,border:`1px solid ${C.border}`,borderRadius:8,padding:'8px 12px',fontSize:'0.78em',color:C.slate,lineHeight:1.7,marginTop:10,fontFamily:mono}}>
          <span style={{color:C.teal,fontWeight:700}}>Key insight: </span>
          The model has <span style={{color:C.red}}>no database</span>, <span style={{color:C.red}}>no persistent state</span>, and <span style={{color:C.red}}>no memory system</span>.
          &ldquo;Memory&rdquo; only exists because the <span style={{color:C.accent}}>chat app</span> saves history and re-bundles it with every call.
          After each response, the model&rsquo;s state is completely wiped. The context window IS the memory &mdash; nothing more.
        </div>
      </div>
    );
  }

  window._renderTab3 = function() {
    const root = document.getElementById('tab3-react-root');
    if (!root) return;
    if (!window._tab3ReactRoot) window._tab3ReactRoot = ReactDOM.createRoot(root);
    window._tab3ReactRoot.render(<Tab3/>);
  };
})();
</script>

<!-- ══ Tab routing ══ -->
<script>
// Override switchTab to handle React roots
const _origSwitch = window.switchTab;
window.switchTab = function(i) {
  // Hide all content
  document.getElementById('tab-content').style.display  = 'none';
  document.getElementById('tab2-react-root').style.display = 'none';
  document.getElementById('tab3-react-root').style.display = 'none';

  activeTab = i;
  renderTabBar();

  if (i === 0) {
    document.getElementById('tab-content').style.display = 'block';
    renderTab1();
  } else if (i === 1) {
    document.getElementById('tab2-react-root').style.display = 'block';
    window._renderTab2();
  } else if (i === 2) {
    document.getElementById('tab3-react-root').style.display = 'block';
    window._renderTab3();
  }
};

// Ctrl+scroll zoom (matches perceptron.py)
var ZOOM = 1.0, zoomTimer = null;
function applyZoom() {
  document.body.style.zoom = ZOOM; clearTimeout(zoomTimer);
  var t = document.getElementById('zoom-toast');
  if (!t) {
    t = document.createElement('div'); t.id = 'zoom-toast';
    t.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#111118;border:1px solid #ff6b35;color:#ff6b35;font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;padding:8px 14px;border-radius:8px;z-index:9999;pointer-events:none;transition:opacity .3s;';
    document.body.appendChild(t);
  }
  t.textContent = 'zoom ' + Math.round(ZOOM * 100) + '%'; t.style.opacity = '1';
  zoomTimer = setTimeout(() => t.style.opacity = '0', 1200);
}
document.addEventListener('wheel', e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  ZOOM = Math.min(3.0, Math.max(0.4, Math.round((ZOOM + (e.deltaY > 0 ? -0.05 : 0.05)) * 100) / 100));
  applyZoom();
}, { passive: false });
</script>

</body>
</html>"""

CONTEXT_WINDOW_VISUAL_HEIGHT = 860