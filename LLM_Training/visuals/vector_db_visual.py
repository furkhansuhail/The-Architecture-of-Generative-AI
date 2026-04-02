"""
Self-contained HTML for the Vector Database interactive walkthrough.
Covers: Embeddings, IVF Algorithm, HNSW Algorithm, Recall & Speed, DB Comparison.
Embed in Streamlit via st.components.v1.html(VECTORDB_VISUAL_HTML, height=VECTORDB_VISUAL_HEIGHT).
"""

VECTORDB_VISUAL_HTML = """
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

var useState = React.useState;
var useEffect = React.useEffect;

var C = {
  bg:"#0a0a0f", card:"#12121a", border:"#1e1e2e",
  accent:"#ff6b35", blue:"#4ecdc4", purple:"#a78bfa",
  yellow:"#fbbf24", text:"#e4e4e7", muted:"#71717a",
  dim:"#3f3f46", red:"#ef4444", green:"#4ade80",
  cyan:"#38bdf8", pink:"#f472b6", orange:"#fb923c",
};

var ARR  = "\\u2192";
var DASH = "\\u2014";
var PLAY = "\\u25B6";
var PAUSE= "\\u23F8";
var BULB = "\\uD83D\\uDCA1";
var TARG = "\\uD83C\\uDFAF";
var CHK  = "\\u2713";

function TabBar(props) {
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

function Card(props) {
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

function StepButtons(props) {
  var n=props.total, step=props.step, setStep=props.setStep;
  var auto=props.auto, setAuto=props.setAuto;
  return (
    <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:16,flexWrap:"wrap"}}>
      {Array.from({length:n},function(_,i){
        return (
          <button key={i} onClick={function(){setStep(i);if(setAuto)setAuto(false);}} style={{
            width:32,height:32,borderRadius:6,
            border:"1.5px solid "+(step===i?C.accent:C.border),
            background:step===i?C.accent+"25":C.card,
            color:step===i?C.accent:C.muted,
            cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace",
          }}>{i+1}</button>
        );
      })}
      {setAuto&&(
        <button onClick={function(){setAuto(!auto);}} style={{
          padding:"0 14px",borderRadius:6,
          border:"1.5px solid "+(auto?C.yellow:C.border),
          background:auto?C.yellow+"20":C.card,
          color:auto?C.yellow:C.muted,cursor:"pointer",fontSize:12,
        }}>{auto?PAUSE:PLAY}</button>
      )}
    </div>
  );
}


/* ===============================================================
   TAB 1: WHAT ARE EMBEDDINGS?
   =============================================================== */
function TabEmbeddings() {
  var _s=useState(0); var step=_s[0]; var setStep=_s[1];
  var _a=useState(false); var auto=_a[0]; var setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%5;});},2500);
    return function(){clearInterval(t);};
  },[auto]);

  var docs=[
    {label:"Dog training", x:72,  y:68, ci:0},
    {label:"Cat care",     x:98,  y:46, ci:0},
    {label:"Pet nutrition",x:56,  y:88, ci:0},
    {label:"Fish tanks",   x:44,  y:60, ci:0},
    {label:"Puppy tricks", x:90,  y:90, ci:0},
    {label:"Exotic birds", x:116, y:66, ci:0},
    {label:"Pasta recipes",x:316, y:60, ci:1},
    {label:"Baking bread", x:342, y:42, ci:1},
    {label:"French cuisine",x:298,y:46, ci:1},
    {label:"Vegan meals",  x:358, y:80, ci:1},
    {label:"Sushi guide",  x:334, y:84, ci:1},
    {label:"Wine pairing", x:368, y:56, ci:1},
    {label:"Neural nets",  x:194, y:200,ci:2},
    {label:"Grad descent", x:220, y:218,ci:2},
    {label:"Python ML",    x:182, y:222,ci:2},
    {label:"Transformers", x:208, y:192,ci:2},
    {label:"GPU compute",  x:236, y:206,ci:2},
    {label:"Data pipelines",x:170,y:210,ci:2},
  ];

  var clCols=[C.blue,C.yellow,C.purple];
  var clNames=["Pets & Animals","Food & Recipes","Machine Learning"];
  var clCenters=[{x:79,y:68},{x:336,y:61},{x:202,y:208}];
  var query={x:313,y:64};

  var dists=docs.map(function(d,i){
    return {i:i,d:Math.sqrt(Math.pow(d.x-query.x,2)+Math.pow(d.y-query.y,2))};
  }).sort(function(a,b){return a.d-b.d;});
  var top3=new Set(dists.slice(0,3).map(function(x){return x.i;}));

  var steps=[
    {title:"18 documents encoded as vectors",desc:"Each piece of text is passed through an embedding model, converting it to a point in high-dimensional space. Here we visualize just 2 of those dimensions."},
    {title:"Similar meanings cluster naturally",desc:"Documents about pets land near other pet content. Recipes land near recipes. This structure was never programmed \u2014 it emerged from training on billions of text examples."},
    {title:"Clusters appear in the 2D slice",desc:"The model learned that \u2018dog training\u2019 and \u2018cat care\u2019 co-occur in similar contexts. Even with no shared keywords, their vectors are neighbors. This is semantic similarity."},
    {title:"A query vector arrives",desc:"Searching for \u2018cooking and food\u2019 produces a query vector. Encoded by the same model, it lands near the food cluster \u2014 even if those exact words don\u2019t appear in any document."},
    {title:"Top-3 nearest neighbors returned",desc:"We find the 3 stored vectors with smallest distance to the query. No keyword matching needed \u2014 results are semantically relevant. This is the power of vector search."},
  ];

  return (
    <div>
      <SectionTitle title="What Are Embeddings?" subtitle={"Meaning as geometry "+DASH+" similar concepts become close neighbors"} />
      <StepButtons total={5} step={step} setStep={setStep} auto={auto} setAuto={setAuto} />

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={800} height={305} viewBox="0 0 430 283" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,width:"100%",maxWidth:800}}>

          <text x={215} y={278} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{"\u2190 semantic dimension 1 \u2192"}</text>
          <text x={10} y={140} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace" transform="rotate(-90,10,140)">{"\u2190 dim 2 \u2192"}</text>
          {[100,200,300,400].map(function(v){return <line key={"gx"+v} x1={v} y1={0} x2={v} y2={270} stroke={C.border} strokeWidth={0.5} opacity={0.5} />;}) }
          {[80,160].map(function(v){return <line key={"gy"+v} x1={18} y1={v} x2={430} y2={v} stroke={C.border} strokeWidth={0.5} opacity={0.5} />;}) }

          {step>=1&&clCenters.map(function(c,ci){
            var col=clCols[ci];
            return (
              <g key={"cl"+ci}>
                <circle cx={c.x} cy={c.y} r={62} fill={col+"07"} stroke={col+"22"} strokeWidth={1} strokeDasharray="5,4" />
                {step>=2&&<text x={c.x} y={c.y-66} textAnchor="middle" fill={col} fontSize={8} fontWeight={700} fontFamily="monospace">{clNames[ci]}</text>}
              </g>
            );
          })}

          {step>=4&&dists.slice(0,3).map(function(item){
            var d=docs[item.i];
            return <line key={"nl"+item.i} x1={query.x} y1={query.y} x2={d.x} y2={d.y} stroke={C.green} strokeWidth={1.5} strokeDasharray="4,3" opacity={0.85} />;
          })}

          {docs.map(function(d,i){
            var col=clCols[d.ci];
            var isN=step>=4&&top3.has(i);
            return (
              <g key={"d"+i}>
                <circle cx={d.x} cy={d.y} r={isN?7:5} fill={isN?C.green:col} stroke={isN?C.green:col+"80"} strokeWidth={isN?2:0.5} />
                <text x={d.x} y={d.y-9} textAnchor="middle" fill={isN?C.green:col} fontSize={7} fontFamily="monospace" fontWeight={isN?700:400}>{d.label}</text>
              </g>
            );
          })}

          {step>=3&&(
            <g>
              <circle cx={query.x} cy={query.y} r={11} fill={C.accent+"30"} stroke={C.accent} strokeWidth={2} />
              <text x={query.x} y={query.y+4} textAnchor="middle" dominantBaseline="middle" fill={C.accent} fontSize={8} fontWeight={800} fontFamily="monospace">Q</text>
              <text x={query.x} y={query.y-17} textAnchor="middle" fill={C.accent} fontSize={7.5} fontFamily="monospace">"cooking &amp; food"</text>
            </g>
          )}

          {step===0&&(
            <g>
              <rect x={155} y={120} width={126} height={46} rx={8} fill={C.purple+"18"} stroke={C.purple+"50"} strokeWidth={1} />
              <text x={218} y={140} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">Embedding</text>
              <text x={218} y={155} textAnchor="middle" fill={C.purple} fontSize={9} fontWeight={700} fontFamily="monospace">Model</text>
              <text x={218} y={175} textAnchor="middle" fill={C.muted} fontSize={7} fontFamily="monospace">text {ARR} [0.12, -0.84, 0.37 ...]</text>
            </g>
          )}
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+steps[step].title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{steps[step].desc}</div>
      </Card>

      <Insight>
        An embedding model is trained to put <span style={{color:C.accent,fontWeight:700}}>similar meanings close together</span>. The geometry that emerges encodes real relationships: <span style={{color:C.blue,fontWeight:700}}>king \u2212 man + woman \u2248 queen</span>. Distance becomes a proxy for semantic relatedness \u2014 and that\u2019s what vector search exploits.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 2: IVF INDEX
   =============================================================== */
function TabIVF() {
  var _s=useState(0); var step=_s[0]; var setStep=_s[1];
  var _a=useState(false); var auto=_a[0]; var setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%6;});},2000);
    return function(){clearInterval(t);};
  },[auto]);

  var cls=[
    {cx:108,cy:88, col:C.blue,   name:"A"},
    {cx:388,cy:88, col:C.accent, name:"B"},
    {cx:108,cy:222,col:C.purple, name:"C"},
    {cx:388,cy:222,col:C.yellow, name:"D"},
  ];
  var pts=[
    {x:55,y:62,ci:0},{x:80,y:88,ci:0},{x:110,y:58,ci:0},{x:58,y:95,ci:0},{x:118,y:88,ci:0},{x:78,y:55,ci:0},{x:128,y:72,ci:0},
    {x:338,y:62,ci:1},{x:358,y:82,ci:1},{x:382,y:52,ci:1},{x:405,y:68,ci:1},{x:345,y:92,ci:1},{x:378,y:98,ci:1},{x:418,y:82,ci:1},
    {x:55,y:198,ci:2},{x:80,y:222,ci:2},{x:110,y:198,ci:2},{x:58,y:248,ci:2},{x:118,y:238,ci:2},{x:78,y:252,ci:2},{x:128,y:218,ci:2},
    {x:338,y:198,ci:3},{x:358,y:218,ci:3},{x:382,y:198,ci:3},{x:405,y:215,ci:3},{x:345,y:238,ci:3},{x:378,y:245,ci:3},{x:418,y:210,ci:3},
  ];
  var query={x:376,y:82};
  var nearCI=1;

  var dists=cls.map(function(c){return Math.sqrt(Math.pow(c.cx-query.x,2)+Math.pow(c.cy-query.y,2)).toFixed(1);});
  var clPts=pts.filter(function(p){return p.ci===nearCI;});
  clPts.sort(function(a,b){return(Math.pow(a.x-query.x,2)+Math.pow(a.y-query.y,2))-(Math.pow(b.x-query.x,2)+Math.pow(b.y-query.y,2));});
  var top3Pts=clPts.slice(0,3);

  var steps=[
    {title:"Index built: 28 vectors in 4 clusters",desc:"During indexing (offline), k-means clustering partitions all vectors. Each centroid represents the center of its cluster. This pre-computation happens once and is cached."},
    {title:"Query vector arrives",desc:"A new query comes in at runtime. Brute-force would compare it to all 28 vectors. IVF finds a shortcut \u2014 compare to centroids first."},
    {title:"4 centroid distances computed",desc:"Just 4 comparisons \u2014 one per centroid. At 1M vectors with 1000 clusters, that\u2019s 1000 distance calculations instead of 1,000,000."},
    {title:"Nearest centroid: Cluster B wins",desc:"Cluster B\u2019s centroid is closest to the query. All other clusters are eliminated. Only Cluster B\u2019s 7 vectors will be searched."},
    {title:"Full search within Cluster B",desc:"7 vectors checked. Top-3 nearest are highlighted in green. Results returned in a fraction of the brute-force cost."},
    {title:"Done: 7 checked vs 28 brute-force",desc:"4\u00D7 fewer comparisons here. At 1M vectors with 256 clusters, this becomes ~3,900 comparisons vs 1,000,000. Increase nprobe to search more clusters if recall drops."},
  ];

  return (
    <div>
      <SectionTitle title="IVF: Inverted File Index" subtitle={"Cluster during indexing "+DASH+" search one cluster at query time"} />
      <StepButtons total={6} step={step} setStep={setStep} auto={auto} setAuto={setAuto} />

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={800} height={315} viewBox="0 0 760 310" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,width:"100%",maxWidth:800}}>

          {cls.map(function(cl,i){
            var isW=i===nearCI, dim=step>=3&&!isW;
            return <circle key={"h"+i} cx={cl.cx} cy={cl.cy} r={84} fill={dim?"transparent":cl.col+(isW&&step>=3?"18":"08")} stroke={dim?cl.col+"18":cl.col+(isW&&step>=3?"55":"24")} strokeWidth={isW&&step>=3?2:1} strokeDasharray={isW&&step>=3?"0":"6,4"} />;
          })}

          {step===2&&cls.map(function(cl,i){
            var isN=i===nearCI;
            var mx=(cl.cx+query.x)/2, my=(cl.cy+query.y)/2;
            return (
              <g key={"dl"+i}>
                <line x1={query.x} y1={query.y} x2={cl.cx} y2={cl.cy} stroke={isN?cl.col:C.dim} strokeWidth={isN?2:1} strokeDasharray="5,4" opacity={isN?1:0.35} />
                <text x={mx} y={my-5} textAnchor="middle" fill={isN?cl.col:C.dim} fontSize={9} fontFamily="monospace" fontWeight={isN?700:400}>{"d="+dists[i]}</text>
              </g>
            );
          })}

          {step>=4&&top3Pts.map(function(p,i){
            return <line key={"rl"+i} x1={query.x} y1={query.y} x2={p.x} y2={p.y} stroke={C.green} strokeWidth={1.5} strokeDasharray="4,3" opacity={0.85} />;
          })}

          {pts.map(function(p,i){
            var col=cls[p.ci].col;
            var inW=p.ci===nearCI;
            var isRes=step>=4&&top3Pts.indexOf(p)>=0;
            var dim=step>=3&&!inW;
            return <circle key={"p"+i} cx={p.x} cy={p.y} r={isRes?7:5} fill={isRes?C.green:(dim?"transparent":col)} stroke={isRes?C.green:(dim?col+"18":col+"80")} strokeWidth={isRes?2:0.5} />;
          })}

          {cls.map(function(cl,i){
            var dim=step>=3&&i!==nearCI;
            return (
              <g key={"c"+i}>
                <rect x={cl.cx-9} y={cl.cy-9} width={18} height={18} fill={cl.col+(dim?"20":"38")} stroke={dim?C.dim:cl.col} strokeWidth={1.5} transform={"rotate(45,"+cl.cx+","+cl.cy+")"} />
                <text x={cl.cx} y={cl.cy-20} textAnchor="middle" fill={dim?C.dim:cl.col} fontSize={9} fontWeight={700} fontFamily="monospace">{cl.name}</text>
              </g>
            );
          })}

          {step>=1&&(
            <g>
              <circle cx={query.x} cy={query.y} r={12} fill={C.accent+"30"} stroke={C.accent} strokeWidth={2.5} />
              <text x={query.x} y={query.y+4} textAnchor="middle" dominantBaseline="middle" fill={C.accent} fontSize={10} fontWeight={800} fontFamily="monospace">Q</text>
            </g>
          )}

          {step===5&&(
            <g>
              <rect x={490} y={108} width={256} height={98} rx={10} fill={C.card} stroke={C.green+"40"} strokeWidth={1} />
              <text x={618} y={134} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">VECTORS CHECKED</text>
              <text x={618} y={176} textAnchor="middle" fill={C.green} fontSize={44} fontWeight={800} fontFamily="monospace">7</text>
              <text x={618} y={198} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">vs 28 brute-force</text>
            </g>
          )}
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+steps[step].title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{steps[step].desc}</div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.purple,marginBottom:8}}>Tuning nprobe</div>
        <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
          {[{v:"nprobe=1",r:"~82%",q:"10K"},{v:"nprobe=4",r:"~91%",q:"3K"},{v:"nprobe=16",r:"~95%",q:"1K"},{v:"nprobe=64",r:"~99%",q:"300"}].map(function(d,i){
            var cols=[C.green,C.yellow,C.accent,C.red];
            var col=cols[i];
            return (
              <div key={i} style={{flex:1,minWidth:90,background:"#08080d",borderRadius:8,padding:"10px 12px",border:"1px solid "+col+"30",textAlign:"center"}}>
                <div style={{fontSize:10,color:col,fontWeight:700,fontFamily:"monospace"}}>{d.v}</div>
                <div style={{fontSize:20,color:col,fontWeight:800,margin:"4px 0"}}>{d.r}</div>
                <div style={{fontSize:8,color:C.dim,fontFamily:"monospace"}}>recall</div>
                <div style={{fontSize:10,color:C.muted,marginTop:4,fontFamily:"monospace"}}>{d.q+" QPS"}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight>
        IVF\u2019s main weakness: the <span style={{color:C.red,fontWeight:700}}>cluster boundary problem</span>. A query near the edge of two clusters misses results in the neighbor. Fix by increasing <span style={{color:C.accent,fontWeight:700}}>nprobe</span> \u2014 search more centroids at the cost of speed.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 3: HNSW INDEX
   =============================================================== */
function TabHNSW() {
  var _s=useState(0); var step=_s[0]; var setStep=_s[1];
  var _a=useState(false); var auto=_a[0]; var setAuto=_a[1];

  useEffect(function(){
    if(!auto)return;
    var t=setInterval(function(){setStep(function(s){return(s+1)%7;});},2000);
    return function(){clearInterval(t);};
  },[auto]);

  var NX=[48,128,208,288,368,448,528,608,672,750];
  var LY={2:72,1:188,0:304};
  var LBand={2:[40,105],1:[155,220],0:[271,336]};
  var inL2=[0,5,8], inL1=[0,2,5,7,8];
  var E2=[[0,5],[5,8]];
  var E1=[[0,2],[2,5],[5,7],[7,8],[0,5],[2,7]];
  var E0=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[0,2],[4,6],[6,8]];
  var TARGET=9, ENTRY=0;
  var lCols=[C.blue,C.purple,C.accent];
  var lNames=["Layer 0 \u2014 all nodes, dense","Layer 1 \u2014 subset, medium range","Layer 2 \u2014 few nodes, long range"];

  var SS=[
    {L:2,cur:0,vis:[],path:[],title:"Enter at the entry point",desc:"Layer 2 has only 3 nodes with long-range connections spanning the whole space. Every search starts at the same entry node, regardless of the query."},
    {L:2,cur:5,vis:[0],path:[[0,5,2]],title:"Layer 2: greedy step toward target",desc:"Node 5 (x=448) is closer to the target than node 0 (x=48). Move there greedily."},
    {L:2,cur:8,vis:[0,5],path:[[0,5,2],[5,8,2]],title:"Layer 2: local optimum reached",desc:"Node 8 (x=672) is closer still. No remaining neighbor is closer to the target \u2014 local optimum in Layer 2."},
    {L:1,cur:8,vis:[],path:[],drop:2,title:"Drop to Layer 1",desc:"Enter Layer 1 at node 8. This layer has more nodes and shorter-range connections, allowing finer navigation."},
    {L:1,cur:8,vis:[],path:[],title:"Layer 1: also local optimum",desc:"From node 8, the neighbor node 7 (x=608) is farther from the target. Immediate local optimum \u2014 drop to Layer 0."},
    {L:0,cur:8,vis:[],path:[],drop:1,title:"Drop to Layer 0 (all nodes)",desc:"Enter the base layer \u2014 all 10 nodes with dense short-range connections. Fine-grained search begins here."},
    {L:0,cur:9,vis:[8],path:[[8,9,0]],title:"Result found in Layer 0!",desc:"Node 9 is adjacent to node 8 and is the nearest neighbor. Found in 6 steps across 3 layers."},
  ];

  var s=SS[step];

  function gp(nid,layer){return {x:NX[nid],y:LY[layer]};}
  function nil(nid,layer){if(layer===0)return nid<=9;if(layer===1)return inL1.indexOf(nid)>=0;return inL2.indexOf(nid)>=0;}
  function ge(layer){return layer===2?E2:(layer===1?E1:E0);}

  return (
    <div>
      <SectionTitle title="HNSW: Hierarchical Navigable Small World" subtitle={"Navigate coarse "+ARR+" fine: highway first, local streets last"} />
      <StepButtons total={7} step={step} setStep={setStep} auto={auto} setAuto={setAuto} />

      <div style={{display:"flex",justifyContent:"center",marginBottom:16}}>
        <svg width={800} height={370} viewBox="0 0 800 370" style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,width:"100%",maxWidth:800}}>

          {[2,1,0].map(function(layer){
            var band=LBand[layer], isAct=s.L===layer, col=lCols[layer];
            return (
              <g key={"band"+layer}>
                <rect x={16} y={band[0]} width={768} height={band[1]-band[0]} rx={8} fill={isAct?col+"08":col+"03"} stroke={isAct?col+"35":C.border} strokeWidth={isAct?1.5:0.5} />
                <text x={26} y={band[0]+16} fill={isAct?col:C.dim} fontSize={9} fontWeight={isAct?700:400} fontFamily="monospace">{lNames[layer]}</text>
              </g>
            );
          })}

          {s.drop!==undefined&&(function(){
            var p1=gp(s.cur,s.drop), p2=gp(s.cur,s.drop-1);
            return (
              <g>
                <line x1={p1.x} y1={p1.y+10} x2={p2.x} y2={p2.y-13} stroke={C.yellow} strokeWidth={2} strokeDasharray="5,3" />
                <polygon points={p2.x+","+p2.y+" "+(p2.x-5)+","+(p2.y-12)+" "+(p2.x+5)+","+(p2.y-12)} fill={C.yellow} />
                <text x={p2.x+14} y={(p1.y+p2.y)/2+4} fill={C.yellow} fontSize={9} fontFamily="monospace">drop down</text>
              </g>
            );
          })()}

          {[2,1,0].map(function(layer){
            return ge(layer).map(function(e,ei){
              if(!nil(e[0],layer)||!nil(e[1],layer))return null;
              var onP=s.path.some(function(p){return p[2]===layer&&((p[0]===e[0]&&p[1]===e[1])||(p[0]===e[1]&&p[1]===e[0]));});
              var p1=gp(e[0],layer),p2=gp(e[1],layer);
              var isAct=s.L===layer;
              return <line key={"e"+layer+ei} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke={onP?C.green:(isAct?lCols[layer]+"45":C.dim+"20")} strokeWidth={onP?2.5:1} />;
            });
          })}

          {[2,1,0].map(function(layer){
            var nodes=layer===2?inL2:(layer===1?inL1:[0,1,2,3,4,5,6,7,8,9]);
            var isAct=s.L===layer;
            return nodes.map(function(nid){
              var pos=gp(nid,layer);
              var isCur=nid===s.cur&&s.L===layer;
              var isVis=s.vis.indexOf(nid)>=0&&s.L===layer;
              var isDone=step===6&&nid===TARGET&&layer===0;
              var isTarg=nid===TARGET&&layer===0;
              var isEnt=nid===ENTRY&&layer===2&&step===0;
              var r=isCur||isTarg?9:6;
              var fill,stroke;
              if(isDone){fill=C.green;stroke=C.green;r=11;}
              else if(isCur){fill=C.accent;stroke=C.accent+"cc";}
              else if(isEnt){fill=C.purple;stroke=C.purple;}
              else if(isTarg){fill=C.green+"30";stroke=C.green;}
              else if(isVis){fill=C.dim;stroke=C.muted;r=5;}
              else{fill=isAct?lCols[layer]+"40":lCols[layer]+"14";stroke=isAct?lCols[layer]+"65":lCols[layer]+"22";}
              return (
                <g key={"n"+layer+nid}>
                  <circle cx={pos.x} cy={pos.y} r={r} fill={fill} stroke={stroke} strokeWidth={1.5} />
                  {isCur&&!isDone&&<text x={pos.x} y={pos.y-14} textAnchor="middle" fill={C.accent} fontSize={8} fontFamily="monospace">here</text>}
                  {isEnt&&<text x={pos.x} y={pos.y-14} textAnchor="middle" fill={C.purple} fontSize={8} fontFamily="monospace">entry</text>}
                  {isTarg&&layer===0&&<text x={pos.x} y={pos.y-14} textAnchor="middle" fill={C.green} fontSize={8} fontWeight={isDone?700:400} fontFamily="monospace">{isDone?"result "+CHK:"target"}</text>}
                </g>
              );
            });
          })}

          {step===6&&(
            <g>
              <rect x={508} y={228} width={274} height={40} rx={6} fill={C.card} stroke={C.green+"40"} strokeWidth={1} />
              <text x={645} y={244} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">Visited 6 of 10 nodes</text>
              <text x={645} y={260} textAnchor="middle" fill={C.green} fontSize={9} fontWeight={700} fontFamily="monospace">Scales to O(log n) across millions</text>
            </g>
          )}
        </svg>
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:13,fontWeight:700,color:C.accent,marginBottom:4}}>{"Step "+(step+1)+": "+s.title}</div>
        <div style={{fontSize:11,color:C.muted,lineHeight:1.7}}>{s.desc}</div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Key parameters</div>
        <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
          {[
            {p:"M",r:"16\u201332",d:"Connections per node. Higher = better graph quality and recall, more memory and slower indexing."},
            {p:"ef_construction",r:"100\u2013200",d:"Beam width at index build time. Higher = better quality graph. Set once; not tuned at query time."},
            {p:"ef_search",r:"64\u2013128",d:"Beam width at query time. The primary recall\u2013speed knob. ef=64 gives ~97% recall as a default."},
          ].map(function(d,i){
            return (
              <div key={i} style={{flex:1,minWidth:150,background:"#08080d",borderRadius:8,padding:"10px 12px",border:"1px solid "+C.purple+"30"}}>
                <div style={{fontSize:11,color:C.purple,fontWeight:700,fontFamily:"monospace"}}>{d.p}</div>
                <div style={{fontSize:9,color:C.accent,marginTop:2,fontFamily:"monospace"}}>{"typical: "+d.r}</div>
                <div style={{fontSize:10,color:C.muted,marginTop:4,lineHeight:1.5}}>{d.d}</div>
              </div>
            );
          })}
        </div>
      </Card>

      <Insight>
        HNSW\u2019s key insight: the same graph at different densities. Top layers provide <span style={{color:C.accent,fontWeight:700}}>long-range teleportation</span> past irrelevant regions. Bottom layers provide <span style={{color:C.blue,fontWeight:700}}>fine-grained accuracy</span>. The hierarchy enables sub-linear search time: O(log n).
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 4: RECALL & SPEED TRADEOFF
   =============================================================== */
function TabRecallSpeed() {
  var _m=useState("hnsw"); var mode=_m[0]; var setMode=_m[1];
  var _e=useState(64); var efVal=_e[0]; var setEfVal=_e[1];
  var _n=useState(8); var nprobeVal=_n[0]; var setNprobeVal=_n[1];

  var hnswCurve=[
    {p:8,r:80,q:12000},{p:16,r:88,q:8000},{p:32,r:94,q:5000},
    {p:64,r:97,q:2500},{p:128,r:99,q:1200},{p:256,r:99.5,q:600}
  ];
  var ivfCurve=[
    {p:1,r:65,q:10000},{p:4,r:82,q:6000},{p:8,r:87,q:3500},
    {p:16,r:91,q:2000},{p:32,r:95,q:1100},{p:64,r:97,q:600}
  ];

  function interp(curve,val) {
    for(var i=0;i<curve.length-1;i++){
      if(val>=curve[i].p&&val<=curve[i+1].p){
        var t=(val-curve[i].p)/(curve[i+1].p-curve[i].p);
        return {r:curve[i].r+t*(curve[i+1].r-curve[i].r),q:curve[i].q+t*(curve[i+1].q-curve[i].q)};
      }
    }
    return val<=curve[0].p?{r:curve[0].r,q:curve[0].q}:{r:curve[curve.length-1].r,q:curve[curve.length-1].q};
  }

  var curCurve=mode==="hnsw"?hnswCurve:ivfCurve;
  var curVal=mode==="hnsw"?efVal:nprobeVal;
  var cur=interp(curCurve,curVal);

  var W=740,H=262,PL=50,PR=16,PT=16,PB=38;
  var pw=W-PL-PR, ph=H-PT-PB;
  function sx(q){return PL+(q/13000)*pw;}
  function sy(r){return PT+(1-(r-60)/40)*ph;}
  function mkPath(curve){
    var sorted=curve.slice().sort(function(a,b){return a.q-b.q;});
    return sorted.map(function(p,i){return(i===0?"M":"L")+sx(p.q).toFixed(1)+","+sy(p.r).toFixed(1);}).join(" ");
  }

  var tooltipLeft=sx(cur.q)>W-150;
  var tx=tooltipLeft?sx(cur.q)-138:sx(cur.q)+14;

  return (
    <div>
      <SectionTitle title="Recall vs Speed Tradeoff" subtitle={"Every index has a dial "+DASH+" push recall up and QPS comes down"} />

      <div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:20}}>
        {[{v:"hnsw",l:"HNSW",c:C.purple},{v:"ivf",l:"IVF",c:C.green}].map(function(m){
          var on=mode===m.v;
          return <button key={m.v} onClick={function(){setMode(m.v);}} style={{padding:"8px 22px",borderRadius:20,border:"1.5px solid "+(on?m.c:C.border),background:on?m.c+"20":C.card,color:on?m.c:C.muted,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"monospace"}}>{m.l}</button>;
        })}
      </div>

      <div style={{display:"flex",justifyContent:"center",marginBottom:14}}>
        <svg width={800} height={262} viewBox={"0 0 "+W+" "+H} style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,width:"100%",maxWidth:800}}>
          {[0,2000,4000,6000,8000,10000,12000].map(function(q){return <line key={"gx"+q} x1={sx(q)} y1={PT} x2={sx(q)} y2={PT+ph} stroke={C.border} strokeWidth={0.5} />;}) }
          {[60,70,80,90,100].map(function(r){return <line key={"gy"+r} x1={PL} y1={sy(r)} x2={PL+pw} y2={sy(r)} stroke={C.border} strokeWidth={0.5} />;}) }
          {[0,2,4,6,8,10,12].map(function(k){return <text key={"xl"+k} x={sx(k*1000)} y={H-6} textAnchor="middle" fill={C.dim} fontSize={9} fontFamily="monospace">{k===0?"0":k+"K"}</text>;}) }
          {[60,70,80,90,100].map(function(r){return <text key={"yl"+r} x={PL-6} y={sy(r)+3} textAnchor="end" fill={C.dim} fontSize={9} fontFamily="monospace">{r+"%"}</text>;}) }
          <text x={W/2} y={H-1} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace">{"Queries per second "+ARR}</text>
          <text x={12} y={PT+ph/2} textAnchor="middle" fill={C.muted} fontSize={9} fontFamily="monospace" transform={"rotate(-90,12,"+(PT+ph/2)+")"}>{"Recall@10 "+ARR}</text>

          <path d={mkPath(hnswCurve)} fill="none" stroke={C.purple} strokeWidth={2} opacity={mode==="hnsw"?1:0.2} />
          {hnswCurve.map(function(p,i){return <circle key={"hp"+i} cx={sx(p.q)} cy={sy(p.r)} r={4} fill={C.purple} opacity={mode==="hnsw"?1:0.2} />;}) }
          {mode==="hnsw"&&hnswCurve.map(function(p,i){return <text key={"hl"+i} x={sx(p.q)+6} y={sy(p.r)-5} fill={C.purple} fontSize={8} fontFamily="monospace" opacity={0.8}>{"ef="+p.p}</text>;}) }

          <path d={mkPath(ivfCurve)} fill="none" stroke={C.green} strokeWidth={2} opacity={mode==="ivf"?1:0.2} />
          {ivfCurve.map(function(p,i){return <circle key={"ip"+i} cx={sx(p.q)} cy={sy(p.r)} r={4} fill={C.green} opacity={mode==="ivf"?1:0.2} />;}) }
          {mode==="ivf"&&ivfCurve.map(function(p,i){return <text key={"il"+i} x={sx(p.q)+6} y={sy(p.r)-5} fill={C.green} fontSize={8} fontFamily="monospace" opacity={0.8}>{"nprobe="+p.p}</text>;}) }

          <circle cx={sx(cur.q)} cy={sy(cur.r)} r={11} fill={C.accent+"40"} stroke={C.accent} strokeWidth={2.5} />
          <circle cx={sx(cur.q)} cy={sy(cur.r)} r={4} fill={C.accent} />
          <rect x={tx} y={sy(cur.r)-22} width={124} height={38} rx={6} fill={C.card} stroke={C.accent+"55"} strokeWidth={1} />
          <text x={tx+62} y={sy(cur.r)-7} textAnchor="middle" fill={C.accent} fontSize={9} fontWeight={700} fontFamily="monospace">{cur.r.toFixed(1)+"% recall"}</text>
          <text x={tx+62} y={sy(cur.r)+9} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{Math.round(cur.q).toLocaleString()+" QPS"}</text>

          <rect x={W-148} y={PT+4} width={130} height={46} rx={6} fill={C.card} stroke={C.border} />
          <circle cx={W-128} cy={PT+16} r={5} fill={C.purple} />
          <text x={W-118} y={PT+20} fill={C.purple} fontSize={10} fontFamily="monospace">HNSW</text>
          <circle cx={W-128} cy={PT+36} r={5} fill={C.green} />
          <text x={W-118} y={PT+40} fill={C.green} fontSize={10} fontFamily="monospace">IVF</text>
        </svg>
      </div>

      <div style={{maxWidth:750,margin:"0 auto 14px"}}>
        <div style={{display:"flex",alignItems:"center",gap:14}}>
          <span style={{fontSize:11,color:C.muted,fontFamily:"monospace",minWidth:90}}>{mode==="hnsw"?"ef_search":"nprobe"}</span>
          {mode==="hnsw"
            ?<input type="range" min={8} max={256} step={4} value={efVal} onChange={function(e){setEfVal(parseInt(e.target.value));}} style={{flex:1}} />
            :<input type="range" min={1} max={64} step={1} value={nprobeVal} onChange={function(e){setNprobeVal(parseInt(e.target.value));}} style={{flex:1}} />
          }
          <span style={{fontSize:16,color:C.accent,fontFamily:"monospace",fontWeight:700,minWidth:36}}>{mode==="hnsw"?efVal:nprobeVal}</span>
        </div>
      </div>

      <div style={{display:"flex",gap:10,maxWidth:750,margin:"0 auto 16px",flexWrap:"wrap"}}>
        {[
          {label:"RECALL@10",val:cur.r.toFixed(1)+"%",col:C.green},
          {label:"QUERIES/SEC",val:Math.round(cur.q).toLocaleString(),col:C.blue},
          {label:"LATENCY P99",val:(1000/cur.q*2.5).toFixed(1)+"ms",col:C.yellow},
        ].map(function(m,i){
          return (
            <Card key={i} style={{flex:1,minWidth:140,textAlign:"center"}}>
              <div style={{fontSize:9,color:C.muted,fontFamily:"monospace",marginBottom:4}}>{m.label}</div>
              <div style={{fontSize:30,color:m.col,fontWeight:800}}>{m.val}</div>
            </Card>
          );
        })}
      </div>

      <Insight icon={TARG} title="Production Tuning">
        Most teams target 95%+ recall. For HNSW this means ef_search {"\u2265"} 48. HNSW <span style={{color:C.purple,fontWeight:700}}>Pareto-dominates</span> IVF on this chart {DASH} at the same recall level, HNSW achieves higher QPS. IVF wins when <span style={{color:C.green,fontWeight:700}}>memory is severely constrained</span> {DASH} it stores far less per vector.
      </Insight>
    </div>
  );
}


/* ===============================================================
   TAB 5: VECTOR DATABASES
   =============================================================== */
function TabDatabases() {
  var _s=useState(0); var sel=_s[0]; var setSel=_s[1];

  var dbs=[
    {name:"Pinecone",year:2019,col:C.blue,type:"Managed",scale:5,filter:4,hybrid:3,ease:5,index:"HNSW + PQ",max:"Billions",
     short:"Fully managed, zero-infra. Fastest path to production.",
     note:"No self-hosting option. Costs scale with vector volume. Ideal when ops complexity is the bottleneck.",
     arch:"Proprietary distributed index with automatic sharding and replication."},
    {name:"Qdrant",year:2021,col:C.accent,type:"Open source",scale:5,filter:5,hybrid:4,ease:4,index:"HNSW",max:"Billions",
     short:"Rust-built, best-in-class payload filtering.",
     note:"Filtering is pre-search (not post), preserving recall even with restrictive filters. Strong default choice for new projects.",
     arch:"Single-file storage engine with columnar payload storage alongside the HNSW graph."},
    {name:"Weaviate",year:2019,col:C.purple,type:"Open source",scale:4,filter:4,hybrid:5,ease:3,index:"HNSW",max:"~500M",
     short:"Best native hybrid search \u2014 BM25 + vector in one query.",
     note:"Built-in modules for auto-embedding, generative search, and reranking. GraphQL + REST APIs.",
     arch:"Per-class indices combining HNSW with a BM25 inverted index for hybrid queries."},
    {name:"pgvector",year:2021,col:C.green,type:"Extension",scale:3,filter:5,hybrid:4,ease:4,index:"IVFFlat / HNSW",max:"~10M",
     short:"Postgres extension. Full SQL, zero new infrastructure.",
     note:"JOINs, WHERE, transactions, access control. Tops out around 10M vectors without dedicated tuning.",
     arch:"GiST index on Postgres pages. HNSW available since pgvector 0.5."},
    {name:"Chroma",year:2022,col:C.yellow,type:"Open source",scale:2,filter:3,hybrid:2,ease:5,index:"hnswlib",max:"~500K",
     short:"Lightweight Python-native. Fastest from zero to working prototype.",
     note:"Not designed for production at scale. Excellent for local dev, notebooks, and demos.",
     arch:"SQLite for metadata + hnswlib for vectors. Client-server mode available."},
    {name:"Milvus",year:2019,col:C.pink,type:"Open source",scale:5,filter:4,hybrid:3,ease:2,index:"HNSW, IVF, DiskANN",max:"Trillions",
     short:"Billion-scale, Kubernetes-native, fully distributed.",
     note:"Significant infra investment required. Best for teams with dedicated MLOps and 100M+ vectors.",
     arch:"Separated compute/storage with etcd for metadata and multiple pluggable index types."},
  ];

  var db=dbs[sel];
  var attrs=["scale","filter","hybrid","ease"];
  var attrLabels=["Scale","Filtering","Hybrid search","Setup ease"];

  return (
    <div>
      <SectionTitle title="Choosing a Vector Database" subtitle={"Six options, each with a distinct niche "+DASH+" pick the one that fits your constraints"} />

      <div style={{display:"flex",gap:6,justifyContent:"center",marginBottom:20,flexWrap:"wrap"}}>
        {dbs.map(function(d,i){
          var on=sel===i;
          return <button key={i} onClick={function(){setSel(i);}} style={{padding:"8px 16px",borderRadius:8,border:"1.5px solid "+(on?d.col:C.border),background:on?d.col+"20":C.card,color:on?d.col:C.muted,cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace"}}>{d.name}</button>;
        })}
      </div>

      <Card highlight={true} style={{maxWidth:750,margin:"0 auto 16px",borderColor:db.col}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
          <div style={{flex:1,minWidth:260}}>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
              <div style={{fontSize:22,fontWeight:800,color:db.col}}>{db.name}</div>
              <div style={{fontSize:10,padding:"2px 8px",borderRadius:12,background:db.col+"20",border:"1px solid "+db.col+"40",color:db.col,fontFamily:"monospace"}}>{db.type}</div>
              <div style={{fontSize:10,color:C.dim,fontFamily:"monospace"}}>{db.year}</div>
            </div>
            <div style={{fontSize:12,color:C.text,lineHeight:1.6,marginBottom:8}}>{db.short}</div>
            <div style={{fontSize:11,color:C.muted,lineHeight:1.7,borderLeft:"3px solid "+db.col+"40",paddingLeft:12,fontStyle:"italic"}}>{db.note}</div>
          </div>
          <div style={{minWidth:170}}>
            {attrs.map(function(a,i){
              var v=db[a];
              return (
                <div key={a} style={{display:"flex",alignItems:"center",gap:8,marginBottom:7}}>
                  <div style={{width:80,fontSize:9,color:C.muted,textAlign:"right",fontFamily:"monospace"}}>{attrLabels[i]}</div>
                  <div style={{flex:1,height:16,background:C.dim+"20",borderRadius:4,overflow:"hidden"}}>
                    <div style={{width:(v/5*100)+"%",height:"100%",background:db.col+"55",border:"1px solid "+db.col+"45",borderRadius:4,transition:"width 0.4s"}} />
                  </div>
                  <div style={{fontSize:10,color:db.col,fontFamily:"monospace",fontWeight:700}}>{v+"/5"}</div>
                </div>
              );
            })}
          </div>
        </div>
        <div style={{marginTop:14,padding:"10px 14px",background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
          <div style={{display:"flex",gap:24,marginBottom:6,flexWrap:"wrap"}}>
            <div><span style={{fontSize:9,color:C.muted,fontFamily:"monospace"}}>INDEX </span><span style={{fontSize:10,color:db.col,fontFamily:"monospace"}}>{db.index}</span></div>
            <div><span style={{fontSize:9,color:C.muted,fontFamily:"monospace"}}>MAX SCALE </span><span style={{fontSize:10,color:db.col,fontFamily:"monospace"}}>{db.max}</span></div>
          </div>
          <div style={{fontSize:10,color:C.muted,lineHeight:1.6}}>{db.arch}</div>
        </div>
      </Card>

      <Card style={{maxWidth:750,margin:"0 auto 16px"}}>
        <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:12}}>Comparison</div>
        {attrLabels.map(function(lbl,ai){
          var key=attrs[ai];
          return (
            <div key={lbl} style={{marginBottom:12}}>
              <div style={{fontSize:9,color:C.muted,marginBottom:5,fontFamily:"monospace"}}>{lbl.toUpperCase()}</div>
              {dbs.map(function(d,i){
                var v=d[key];
                return (
                  <div key={i} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                    <div style={{width:68,fontSize:9,color:sel===i?d.col:C.dim,fontFamily:"monospace",textAlign:"right",fontWeight:sel===i?700:400}}>{d.name}</div>
                    <div style={{flex:1,height:14,background:C.dim+"20",borderRadius:3}}>
                      <div style={{width:(v/5*100)+"%",height:"100%",borderRadius:3,background:d.col+(sel===i?"60":"25"),border:"1px solid "+d.col+(sel===i?"80":"28"),transition:"all 0.3s"}} />
                    </div>
                    <div style={{width:20,fontSize:9,color:sel===i?d.col:C.dim,fontFamily:"monospace"}}>{v}</div>
                  </div>
                );
              })}
            </div>
          );
        })}
      </Card>

      <Insight icon={TARG} title="Decision Guide">
        Already on Postgres? {ARR} <span style={{color:C.green,fontWeight:700}}>pgvector</span>. Prototyping? {ARR} <span style={{color:C.yellow,fontWeight:700}}>Chroma</span>. Complex filters? {ARR} <span style={{color:C.accent,fontWeight:700}}>Qdrant</span>. Hybrid BM25+vector? {ARR} <span style={{color:C.purple,fontWeight:700}}>Weaviate</span>. Zero ops? {ARR} <span style={{color:C.blue,fontWeight:700}}>Pinecone</span>. Billion-scale? {ARR} <span style={{color:C.pink,fontWeight:700}}>Milvus</span>.
      </Insight>
    </div>
  );
}


/* ===============================================================
   ROOT APP
   =============================================================== */
function App() {
  var _t=useState(0); var tab=_t[0]; var setTab=_t[1];
  var tabs=["Embeddings","IVF Index","HNSW Index","Recall & Speed","Vector DBs"];
  return (
    <div style={{background:C.bg,minHeight:"100vh",padding:"24px 16px",fontFamily:"'JetBrains Mono','SF Mono',monospace",color:C.text,maxWidth:960,margin:"0 auto"}}>
      <div style={{textAlign:"center",marginBottom:16}}>
        <div style={{fontSize:22,fontWeight:800,background:"linear-gradient(135deg,"+C.blue+","+C.purple+")",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",display:"inline-block"}}>Vector Databases</div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>{"Interactive visual walkthrough "+DASH+" from embeddings to choosing the right DB"}</div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab===0&&<TabEmbeddings />}
      {tab===1&&<TabIVF />}
      {tab===2&&<TabHNSW />}
      {tab===3&&<TabRecallSpeed />}
      {tab===4&&<TabDatabases />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

</script>
</body>
</html>
"""

VECTORDB_VISUAL_HEIGHT = 1100