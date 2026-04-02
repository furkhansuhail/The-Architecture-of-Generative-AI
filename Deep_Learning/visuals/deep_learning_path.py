"""
Self-contained HTML for the Deep Learning Path interactive walkthrough.
Covers: Foundations, Core Neural Nets, Architectures, Training & Optimization,
Advanced Topics, and Specializations & Deployment.
Embed in Streamlit via st.components.v1.html(DL_PATH_HTML, height=DL_PATH_HEIGHT).
"""

DL_PATH_HTML = """
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
  @keyframes glow { 0%,100%{box-shadow:0 0 6px rgba(255,107,53,0.3)} 50%{box-shadow:0 0 18px rgba(255,107,53,0.7)} }
  @keyframes fadeSlide { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  .fade-in { animation: fadeSlide 0.3s ease forwards; }
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">

var useState = React.useState;
var useEffect = React.useEffect;
var useMemo = React.useMemo;

var C = {
  bg:"#0a0a0f", card:"#12121a", border:"#1e1e2e",
  accent:"#ff6b35", blue:"#4ecdc4", purple:"#a78bfa",
  yellow:"#fbbf24", text:"#e4e4e7", muted:"#71717a",
  dim:"#3f3f46", red:"#ef4444", green:"#4ade80",
  cyan:"#38bdf8", pink:"#f472b6", orange:"#fb923c",
};

var ARR = "\\u2192"; var DASH = "\\u2014"; var CHK = "\\u2713";
var BULB = "\\uD83D\\uDCA1"; var TARG = "\\uD83C\\uDFAF";
var WARN = "\\u26A0"; var STAR = "\\u2605"; var DOT = "\\u25CF";
var BOOK = "\\uD83D\\uDCDA";

function TabBar(props) {
  var tabs=props.tabs, active=props.active, onChange=props.onChange;
  return (
    <div style={{display:"flex",gap:0,borderBottom:"2px solid "+C.border,marginBottom:24,overflowX:"auto"}}>
      {tabs.map(function(t,i){return(
        <button key={i} onClick={function(){onChange(i);}} style={{
          padding:"12px 16px",background:"none",border:"none",
          borderBottom:active===i?"2px solid "+C.accent:"2px solid transparent",
          color:active===i?C.accent:C.muted,cursor:"pointer",
          fontSize:10,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",
          transition:"all 0.2s",whiteSpace:"nowrap",marginBottom:-2,
        }}>{t}</button>
      );})}
    </div>
  );
}

function Card(props) {
  return (
    <div style={Object.assign({
      background:C.card,borderRadius:10,padding:"18px 22px",
      border:"1px solid "+(props.highlight?C.accent:props.color||C.border),
      transition:"border 0.3s",
    },props.style||{})}>
      {props.children}
    </div>
  );
}

function Insight(props) {
  return (
    <div style={Object.assign({
      maxWidth:780,margin:"16px auto 0",
      padding:"14px 20px",background:"rgba(255,107,53,0.06)",
      borderRadius:10,border:"1px solid rgba(255,107,53,0.2)",
    },props.style||{})}>
      <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:5}}>{(props.icon||BULB)+" "+(props.title||"Key Insight")}</div>
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

function Badge(props) {
  return (
    <span style={{
      display:"inline-block",padding:"2px 8px",borderRadius:4,
      background:(props.color||C.accent)+"20",border:"1px solid "+(props.color||C.accent)+"50",
      color:props.color||C.accent,fontSize:9,fontWeight:700,fontFamily:"monospace",
      marginRight:4,marginTop:3,
    }}>{props.children}</span>
  );
}

function Pill(props) {
  var on=props.active;
  return (
    <button onClick={props.onClick} style={{
      padding:"6px 14px",borderRadius:20,border:"1.5px solid "+(on?props.color||C.accent:C.border),
      background:on?(props.color||C.accent)+"20":C.card,
      color:on?props.color||C.accent:C.muted,
      cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace",
      transition:"all 0.2s",
    }}>{props.children}</button>
  );
}

/* ================================================================
   TAB 1: LEARNING ROADMAP (visual flowchart)
   ================================================================ */
function TabRoadmap() {
  var _s=useState(null); var sel=_s[0]; var setSel=_s[1];

  var phases=[
    {id:0,label:"Phase 1",name:"Foundations",color:C.blue,icon:"\\uD83D\\uDCDA",x:60,y:60,
     duration:"4-6 weeks",
     topics:["Linear Algebra","Calculus & Gradients","Probability & Stats","Information Theory","Python & NumPy","Data Wrangling (Pandas)","Visualization (Matplotlib)","ML Fundamentals"],
     prereqs:[],
     why:"Every DL operation is a matrix multiply. Every training step is gradient descent. Without math fluency you\\'re flying blind."},
    {id:1,label:"Phase 2",name:"Core Neural Nets",color:C.accent,icon:"\\uD83E\\uDDE0",x:60,y:190,
     duration:"3-4 weeks",
     topics:["Perceptron & MLP","Forward Propagation","Backpropagation","Activation Functions","Loss Functions","Optimizers (SGD→Adam)","Regularization Basics","Hyperparameter Tuning"],
     prereqs:[0],
     why:"This is the engine of everything. Master backprop by hand once. Understand Adam. Know your loss functions. The rest is just architecture."},
    {id:2,label:"Phase 3",name:"Architectures",color:C.purple,icon:"\\uD83C\\uDFD7",x:60,y:320,
     duration:"6-8 weeks",
     topics:["CNNs & Feature Maps","ResNet / Skip Connections","RNNs & Vanishing Gradients","LSTMs & GRUs","Seq2Seq & Attention","Transformers & Self-Attention","GANs & Discriminators","Autoencoders / VAEs","Graph Neural Networks"],
     prereqs:[1],
     why:"Architectures are design patterns. Learn why each was invented, what problem it solved, and what its limitations are."},
    {id:3,label:"Phase 4",name:"Training Mastery",color:C.green,icon:"\\uD83D\\uDE80",x:60,y:450,
     duration:"3-4 weeks",
     topics:["Weight Initialization","Batch / Layer / Group Norm","Dropout & Weight Decay","LR Scheduling & Warmup","Gradient Clipping","Mixed Precision (FP16)","Data Augmentation","Early Stopping & Callbacks","Debugging Training Loops"],
     prereqs:[2],
     why:"90% of DL engineering is making training stable and fast. These techniques separate practitioners from researchers."},
    {id:4,label:"Phase 5",name:"Advanced Topics",color:C.yellow,icon:"\\u2728",x:60,y:580,
     duration:"6-10 weeks",
     topics:["Transfer Learning & Fine-Tuning","Self-Supervised Learning","Contrastive Learning (SimCLR)","Vision Transformers (ViT)","Large Language Models","RLHF & Alignment","Diffusion Models","Meta-Learning (MAML)","Neural Architecture Search"],
     prereqs:[3],
     why:"This is where you reach the frontier. Pick 1-2 areas and go deep rather than surveying all."},
    {id:5,label:"Phase 6",name:"Specialization & Deploy",color:C.pink,icon:"\\uD83C\\uDF10",x:60,y:710,
     duration:"Ongoing",
     topics:["Computer Vision (Detection, Segmentation)","NLP & LLMs (BERT, GPT, LLaMA)","Speech & Audio (Wav2Vec, Whisper)","Multimodal Models (CLIP, Flamingo)","PyTorch / JAX / TensorFlow","Experiment Tracking (W&B, MLflow)","Model Serving & Quantization","ONNX & Edge Deployment","MLOps & CI/CD for ML"],
     prereqs:[4],
     why:"Choose your domain. Build real projects. Deploy something real. Read papers in your niche weekly."},
  ];

  var selPhase=sel!==null?phases[sel]:null;

  return (
    <div>
      <SectionTitle title="Deep Learning Mastery Path" subtitle={"Six phases from zero to frontier "+DASH+" click any phase to explore"} />

      <div style={{display:"flex",gap:20,alignItems:"flex-start",flexWrap:"wrap"}}>

        {/* Left: SVG roadmap */}
        <div style={{flex:"0 0 auto"}}>
          <svg width={280} height={820} viewBox="0 0 280 820" style={{background:"#08080d",borderRadius:12,border:"1px solid "+C.border}}>
            {/* Connecting spine */}
            <line x1={140} y1={100} x2={140} y2={770} stroke={C.border} strokeWidth={2} strokeDasharray="6,4" />

            {phases.map(function(ph,i){
              var cy=ph.y+55; var isActive=sel===ph.id;
              return (
                <g key={ph.id} onClick={function(){setSel(sel===ph.id?null:ph.id);}} style={{cursor:"pointer"}}>
                  {/* Node circle */}
                  <circle cx={140} cy={cy} r={44}
                    fill={isActive?ph.color+"25":"#0d0d14"}
                    stroke={ph.color} strokeWidth={isActive?3:1.5}
                    style={{transition:"all 0.25s",filter:isActive?"drop-shadow(0 0 12px "+ph.color+"80)":"none"}}
                  />
                  <text x={140} y={cy-10} textAnchor="middle" dominantBaseline="middle"
                    fill={ph.color} fontSize={16} fontFamily="monospace">{ph.icon}</text>
                  <text x={140} y={cy+8} textAnchor="middle"
                    fill={isActive?ph.color:C.text} fontSize={9} fontWeight={800} fontFamily="monospace">{ph.label}</text>
                  <text x={140} y={cy+22} textAnchor="middle"
                    fill={isActive?ph.color:C.muted} fontSize={8} fontFamily="monospace">{ph.name}</text>
                  {/* Duration badge */}
                  <rect x={87} y={cy+32} width={106} height={16} rx={8}
                    fill={ph.color+"18"} stroke={ph.color+"40"} />
                  <text x={140} y={cy+44} textAnchor="middle" dominantBaseline="middle"
                    fill={ph.color} fontSize={7.5} fontWeight={700} fontFamily="monospace">{ph.duration}</text>
                  {/* Arrow to next */}
                  {i<5&&<polygon points={"140,"+(cy+55)+" 135,"+(cy+65)+" 145,"+(cy+65)} fill={C.dim} opacity={0.5} />}
                </g>
              );
            })}

            {/* Title */}
            <text x={140} y={800} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">click a phase to expand</text>
          </svg>
        </div>

        {/* Right: Detail panel */}
        <div style={{flex:1,minWidth:280}}>
          {selPhase?(
            <div className="fade-in">
              <Card highlight={true} style={{borderColor:selPhase.color,marginBottom:16}}>
                <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
                  <div style={{fontSize:28}}>{selPhase.icon}</div>
                  <div>
                    <div style={{fontSize:16,fontWeight:800,color:selPhase.color}}>{selPhase.name}</div>
                    <div style={{fontSize:10,color:C.muted}}>{selPhase.label+" "+DASH+" "+selPhase.duration}</div>
                  </div>
                </div>
                <div style={{fontSize:11,color:C.muted,lineHeight:1.8,marginBottom:14,fontStyle:"italic",
                  borderLeft:"3px solid "+selPhase.color+"50",paddingLeft:12}}>
                  {selPhase.why}
                </div>
                <div style={{fontSize:10,fontWeight:700,color:C.text,marginBottom:8}}>TOPICS COVERED</div>
                <div style={{display:"flex",flexWrap:"wrap",gap:4}}>
                  {selPhase.topics.map(function(t,i){return <Badge key={i} color={selPhase.color}>{t}</Badge>;})}
                </div>
              </Card>

              {/* Path prerequisites */}
              {selPhase.prereqs.length>0&&(
                <Card style={{marginBottom:12}}>
                  <div style={{fontSize:10,fontWeight:700,color:C.muted,marginBottom:8}}>PREREQUISITES</div>
                  {selPhase.prereqs.map(function(pid){
                    var ph=phases[pid];
                    return <div key={pid} style={{fontSize:11,color:ph.color,fontFamily:"monospace"}}>
                      {CHK+" "+ph.name}
                    </div>;
                  })}
                </Card>
              )}
            </div>
          ):(
            <div>
              <Card style={{marginBottom:12}}>
                <div style={{fontSize:13,fontWeight:800,color:C.text,marginBottom:8}}>How to use this path</div>
                <div style={{fontSize:11,color:C.muted,lineHeight:1.9}}>
                  Click any phase on the left to see its topics, rationale, and prerequisites. The path is sequential by design {DASH} each phase builds on the last. However, you can return to earlier phases at any time to deepen understanding.
                </div>
              </Card>
              <Card style={{marginBottom:12}}>
                <div style={{fontSize:12,fontWeight:800,color:C.accent,marginBottom:10}}>The Golden Rules</div>
                {[
                  {c:C.blue,t:"Implement from scratch once",d:"Write a backprop loop, an LSTM, an attention head. Understanding beats memorization."},
                  {c:C.purple,t:"Read papers, not just blogs",d:"Papers have the precise details. Start with the classics: AlexNet, Attention Is All You Need, ResNet."},
                  {c:C.green,t:"Build a project per phase",d:"Nothing cements learning like a real project. Kaggle, personal datasets, or replicate a paper."},
                  {c:C.yellow,t:"Specialize after Phase 4",d:"DL is too broad to master all of it. Pick NLP, CV, RL, or Audio and go deep in one area."},
                  {c:C.pink,t:"Track experiments",d:"Use Weights & Biases or MLflow from day one. Reproducibility is a professional habit."},
                ].map(function(r,i){return(
                  <div key={i} style={{display:"flex",gap:10,marginBottom:10,alignItems:"flex-start"}}>
                    <div style={{width:6,height:6,borderRadius:"50%",background:r.c,marginTop:5,flexShrink:0}} />
                    <div>
                      <div style={{fontSize:11,fontWeight:700,color:r.c}}>{r.t}</div>
                      <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{r.d}</div>
                    </div>
                  </div>
                );})}
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ================================================================
   TAB 2: FOUNDATIONS
   ================================================================ */
function TabFoundations() {
  var _s=useState(0); var sel=_s[0]; var setSel=_s[1];

  var topics=[
    {name:"Linear Algebra",color:C.blue,icon:"\\uD83D\\uDCCF",
     subtopics:[
       {t:"Scalars, Vectors, Matrices, Tensors",d:"Tensors are the lingua franca of DL. A grayscale image is a 2D tensor, RGB is 3D, a batch is 4D."},
       {t:"Matrix Multiplication",d:"The core operation. A fully-connected layer is y = Wx + b. Shape: (out, in) @ (in, batch) = (out, batch)."},
       {t:"Dot Product & Norms",d:"Dot product measures similarity. L1/L2 norms measure magnitude. Cosine similarity = dot / (|a||b|)."},
       {t:"Eigenvalues & PCA",d:"PCA uses eigenvectors of the covariance matrix. Useful for dimensionality reduction and understanding transformations."},
       {t:"Singular Value Decomposition",d:"M = U\u03A3V\u1D40. Underpins low-rank approximations, LoRA fine-tuning, and data compression."},
       {t:"Broadcasting Rules",d:"NumPy/PyTorch auto-expand dims. (1,3) + (4,3) = (4,3). Crucial for batch computations without loops."},
     ],
     resources:["3Blue1Brown: Essence of Linear Algebra","Gilbert Strang: Linear Algebra (MIT OCW)","fast.ai Numerical Linear Algebra"]},
    {name:"Calculus & Gradients",color:C.accent,icon:"\\u222B",
     subtopics:[
       {t:"Derivatives & Partial Derivatives",d:"df/dx measures rate of change. In DL, \u2202L/\u2202w tells us how the loss changes with each weight."},
       {t:"Chain Rule",d:"The backbone of backpropagation. d(f\u2218g)/dx = f\\'(g(x)) \u00B7 g\\'(x). Composed layer after layer."},
       {t:"Gradient & Jacobian",d:"Gradient: vector of all partials (points uphill). Jacobian: matrix of partials for vector functions."},
       {t:"Hessian & Curvature",d:"Matrix of second derivatives. High curvature = sharp loss landscape = learning rate sensitivity."},
       {t:"Automatic Differentiation",d:"PyTorch\\'s autograd builds a computation graph and applies chain rule automatically. You never derive by hand."},
       {t:"Taylor Series & Approximations",d:"Loss landscape can be locally approximated. Second-order optimizers use the Hessian for better step sizes."},
     ],
     resources:["3Blue1Brown: Calculus series","Karpathy: micrograd (build autograd from scratch)","The Matrix Calculus You Need for DL (Parr & Howard)"]},
    {name:"Probability & Stats",color:C.purple,icon:"\\uD83C\\uDFB2",
     subtopics:[
       {t:"Probability Distributions",d:"Gaussian (weight init, VAEs), Bernoulli (binary classification), Categorical (softmax output), Dirichlet (topic models)."},
       {t:"Bayes\\' Theorem",d:"P(A|B) = P(B|A)P(A)/P(B). Foundation for Bayesian NNs, uncertainty estimation, and probabilistic ML."},
       {t:"MLE & MAP Estimation",d:"MLE: find params maximizing likelihood. MAP: MLE + prior = L2 regularization is MAP with Gaussian prior."},
       {t:"Expectations & Variance",d:"Expected gradient = mean gradient. Variance tells you how noisy your estimates are (batch size effect)."},
       {t:"Sampling Methods",d:"Monte Carlo estimation. MCMC for posterior sampling. Used heavily in RL and generative models."},
       {t:"Central Limit Theorem",d:"Sum of many independent vars \u2192 Gaussian. Explains why weight init and batch stats matter."},
     ],
     resources:["Seeing Theory (visual prob/stats)","Bishop: Pattern Recognition and ML","StatQuest with Josh Starmer (YouTube)"]},
    {name:"Information Theory",color:C.green,icon:"\\uD83D\\uDCE1",
     subtopics:[
       {t:"Entropy H(X)",d:"H(X) = -\u03A3 p log p. Measures uncertainty/information. Max entropy = uniform distribution."},
       {t:"Cross-Entropy Loss",d:"H(p,q) = -\u03A3 p log q. The most common classification loss. Minimizing CE = maximizing log likelihood."},
       {t:"KL Divergence",d:"D_KL(p||q) = \u03A3 p log(p/q). Measures distribution distance. Used in VAEs and RL (PPO)."},
       {t:"Mutual Information",d:"How much knowing X reduces uncertainty about Y. Used in self-supervised learning and feature selection."},
       {t:"Bits & Nats",d:"Entropy in base-2 = bits. In base-e = nats. Perplexity = 2^H for language models."},
       {t:"Channel Capacity",d:"Maximum mutual information between input and output. Foundation for understanding model compression."},
     ],
     resources:["Elements of Information Theory (Cover & Thomas)","Colah\\'s Blog: Visual Information Theory","David MacKay: Information Theory (free online)"]},
    {name:"Python & NumPy",color:C.yellow,icon:"\\uD83D\\uDC0D",
     subtopics:[
       {t:"Python Fundamentals",d:"Functions, classes, decorators, generators, context managers. Clean Python is clean DL code."},
       {t:"NumPy Arrays & Operations",d:"Vectorized ops instead of loops. np.dot, np.einsum, broadcasting. 100x faster than pure Python."},
       {t:"Pandas for Data",d:"DataFrames, groupby, merge, apply. Essential for EDA and preprocessing tabular/text datasets."},
       {t:"Matplotlib & Seaborn",d:"Visualize loss curves, activations, confusion matrices, attention maps. Always plot before you train."},
       {t:"Jupyter & Colab Workflow",d:"Interactive development. Colab provides free GPU. Learn magic commands, widgets, and cell organization."},
       {t:"Profiling & Debugging",d:"cProfile, line_profiler, torch.profiler. Know your bottlenecks before optimizing."},
     ],
     resources:["Python for Data Analysis (Wes McKinney)","NumPy official docs + exercises","CS231n Python/NumPy tutorial"]},
    {name:"ML Fundamentals",color:C.cyan,icon:"\\uD83E\\uDD16",
     subtopics:[
       {t:"Supervised vs Unsupervised",d:"Supervised: labeled pairs (x,y). Unsupervised: find structure in x alone. Semi-supervised: mix of both."},
       {t:"Bias-Variance Tradeoff",d:"High bias = underfitting (too simple). High variance = overfitting (too complex). Test error = bias\u00B2 + variance + noise."},
       {t:"Train / Val / Test Split",d:"Train: optimize. Val: select hyperparams. Test: one final honest evaluation. Never tune on test set."},
       {t:"Cross-Validation (k-fold)",d:"Rotate val set through k folds for robust evaluation. Essential for small datasets."},
       {t:"Evaluation Metrics",d:"Accuracy, Precision, Recall, F1, AUC-ROC. Task-dependent: MSE/MAE for regression, perplexity for LMs."},
       {t:"Feature Engineering",d:"Normalization, one-hot encoding, embedding lookup. In DL, the model learns features \u2014 but preprocessing still matters."},
     ],
     resources:["Hands-On ML (Geron)","fast.ai Practical Deep Learning","CS229 Andrew Ng lecture notes"]},
  ];

  var t=topics[sel];
  return (
    <div>
      <SectionTitle title="Phase 1: Foundations" subtitle="The mathematical and programming bedrock every DL practitioner must master" />

      <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:20}}>
        {topics.map(function(tp,i){return(
          <Pill key={i} active={sel===i} color={tp.color} onClick={function(){setSel(i);}}>
            {tp.icon+" "+tp.name}
          </Pill>
        );})}
      </div>

      <div className="fade-in" key={sel}>
        <Card highlight={true} style={{maxWidth:780,margin:"0 auto 16px",borderColor:t.color}}>
          <div style={{fontSize:15,fontWeight:800,color:t.color,marginBottom:2}}>{t.icon+" "+t.name}</div>
          <div style={{fontSize:10,color:C.muted,marginBottom:16}}>6 essential concepts {DASH} expand each to understand</div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit, minmax(340px, 1fr))",gap:10}}>
            {t.subtopics.map(function(s,i){
              return (
                <div key={i} style={{
                  padding:"12px 14px",background:"#08080d",borderRadius:8,
                  border:"1px solid "+t.color+"30",
                }}>
                  <div style={{display:"flex",gap:8,alignItems:"flex-start"}}>
                    <span style={{color:t.color,fontSize:10,fontWeight:800,fontFamily:"monospace",marginTop:1,flexShrink:0}}>
                      {"0"+(i+1)}
                    </span>
                    <div>
                      <div style={{fontSize:11,fontWeight:700,color:t.color,marginBottom:4}}>{s.t}</div>
                      <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{s.d}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card style={{maxWidth:780,margin:"0 auto 16px"}}>
          <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>{BOOK+" Recommended Resources"}</div>
          {t.resources.map(function(r,i){return(
            <div key={i} style={{display:"flex",gap:8,alignItems:"center",marginBottom:6}}>
              <span style={{color:t.color,fontSize:9}}>{"\\u25B6"}</span>
              <span style={{fontSize:10,color:C.muted}}>{r}</span>
            </div>
          );})}
        </Card>
      </div>

      <Insight title="Foundation Mindset">
        Don\\'t rush through this phase. A researcher who can derive backprop has a <span style={{color:C.green,fontWeight:700}}>permanent edge</span> over one who can\\'t. Spend real time with linear algebra and calculus {DASH} everything in DL is built on these two pillars. The rest is engineering.
      </Insight>
    </div>
  );
}

/* ================================================================
   TAB 3: CORE NEURAL NETWORKS
   ================================================================ */
function TabCoreNets() {
  var _st=useState(0); var subtab=_st[0]; var setSubtab=_st[1];
  var _xv=useState(0); var xval=_xv[0]; var setXval=_xv[1];

  var stabs=["Activation Fns","Loss Functions","Optimizers","Backprop","Architecture"];

  /* Activation function data */
  var acts=[
    {name:"Sigmoid",color:C.blue,
     fn:function(x){return 1/(1+Math.exp(-x));},
     deriv:function(x){var s=1/(1+Math.exp(-x));return s*(1-s);},
     pros:"Squashes to (0,1). Good for binary output probabilities.",
     cons:"Vanishing gradient for |x|>3. Saturated neurons learn nothing. Not zero-centered.",
     use:"Output layer for binary classification."},
    {name:"Tanh",color:C.cyan,
     fn:function(x){return Math.tanh(x);},
     deriv:function(x){return 1-Math.pow(Math.tanh(x),2);},
     pros:"Zero-centered. Stronger gradients than sigmoid. Range (-1, 1).",
     cons:"Still saturates. Vanishing gradients for deep nets.",
     use:"RNNs and LSTMs (gates). Hidden layers when zero-centering matters."},
    {name:"ReLU",color:C.accent,
     fn:function(x){return Math.max(0,x);},
     deriv:function(x){return x>0?1:0;},
     pros:"No vanishing gradient for x>0. Sparse activation. Very fast to compute.",
     cons:"Dead neurons (x<0 permanently off). Not zero-centered.",
     use:"Default for hidden layers in CNNs and MLPs. Start here."},
    {name:"Leaky ReLU",color:C.orange,
     fn:function(x){return x>0?x:0.01*x;},
     deriv:function(x){return x>0?1:0.01;},
     pros:"Fixes dying ReLU: small gradient for x<0. Never truly dead.",
     cons:"Extra hyperparameter (slope). Slightly more compute.",
     use:"When standard ReLU causes dead neuron problems."},
    {name:"GELU",color:C.purple,
     fn:function(x){return x*0.5*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3))));},
     deriv:function(x){var t=Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3)));var dt=(1-t*t)*Math.sqrt(2/Math.PI)*(1+3*0.044715*x*x);return 0.5*(1+t)+0.5*x*dt;},
     pros:"Smooth, probabilistic gating. SOTA in transformers. Better than ReLU in practice.",
     cons:"More expensive to compute. Harder to interpret.",
     use:"Transformers (BERT, GPT), modern MLPs. Preferred in NLP."},
    {name:"Swish",color:C.pink,
     fn:function(x){return x*(1/(1+Math.exp(-x)));},
     deriv:function(x){var s=1/(1+Math.exp(-x));return s+x*s*(1-s);},
     pros:"Smooth, non-monotonic. Slightly outperforms ReLU on deep nets.",
     cons:"No exact advantage over GELU. More compute than ReLU.",
     use:"EfficientNet and other efficient architectures."},
  ];

  var _sa=useState(0); var selAct=_sa[0]; var setSelAct=_sa[1];
  var act=acts[selAct];

  /* Plot helper */
  var W=340, H=160, pw=W-40, ph=H-30;
  function toSVGX(x){return 20+((x+5)/10)*pw;}
  function toSVGY(y,ymin,ymax){return H-15-((y-ymin)/(ymax-ymin))*ph;}

  var xs=[];
  for(var i=-50;i<=50;i++) xs.push(i/10);

  function makePolyline(fn,ymin,ymax){
    return xs.map(function(x){return toSVGX(x)+","+toSVGY(fn(x),ymin,ymax);}).join(" ");
  }

  /* Loss functions data */
  var losses=[
    {name:"MSE / L2 Loss",color:C.blue,
     formula:"L = (1/N) \u03A3 (y\u0302 - y)\u00B2",
     use:"Regression. When errors should be penalized quadratically.",
     props:["Differentiable everywhere","Sensitive to outliers (squares errors)","Gradient: 2(y\u0302-y)","Output: any real value"]},
    {name:"MAE / L1 Loss",color:C.cyan,
     formula:"L = (1/N) \u03A3 |y\u0302 - y|",
     use:"Robust regression. When outliers are common in your data.",
     props:["Robust to outliers","Not differentiable at 0 (use Huber instead)","Gradient: sign(y\u0302-y)","Leads to sparse solutions"]},
    {name:"Huber Loss",color:C.orange,
     formula:"L = { 0.5(y\u0302-y)\u00B2  if |e|<\u03B4 ; \u03B4|e| - 0.5\u03B4\u00B2 otherwise }",
     use:"Best of MSE+MAE. Object detection (bounding box regression).",
     props:["Smooth everywhere (differentiable)","MSE for small errors, MAE for large","\u03B4 is a tunable hyperparameter","Used in Faster R-CNN, Hugging Face"]},
    {name:"Binary Cross-Entropy",color:C.accent,
     formula:"L = -[y log(p) + (1-y) log(1-p)]",
     use:"Binary classification. Single sigmoid output.",
     props:["Pairs with sigmoid activation","Gradient: p - y (clean!)","Explodes for p\u21920 or p\u21921","Use logits version for numerical stability"]},
    {name:"Categorical Cross-Entropy",color:C.purple,
     formula:"L = -\u03A3 y_c log(p_c)",
     use:"Multi-class classification. Softmax output layer.",
     props:["Pairs with softmax activation","Gradient: p - y (one-hot y)","Most common classification loss","Label smoothing: y = 0.9 instead of 1.0"]},
    {name:"Contrastive / NT-Xent",color:C.green,
     formula:"L = -log[ sim(z,z+) / \u03A3 sim(z,z-) ]",
     use:"Self-supervised learning. SimCLR, CLIP, sentence embeddings.",
     props:["Pulls positives together","Pushes negatives apart","Temperature \u03C4 controls softness","Needs large batch or memory bank"]},
    {name:"KL Divergence",color:C.yellow,
     formula:"L = \u03A3 p(x) log[ p(x) / q(x) ]",
     use:"VAEs (regularize latent space), knowledge distillation.",
     props:["Measures distribution distance","Asymmetric: KL(p||q) \u2260 KL(q||p)","VAE: KL(q(z|x)||N(0,I))","Not a true metric (no triangle inequality)"]},
    {name:"Focal Loss",color:C.pink,
     formula:"L = -\u03B1(1-p)\u03B3 log(p)",
     use:"Class imbalance. Object detection (RetinaNet). Medical imaging.",
     props:["\u03B3 downweights easy examples","Hard examples get full loss weight","\u03B3=0 \u2192 standard cross-entropy","Standard choice for detection tasks"]},
  ];

  var _sl=useState(0); var selLoss=_sl[0]; var setSelLoss=_sl[1];
  var loss=losses[selLoss];

  /* Optimizers data */
  var opts=[
    {name:"SGD",color:C.muted,
     update:"w \u2190 w - \u03B7 \u2207L",
     params:["lr: 0.01-0.1","momentum: 0 (vanilla)"],
     pros:"Simple. Well understood. Generalizes well with momentum.",
     cons:"Noisy. Slow convergence. LR hard to tune.",
     when:"Training from scratch with careful LR tuning. CNNs often prefer SGD+momentum."},
    {name:"SGD + Momentum",color:C.blue,
     update:"v \u2190 \u03B2v - \u03B7\u2207L ;  w \u2190 w + v",
     params:["lr: 0.01","momentum \u03B2: 0.9"],
     pros:"Damps oscillations. Accelerates in consistent directions.",
     cons:"Overshoots sharp minima. Still needs good LR.",
     when:"Most CNN training. ResNets, VGGs often trained with SGD+0.9 momentum."},
    {name:"AdaGrad",color:C.cyan,
     update:"G \u2190 G + \u2207L\u00B2 ;  w \u2190 w - (\u03B7/\u221AG) \u2207L",
     params:["lr: 0.01","eps: 1e-8"],
     pros:"Adapts LR per parameter. Great for sparse gradients.",
     cons:"LR monotonically decreases to 0. Dies in long training.",
     when:"NLP with sparse features. Embedding layers. Short training runs."},
    {name:"RMSProp",color:C.orange,
     update:"E[g\u00B2] \u2190 \u03B2E[g\u00B2] + (1-\u03B2)\u2207L\u00B2 ;  w \u2190 w - (\u03B7/\u221AE[g\u00B2])\u2207L",
     params:["lr: 0.001","decay \u03B2: 0.9","eps: 1e-8"],
     pros:"Fixes AdaGrad dying LR. Good for RNNs.",
     cons:"Not as stable as Adam. Less popular now.",
     when:"RNNs. When Adam is unstable. Historically Hinton\\'s default."},
    {name:"Adam",color:C.accent,
     update:"m \u2190 \u03B2\u2081m + (1-\u03B2\u2081)\u2207L ;  v \u2190 \u03B2\u2082v + (1-\u03B2\u2082)\u2207L\u00B2 ;  w \u2190 w - \u03B7m\u0302/(\u221Av\u0302+\u03B5)",
     params:["lr: 3e-4 (Karpathy const.)","\\u03B2\u2081: 0.9","\\u03B2\u2082: 0.999","\\u03B5: 1e-8"],
     pros:"Adaptive per-param LR. Bias correction. Robust across tasks.",
     cons:"May not generalize as well as SGD. Weight decay is wrong (use AdamW).",
     when:"Default for most DL. Transformers, GANs, NLP, RL. Start here."},
    {name:"AdamW",color:C.green,
     update:"Same as Adam but weight decay applied directly: w \u2190 w(1-\u03BB) - \u03B7m\u0302/(\u221Av\u0302+\u03B5)",
     params:["lr: 1e-4 to 3e-4","weight_decay \u03BB: 0.01-0.1"],
     pros:"Decoupled weight decay = correct L2 regularization. Better generalization.",
     cons:"Slightly more complex. Small perf difference from Adam.",
     when:"Modern transformers (BERT, GPT, ViT). Preferred over Adam for most tasks."},
    {name:"Lion",color:C.purple,
     update:"u \u2190 \u03B2\u2081m + (1-\u03B2\u2081)\u2207L ;  w \u2190 w(1-\u03BB) - \u03B7 sign(u) ;  m \u2190 \u03B2\u2082m + (1-\u03B2\u2082)\u2207L",
     params:["lr: 1e-5 to 1e-4","\\u03B2\u2081: 0.9","\\u03B2\u2082: 0.99"],
     pros:"Uses sign update = memory efficient. Competitive with AdamW.",
     cons:"Requires smaller LR. Less battle-tested.",
     when:"Large models where memory matters. Google\\'s preferred optimizer internally."},
    {name:"Muon / Shampoo",color:C.yellow,
     update:"Approximates natural gradient: w \u2190 w - \u03B7 F\u207B\u00B9 \u2207L",
     params:["Various second-order","approximation params"],
     pros:"Follows loss landscape curvature. Faster convergence.",
     cons:"High compute/memory. Complex implementation.",
     when:"Large-scale training with compute budget. Active research area."},
  ];

  var _so=useState(4); var selOpt=_so[0]; var setSelOpt=_so[1];
  var opt=opts[selOpt];

  return (
    <div>
      <SectionTitle title="Phase 2: Core Neural Networks" subtitle="The building blocks every DL practitioner must understand deeply" />

      <TabBar tabs={stabs} active={subtab} onChange={setSubtab} />

      {subtab===0&&(
        <div className="fade-in">
          <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:16}}>
            {acts.map(function(a,i){return(
              <Pill key={i} active={selAct===i} color={a.color} onClick={function(){setSelAct(i);}}>
                {a.name}
              </Pill>
            );})}
          </div>

          <div style={{display:"flex",gap:16,flexWrap:"wrap",justifyContent:"center",marginBottom:16}}>
            {/* Function plot */}
            <div style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,padding:12}}>
              <div style={{fontSize:10,fontWeight:700,color:act.color,marginBottom:8,fontFamily:"monospace"}}>
                {act.name+" "+DASH+" f(x)"}
              </div>
              <svg width={W} height={H} viewBox={"0 0 "+W+" "+H}>
                {/* Axes */}
                <line x1={20} y1={H-15} x2={W-10} y2={H-15} stroke={C.dim} strokeWidth={1}/>
                <line x1={toSVGX(0)} y1={10} x2={toSVGX(0)} y2={H-15} stroke={C.dim} strokeWidth={1}/>
                {[-4,-2,0,2,4].map(function(v){return(
                  <text key={v} x={toSVGX(v)} y={H-3} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{v}</text>
                );})}
                <polyline points={makePolyline(act.fn,-1.1,1.5)} fill="none" stroke={act.color} strokeWidth={2.5} />
                {/* Selected x marker */}
                {xval!=null&&<>
                  <line x1={toSVGX(xval)} y1={10} x2={toSVGX(xval)} y2={H-15} stroke={act.color} strokeWidth={1} strokeDasharray="3,2" opacity={0.5}/>
                  <circle cx={toSVGX(xval)} cy={toSVGY(act.fn(xval),-1.1,1.5)} r={5} fill={act.color} />
                </>}
              </svg>
              <input type="range" min={-50} max={50} value={Math.round(xval*10)} onChange={function(e){setXval(parseInt(e.target.value)/10);}}
                style={{width:"100%",marginTop:6}} />
              <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.muted,marginTop:4,fontFamily:"monospace"}}>
                <span>{"x = "+xval.toFixed(1)}</span>
                <span>{"f(x) = "+act.fn(xval).toFixed(4)}</span>
                <span>{"f\\'(x) = "+act.deriv(xval).toFixed(4)}</span>
              </div>
            </div>

            {/* Derivative plot */}
            <div style={{background:"#08080d",borderRadius:10,border:"1px solid "+C.border,padding:12}}>
              <div style={{fontSize:10,fontWeight:700,color:act.color,marginBottom:8,fontFamily:"monospace"}}>
                {act.name+" "+DASH+" f\\'(x) gradient"}
              </div>
              <svg width={W} height={H} viewBox={"0 0 "+W+" "+H}>
                <line x1={20} y1={H-15} x2={W-10} y2={H-15} stroke={C.dim} strokeWidth={1}/>
                <line x1={toSVGX(0)} y1={10} x2={toSVGX(0)} y2={H-15} stroke={C.dim} strokeWidth={1}/>
                {[-4,-2,0,2,4].map(function(v){return(
                  <text key={v} x={toSVGX(v)} y={H-3} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{v}</text>
                );})}
                <polyline points={makePolyline(act.deriv,0,1.2)} fill="none" stroke={act.color} strokeWidth={2.5} strokeDasharray="6,3" />
                {xval!=null&&<circle cx={toSVGX(xval)} cy={toSVGY(act.deriv(xval),0,1.2)} r={5} fill={act.color} />}
              </svg>
              <div style={{marginTop:6,fontSize:9,color:C.muted,fontFamily:"monospace",textAlign:"center"}}>
                {"Gradient magnitude at x="+xval.toFixed(1)+": "+(Math.abs(act.deriv(xval))>0.001?"healthy":"vanishing!")}
              </div>
            </div>
          </div>

          <Card highlight={true} style={{maxWidth:780,margin:"0 auto 12px",borderColor:act.color}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16,flexWrap:"wrap"}}>
              <div><div style={{fontSize:9,color:C.muted,marginBottom:4}}>PROS</div><div style={{fontSize:11,color:C.green,lineHeight:1.7}}>{act.pros}</div></div>
              <div><div style={{fontSize:9,color:C.muted,marginBottom:4}}>CONS</div><div style={{fontSize:11,color:C.red,lineHeight:1.7}}>{act.cons}</div></div>
              <div><div style={{fontSize:9,color:C.muted,marginBottom:4}}>BEST USED FOR</div><div style={{fontSize:11,color:act.color,lineHeight:1.7}}>{act.use}</div></div>
            </div>
          </Card>

          <Insight title="Activation Rule of Thumb">
            <span style={{color:C.accent,fontWeight:700}}>ReLU</span> for CNNs and general MLPs.{" "}
            <span style={{color:C.purple,fontWeight:700}}>GELU</span> for transformers.{" "}
            <span style={{color:C.cyan,fontWeight:700}}>Tanh</span> for LSTM gates.{" "}
            <span style={{color:C.blue,fontWeight:700}}>Sigmoid</span> for binary output only. Avoid sigmoid in hidden layers {DASH} vanishing gradients guaranteed.
          </Insight>
        </div>
      )}

      {subtab===1&&(
        <div className="fade-in">
          <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:16}}>
            {losses.map(function(l,i){return(
              <Pill key={i} active={selLoss===i} color={l.color} onClick={function(){setSelLoss(i);}}>
                {l.name.split(" ")[0]}
              </Pill>
            );})}
          </div>
          <Card highlight={true} style={{maxWidth:780,margin:"0 auto 12px",borderColor:loss.color}}>
            <div style={{fontSize:14,fontWeight:800,color:loss.color,marginBottom:4}}>{loss.name}</div>
            <div style={{fontSize:13,fontFamily:"monospace",color:C.text,background:"#08080d",
              padding:"10px 14px",borderRadius:6,marginBottom:14,border:"1px solid "+loss.color+"30"}}>
              {loss.formula}
            </div>
            <div style={{fontSize:10,color:C.muted,marginBottom:12}}>
              <span style={{color:C.text,fontWeight:700}}>When to use: </span>{loss.use}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:6}}>
              {loss.props.map(function(p,i){return(
                <div key={i} style={{display:"flex",gap:6,alignItems:"flex-start",padding:"8px 10px",
                  background:"#08080d",borderRadius:6,border:"1px solid "+C.border}}>
                  <span style={{color:loss.color,fontSize:9,marginTop:1}}>{"\\u25CF"}</span>
                  <span style={{fontSize:10,color:C.muted,lineHeight:1.6}}>{p}</span>
                </div>
              );})}
            </div>
          </Card>
          <Insight title="Loss Selection Guide">
            <span style={{color:C.blue,fontWeight:700}}>Regression</span>: MSE default, Huber if outliers, MAE for robustness.{" "}
            <span style={{color:C.purple,fontWeight:700}}>Multi-class</span>: Categorical CE always.{" "}
            <span style={{color:C.accent,fontWeight:700}}>Binary</span>: BCE.{" "}
            <span style={{color:C.pink,fontWeight:700}}>Imbalanced</span>: Focal loss.{" "}
            <span style={{color:C.green,fontWeight:700}}>Self-supervised</span>: Contrastive / NT-Xent.{" "}
            <span style={{color:C.yellow,fontWeight:700}}>Generative</span>: KL divergence + reconstruction.
          </Insight>
        </div>
      )}

      {subtab===2&&(
        <div className="fade-in">
          <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:16}}>
            {opts.map(function(o,i){return(
              <Pill key={i} active={selOpt===i} color={o.color} onClick={function(){setSelOpt(i);}}>
                {o.name}
              </Pill>
            );})}
          </div>
          <Card highlight={true} style={{maxWidth:780,margin:"0 auto 12px",borderColor:opt.color}}>
            <div style={{fontSize:14,fontWeight:800,color:opt.color,marginBottom:10}}>{opt.name}</div>
            <div style={{background:"#08080d",borderRadius:6,padding:"10px 14px",marginBottom:12,border:"1px solid "+opt.color+"30"}}>
              <div style={{fontSize:9,color:C.muted,marginBottom:4}}>UPDATE RULE</div>
              <div style={{fontSize:10,fontFamily:"monospace",color:opt.color,lineHeight:2.0}}>{opt.update}</div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16,marginBottom:12}}>
              <div>
                <div style={{fontSize:9,color:C.muted,marginBottom:4}}>DEFAULT HYPERPARAMETERS</div>
                {opt.params.map(function(p,i){return<div key={i} style={{fontSize:10,color:C.cyan,fontFamily:"monospace",marginBottom:3}}>{p}</div>;})}
              </div>
              <div>
                <div style={{fontSize:9,color:C.muted,marginBottom:4}}>BEST USED WHEN</div>
                <div style={{fontSize:10,color:opt.color,lineHeight:1.7}}>{opt.when}</div>
              </div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div style={{padding:"8px 12px",background:"#08080d",borderRadius:6,border:"1px solid "+C.green+"30"}}>
                <div style={{fontSize:9,color:C.green,marginBottom:4,fontWeight:700}}>PROS</div>
                <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{opt.pros}</div>
              </div>
              <div style={{padding:"8px 12px",background:"#08080d",borderRadius:6,border:"1px solid "+C.red+"30"}}>
                <div style={{fontSize:9,color:C.red,marginBottom:4,fontWeight:700}}>CONS</div>
                <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{opt.cons}</div>
              </div>
            </div>
          </Card>

          {/* Convergence comparison visual */}
          <Card style={{maxWidth:780,margin:"0 auto 12px"}}>
            <div style={{fontSize:11,fontWeight:700,color:C.text,marginBottom:10}}>Optimizer Family Tree</div>
            <svg width={"100%"} height={120} viewBox="0 0 760 120" style={{overflow:"visible"}}>
              {[
                {x:60,label:"SGD",color:C.muted,year:"1951"},
                {x:200,label:"Momentum",color:C.blue,year:"1964"},
                {x:340,label:"AdaGrad",color:C.cyan,year:"2011"},
                {x:470,label:"RMSProp",color:C.orange,year:"2012"},
                {x:580,label:"Adam",color:C.accent,year:"2014"},
                {x:680,label:"AdamW",color:C.green,year:"2019"},
              ].map(function(n,i,arr){
                var isActive=selOpt===i;
                return (
                  <g key={i}>
                    {i>0&&<line x1={arr[i-1].x} y1={50} x2={n.x} y2={50} stroke={C.dim} strokeWidth={1.5} />}
                    <circle cx={n.x} cy={50} r={isActive?18:12} fill={isActive?n.color+"30":"#0d0d14"} stroke={n.color} strokeWidth={isActive?2.5:1.5} />
                    <text x={n.x} y={54} textAnchor="middle" fill={n.color} fontSize={8} fontWeight={700} fontFamily="monospace">{n.label}</text>
                    <text x={n.x} y={78} textAnchor="middle" fill={C.dim} fontSize={8} fontFamily="monospace">{n.year}</text>
                  </g>
                );
              })}
            </svg>
          </Card>
          <Insight icon={TARG} title="Optimizer Rule">
            Default to <span style={{color:C.green,fontWeight:700}}>AdamW</span> for transformers and modern architectures. Use <span style={{color:C.blue,fontWeight:700}}>SGD + Momentum</span> for CNNs if you want better generalization. The <span style={{color:C.accent,fontWeight:700}}>learning rate</span> matters more than optimizer choice. Karpathy\\'s constant: <span style={{color:C.yellow,fontWeight:700}}>3e-4</span> is a safe Adam LR.
          </Insight>
        </div>
      )}

      {subtab===3&&(
        <div className="fade-in">
          <SectionTitle title="Backpropagation" subtitle="Applying the chain rule layer-by-layer to compute all gradients" />
          <Card style={{maxWidth:780,margin:"0 auto 16px"}}>
            <div style={{fontSize:11,fontWeight:700,color:C.accent,marginBottom:16}}>Chain Rule Through a Simple Network</div>
            <svg width={"100%"} height={320} viewBox="0 0 760 320" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
              {/* Forward pass layers */}
              {[
                {x:60,label:"Input",sub:"x",color:C.blue},
                {x:200,label:"Linear",sub:"z = Wx+b",color:C.purple},
                {x:340,label:"Activation",sub:"a = \u03C3(z)",color:C.accent},
                {x:480,label:"Linear",sub:"z\u2082 = W\u2082a+b\u2082",color:C.purple},
                {x:620,label:"Loss",sub:"L = CE(y\u0302,y)",color:C.red},
              ].map(function(n,i,arr){
                return (
                  <g key={i}>
                    {i>0&&<>
                      <line x1={arr[i-1].x+40} y1={80} x2={n.x-40} y2={80} stroke={C.green} strokeWidth={1.5} />
                      <polygon points={(n.x-42)+",76 "+(n.x-42)+",84 "+(n.x-35)+",80"} fill={C.green} />
                    </>}
                    <rect x={n.x-40} y={55} width={80} height={50} rx={6} fill={n.color+"15"} stroke={n.color} strokeWidth={1.5} />
                    <text x={n.x} y={75} textAnchor="middle" fill={n.color} fontSize={9} fontWeight={700} fontFamily="monospace">{n.label}</text>
                    <text x={n.x} y={92} textAnchor="middle" fill={C.muted} fontSize={8} fontFamily="monospace">{n.sub}</text>
                  </g>
                );
              })}
              {/* Forward label */}
              <text x={340} y={30} textAnchor="middle" fill={C.green} fontSize={10} fontWeight={700} fontFamily="monospace">{"Forward Pass \u2192 compute outputs"}</text>

              {/* Backward arrows */}
              {[
                {x1:580,x2:480,grad:"\u2202L/\u2202z\u2082 = y\u0302-y",color:C.yellow},
                {x1:440,x2:340,grad:"\u2202L/\u2202a = W\u2082\u1D40\u2202L/\u2202z\u2082",color:C.yellow},
                {x1:300,x2:200,grad:"\u2202L/\u2202z = \u03C3\\'(z)\u00B7\u2202L/\u2202a",color:C.orange},
                {x1:160,x2:60,grad:"\u2202L/\u2202x = W\u1D40\u2202L/\u2202z",color:C.orange},
              ].map(function(g,i){
                return (
                  <g key={i}>
                    <line x1={g.x1} y1={160} x2={g.x2+40} y2={160} stroke={g.color} strokeWidth={1.5} strokeDasharray="5,3"/>
                    <polygon points={(g.x2+42)+",156 "+(g.x2+42)+",164 "+(g.x2+35)+",160"} fill={g.color} />
                    <text x={(g.x1+g.x2)/2+20} y={155} textAnchor="middle" fill={g.color} fontSize={7.5} fontFamily="monospace">{g.grad}</text>
                  </g>
                );
              })}
              <text x={340} y={195} textAnchor="middle" fill={C.yellow} fontSize={10} fontWeight={700} fontFamily="monospace">{"Backward Pass \u2190 chain rule gradients"}</text>

              {/* Weight update section */}
              <rect x={30} y={220} width={700} height={85} rx={6} fill={C.card} stroke={C.border} />
              <text x={380} y={242} textAnchor="middle" fill={C.cyan} fontSize={10} fontWeight={700} fontFamily="monospace">Weight Updates (after backward pass)</text>
              {[
                {x:120,label:"W\u2082",update:"W\u2082 \u2190 W\u2082 - \u03B7 \u2202L/\u2202W\u2082",where:"\u2202L/\u2202W\u2082 = \u2202L/\u2202z\u2082 \u00B7 a\u1D40",color:C.purple},
                {x:380,label:"W\u2081",update:"W\u2081 \u2190 W\u2081 - \u03B7 \u2202L/\u2202W\u2081",where:"\u2202L/\u2202W\u2081 = \u2202L/\u2202z \u00B7 x\u1D40",color:C.purple},
                {x:620,label:"\u03B7 (LR)",update:"Controls step size",where:"Too big: diverge. Too small: slow.",color:C.accent},
              ].map(function(w,i){return(
                <g key={i}>
                  <text x={w.x} y={265} textAnchor="middle" fill={w.color} fontSize={9} fontWeight={700} fontFamily="monospace">{w.label}</text>
                  <text x={w.x} y={280} textAnchor="middle" fill={C.text} fontSize={8} fontFamily="monospace">{w.update}</text>
                  <text x={w.x} y={295} textAnchor="middle" fill={C.muted} fontSize={7.5} fontFamily="monospace">{w.where}</text>
                </g>
              );})}
            </svg>
          </Card>
          <Insight title="Backprop Key Facts">
            Backprop is just the <span style={{color:C.accent,fontWeight:700}}>chain rule applied recursively</span>. Each layer computes its local gradient and passes it upstream. The key insight: <span style={{color:C.green,fontWeight:700}}>gradients flow backward through the same graph built during the forward pass</span>. Vanishing gradients occur when many derivatives less than 1 are multiplied together. Skip connections (ResNet) and normalization fix this.
          </Insight>
        </div>
      )}

      {subtab===4&&(
        <div className="fade-in">
          <SectionTitle title="MLP Architecture" subtitle="Building blocks: how layers stack to create representations" />
          <Card style={{maxWidth:780,margin:"0 auto 16px"}}>
            <svg width={"100%"} height={300} viewBox="0 0 760 300" style={{background:"#08080d",borderRadius:8,border:"1px solid "+C.border}}>
              {/* Layers */}
              {[
                {x:60,neurons:3,label:"Input",sub:"x\u2080,x\u2081,x\u2082",color:C.blue},
                {x:220,neurons:5,label:"Hidden 1",sub:"z=Wx+b\na=ReLU(z)",color:C.purple},
                {x:400,neurons:4,label:"Hidden 2",sub:"z=Wa+b\na=ReLU(z)",color:C.purple},
                {x:570,neurons:3,label:"Hidden 3",sub:"z=Wa+b\na=ReLU(z)",color:C.purple},
                {x:700,neurons:2,label:"Output",sub:"Softmax(z)",color:C.green},
              ].map(function(lay,li,layers){
                var totalH=300; var spacing=totalH/(lay.neurons+1);
                return (
                  <g key={li}>
                    {/* Weights to next layer */}
                    {li<layers.length-1&&Array.from({length:lay.neurons},function(_,ni){
                      var nextLay=layers[li+1]; var nspace=totalH/(nextLay.neurons+1);
                      return Array.from({length:nextLay.neurons},function(_,nj){
                        return <line key={ni+"-"+nj}
                          x1={lay.x+16} y1={spacing*(ni+1)}
                          x2={nextLay.x-16} y2={nspace*(nj+1)}
                          stroke={li===0?C.blue+"30":C.purple+"20"} strokeWidth={0.5} />;
                      });
                    })}
                    {/* Neurons */}
                    {Array.from({length:lay.neurons},function(_,ni){
                      return (
                        <g key={ni}>
                          <circle cx={lay.x} cy={spacing*(ni+1)} r={16}
                            fill={lay.color+"20"} stroke={lay.color} strokeWidth={1.5}/>
                          <text x={lay.x} y={spacing*(ni+1)+1} textAnchor="middle" dominantBaseline="middle"
                            fill={lay.color} fontSize={8} fontFamily="monospace">{lay.label[0]+ni}</text>
                        </g>
                      );
                    })}
                    {/* Labels */}
                    <text x={lay.x} y={270} textAnchor="middle" fill={lay.color} fontSize={9} fontWeight={700} fontFamily="monospace">{lay.label}</text>
                    <text x={lay.x} y={284} textAnchor="middle" fill={C.muted} fontSize={7} fontFamily="monospace">{lay.sub.split("\\n")[0]}</text>
                    {lay.neurons>0&&<text x={lay.x} y={7} textAnchor="middle" fill={C.dim} fontSize={7} fontFamily="monospace">{"n="+lay.neurons}</text>}
                  </g>
                );
              })}
            </svg>
          </Card>
          {[
            {label:"Universal Approximation Theorem",color:C.accent,desc:"A single hidden layer MLP with enough neurons can approximate any continuous function. But deeper networks learn more efficiently with fewer parameters."},
            {label:"Depth vs Width",color:C.purple,desc:"Deeper networks (more layers) learn hierarchical features. Wider networks (more neurons/layer) capture more complex patterns per layer. In practice: deeper wins for most tasks."},
            {label:"Dead Neurons",color:C.red,desc:"If a neuron always outputs 0 (ReLU with negative input), its gradient is 0 and it never updates. Proper weight initialization and learning rate prevent this."},
            {label:"Representation Learning",color:C.green,desc:"Each hidden layer learns increasingly abstract features. In vision: pixels \u2192 edges \u2192 shapes \u2192 objects. This hierarchical learning is the core superpower of deep networks."},
          ].map(function(item,i){return(
            <Card key={i} style={{maxWidth:780,margin:"0 auto 8px",borderColor:item.color+"40"}}>
              <div style={{display:"flex",gap:10,alignItems:"flex-start"}}>
                <div style={{width:4,background:item.color,borderRadius:2,flexShrink:0,alignSelf:"stretch"}} />
                <div>
                  <div style={{fontSize:11,fontWeight:700,color:item.color,marginBottom:4}}>{item.label}</div>
                  <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{item.desc}</div>
                </div>
              </div>
            </Card>
          );})}
        </div>
      )}
    </div>
  );
}


/* ================================================================
   TAB 4: ARCHITECTURES
   ================================================================ */
function TabArchitectures() {
  var _s=useState(0); var sel=_s[0]; var setSel=_s[1];

  var archs=[
    {name:"CNN",full:"Convolutional Neural Network",color:C.accent,year:"1989/2012",icon:"\\uD83D\\uDDBC",
     tagline:"Translation-invariant local feature extraction",
     keyIdea:"Shared weights + local connectivity. A filter applied everywhere detects the same feature regardless of position.",
     layers:["Conv2D (filters, kernel, stride, padding)","Batch Norm + ReLU","MaxPool / AvgPool","Residual Blocks (ResNet)","Global Avg Pool","FC + Softmax"],
     breakthroughs:["LeNet (1989): First working CNN","AlexNet (2012): GPU + ReLU revival","VGG (2014): depth beats filter size","ResNet (2015): skip connections enable 152 layers"],
     applications:["Image classification","Object detection (YOLO, Faster R-CNN)","Semantic segmentation","Medical imaging","Video understanding"],
     params:"1M \u2013 600M",complexity:"O(k\u00B2 \u00B7 C_in \u00B7 C_out \u00B7 H \u00B7 W)"},
    {name:"RNN/LSTM",full:"Recurrent Networks & Long Short-Term Memory",color:C.blue,year:"1986/1997",icon:"\\uD83D\\uDD04",
     tagline:"Sequential modeling with persistent hidden state",
     keyIdea:"Hidden state h_t = f(h_{t-1}, x_t) carries information across time steps. LSTMs add gates to control what to remember, forget, and output.",
     layers:["Input embedding","RNN: h_t = tanh(Wh_{t-1} + Ux_t + b)","LSTM: 4 gates (input, forget, output, cell)","GRU: Simplified LSTM (2 gates)","Final hidden state / sequence output","FC decoder"],
     breakthroughs:["Vanilla RNN (1986): sequential processing","LSTM (1997): long-range dependencies","GRU (2014): simplified LSTM","Bidirectional RNNs: process both directions"],
     applications:["Machine translation (before Transformers)","Speech recognition","Time series forecasting","Sequence labeling (NER, POS)","Music generation"],
     params:"1M \u2013 100M",complexity:"O(T \u00B7 H\u00B2) per sequence"},
    {name:"Transformer",full:"Transformer & Self-Attention",color:C.purple,year:"2017",icon:"\\u26A1",
     tagline:"Every token attends to every other token in parallel",
     keyIdea:"Attention(Q,K,V) = softmax(QK\u1D40/\u221Ad_k)V. Self-attention allows each position to aggregate information from all positions simultaneously, enabling parallelism and long-range dependencies.",
     layers:["Token + Positional Embedding","Multi-Head Self-Attention","Add & LayerNorm","Feed-Forward Network (2 linear)","Add & LayerNorm","Final projection / decoder"],
     breakthroughs:["\"Attention is All You Need\" (2017)","BERT (2018): bidirectional pre-training","GPT-2/3 (2019/2020): scaling laws","ViT (2020): transformers for vision","GPT-4 / LLaMA (2023+): frontier LLMs"],
     applications:["Language modeling (GPT, LLaMA)","Text classification (BERT)","Machine translation","Vision (ViT, DINO)","Multimodal (CLIP, Flamingo)"],
     params:"100M \u2013 1T+",complexity:"O(n\u00B2 \u00B7 d) \u2013 quadratic in sequence length"},
    {name:"GAN",full:"Generative Adversarial Network",color:C.pink,year:"2014",icon:"\\uD83C\\uDFAD",
     tagline:"Generator vs Discriminator in a minimax game",
     keyIdea:"min_G max_D E[log D(x)] + E[log(1-D(G(z)))]. Generator fools Discriminator; Discriminator distinguishes real from fake. Nash equilibrium: G produces real-looking data.",
     layers:["Noise vector z \u223C N(0,I)","Generator G(z): Upsample \u2192 image","Discriminator D(x): Conv \u2192 real/fake","Adversarial loss + optional L1/VGG loss","Training alternates G and D updates","Optional: Conditional GAN (class label)"],
     breakthroughs:["Original GAN (Goodfellow 2014)","DCGAN (2015): stable CNN-based GAN","StyleGAN (2019): SOTA face synthesis","BigGAN (2018): class-conditional","CycleGAN: unpaired image translation"],
     applications:["Image synthesis (faces, art)","Image-to-image translation","Data augmentation","Video generation","Super-resolution"],
     params:"10M \u2013 500M",complexity:"Training instability is the main challenge"},
    {name:"VAE",full:"Variational Autoencoder",color:C.yellow,year:"2013",icon:"\\uD83D\\uDD2C",
     tagline:"Encoder learns a structured latent space; Decoder reconstructs",
     keyIdea:"Encode x into q(z|x) = N(\u03BC,\u03C3\u00B2). Reparameterize z = \u03BC + \u03C3\u22C5\u03B5. Decode to p(x|z). Loss = reconstruction + KL(q||p). The latent space is continuous and structured.",
     layers:["Encoder CNN/MLP \u2192 (\u03BC, log\u03C3\u00B2)","Reparameterization trick (enables backprop)","Latent z \u223C N(\u03BC, \u03C3\u00B2)","Decoder MLP/CNN \u2192 reconstruction","ELBO = E[log p(x|z)] - KL(q||p)","Optional: \u03B2-VAE for disentanglement"],
     breakthroughs:["Original VAE (Kingma & Welling 2013)","Conditional VAE (CVAE)","VQ-VAE: discrete latent codes","VQ-VAE-2: hierarchical latents","Used in latent diffusion models"],
     applications:["Image generation","Anomaly detection","Drug discovery (molecular generation)","Representation learning","Latent space interpolation"],
     params:"1M \u2013 100M",complexity:"O(encoder + decoder) \u2013 efficient"},
    {name:"GNN",full:"Graph Neural Network",color:C.cyan,year:"2016",icon:"\\uD83D\\uDD78",
     tagline:"Message passing over graph structures",
     keyIdea:"h_v\u207A = UPDATE(h_v, AGG({h_u : u\u2208N(v)})). Each node aggregates messages from its neighbors and updates its representation. Stacking layers = expanding receptive field.",
     layers:["Node feature matrix X","Adjacency matrix A (or edge list)","Graph Conv: H = \u03C3(\u00C3 H W) where \u00C3 = D\u207B\u00BD A D\u207B\u00BD","Message Passing layers","Graph Pooling (mean, sum, hierarchical)","Graph-level readout + FC"],
     breakthroughs:["GCN (Kipf & Welling 2016)","GraphSAGE: inductive learning","GAT: attention-weighted aggregation","PNA / DGN: diverse aggregators","AlphaFold uses EvoFormer (attention on graphs)"],
     applications:["Molecular property prediction","Social network analysis","Recommendation systems","Drug discovery","Protein structure (AlphaFold)"],
     params:"100K \u2013 10M",complexity:"O(|E| \u00B7 d) per layer \u2013 edge-dependent"},
    {name:"Diffusion",full:"Diffusion Model (DDPM / Score-Based)",color:C.orange,year:"2020",icon:"\\u2601",
     tagline:"Learn to reverse a gradual noising process",
     keyIdea:"Forward: q(x_t|x_0) = N(\u221A\u03B1_t x_0, (1-\u03B1_t)I). Reverse: train \u03B5_\u03B8 to predict noise added at each step. Generation: start from pure noise, iteratively denoise for T steps.",
     layers:["Forward diffusion: add noise for T steps","UNet with time-step conditioning","Cross-attention for text conditioning","Classifier-Free Guidance","DDIM sampling (fast, deterministic)","Latent Diffusion: operate in VAE latent space"],
     breakthroughs:["DDPM (Ho et al. 2020): theoretical clarity","Score matching (Song et al.)","DALL-E 2 / Imagen (2022): text-to-image","Stable Diffusion: latent + open source","SDXL / Flux (2024): next-gen quality"],
     applications:["Text-to-image (SD, Midjourney)","Image editing & inpainting","Video generation (Sora)","Audio synthesis","Protein structure design"],
     params:"100M \u2013 10B",complexity:"O(T \u00B7 UNet) \u2013 slow sampling"},
    {name:"ViT",full:"Vision Transformer",color:C.green,year:"2020",icon:"\\uD83D\\uDC41",
     tagline:"Patch embeddings + pure transformer for vision",
     keyIdea:"Split image into N 16\u00D716 patches. Linear embed each patch. Prepend [CLS] token. Run through standard transformer encoder. The CLS token is used for classification.",
     layers:["Patch + Position Embedding","[CLS] token prepend","L \u00D7 Transformer Encoder blocks","Multi-Head Self-Attention (global)","Layer Norm + FFN","MLP head on [CLS] token"],
     breakthroughs:["Original ViT (Dosovitskiy 2020)","DeiT: data-efficient training","DINO: self-supervised ViT","Swin Transformer: hierarchical vision","MAE: masked autoencoder pretraining"],
     applications:["Image classification","Object detection (DETR)","Semantic segmentation","Self-supervised vision pretraining","Multimodal (CLIP uses ViT)"],
     params:"86M \u2013 2B",complexity:"O(N\u00B2 \u00B7 d) where N = num patches"},
  ];

  var arch=archs[sel];
  return (
    <div>
      <SectionTitle title="Phase 3: Major Architectures" subtitle={"Every major architecture family "+DASH+" click to explore"} />

      <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:20}}>
        {archs.map(function(a,i){return(
          <button key={i} onClick={function(){setSel(i);}} style={{
            padding:"8px 14px",borderRadius:8,
            border:"1.5px solid "+(sel===i?a.color:C.border),
            background:sel===i?a.color+"20":C.card,
            color:sel===i?a.color:C.muted,
            cursor:"pointer",fontSize:10,fontWeight:700,fontFamily:"monospace",
          }}>{a.icon+" "+a.name}</button>
        );})}
      </div>

      <div className="fade-in" key={sel}>
        <Card highlight={true} style={{maxWidth:780,margin:"0 auto 14px",borderColor:arch.color}}>
          <div style={{display:"flex",gap:16,alignItems:"flex-start",flexWrap:"wrap",marginBottom:14}}>
            <div style={{fontSize:32}}>{arch.icon}</div>
            <div style={{flex:1}}>
              <div style={{fontSize:16,fontWeight:800,color:arch.color}}>{arch.full}</div>
              <div style={{fontSize:10,color:C.muted,marginTop:2}}>{arch.year+" "+DASH+" "+arch.params+" parameters"}</div>
              <div style={{fontSize:11,color:C.text,marginTop:6,lineHeight:1.6,fontStyle:"italic"}}>{arch.tagline}</div>
            </div>
          </div>
          <div style={{background:"#08080d",borderRadius:6,padding:"12px 14px",marginBottom:14,border:"1px solid "+arch.color+"30"}}>
            <div style={{fontSize:9,color:C.muted,marginBottom:4,fontWeight:700}}>KEY IDEA</div>
            <div style={{fontSize:11,color:C.text,lineHeight:1.8}}>{arch.keyIdea}</div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",gap:14}}>
            <div>
              <div style={{fontSize:9,color:C.muted,marginBottom:6,fontWeight:700}}>LAYER STACK</div>
              {arch.layers.map(function(l,i){return(
                <div key={i} style={{display:"flex",gap:6,alignItems:"flex-start",marginBottom:4}}>
                  <span style={{color:arch.color,fontSize:8,fontFamily:"monospace",flexShrink:0,marginTop:2}}>{i+1+"."}</span>
                  <span style={{fontSize:9,color:C.muted,lineHeight:1.5}}>{l}</span>
                </div>
              );})}
            </div>
            <div>
              <div style={{fontSize:9,color:C.muted,marginBottom:6,fontWeight:700}}>BREAKTHROUGHS</div>
              {arch.breakthroughs.map(function(b,i){return(
                <div key={i} style={{display:"flex",gap:6,alignItems:"flex-start",marginBottom:4}}>
                  <span style={{color:arch.color,fontSize:8,flexShrink:0,marginTop:2}}>{STAR}</span>
                  <span style={{fontSize:9,color:C.muted,lineHeight:1.5}}>{b}</span>
                </div>
              );})}
            </div>
            <div>
              <div style={{fontSize:9,color:C.muted,marginBottom:6,fontWeight:700}}>APPLICATIONS</div>
              {arch.applications.map(function(ap,i){return(
                <Badge key={i} color={arch.color}>{ap}</Badge>
              );})}
              <div style={{marginTop:10,fontSize:9,color:C.muted}}>
                <span style={{color:C.text,fontWeight:700}}>Complexity: </span>{arch.complexity}
              </div>
            </div>
          </div>
        </Card>
      </div>

      <Insight icon={TARG} title="Architecture Selection Guide">
        <span style={{color:C.accent,fontWeight:700}}>Images/Video</span>: CNN or ViT (ViT wins with enough data).{" "}
        <span style={{color:C.blue,fontWeight:700}}>Sequences/Text</span>: Transformer (RNN is legacy).{" "}
        <span style={{color:C.pink,fontWeight:700}}>Image Generation</span>: Diffusion model (GAN for speed, Diffusion for quality).{" "}
        <span style={{color:C.yellow,fontWeight:700}}>Structured latent space</span>: VAE.{" "}
        <span style={{color:C.cyan,fontWeight:700}}>Graph-structured data</span>: GNN.
      </Insight>
    </div>
  );
}

/* ================================================================
   TAB 5: TRAINING & OPTIMIZATION
   ================================================================ */
function TabTraining() {
  var _s=useState(0); var sel=_s[0]; var setSel=_s[1];

  var sections=[
    {name:"Initialization",color:C.blue,icon:"\\uD83C\\uDF31",
     items:[
       {t:"Zero Init",d:"All weights = 0. Symmetric breaking fails: all neurons learn the same gradient. Never do this.",bad:true},
       {t:"Random Normal",d:"Small random values N(0, 0.01). Works for shallow nets, but variance explodes/vanishes in deep nets.",bad:true},
       {t:"Xavier / Glorot",d:"W ~ N(0, 2/(fan_in + fan_out)). Designed for tanh/sigmoid. Variance preserved across layers.",bad:false},
       {t:"He / Kaiming",d:"W ~ N(0, 2/fan_in). Designed for ReLU. Accounts for the ~50% dead neurons at init. Default for ReLU nets.",bad:false},
       {t:"Orthogonal Init",d:"Initialize weight matrices as orthogonal matrices. Good for RNNs to preserve gradient norms over time.",bad:false},
       {t:"Pre-trained Weights",d:"Transfer learning: start from ImageNet/BERT weights. Almost always better than random init.",bad:false},
     ]},
    {name:"Normalization",color:C.purple,icon:"\\u2696",
     items:[
       {t:"Batch Norm (BN)",d:"Normalize per-feature across batch. y = \u03B3 \u00B7 (x-\u03BC_B)/\u03C3_B + \u03B2. Batch-dependent: bad for small batches or inference."},
       {t:"Layer Norm (LN)",d:"Normalize per-token across features. Independent of batch size. Default for Transformers and RNNs."},
       {t:"Group Norm (GN)",d:"Normalize across groups of channels. Better than BN for small batches. Used in object detection."},
       {t:"Instance Norm",d:"Normalize per-sample per-channel. Used in style transfer where per-image statistics matter."},
       {t:"RMS Norm",d:"y = x / RMS(x) \u00B7 \u03B3. Simpler than LayerNorm (no mean centering). Used in LLaMA and modern LLMs."},
       {t:"When to use what",d:"CNN: BatchNorm. Transformer: LayerNorm or RMSNorm. Small batch: GroupNorm. Style: InstanceNorm."},
     ]},
    {name:"Regularization",color:C.accent,icon:"\\uD83D\\uDEE1",
     items:[
       {t:"L2 / Weight Decay",d:"\u03A9 = \u03BB\u03A3w\u00B2 added to loss. Penalizes large weights. Equivalent to MAP with Gaussian prior. Use AdamW for correct decoupled decay."},
       {t:"L1 Regularization",d:"\u03A9 = \u03BB\u03A3|w|. Induces sparsity \u2014 many weights go exactly to 0. Useful for feature selection."},
       {t:"Dropout",d:"Randomly zero p% of neurons each forward pass. Forces ensemble learning. p=0.1 for Transformers, 0.5 for FC layers."},
       {t:"DropPath / Stochastic Depth",d:"Drop entire residual blocks at random during training. Scales regularization with network depth. Used in ViT."},
       {t:"Label Smoothing",d:"Replace one-hot y with y=(1-\u03B5) + \u03B5/K. Prevents overconfident predictions. Typical \u03B5=0.1 improves top-1 accuracy."},
       {t:"Data Augmentation",d:"Random crop, flip, color jitter, mixup, cutmix, random erasing. Effectively multiplies dataset size."},
     ]},
    {name:"LR Scheduling",color:C.green,icon:"\\uD83D\\uDCC8",
     items:[
       {t:"Constant LR",d:"Simple but misses the benefits of scheduling. Use only for quick experiments."},
       {t:"Step Decay",d:"Multiply LR by \u03B3 every N epochs. Classic approach for CNNs. LR drops at fixed milestones."},
       {t:"Cosine Annealing",d:"LR = \u03B7_min + 0.5(\u03B7_max-\u03B7_min)(1+cos(\u03C0 t/T)). Smooth decay to minimum. Add warm restarts for SGDR."},
       {t:"Linear Warmup",d:"Ramp LR from 0 to target over first N steps. Critical for transformers: prevents early instability."},
       {t:"Warmup + Cosine",d:"Standard for LLMs and ViT: warm up for 1-5% of steps, then cosine decay. Default in HuggingFace."},
       {t:"OneCycleLR",d:"Fast.ai discovery: LR increases then decreases in one cycle. Achieves great results in fewer epochs."},
     ]},
    {name:"Stability Tricks",color:C.yellow,icon:"\\uD83D\\uDD27",
     items:[
       {t:"Gradient Clipping",d:"If ||g|| > threshold, scale g down. Prevents exploding gradients in RNNs and transformers. clip_norm=1.0 default."},
       {t:"Mixed Precision (FP16/BF16)",d:"Forward/backward in FP16, master weights in FP32. 2x memory saving, ~2x speedup on modern GPUs. Use BF16 for stability."},
       {t:"Gradient Checkpointing",d:"Recompute activations during backward pass instead of storing them. 10x memory reduction for 30% speed cost."},
       {t:"Gradient Accumulation",d:"Accumulate gradients over N micro-batches before stepping. Simulates large batch size on limited GPU memory."},
       {t:"EMA of Weights",d:"Maintain exponential moving average of weights. Use EMA weights for inference \u2014 much more stable than step weights."},
       {t:"Loss Scaling",d:"Multiply loss by large constant before FP16 backward to prevent underflow. PyTorch GradScaler handles this."},
     ]},
    {name:"Debugging Training",color:C.pink,icon:"\\uD83D\\uDC1B",
     items:[
       {t:"Overfit a tiny batch first",d:"Train on 1-10 examples until loss \u22480. If it doesn\\'t overfit, your model or loss has a bug."},
       {t:"Watch loss curves",d:"Training loss not decreasing: LR too low, bug in data pipeline, bad init. Val loss diverging: overfitting."},
       {t:"Log gradient norms",d:"If gradient norms explode or vanish, fix init or add gradient clipping. Normal range: 0.1 to 10."},
       {t:"Check activation stats",d:"Mean and std of activations should be reasonable after each layer. BN/LN makes this automatic."},
       {t:"Learning rate finder",d:"Run LR sweep from 1e-7 to 10. Plot loss vs LR. Choose LR just before loss starts rising."},
       {t:"Inspect predictions",d:"Visualize model outputs, attention maps, confusion matrices early. Catch data/label bugs before long runs."},
     ]},
  ];

  var sec=sections[sel];
  return (
    <div>
      <SectionTitle title="Phase 4: Training Mastery" subtitle="The techniques that separate engineers who get results from those who don't" />

      <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:20}}>
        {sections.map(function(s,i){return(
          <Pill key={i} active={sel===i} color={s.color} onClick={function(){setSel(i);}}>
            {s.icon+" "+s.name}
          </Pill>
        );})}
      </div>

      <div className="fade-in" key={sel}>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(340px,1fr))",gap:10,maxWidth:780,margin:"0 auto 16px"}}>
          {sec.items.map(function(item,i){return(
            <div key={i} style={{
              padding:"14px 16px",background:C.card,borderRadius:8,
              border:"1px solid "+(item.bad?C.red+"40":sec.color+"30"),
            }}>
              <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:6}}>
                <span style={{fontSize:10,fontWeight:800,fontFamily:"monospace",
                  color:item.bad===true?C.red:item.bad===false?C.green:sec.color}}>
                  {item.bad===true?"\\u2717":item.bad===false?CHK:DOT}
                </span>
                <div style={{fontSize:11,fontWeight:700,color:item.bad===true?C.red:sec.color}}>{item.t}</div>
              </div>
              <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{item.d}</div>
            </div>
          );})}
        </div>
      </div>

      <Insight title="Training Checklist">
        Before every run: <span style={{color:C.blue,fontWeight:700}}>He init</span> + <span style={{color:C.purple,fontWeight:700}}>LayerNorm/BatchNorm</span> + <span style={{color:C.green,fontWeight:700}}>warmup schedule</span> + <span style={{color:C.yellow,fontWeight:700}}>gradient clipping</span> + <span style={{color:C.cyan,fontWeight:700}}>mixed precision</span> + <span style={{color:C.pink,fontWeight:700}}>experiment tracking</span>. Overfit one batch first. Then scale.
      </Insight>
    </div>
  );
}

/* ================================================================
   TAB 6: ADVANCED TOPICS
   ================================================================ */
function TabAdvanced() {
  var _s=useState(0); var sel=_s[0]; var setSel=_s[1];

  var topics=[
    {name:"Transfer Learning",color:C.blue,icon:"\\uD83D\\uDD01",
     overview:"Use weights from a model trained on a large dataset as initialization for your task. Almost always outperforms training from scratch.",
     strategies:[
       {t:"Feature Extraction",d:"Freeze all pretrained layers. Train only the new head. Fast, few params, works with small datasets."},
       {t:"Fine-tuning",d:"Unfreeze all or some layers. Train end-to-end with small LR (1e-5 to 1e-4). Better with more data."},
       {t:"Progressive Unfreezing",d:"Start frozen, gradually unfreeze from top layers. Avoids catastrophic forgetting. fast.ai approach."},
       {t:"LoRA (Low-Rank Adaptation)",d:"Add trainable low-rank matrices alongside frozen weights. Only 0.1-1% of params. Standard for LLM fine-tuning."},
       {t:"Domain Adaptation",d:"Source domain \u2260 target domain. Techniques: adversarial DA, CORAL, pseudo-labels."},
       {t:"Few-Shot Fine-Tuning",d:"Fine-tune on 5-100 examples per class. Use MAML or prototypical networks as meta-init."},
     ]},
    {name:"Self-Supervised",color:C.green,icon:"\\uD83E\\uDDE9",
     overview:"Learn representations from unlabeled data by predicting parts of the input from other parts. No human labels needed.",
     strategies:[
       {t:"Contrastive (SimCLR)",d:"Augment x twice \u2192 x+, x++. Pull embeddings together, push other samples away. NT-Xent loss."},
       {t:"BYOL / DINO",d:"Bootstrap: no negative samples needed. Teacher-student self-distillation. DINO for ViT generates excellent features."},
       {t:"Masked Autoencode (MAE)",d:"Mask 75% of image patches. Predict masked patches. Very efficient. Scales to large ViTs."},
       {t:"BERT / MLM",d:"Mask 15% of tokens. Predict masked words. Foundation of most NLP models."},
       {t:"Causal LM (GPT-style)",d:"Predict next token. Entire sequence is self-supervised signal. Scales to trillion tokens."},
       {t:"CLIP",d:"Match image-text pairs. Contrastive loss across modalities. Enables zero-shot classification."},
     ]},
    {name:"Reinforcement Learning",color:C.orange,icon:"\\uD83C\\uDFAE",
     overview:"Agent learns from reward signals by interacting with an environment. No labeled data, only reward feedback.",
     strategies:[
       {t:"Q-Learning / DQN",d:"Learn Q(s,a): expected return. DQN adds replay buffer + target network. Atari breakthrough."},
       {t:"Policy Gradient (REINFORCE)",d:"Directly optimize policy \u03C0_\u03B8. High variance. Use baseline (advantage) to reduce variance."},
       {t:"Actor-Critic (A3C, A2C)",d:"Actor: policy. Critic: value function estimates advantage. Reduce variance with critic baseline."},
       {t:"PPO (Proximal Policy Opt.)",d:"Clip policy update ratio to prevent too-large steps. Most popular RL algo. Robust and simple."},
       {t:"RLHF",d:"Reinforcement Learning from Human Feedback. Train reward model on preferences, then PPO. Powers ChatGPT/Claude."},
       {t:"Model-Based RL",d:"Learn world model p(s\\'|s,a). Plan inside model (Dreamer, MuZero). Sample efficient."},
     ]},
    {name:"LLM & Alignment",color:C.purple,icon:"\\uD83E\\uDD16",
     overview:"Large language models trained on internet-scale text. Fine-tuning and alignment techniques to make them safe and useful.",
     strategies:[
       {t:"Scaling Laws",d:"Loss ~ (params, data, compute)^{-\u03B1}. Chinchilla: optimal to scale params and tokens equally."},
       {t:"Instruction Fine-Tuning",d:"Fine-tune on (instruction, response) pairs. Makes base models follow instructions. FLAN, Alpaca."},
       {t:"RLHF / DPO",d:"RLHF: reward model + PPO. DPO (Direct Preference Optimization): simpler, no RL needed. DPO is current standard."},
       {t:"Prompt Engineering",d:"Chain-of-thought, few-shot examples, structured output, system prompts. Major performance driver."},
       {t:"RAG",d:"Retrieval-Augmented Generation: retrieve relevant docs, inject into context. Reduces hallucination."},
       {t:"Quantization (GGUF, AWQ)",d:"Reduce weights to INT8/INT4. 4x memory reduction. Enables LLMs on consumer hardware."},
     ]},
    {name:"Diffusion & Generative",color:C.pink,icon:"\\uD83C\\uDFA8",
     overview:"State-of-the-art generative models. Diffusion models dominate image/video/audio synthesis.",
     strategies:[
       {t:"DDPM",d:"Learn to denoise at each step t. UNet predicts noise \u03B5 given noisy image x_t and timestep t."},
       {t:"Latent Diffusion",d:"Diffuse in VAE latent space instead of pixel space. 8x smaller, enables Stable Diffusion."},
       {t:"Classifier-Free Guidance",d:"Train conditional and unconditional jointly. Guide: \u03B5_guided = \u03B5_uncond + w(\u03B5_cond - \u03B5_uncond). w controls strength."},
       {t:"ControlNet",d:"Add spatial conditioning (pose, depth, edges) to pretrained diffusion. Adapter architecture."},
       {t:"Flow Matching",d:"Simpler than diffusion: directly learn flow from noise to data. Flux, SD3 use this. Faster training."},
       {t:"Video Diffusion (Sora)",d:"3D UNet or DiT over video latents. Temporal attention across frames. Massive compute."},
     ]},
    {name:"MLOps & Deploy",color:C.yellow,icon:"\\uD83D\\uDE80",
     overview:"Production ML: tracking, serving, monitoring, and optimizing models at scale.",
     strategies:[
       {t:"Experiment Tracking",d:"Weights & Biases, MLflow, Neptune. Log: hyperparams, metrics, artifacts, system stats. Do this from day 1."},
       {t:"Model Serving",d:"TorchServe, TensorFlow Serving, Triton Inference Server. REST/gRPC API. Batching for throughput."},
       {t:"Quantization",d:"Post-training quantization (INT8, INT4). QAT (quantization-aware training). PyTorch torch.ao module."},
       {t:"ONNX Export",d:"Convert PyTorch/TF model to ONNX graph. Run with ONNX Runtime on CPU/GPU. Hardware-independent."},
       {t:"Edge Deployment",d:"TFLite, CoreML, TensorRT. Pruning + quantization + distillation to fit model on device."},
       {t:"CI/CD for ML",d:"Data validation (Great Expectations), model testing (pytest), auto-retraining pipelines. MLOps maturity model."},
     ]},
  ];

  var t=topics[sel];
  return (
    <div>
      <SectionTitle title="Phase 5+6: Advanced Topics" subtitle="Frontier techniques and production deployment" />

      <div style={{display:"flex",gap:6,flexWrap:"wrap",justifyContent:"center",marginBottom:20}}>
        {topics.map(function(tp,i){return(
          <Pill key={i} active={sel===i} color={tp.color} onClick={function(){setSel(i);}}>
            {tp.icon+" "+tp.name}
          </Pill>
        );})}
      </div>

      <div className="fade-in" key={sel}>
        <Card style={{maxWidth:780,margin:"0 auto 14px",borderColor:t.color+"50"}}>
          <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:10}}>
            <span style={{fontSize:24}}>{t.icon}</span>
            <div>
              <div style={{fontSize:14,fontWeight:800,color:t.color}}>{t.name}</div>
              <div style={{fontSize:10,color:C.muted,marginTop:2,lineHeight:1.6}}>{t.overview}</div>
            </div>
          </div>
        </Card>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(340px,1fr))",gap:10,maxWidth:780,margin:"0 auto 16px"}}>
          {t.strategies.map(function(s,i){return(
            <div key={i} style={{padding:"14px 16px",background:C.card,borderRadius:8,border:"1px solid "+t.color+"25"}}>
              <div style={{fontSize:11,fontWeight:700,color:t.color,marginBottom:5}}>{s.t}</div>
              <div style={{fontSize:10,color:C.muted,lineHeight:1.7}}>{s.d}</div>
            </div>
          );})}
        </div>
      </div>

      <Insight icon={TARG} title="Advice for Phase 5+">
        Pick <span style={{color:C.accent,fontWeight:700}}>one specialization</span> and go deep. Read the seminal papers: Attention Is All You Need, DDPM, SimCLR, LoRA, PPO. Build something real. Follow DL Twitter/X, Yannic Kilcher, Andrej Karpathy, Sebastien Raschka. The field moves fast {DASH} reading weekly is essential.
      </Insight>
    </div>
  );
}


/* ================================================================
   ROOT APP
   ================================================================ */
function App() {
  var _t=useState(0); var tab=_t[0]; var setTab=_t[1];
  var tabs=["\\uD83D\\uDDFA Roadmap","\\uD83D\\uDCDA Foundations","\\uD83E\\uDDE0 Core Nets","\\uD83C\\uDFD7 Architectures","\\uD83D\\uDE80 Training","\\u2728 Advanced"];
  return (
    <div style={{
      background:C.bg,minHeight:"100vh",padding:"24px 16px",
      fontFamily:"'JetBrains Mono','SF Mono',monospace",
      color:C.text,maxWidth:980,margin:"0 auto",
    }}>
      <div style={{textAlign:"center",marginBottom:20}}>
        <div style={{
          fontSize:24,fontWeight:800,
          background:"linear-gradient(135deg,"+C.accent+","+C.purple+","+C.cyan+")",
          WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",display:"inline-block",
        }}>Deep Learning Mastery Path</div>
        <div style={{fontSize:11,color:C.muted,marginTop:4}}>
          {"Interactive guide "+DASH+" from math foundations to frontier models"}
        </div>
      </div>
      <TabBar tabs={tabs} active={tab} onChange={setTab} />
      {tab===0&&<TabRoadmap />}
      {tab===1&&<TabFoundations />}
      {tab===2&&<TabCoreNets />}
      {tab===3&&<TabArchitectures />}
      {tab===4&&<TabTraining />}
      {tab===5&&<TabAdvanced />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
</script>
</body>
</html>
"""

DL_PATH_HEIGHT = 1400