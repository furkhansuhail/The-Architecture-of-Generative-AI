"""
The Architecture of Machine Learning
======================================
A multi-paradigm Streamlit reference hub — theory, visual breakdowns,
and runnable implementations across all major ML paradigms.

Paradigms:
  - Generative AI        → topics in generative_ai/topics/
  - Supervised Learning  → topics in supervised/topics/
  - Unsupervised Learning→ topics in unsupervised/topics/
  - Reinforcement Learning→ topics in reinforcement_learning/topics/

Each paradigm shares the same 3-panel structure:
  Topics (Theory + Visual + Step-by-Step) | Code Runner | AI Assistant
"""

import os
import sys
import subprocess
import tempfile
import re
import base64
from pathlib import Path
from typing import Dict, Optional

# ── Matplotlib interceptor ─────────────────────────────────────────────────────
# Injected at the top of every subprocess execution so plt.savefig/plt.show
# serialise figures to stdout (base64) instead of writing to disk.
_MATPLOTLIB_INTERCEPTOR = '''
import sys as _sys, io as _io, base64 as _b64
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as _plt
 
_orig_savefig = _plt.savefig
_orig_show    = _plt.show
 
def _intercept_savefig(fname, *args, **kwargs):
    buf = _io.BytesIO()
    kwargs.pop('format', None)
    _orig_savefig(buf, *args, format='png', **kwargs)
    buf.seek(0)
    enc = _b64.b64encode(buf.read()).decode('ascii')
    print(f"###STREAMLIT_FIG###{enc}###END_FIG###", flush=True)
 
def _intercept_show(*args, **kwargs):
    for _fn in _plt.get_fignums():
        buf = _io.BytesIO()
        _plt.figure(_fn).savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        enc = _b64.b64encode(buf.read()).decode('ascii')
        print(f"###STREAMLIT_FIG###{enc}###END_FIG###", flush=True)
    _plt.close('all')
 
_plt.savefig = _intercept_savefig
_plt.show    = _intercept_show
'''

_FIG_PATTERN = re.compile(r'###STREAMLIT_FIG###([A-Za-z0-9+/=]+)###END_FIG###')

import streamlit as st
import streamlit.components.v1 as st_components

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Architecture of ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: "Times New Roman", Times, serif; }
    </style>
""", unsafe_allow_html=True)

# ── Paradigm registry ──────────────────────────────────────────────────────────
PARADIGMS ={

    "00_Automation"                 :                                   "Automation",
    "01_Basic_ML_Libraries"         :                        "00_Basic_ML_Libraries",
    "02_Core Math"                  :                   "01_ML_Foundation_Math_Core",
    "03_Probability_&_Information"  :   "02_ML_Foundation_Probability_&_Information",
    "04_Data"                       :                        "03_ML_Foundation_Data",
    "05_Model Behaviour"            :             "04_ML_Foundation_Model_Behaviour",
    "06_Learning & Optimization"    :     "05_ML_Foundation_Learning_&_Optimization",
    "07_Evaluation & Generalization": "06_ML_Foundation_Evaluation_&_Generalization",
    "08_Advanced Foundations"       :        "07_ML_Foundation_Advanced_Foundations",
    "09_Supervised Learning Core"   :                          "Supervised_Learning",
    "10_Unsupervised Learning Core" :                        "Unsupervised_Learning",
    "11_Deep Learning"              :                                "Deep_Learning",
    "12_Training Core"              :                                "Training_Core",
    "13_Reinforcement Learning"     :                       "Reinforcement_learning",
    "14_Generative AI"              :                                "Generative_AI",
    "15_Frameworks"                 :                                   "Frameworks",
    "16_Interpretability"           :                             "Interpretability",
    "17_Compiler Runtime"           :                           "Compilers_Runtimes",
    "18_GPU Kernels"                :                                  "GPU_Kernels",
    "19_Training LLM"               :                                 "LLM_Training",
}



#     {
#     "Automation"                                 :                                   "Automation",
#     "ML Foundation - Core Math"                  :                   "00_ML_Foundation_Math_Core",
#     "ML Foundation - Data"                       :                       "02__ML_Foundation_Data",
#     "ML Foundation - Model Behaviour"            :            "03__ML_Foundation_Model Behaviour",
#     "ML Foundation - Learning & Optimization"    :    "04__ML_Foundation_Learning & Optimization",
#     "ML Foundation - Evaluation & Generalization":"05__ML_Foundation_Evaluation & Generalization",
#     "ML Foundation - Advanced Foundations"       :       "06__ML_Foundation_Advanced Foundations",
#     "AIML Training Core"                         :                             "07_Training_Core",
#     "AIML Interpretability"                      :                          "08_Interpretability",
#     "AIML Frameworks"                            :                                "09_Frameworks",
#     "Compiler Runtime"                           :                        "10_compilers_runtimes",
#     "GPU Kernels"                                :                               "11_gpu_kernels",
#     "Training LLM"                               :                              "12_LLM_Training",
#     "Supervised Learning"                        :                                   "supervised",
#     "Unsupervised Learning"                      :                                 "unsupervised",
#     "Deep Learning"                              :                                "deep_learning",
#     "Generative AI"                              :                                "generative_ai",
#     "Reinforcement Learning"                     :                       "reinforcement_learning",
# }

# ── Dynamic topic loader ───────────────────────────────────────────────────────
@st.cache_resource
def load_topics_for(paradigm_key: str) -> dict:
    """Dynamically import and return all topics for the given paradigm package.

    Uses @st.cache_resource (not @st.cache_data) so the result is stored in
    memory rather than pickled to disk.  Pickling fails when the returned dict
    contains function objects (e.g. 'render_operations') because pickle tries
    to serialise them as references to their owning module
    (e.g. 'deep_learning.topics.01_…') which isn't on sys.modules yet during
    the next startup → KeyError: 'deep_learning.topics'.
    """
    base_dir = Path(__file__).parent
    pkg_dir  = base_dir / paradigm_key
    if not pkg_dir.exists():
        return {}

    # Make sure the project root is on sys.path so package imports resolve.
    parent = str(base_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        import importlib
        # Reload the sub-package each time so stale cached modules don't hide
        # newly added topic files during development.
        mod_name = f"{paradigm_key}.topics"
        pkg = importlib.import_module(mod_name)
        importlib.reload(pkg)
        raw = pkg.get_all_topics()

        # Strip non-serialisable callables so the dict is safe if the cache
        # backend ever switches back to pickle-based storage.
        clean: dict = {}
        for topic, data in raw.items():
            clean[topic] = {k: v for k, v in data.items()
                            if not callable(v)}
        return clean
    except Exception as e:
        st.error(f"Could not load topics for '{paradigm_key}': {e}")
        return {}


@st.cache_resource
def load_implementations_for(paradigm_key: str) -> dict:
    """Load all implementation .py files from paradigm's Implementation folder."""
    base_dir = Path(__file__).parent
    impl_dir = base_dir / paradigm_key / "Implementation"

    if not impl_dir.exists():
        return {}

    implementations = {}
    for py_file in sorted(impl_dir.rglob("*.py")):
        if py_file.name.startswith("_"):
            continue
        try:
            code_text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        key = py_file.stem
        level, concepts, module = "Unknown", [], "General"

        for line in code_text.split("\n"):
            s = line.strip()
            if s.lower().startswith("level:"):
                level = s.split(":", 1)[1].strip()
            elif s.lower().startswith("concepts:"):
                concepts = [c.strip() for c in s.split(":", 1)[1].split(",")]
            elif s.lower().startswith("module:"):
                module = s.split(":", 1)[1].strip()

        implementations[key] = {
            "display_name": key.replace("_", " ").title(),
            "code": code_text, "path": str(py_file),
            "level": level, "concepts": concepts, "module": module,
        }
    return implementations


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stExpander { border-radius: 8px; margin-bottom: 0.5rem; }
    .paradigm-badge {
        display: inline-block;
        background: linear-gradient(90deg, #0f3460, #e94560);
        color: white;
        border-radius: 6px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .concept-tag {
        display: inline-block;
        background: #0f3460;
        color: #53d8fb;
        border: 1px solid #53d8fb;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .learning-path-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        font-family: monospace;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 2;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def switch_view(view: str):
    st.session_state.main_view = view


def run_code_subprocess(code: str, timeout: int = 30) -> dict:
    # Prepend UTF-8 declaration + matplotlib interceptor so figures are
    # captured in memory instead of saved to disk.
    full_code = "# -*- coding: utf-8 -*-\n" + _MATPLOTLIB_INTERCEPTOR + "\n" + code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8"
        )
        # Strip figure tokens from stdout and decode each figure
        raw_stdout = result.stdout
        figures_b64 = _FIG_PATTERN.findall(raw_stdout)
        clean_stdout = _FIG_PATTERN.sub("", raw_stdout).strip()
        return {
            "success": result.returncode == 0,
            "stdout":  clean_stdout,
            "stderr":  result.stderr,
            "figures": figures_b64,          # list of base64-encoded PNGs
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "",
                "stderr": f"⏱️ Execution timed out ({timeout}s)", "figures": []}
    finally:
        os.unlink(tmp_path)


def render_operation(op_name: str, op_data: dict, key_prefix: str = ""):
    st.markdown(f"**{op_data.get('description', '')}**")
    code = op_data.get("code", "")
    timeout = op_data.get("timeout", 30)
    if code:
        st.code(code, language="python")
        run_key = f"run_{key_prefix}"
        if st.button("▶️ Run", key=f"btn_{run_key}"):
            with st.spinner("Running..."):
                st.session_state[run_key] = run_code_subprocess(code, timeout=timeout)
        if run_key in st.session_state:
            res = st.session_state[run_key]
            st.markdown("---")
            st.markdown("#### 📤 Output")
            if res["success"]:
                st.success("✅ OK")
            else:
                st.warning("⚠️ Errors")

            if res["stdout"]:
                st.code(res["stdout"], language="text")
            if res["stderr"]:
                st.markdown("🔴 **Stderr**")
                with st.container(border=True):
                    st.code(res["stderr"], language="text")

            # Display any matplotlib figures captured from the subprocess
            if res.get("figures"):
                st.markdown("#### 📊 Figures")
                for fig_b64 in res["figures"]:
                    st.image(base64.b64decode(fig_b64), width='stretch')


# ── Session state ─────────────────────────────────────────────────────────────
if "paradigm" not in st.session_state:
    st.session_state.paradigm = list(PARADIGMS.keys())[0]
if "main_view" not in st.session_state:
    st.session_state.main_view = "topics"
if "font_size" not in st.session_state:
    st.session_state.font_size = 16
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load content for the active paradigm
active_pkg   = PARADIGMS[st.session_state.paradigm]
CONTENT      = load_topics_for(active_pkg)
TOPIC_LIST   = list(CONTENT.keys())
IMPLEMENTATIONS = load_implementations_for(active_pkg)
IMPL_KEYS    = list(IMPLEMENTATIONS.keys())

# Dynamic font-size CSS
_fs = st.session_state.font_size
st.markdown(f"""
<style>
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown td, .stMarkdown th {{
        font-size: {_fs}px !important; line-height: 1.7 !important;
    }}
    .stMarkdown h1 {{ font-size: {_fs*2.0:.0f}px !important; }}
    .stMarkdown h2 {{ font-size: {_fs*1.6:.0f}px !important; }}
    .stMarkdown h3 {{ font-size: {_fs*1.3:.0f}px !important; }}
    .stMarkdown h4 {{ font-size: {_fs*1.1:.0f}px !important; }}
    .stCodeBlock code {{ font-size: {max(_fs-2,12)}px !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🧠 Architecture of ML")
st.sidebar.markdown("*Theory · Visuals · Implementations*")
st.sidebar.markdown("---")

# ── Paradigm selector ──────────────────────────────────────────────────────────
st.sidebar.markdown("### 🗂️ Learning Paradigm")
selected_paradigm = st.sidebar.radio(
    "Choose a paradigm:",
    list(PARADIGMS.keys()),
    index=list(PARADIGMS.keys()).index(st.session_state.paradigm),
    label_visibility="collapsed",
    key="paradigm_radio",
)

if selected_paradigm != st.session_state.paradigm:
    st.session_state.paradigm   = selected_paradigm
    st.session_state.main_view  = "topics"
    st.session_state.chat_history = []
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")

# ── View selector ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.sidebar.columns(3)
with c1:
    st.button("📚 Topics",
              type="primary" if st.session_state.main_view == "topics" else "secondary",
              use_container_width=True, on_click=switch_view, args=("topics",))
with c2:
    st.button("⚙️ Code",
              type="primary" if st.session_state.main_view == "implementation" else "secondary",
              use_container_width=True, on_click=switch_view, args=("implementation",))
with c3:
    st.button("🤖 Ask AI",
              type="primary" if st.session_state.main_view == "ai_assistant" else "secondary",
              use_container_width=True, on_click=switch_view, args=("ai_assistant",))

st.sidebar.markdown("---")

# Font size
with st.sidebar.expander("🔤 Font Size", expanded=False):
    fs = st.slider("Text size", 12, 28, st.session_state.font_size, step=1,
                   format="%dpx", label_visibility="collapsed")
    st.session_state.font_size = fs

# Topic/Impl list
if st.session_state.main_view == "topics":
    st.sidebar.markdown(f"## {selected_paradigm}")
    if TOPIC_LIST:
        # Reset topic selection when paradigm changes
        topic_key = f"topic_radio_{active_pkg}"
        if topic_key not in st.session_state:
            st.session_state[topic_key] = TOPIC_LIST[0]
        st.sidebar.radio(
            "Select a topic:", TOPIC_LIST,
            label_visibility="collapsed", key=topic_key
        )
    else:
        st.sidebar.info("No topics found. Add modules to the topics/ folder.")

elif st.session_state.main_view == "implementation":
    st.sidebar.markdown("## ⚙️ Implementations")
    if IMPLEMENTATIONS:
        impl_search = st.sidebar.text_input("🔍 Search", placeholder="Filter...",
                                            key=f"impl_search_{active_pkg}")
        all_levels  = sorted(set(v["level"] for v in IMPLEMENTATIONS.values()))
        level_icons = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴", "Unknown": "⚪"}

        selected_levels = st.sidebar.multiselect(
            "Level", options=all_levels,
            format_func=lambda x: f"{level_icons.get(x,'⚪')} {x}",
            key=f"level_filter_{active_pkg}")

        filtered_keys = [
            k for k in IMPL_KEYS
            if (not impl_search or
                impl_search.lower() in IMPLEMENTATIONS[k]["display_name"].lower() or
                any(impl_search.lower() in c.lower() for c in IMPLEMENTATIONS[k].get("concepts", [])))
            and (not selected_levels or IMPLEMENTATIONS[k].get("level") in selected_levels)
        ]

        st.sidebar.caption(f"Showing {len(filtered_keys)} of {len(IMPL_KEYS)}")
        st.sidebar.markdown("---")
        impl_radio_key = f"impl_radio_{active_pkg}"
        if impl_radio_key not in st.session_state:
            st.session_state[impl_radio_key] = (filtered_keys or IMPL_KEYS)[0]
        st.sidebar.radio(
            "Select implementation:", filtered_keys or IMPL_KEYS,
            format_func=lambda k: IMPLEMENTATIONS[k]["display_name"],
            label_visibility="collapsed", key=impl_radio_key)
    else:
        st.sidebar.info("No implementations yet. Add .py files to Implementation/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — TOPICS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.main_view == "topics":
    topic_key    = f"topic_radio_{active_pkg}"
    selected_topic = st.session_state.get(topic_key, TOPIC_LIST[0] if TOPIC_LIST else None)

    if not selected_topic or not CONTENT:
        st.info("👈 Select a topic from the sidebar.")
        st.stop()

    topic_data = CONTENT[selected_topic]

    st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
    st.markdown(f"# {topic_data.get('icon','📖')} {selected_topic}")
    st.caption(topic_data.get("subtitle", ""))
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📖 Theory", "🎨 Visual Breakdown", "🔬 Step-by-Step", "📊 Complexity"])

    with tab1:
        with st.container(border=True):
            st.markdown(topic_data.get("theory", "_Theory not yet added._"),
                        unsafe_allow_html=True)

    with tab2:
        visual_tabs = topic_data.get("visual_tabs", [])
        if visual_tabs:
            # Multiple visuals: render each in its own native Streamlit sub-tab
            # so every React app gets an independent, properly-initialised iframe.
            sub_labels = [vt["label"] for vt in visual_tabs]
            sub_tabs   = st.tabs(sub_labels)
            for sub_tab, vt in zip(sub_tabs, visual_tabs):
                with sub_tab:
                    st_components.html(vt["html"], height=vt.get("height", 700), scrolling=True)
        else:
            visual_html = topic_data.get("visual_html", "")
            if visual_html:
                visual_height = topic_data.get("visual_height", 700)
                st_components.html(visual_html, height=visual_height, scrolling=True)
            else:
                st.info("🎨 Visual breakdown coming soon for this topic.")

    with tab3:
        operations = topic_data.get("operations", {})
        if not operations:
            st.info("🔬 Step-by-step implementations coming soon.")
        else:
            search = st.text_input("🔍 Search steps", placeholder="Filter...",
                                   key=f"op_search_{selected_topic}")
            filtered_ops = {k: v for k, v in operations.items()
                            if not search or search.lower() in k.lower()
                            or search.lower() in v.get("description", "").lower()}
            st.caption(f"{len(filtered_ops)} of {len(operations)} steps shown")
            st.markdown("---")
            for op_name, op_data in filtered_ops.items():
                with st.expander(f"▶️ {op_name}", expanded=False):
                    render_operation(op_name, op_data,
                                     key_prefix=f"{selected_topic}_{op_name}")

    with tab4:
        complexity = topic_data.get("complexity", "")
        if complexity:
            with st.container(border=True):
                st.code(complexity.strip(), language=None)
        else:
            st.info("📊 Complexity breakdown coming soon for this topic.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.main_view == "implementation":
    impl_key     = f"impl_radio_{active_pkg}"
    selected_impl = st.session_state.get(impl_key)

    if not selected_impl or not IMPLEMENTATIONS:
        st.info("👈 Select an implementation from the sidebar.")
        st.stop()

    impl = IMPLEMENTATIONS[selected_impl]
    level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}

    st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
    st.markdown(f"## ⚙️ {impl['display_name']}")
    st.caption(
        f"{level_colors.get(impl['level'],'⚪')} **{impl['level']}** &nbsp;|&nbsp; "
        f"Module: `{impl.get('module','General')}`"
    )

    if impl.get("concepts"):
        tags = " ".join(f'<span class="concept-tag">{c}</span>' for c in impl["concepts"])
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown("---")
    tab_code, tab_run = st.tabs(["📄 Code", "▶️ Run"])

    with tab_code:
        st.code(impl["code"], language="python")

    with tab_run:
        run_key = f"impl_run_{active_pkg}_{selected_impl}"
        if st.button("▶️ Run Implementation", type="primary", key=f"btn_{run_key}"):
            with st.spinner("⏳ Running..."):
                st.session_state[run_key] = run_code_subprocess(impl["code"])

        if run_key in st.session_state:
            res = st.session_state[run_key]
            st.success("✅ Completed") if res["success"] else st.warning("⚠️ Errors")
            if res["stdout"]:
                st.code(res["stdout"], language="text")
            if res["stderr"]:
                st.markdown("🔴 **Stderr**")
                with st.container(border=True):
                    st.code(res["stderr"], language="text")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.main_view == "ai_assistant":
    st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
    st.markdown("## 🤖 ML Learning Assistant")
    st.markdown(f"*Ask questions about any **{selected_paradigm}** concept covered in this hub.*")
    st.markdown("---")
    st.info("🔧 Wire up your LLM module here (same pattern as Architecture of Intelligence).")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Ask a {selected_paradigm} question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "🔧 AI Assistant not yet connected. Add your LLM_module.py to activate."
        })
        st.rerun()





# """
# The Architecture of Machine Learning
# ======================================
# A multi-paradigm Streamlit reference hub — theory, visual breakdowns,
# and runnable implementations across all major ML paradigms.
#
# Paradigms:
#   - Generative AI        → topics in generative_ai/topics/
#   - Supervised Learning  → topics in supervised/topics/
#   - Unsupervised Learning→ topics in unsupervised/topics/
#   - Reinforcement Learning→ topics in reinforcement_learning/topics/
#
# Each paradigm shares the same 3-panel structure:
#   Topics (Theory + Visual + Step-by-Step) | Code Runner | AI Assistant
# """
#
# import os
# import sys
# import subprocess
# import tempfile
# from pathlib import Path
# from typing import Dict, Optional
#
# import streamlit as st
# import streamlit.components.v1 as st_components
#
# # ── Page config ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="The Architecture of ML",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
#
# st.markdown("""
#     <style>
#     html, body, [class*="css"] { font-family: "Times New Roman", Times, serif; }
#     </style>
# """, unsafe_allow_html=True)
#
# # ── Paradigm registry ──────────────────────────────────────────────────────────
# PARADIGMS = {
#     "Automation"                                 :                                   "Automation",
#     "ML Foundation - Core Math"                  :                   "00_ML_Foundation_Math_Core",
#     "ML Foundation - Probability_&_Information"  :  "01__ML_Foundation_Probability_&_Information",
#     "ML Foundation - Data"                       :                       "02__ML_Foundation_Data",
#     "ML Foundation - Model Behaviour"            :            "03__ML_Foundation_Model Behaviour",
#     "ML Foundation - Learning & Optimization"    :    "04__ML_Foundation_Learning & Optimization",
#     "ML Foundation - Evaluation & Generalization":"05__ML_Foundation_Evaluation & Generalization",
#     "ML Foundation - Advanced Foundations"       :       "06__ML_Foundation_Advanced Foundations",
#     "AIML Training Core"                         :                             "07_Training_Core",
#     "AIML Interpretability"                      :                          "08_Interpretability",
#     "AIML Frameworks"                            :                                "09_Frameworks",
#     "Compiler Runtime"                           :                        "10_compilers_runtimes",
#     "GPU Kernels"                                :                               "11_gpu_kernels",
#     "Training LLM"                               :                              "12_LLM_Training",
#     "Supervised Learning"                        :                                   "supervised",
#     "Unsupervised Learning"                      :                                 "unsupervised",
#     "Deep Learning"                              :                                "deep_learning",
#     "Generative AI"                              :                                "generative_ai",
#     "Reinforcement Learning"                     :                       "reinforcement_learning",
# }
#
# # ── Dynamic topic loader ───────────────────────────────────────────────────────
# @st.cache_resource
# def load_topics_for(paradigm_key: str) -> dict:
#     """Dynamically import and return all topics for the given paradigm package.
#
#     Uses @st.cache_resource (not @st.cache_data) so the result is stored in
#     memory rather than pickled to disk.  Pickling fails when the returned dict
#     contains function objects (e.g. 'render_operations') because pickle tries
#     to serialise them as references to their owning module
#     (e.g. 'deep_learning.topics.01_…') which isn't on sys.modules yet during
#     the next startup → KeyError: 'deep_learning.topics'.
#     """
#     base_dir = Path(__file__).parent
#     pkg_dir  = base_dir / paradigm_key
#     if not pkg_dir.exists():
#         return {}
#
#     # Make sure the project root is on sys.path so package imports resolve.
#     parent = str(base_dir)
#     if parent not in sys.path:
#         sys.path.insert(0, parent)
#
#     try:
#         import importlib
#         # Reload the sub-package each time so stale cached modules don't hide
#         # newly added topic files during development.
#         mod_name = f"{paradigm_key}.topics"
#         pkg = importlib.import_module(mod_name)
#         importlib.reload(pkg)
#         raw = pkg.get_all_topics()
#
#         # Strip non-serialisable callables so the dict is safe if the cache
#         # backend ever switches back to pickle-based storage.
#         clean: dict = {}
#         for topic, data in raw.items():
#             clean[topic] = {k: v for k, v in data.items()
#                             if not callable(v)}
#         return clean
#     except Exception as e:
#         st.error(f"Could not load topics for '{paradigm_key}': {e}")
#         return {}
#
#
# @st.cache_resource
# def load_implementations_for(paradigm_key: str) -> dict:
#     """Load all implementation .py files from paradigm's Implementation folder."""
#     base_dir = Path(__file__).parent
#     impl_dir = base_dir / paradigm_key / "Implementation"
#
#     if not impl_dir.exists():
#         return {}
#
#     implementations = {}
#     for py_file in sorted(impl_dir.rglob("*.py")):
#         if py_file.name.startswith("_"):
#             continue
#         try:
#             code_text = py_file.read_text(encoding="utf-8")
#         except Exception:
#             continue
#
#         key = py_file.stem
#         level, concepts, module = "Unknown", [], "General"
#
#         for line in code_text.split("\n"):
#             s = line.strip()
#             if s.lower().startswith("level:"):
#                 level = s.split(":", 1)[1].strip()
#             elif s.lower().startswith("concepts:"):
#                 concepts = [c.strip() for c in s.split(":", 1)[1].split(",")]
#             elif s.lower().startswith("module:"):
#                 module = s.split(":", 1)[1].strip()
#
#         implementations[key] = {
#             "display_name": key.replace("_", " ").title(),
#             "code": code_text, "path": str(py_file),
#             "level": level, "concepts": concepts, "module": module,
#         }
#     return implementations
#
#
# # ── CSS ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
#     .stExpander { border-radius: 8px; margin-bottom: 0.5rem; }
#     .paradigm-badge {
#         display: inline-block;
#         background: linear-gradient(90deg, #0f3460, #e94560);
#         color: white;
#         border-radius: 6px;
#         padding: 4px 14px;
#         font-size: 0.85rem;
#         font-weight: bold;
#         margin-bottom: 0.5rem;
#     }
#     .concept-tag {
#         display: inline-block;
#         background: #0f3460;
#         color: #53d8fb;
#         border: 1px solid #53d8fb;
#         border-radius: 20px;
#         padding: 2px 10px;
#         font-size: 0.75rem;
#         margin: 2px;
#     }
#     .learning-path-box {
#         background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
#         border: 1px solid #e94560;
#         border-radius: 12px;
#         padding: 1.5rem 2rem;
#         margin: 1rem 0;
#         font-family: monospace;
#         color: #e0e0e0;
#         font-size: 0.95rem;
#         line-height: 2;
#     }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ── Helpers ───────────────────────────────────────────────────────────────────
# def switch_view(view: str):
#     st.session_state.main_view = view
#
#
# def run_code_subprocess(code: str, timeout: int = 30) -> dict:
#     # Always prepend the UTF-8 coding declaration so the subprocess Python
#     # process reads the temp file as UTF-8 on Windows (which otherwise
#     # defaults to cp1252, breaking any Unicode characters in the code).
#     utf8_code = "# -*- coding: utf-8 -*-\n" + code
#     with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
#         f.write(utf8_code)
#         tmp_path = f.name
#     try:
#         result = subprocess.run(
#             [sys.executable, tmp_path],
#             capture_output=True, text=True, timeout=timeout,
#             encoding="utf-8"
#         )
#         return {"success": result.returncode == 0,
#                 "stdout": result.stdout, "stderr": result.stderr}
#     except subprocess.TimeoutExpired:
#         return {"success": False, "stdout": "",
#                 "stderr": f"⏱️ Execution timed out ({timeout}s)"}
#     finally:
#         os.unlink(tmp_path)
#
#
# def render_operation(op_name: str, op_data: dict, key_prefix: str = ""):
#     st.markdown(f"**{op_data.get('description', '')}**")
#     code = op_data.get("code", "")
#     timeout = op_data.get("timeout", 30)
#     if code:
#         st.code(code, language="python")
#         run_key = f"run_{key_prefix}"
#         if st.button("▶️ Run", key=f"btn_{run_key}"):
#             with st.spinner("Running..."):
#                 st.session_state[run_key] = run_code_subprocess(code, timeout=timeout)
#         if run_key in st.session_state:
#             res = st.session_state[run_key]
#             st.markdown("---")
#             st.markdown("#### 📤 Output")
#             if res["success"]:
#                 st.success("✅ OK")
#             else:
#                 st.warning("⚠️ Errors")
#
#             if res["stdout"]:
#                 st.code(res["stdout"], language="text")
#             if res["stderr"]:
#                 st.markdown("🔴 **Stderr**")
#                 with st.container(border=True):
#                     st.code(res["stderr"], language="text")
#
#
# # ── Session state ─────────────────────────────────────────────────────────────
# if "paradigm" not in st.session_state:
#     st.session_state.paradigm = list(PARADIGMS.keys())[0]
# if "main_view" not in st.session_state:
#     st.session_state.main_view = "topics"
# if "font_size" not in st.session_state:
#     st.session_state.font_size = 16
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Load content for the active paradigm
# active_pkg   = PARADIGMS[st.session_state.paradigm]
# CONTENT      = load_topics_for(active_pkg)
# TOPIC_LIST   = list(CONTENT.keys())
# IMPLEMENTATIONS = load_implementations_for(active_pkg)
# IMPL_KEYS    = list(IMPLEMENTATIONS.keys())
#
# # Dynamic font-size CSS
# _fs = st.session_state.font_size
# st.markdown(f"""
# <style>
#     .stMarkdown, .stMarkdown p, .stMarkdown li,
#     .stMarkdown td, .stMarkdown th {{
#         font-size: {_fs}px !important; line-height: 1.7 !important;
#     }}
#     .stMarkdown h1 {{ font-size: {_fs*2.0:.0f}px !important; }}
#     .stMarkdown h2 {{ font-size: {_fs*1.6:.0f}px !important; }}
#     .stMarkdown h3 {{ font-size: {_fs*1.3:.0f}px !important; }}
#     .stMarkdown h4 {{ font-size: {_fs*1.1:.0f}px !important; }}
#     .stCodeBlock code {{ font-size: {max(_fs-2,12)}px !important; }}
# </style>
# """, unsafe_allow_html=True)
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# st.sidebar.markdown("## 🧠 Architecture of ML")
# st.sidebar.markdown("*Theory · Visuals · Implementations*")
# st.sidebar.markdown("---")
#
# # ── Paradigm selector ──────────────────────────────────────────────────────────
# st.sidebar.markdown("### 🗂️ Learning Paradigm")
# selected_paradigm = st.sidebar.radio(
#     "Choose a paradigm:",
#     list(PARADIGMS.keys()),
#     index=list(PARADIGMS.keys()).index(st.session_state.paradigm),
#     label_visibility="collapsed",
#     key="paradigm_radio",
# )
#
# if selected_paradigm != st.session_state.paradigm:
#     st.session_state.paradigm   = selected_paradigm
#     st.session_state.main_view  = "topics"
#     st.session_state.chat_history = []
#     st.cache_resource.clear()
#     st.rerun()
#
# st.sidebar.markdown("---")
#
# # ── View selector ──────────────────────────────────────────────────────────────
# c1, c2, c3 = st.sidebar.columns(3)
# with c1:
#     st.button("📚 Topics",
#               type="primary" if st.session_state.main_view == "topics" else "secondary",
#               use_container_width=True, on_click=switch_view, args=("topics",))
# with c2:
#     st.button("⚙️ Code",
#               type="primary" if st.session_state.main_view == "implementation" else "secondary",
#               use_container_width=True, on_click=switch_view, args=("implementation",))
# with c3:
#     st.button("🤖 Ask AI",
#               type="primary" if st.session_state.main_view == "ai_assistant" else "secondary",
#               use_container_width=True, on_click=switch_view, args=("ai_assistant",))
#
# st.sidebar.markdown("---")
#
# # Font size
# with st.sidebar.expander("🔤 Font Size", expanded=False):
#     fs = st.slider("Text size", 12, 28, st.session_state.font_size, step=1,
#                    format="%dpx", label_visibility="collapsed")
#     st.session_state.font_size = fs
#
# # Topic/Impl list
# if st.session_state.main_view == "topics":
#     st.sidebar.markdown(f"## {selected_paradigm}")
#     if TOPIC_LIST:
#         # Reset topic selection when paradigm changes
#         topic_key = f"topic_radio_{active_pkg}"
#         if topic_key not in st.session_state:
#             st.session_state[topic_key] = TOPIC_LIST[0]
#         st.sidebar.radio(
#             "Select a topic:", TOPIC_LIST,
#             label_visibility="collapsed", key=topic_key
#         )
#     else:
#         st.sidebar.info("No topics found. Add modules to the topics/ folder.")
#
# elif st.session_state.main_view == "implementation":
#     st.sidebar.markdown("## ⚙️ Implementations")
#     if IMPLEMENTATIONS:
#         impl_search = st.sidebar.text_input("🔍 Search", placeholder="Filter...",
#                                             key=f"impl_search_{active_pkg}")
#         all_levels  = sorted(set(v["level"] for v in IMPLEMENTATIONS.values()))
#         level_icons = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴", "Unknown": "⚪"}
#
#         selected_levels = st.sidebar.multiselect(
#             "Level", options=all_levels,
#             format_func=lambda x: f"{level_icons.get(x,'⚪')} {x}",
#             key=f"level_filter_{active_pkg}")
#
#         filtered_keys = [
#             k for k in IMPL_KEYS
#             if (not impl_search or
#                 impl_search.lower() in IMPLEMENTATIONS[k]["display_name"].lower() or
#                 any(impl_search.lower() in c.lower() for c in IMPLEMENTATIONS[k].get("concepts", [])))
#             and (not selected_levels or IMPLEMENTATIONS[k].get("level") in selected_levels)
#         ]
#
#         st.sidebar.caption(f"Showing {len(filtered_keys)} of {len(IMPL_KEYS)}")
#         st.sidebar.markdown("---")
#         impl_radio_key = f"impl_radio_{active_pkg}"
#         if impl_radio_key not in st.session_state:
#             st.session_state[impl_radio_key] = (filtered_keys or IMPL_KEYS)[0]
#         st.sidebar.radio(
#             "Select implementation:", filtered_keys or IMPL_KEYS,
#             format_func=lambda k: IMPLEMENTATIONS[k]["display_name"],
#             label_visibility="collapsed", key=impl_radio_key)
#     else:
#         st.sidebar.info("No implementations yet. Add .py files to Implementation/")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN AREA — TOPICS
# # ══════════════════════════════════════════════════════════════════════════════
# if st.session_state.main_view == "topics":
#     topic_key    = f"topic_radio_{active_pkg}"
#     selected_topic = st.session_state.get(topic_key, TOPIC_LIST[0] if TOPIC_LIST else None)
#
#     if not selected_topic or not CONTENT:
#         st.info("👈 Select a topic from the sidebar.")
#         st.stop()
#
#     topic_data = CONTENT[selected_topic]
#
#     st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
#     st.markdown(f"# {topic_data.get('icon','📖')} {selected_topic}")
#     st.caption(topic_data.get("subtitle", ""))
#     st.markdown("---")
#
#     tab1, tab2, tab3, tab4 = st.tabs(["📖 Theory", "🎨 Visual Breakdown", "🔬 Step-by-Step", "📊 Complexity"])
#
#     with tab1:
#         with st.container(border=True):
#             st.markdown(topic_data.get("theory", "_Theory not yet added._"),
#                         unsafe_allow_html=True)
#
#     with tab2:
#         visual_tabs = topic_data.get("visual_tabs", [])
#         if visual_tabs:
#             # Multiple visuals: render each in its own native Streamlit sub-tab
#             # so every React app gets an independent, properly-initialised iframe.
#             sub_labels = [vt["label"] for vt in visual_tabs]
#             sub_tabs   = st.tabs(sub_labels)
#             for sub_tab, vt in zip(sub_tabs, visual_tabs):
#                 with sub_tab:
#                     st_components.html(vt["html"], height=vt.get("height", 700), scrolling=True)
#         else:
#             visual_html = topic_data.get("visual_html", "")
#             if visual_html:
#                 visual_height = topic_data.get("visual_height", 700)
#                 st_components.html(visual_html, height=visual_height, scrolling=True)
#             else:
#                 st.info("🎨 Visual breakdown coming soon for this topic.")
#
#     with tab3:
#         operations = topic_data.get("operations", {})
#         if not operations:
#             st.info("🔬 Step-by-step implementations coming soon.")
#         else:
#             search = st.text_input("🔍 Search steps", placeholder="Filter...",
#                                    key=f"op_search_{selected_topic}")
#             filtered_ops = {k: v for k, v in operations.items()
#                             if not search or search.lower() in k.lower()
#                             or search.lower() in v.get("description", "").lower()}
#             st.caption(f"{len(filtered_ops)} of {len(operations)} steps shown")
#             st.markdown("---")
#             for op_name, op_data in filtered_ops.items():
#                 with st.expander(f"▶️ {op_name}", expanded=False):
#                     render_operation(op_name, op_data,
#                                      key_prefix=f"{selected_topic}_{op_name}")
#
#     with tab4:
#         complexity = topic_data.get("complexity", "")
#         if complexity:
#             with st.container(border=True):
#                 st.code(complexity.strip(), language=None)
#         else:
#             st.info("📊 Complexity breakdown coming soon for this topic.")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN AREA — IMPLEMENTATION
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.main_view == "implementation":
#     impl_key     = f"impl_radio_{active_pkg}"
#     selected_impl = st.session_state.get(impl_key)
#
#     if not selected_impl or not IMPLEMENTATIONS:
#         st.info("👈 Select an implementation from the sidebar.")
#         st.stop()
#
#     impl = IMPLEMENTATIONS[selected_impl]
#     level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
#
#     st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
#     st.markdown(f"## ⚙️ {impl['display_name']}")
#     st.caption(
#         f"{level_colors.get(impl['level'],'⚪')} **{impl['level']}** &nbsp;|&nbsp; "
#         f"Module: `{impl.get('module','General')}`"
#     )
#
#     if impl.get("concepts"):
#         tags = " ".join(f'<span class="concept-tag">{c}</span>' for c in impl["concepts"])
#         st.markdown(tags, unsafe_allow_html=True)
#
#     st.markdown("---")
#     tab_code, tab_run = st.tabs(["📄 Code", "▶️ Run"])
#
#     with tab_code:
#         st.code(impl["code"], language="python")
#
#     with tab_run:
#         run_key = f"impl_run_{active_pkg}_{selected_impl}"
#         if st.button("▶️ Run Implementation", type="primary", key=f"btn_{run_key}"):
#             with st.spinner("⏳ Running..."):
#                 st.session_state[run_key] = run_code_subprocess(impl["code"])
#
#         if run_key in st.session_state:
#             res = st.session_state[run_key]
#             st.success("✅ Completed") if res["success"] else st.warning("⚠️ Errors")
#             if res["stdout"]:
#                 st.code(res["stdout"], language="text")
#             if res["stderr"]:
#                 st.markdown("🔴 **Stderr**")
#                 with st.container(border=True):
#                     st.code(res["stderr"], language="text")
#
#
# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN AREA — AI ASSISTANT
# # ══════════════════════════════════════════════════════════════════════════════
# elif st.session_state.main_view == "ai_assistant":
#     st.markdown(f'<span class="paradigm-badge">{selected_paradigm}</span>', unsafe_allow_html=True)
#     st.markdown("## 🤖 ML Learning Assistant")
#     st.markdown(f"*Ask questions about any **{selected_paradigm}** concept covered in this hub.*")
#     st.markdown("---")
#     st.info("🔧 Wire up your LLM module here (same pattern as Architecture of Intelligence).")
#
#     for msg in st.session_state.chat_history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])
#
#     user_input = st.chat_input(f"Ask a {selected_paradigm} question...")
#     if user_input:
#         st.session_state.chat_history.append({"role": "user", "content": user_input})
#         st.session_state.chat_history.append({
#             "role": "assistant",
#             "content": "🔧 AI Assistant not yet connected. Add your LLM_module.py to activate."
#         })
#         st.rerun()
#
#