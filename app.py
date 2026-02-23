"""
The Architecture of Generative AI
===================================
A Streamlit reference hub explaining Generative AI concepts — theory,
visual breakdowns, and step-by-step implementations.

Structure:
- app.py              → Main Streamlit application (this file)
- topics/             → Package containing topic modules (auto-discovered)
- Implementation/     → Runnable implementation .py files
- Required_Images/    → HTML visual breakdown files (rendered as iframes)
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as st_components
from topics import get_all_topics

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="The Architecture of Generative AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: "Times New Roman", Times, serif;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD TOPICS
# =============================================================================

@st.cache_data
def load_topics():
    """Load all topics from the topics package."""
    return get_all_topics()


@st.cache_data
def load_implementations():
    """
    Load all implementation .py files from the Implementation folder.

    Expected file format with metadata in docstring:
    ```
    \"\"\"
    RAG Pipeline: Naive to Advanced
    Level: Intermediate
    Concepts: RAG, Embeddings, Vector Store
    Module: 08_RAG
    \"\"\"
    ```

    Returns:
        dict: {key: {"display_name": str, "code": str, "path": str,
                      "level": str, "concepts": list, "module": str}}
    """
    base_dir = Path(__file__).parent
    impl_dir = base_dir / "Implementation"

    if not impl_dir.exists() or not impl_dir.is_dir():
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
        display_name = key.replace("_", " ").title()
        level = "Unknown"
        concepts = []
        module = "General"

        for line in code_text.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("level:"):
                level = line_stripped.split(":", 1)[1].strip()
                level_lower = level.lower()
                if "beginner" in level_lower:
                    level = "Beginner"
                elif "intermediate" in level_lower:
                    level = "Intermediate"
                elif "advanced" in level_lower:
                    level = "Advanced"
            elif line_stripped.lower().startswith("concepts:"):
                concepts_str = line_stripped.split(":", 1)[1].strip()
                concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
            elif line_stripped.lower().startswith("module:"):
                module = line_stripped.split(":", 1)[1].strip()

        implementations[key] = {
            "display_name": display_name,
            "code": code_text,
            "path": str(py_file),
            "level": level,
            "concepts": concepts,
            "module": module,
        }

    return implementations


@st.cache_data
def load_visuals():
    """
    Load all HTML visual files from Required_Images folder.
    These are Python scripts that return HTML strings.
    """
    base_dir = Path(__file__).parent
    visuals_dir = base_dir / "Required_Images"

    if not visuals_dir.exists():
        return {}

    visuals = {}
    for py_file in sorted(visuals_dir.glob("*_visual.py")):
        key = py_file.stem.replace("_visual", "")
        visuals[key] = str(py_file)

    return visuals


# Load all content
CONTENT       = load_topics()
TOPIC_LIST    = list(CONTENT.keys()) if CONTENT else []
IMPLEMENTATIONS = load_implementations()
IMPL_KEYS     = list(IMPLEMENTATIONS.keys()) if IMPLEMENTATIONS else []
VISUALS       = load_visuals()

# =============================================================================
# SESSION STATE
# =============================================================================

defaults = {
    "main_view": "topics",       # "topics" | "implementation" | "ai_assistant"
    "topic_radio": TOPIC_LIST[0] if TOPIC_LIST else None,
    "impl_key": IMPL_KEYS[0] if IMPL_KEYS else None,
    "font_size": 16,
    "chat_history": [],
    "highlighted_operation": None,
    "show_operation_panel": False,
    "target_topic": None,
    "pending_query": None,
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .stExpander { border-radius: 8px; margin-bottom: 0.5rem; }
    .stCodeBlock { border-radius: 8px; }
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
    .module-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 6px;
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
</style>
""", unsafe_allow_html=True)

# Dynamic font-size CSS
_fs = st.session_state.font_size
st.markdown(f"""
<style>
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stMarkdown td, .stMarkdown th {{
        font-size: {_fs}px !important;
        line-height: 1.7 !important;
    }}
    .stMarkdown h1 {{ font-size: {_fs * 2.0:.0f}px !important; }}
    .stMarkdown h2 {{ font-size: {_fs * 1.6:.0f}px !important; }}
    .stMarkdown h3 {{ font-size: {_fs * 1.3:.0f}px !important; }}
    .stMarkdown h4 {{ font-size: {_fs * 1.1:.0f}px !important; }}
    .stCodeBlock, .stCodeBlock code {{ font-size: {max(_fs - 2, 12)}px !important; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CALLBACKS & HELPERS
# =============================================================================

def switch_view(view: str):
    st.session_state.main_view = view

def clear_chat_history():
    st.session_state.chat_history = []

def run_code_subprocess(code_string: str, timeout: int = 60) -> dict:
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        tmp_file.write(code_string)
        tmp_file.flush()
        tmp_file.close()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", env=env,
        )
        return {"success": result.returncode == 0,
                "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "",
                "stderr": f"⏱️ Timed out after {timeout}s."}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}
    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


def render_operation(op_name: str, op_data: dict, key_prefix: str = "op"):
    lang = op_data.get("language", "python")
    st.markdown(f"**{op_data.get('description', '')}**")
    st.markdown("---")
    st.code(op_data["code"], language=lang)

    safe_key  = f"{key_prefix}_{op_name}".replace(" ", "_").replace(":", "_")
    result_key = f"run_result_{safe_key}"

    col_run, col_clear, _ = st.columns([1, 1, 4])
    with col_run:
        run_clicked = st.button("▶️ Run", key=f"run_{safe_key}",
                                type="primary", use_container_width=True)
    with col_clear:
        if result_key in st.session_state:
            if st.button("🗑️ Clear", key=f"clear_{safe_key}", use_container_width=True):
                del st.session_state[result_key]
                st.rerun()

    if run_clicked:
        with st.spinner("⏳ Running..."):
            st.session_state[result_key] = run_code_subprocess(op_data["code"])

    if result_key in st.session_state:
        res = st.session_state[result_key]
        st.markdown("---")
        st.markdown("#### 📤 Output")
        if res["success"]:
            st.success("✅ Completed successfully")
        else:
            st.warning("⚠️ Finished with errors")
        if res["stdout"]:
            st.code(res["stdout"], language="text")
        if res["stderr"]:
            with st.expander("🔴 Stderr", expanded=not res["success"]):
                st.code(res["stderr"], language="text")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("## 🤖 Architecture of GenAI")
st.sidebar.markdown("*From Tokens to Multi-Agent Systems*")
st.sidebar.markdown("---")

# Main section buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    st.button("📚 Topics",
              type="primary" if st.session_state.main_view == "topics" else "secondary",
              use_container_width=True,
              on_click=switch_view, args=("topics",))
with col2:
    st.button("⚙️ Code",
              type="primary" if st.session_state.main_view == "implementation" else "secondary",
              use_container_width=True,
              on_click=switch_view, args=("implementation",))
with col3:
    st.button("🤖 Ask AI",
              type="primary" if st.session_state.main_view == "ai_assistant" else "secondary",
              use_container_width=True,
              on_click=switch_view, args=("ai_assistant",))

st.sidebar.markdown("---")

# Font size
with st.sidebar.expander("🔤 Font Size", expanded=False):
    fs = st.slider("Text size", 12, 28, st.session_state.font_size, step=1,
                   format="%dpx", label_visibility="collapsed")
    st.session_state.font_size = fs

# Topic list in sidebar
if st.session_state.main_view == "topics":
    st.sidebar.markdown("## 📚 Gen AI Topics")
    if CONTENT:
        selected_topic = st.sidebar.radio(
            "Select a topic:",
            TOPIC_LIST,
            label_visibility="collapsed",
            key="topic_radio"
        )
    else:
        st.sidebar.error("No topics found. Add modules to topics/")
        selected_topic = None

elif st.session_state.main_view == "implementation":
    st.sidebar.markdown("## ⚙️ Implementations")
    if IMPLEMENTATIONS:
        impl_search = st.sidebar.text_input("🔍 Search", placeholder="Filter by name or concept...",
                                            key="impl_search")

        all_levels = sorted(set(v["level"] for v in IMPLEMENTATIONS.values()))
        level_icons = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴", "Unknown": "⚪"}

        selected_levels = st.sidebar.multiselect(
            "Level", options=all_levels,
            format_func=lambda x: f"{level_icons.get(x, '⚪')} {x}",
            key="level_filter", label_visibility="visible"
        )

        filtered_keys = []
        for k in IMPL_KEYS:
            impl = IMPLEMENTATIONS[k]
            if impl_search:
                s = impl_search.lower()
                if not (s in impl["display_name"].lower() or
                        any(s in c.lower() for c in impl.get("concepts", []))):
                    continue
            if selected_levels and impl.get("level") not in selected_levels:
                continue
            filtered_keys.append(k)

        st.sidebar.caption(f"Showing {len(filtered_keys)} of {len(IMPL_KEYS)}")
        st.sidebar.markdown("---")

        selected_impl = st.sidebar.radio(
            "Select implementation:",
            filtered_keys if filtered_keys else IMPL_KEYS,
            format_func=lambda k: IMPLEMENTATIONS[k]["display_name"],
            label_visibility="collapsed",
            key="impl_radio"
        )
    else:
        st.sidebar.info("No implementations yet. Add .py files to Implementation/")
        selected_impl = None

# =============================================================================
# MAIN AREA — TOPICS VIEW
# =============================================================================

if st.session_state.main_view == "topics":
    selected_topic = st.session_state.get("topic_radio")

    if not selected_topic or not CONTENT:
        st.info("👈 Select a topic from the sidebar to get started.")
        st.stop()

    topic_data = CONTENT[selected_topic]

    # Header
    st.markdown(f"# {topic_data.get('icon', '📖')} {selected_topic}")
    st.caption(topic_data.get("subtitle", ""))
    st.markdown("---")

    # Tabs: Theory | Visual | Step-by-Step
    tab_labels = ["📖 Theory", "🎨 Visual Breakdown", "🔬 Step-by-Step"]
    tab1, tab2, tab3 = st.tabs(tab_labels)

    with tab1:
        with st.container(border=True):
            st.markdown(topic_data.get("theory", "_Theory not yet added._"),
                        unsafe_allow_html=True)

    with tab2:
        visual_html = topic_data.get("visual_html", "")
        if visual_html:
            st_components.html(visual_html, height=700, scrolling=True)
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
            st.caption(f"{len(filtered_ops)} of {len(operations)} steps")
            st.markdown("---")
            for op_name, op_data in filtered_ops.items():
                with st.expander(f"▶️ {op_name}", expanded=False):
                    render_operation(op_name, op_data,
                                     key_prefix=f"{selected_topic}_{op_name}")

# =============================================================================
# MAIN AREA — IMPLEMENTATION VIEW
# =============================================================================

elif st.session_state.main_view == "implementation":
    selected_impl = st.session_state.get("impl_radio")

    if not selected_impl or not IMPLEMENTATIONS:
        st.info("👈 Select an implementation from the sidebar.")
        st.stop()

    impl = IMPLEMENTATIONS[selected_impl]

    st.markdown(f"## ⚙️ {impl['display_name']}")
    level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.caption(
        f"{level_colors.get(impl['level'], '⚪')} **{impl['level']}** &nbsp;|&nbsp; "
        f"Module: `{impl.get('module', 'General')}`"
    )

    if impl.get("concepts"):
        concepts_html = " ".join(
            f'<span class="concept-tag">{c}</span>' for c in impl["concepts"]
        )
        st.markdown(concepts_html, unsafe_allow_html=True)

    st.markdown("---")

    tab_code, tab_run = st.tabs(["📄 Code", "▶️ Run"])

    with tab_code:
        st.code(impl["code"], language="python")

    with tab_run:
        run_key = f"impl_run_{selected_impl}"
        if st.button("▶️ Run Implementation", type="primary", key=f"btn_{run_key}"):
            with st.spinner("⏳ Running..."):
                st.session_state[run_key] = run_code_subprocess(impl["code"])

        if run_key in st.session_state:
            res = st.session_state[run_key]
            if res["success"]:
                st.success("✅ Completed")
            else:
                st.warning("⚠️ Errors occurred")
            if res["stdout"]:
                st.code(res["stdout"], language="text")
            if res["stderr"]:
                with st.expander("🔴 Stderr", expanded=not res["success"]):
                    st.code(res["stderr"], language="text")

# =============================================================================
# MAIN AREA — AI ASSISTANT VIEW
# =============================================================================

elif st.session_state.main_view == "ai_assistant":
    st.markdown("## 🤖 GenAI Learning Assistant")
    st.markdown("*Ask questions about any Gen AI concept covered in this hub.*")
    st.markdown("---")

    st.info("🔧 Connect your LLM module here (same pattern as Architecture of Intelligence).")

    # Placeholder chat UI — wire up LLM_module.py when ready
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a Gen AI question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "🔧 AI Assistant not yet wired up. Add your LLM_module.py to enable this."
        })
        st.rerun()
