"""
Topics Package
==============
Auto-discovers all topic modules in this directory.
Each module must expose a `get_topic_data()` function that returns a dict.

Expected dict structure:
{
    "display_name": str,          # shown in sidebar (falls back to filename)
    "icon":         str,          # emoji icon
    "subtitle":     str,          # one-line description shown under the title
    "theory":       str,          # markdown / HTML content for the Theory tab
    "visual_html":  str,          # full HTML string for the Visual Breakdown tab
    "operations": {               # Step-by-Step tab
        "Step Name": {
            "description": str,
            "code":        str,
            "language":    str,   # "python" | "bash" | etc.
        }
    }
}

Loading strategy
----------------
Modules are loaded using importlib.util.spec_from_file_location() rather than
importlib.import_module(). This is necessary because:

  • Python identifiers cannot start with a digit, so filenames like
    "01_tokenization_embeddings.py" would silently fail with import_module().
  • spec_from_file_location() loads by filesystem path, bypassing the
    identifier restriction entirely.

Missing optional dependencies (e.g. Required_Images visual modules) are
pre-stubbed in sys.modules so a missing visual file never prevents the topic
from loading — visual_html simply falls back to an empty string.
"""

import importlib.util
import pkgutil
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Optional-dependency stubs
# ---------------------------------------------------------------------------
# Some topic modules do a top-level import such as:
#   from Required_Images.tokenization_visual import get_visual_html
# If Required_Images is not on sys.path at load time, the module crashes
# before get_topic_data() can even be called.  We pre-register lightweight
# stubs for known visual sub-modules so those imports succeed gracefully.

def _ensure_visual_stub(submodule_name: str) -> None:
    """Insert a no-op stub into sys.modules if the real module isn't available."""
    full_name = f"Required_Images.{submodule_name}"
    if full_name in sys.modules:
        return  # real module already loaded — leave it alone
    try:
        importlib.import_module(full_name)
    except ImportError:
        # Build a minimal stub that satisfies common import patterns:
        #   from Required_Images.foo_visual import get_visual_html
        #   from Required_Images.foo_visual import VISUAL_HEIGHT, get_visual_html
        stub = types.ModuleType(full_name)
        stub.get_visual_html   = lambda: ""
        stub.VISUAL_HEIGHT     = 400
        stub.FFT_VISUAL_HTML   = ""
        stub.FFT_VISUAL_HEIGHT = 400

        # Ensure the parent package stub exists too
        parent = "Required_Images"
        if parent not in sys.modules:
            parent_stub = types.ModuleType(parent)
            sys.modules[parent] = parent_stub

        sys.modules[full_name] = stub
        setattr(sys.modules[parent], submodule_name, stub)


# Pre-stub all *_visual submodules we know about.
# Add new entries here when new topic modules are added.
_KNOWN_VISUAL_STUBS = [
    "tokenization_visual",
    "Fft_visual",
    "transformer_visual",
    "attention_visual",
    "rag_visual",
    "lora_visual",
    "diffusion_visual",
]

for _stub in _KNOWN_VISUAL_STUBS:
    _ensure_visual_stub(_stub)


# ---------------------------------------------------------------------------
# Module discovery and loading
# ---------------------------------------------------------------------------

def _load_module_from_path(path: Path):
    """
    Load a Python file as a module regardless of whether its filename is a
    valid Python identifier (e.g. "01_tokenization_embeddings.py" is fine).

    Returns the loaded module object, or raises on genuine import errors.
    """
    # Give the module a legal internal name by prefixing with '_topic_'
    safe_name = f"_topic_{path.stem}"

    spec   = importlib.util.spec_from_file_location(safe_name, path)
    module = importlib.util.module_from_spec(spec)

    # Register before exec so that any intra-package relative imports resolve
    sys.modules[safe_name] = module
    spec.loader.exec_module(module)
    return module


def get_all_topics() -> dict:
    """
    Auto-discover and load all topic modules in this package directory.

    Modules are loaded in alphabetical (filename) order — use numeric prefixes
    like 01_, 02_ to control the sidebar ordering.

    Returns a dict keyed by each topic's display_name.
    """
    topics: dict = {}
    package_dir  = Path(__file__).parent

    # Files to always exclude — not topic modules
    _EXCLUDED_STEMS = {"topic_template", "app"}

    # Collect all .py files that are not private or excluded
    candidate_paths = sorted(
        p for p in package_dir.glob("*.py")
        if not p.name.startswith("_") and p.stem not in _EXCLUDED_STEMS
    )

    for path in candidate_paths:
        stem = path.stem  # e.g. "01_tokenization_embeddings"

        try:
            module = _load_module_from_path(path)

            # Support both naming conventions:
            #   get_topic_data() — original / simple modules
            #   get_content()    — expanded modules (e.g. Full Fine-Tuning)
            if hasattr(module, "get_topic_data"):
                data = module.get_topic_data()
            elif hasattr(module, "get_content"):
                data = module.get_content()
            else:
                # Not a topic module — skip silently
                continue

            # ── Normalise to the common interface app.py expects ──────────
            # Modules using get_content() (e.g. Full Fine-Tuning) may omit keys
            # that simpler modules include.  Fill in sensible defaults from the
            # module's own constants when the key is absent.

            # display_name: declared > TOPIC_NAME constant > filename
            display_name = data.get(
                "display_name",
                getattr(module, "TOPIC_NAME",
                        stem.replace("_", " ").title())
            )

            # icon / subtitle: declared > module constant > safe default
            if "icon" not in data:
                data["icon"] = getattr(module, "ICON", "📖")
            if "subtitle" not in data:
                data["subtitle"] = getattr(module, "SUBTITLE", "")

            # visual_html: declared > first interactive_component html > empty
            if "visual_html" not in data:
                components = data.get("interactive_components") or []
                data["visual_html"] = (
                    components[0].get("html", "") if components else ""
                )

            data["display_name"] = display_name
            topics[display_name] = data

        except Exception as exc:
            # A broken module should never take down the whole app.
            # Show a visible placeholder so the developer can see what failed.
            display_name = stem.replace("_", " ").title()
            topics[display_name] = {
                "icon":        "⚠️",
                "subtitle":    f"Module failed to load: {exc}",
                "theory":      (
                    f"**Error loading `{stem}.py`**\n\n"
                    f"```\n{type(exc).__name__}: {exc}\n```\n\n"
                    f"Check that all imports inside the module are available."
                ),
                "visual_html": "",
                "operations":  {},
            }

    return topics

# """
# Topics Package
# ==============
# Auto-discovers all topic modules in this directory.
# Each module must expose a `get_topic_data()` function that returns a dict.
#
# Expected dict structure:
# {
#     "icon":      str,          # emoji icon
#     "subtitle":  str,          # one-line description shown under the title
#     "theory":    str,          # markdown / HTML content for the Theory tab
#     "visual_html": str,        # full HTML string for the Visual Breakdown tab
#     "operations": {            # Step-by-Step tab
#         "Step Name": {
#             "description": str,
#             "code":        str,
#             "language":    str,  # "python" | "bash" | etc.
#         }
#     }
# }
# """
#
# import importlib
# import pkgutil
# from pathlib import Path
#
#
# def get_all_topics() -> dict:
#     """
#     Auto-discover and load all topic modules in this package.
#     Modules are loaded in alphabetical order (use numeric prefixes to control order).
#     Returns a dict keyed by the topic display name.
#     """
#     topics = {}
#     package_dir = Path(__file__).parent
#
#     for module_info in sorted(pkgutil.iter_modules([str(package_dir)])):
#         name = module_info.name
#
#         # Skip private modules and the template
#         if name.startswith("_") or name == "topic_template":
#             continue
#
#         try:
#             module = importlib.import_module(f"topics.{name}")
#             if hasattr(module, "get_topic_data"):
#                 data = module.get_topic_data()
#                 display_name = data.get("display_name", name.replace("_", " ").title())
#                 topics[display_name] = data
#         except Exception as e:
#             # Gracefully skip broken modules and show a placeholder
#             display_name = name.replace("_", " ").title()
#             topics[display_name] = {
#                 "icon": "⚠️",
#                 "subtitle": f"Module failed to load: {e}",
#                 "theory": f"**Error loading module `{name}`:** `{e}`",
#                 "visual_html": "",
#                 "operations": {},
#             }
#
#     return topics
