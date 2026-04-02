# deep_learning/topics/__init__.py
"""
Topic registry for the Deep Learning paradigm.
Auto-discovers every .py file in this folder and exposes get_all_topics().
"""
import importlib
from pathlib import Path


def get_all_topics() -> dict:
    topics: dict = {}
    pkg_dir = Path(__file__).parent
    for py_file in sorted(pkg_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"Unsupervised_Learning.topics.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            import warnings
            warnings.warn(f"[Unsupervised_Learning.topics] Could not import '{module_name}': {e}", stacklevel=2)
            continue
        if not hasattr(mod, "get_content"):
            continue
        try:
            data = mod.get_content()
        except Exception as e:
            import warnings
            warnings.warn(f"[Unsupervised_Learning.topics] get_content() failed in '{module_name}': {e}", stacklevel=2)
            continue
        key = data.get("display_name", py_file.stem)
        topics[key] = data
    return topics



# """Auto-discovers all topic modules in the unsupervised package."""
# import importlib, pkgutil
# from pathlib import Path
#
# def get_all_topics() -> dict:
#     topics = {}
#     package_dir = Path(__file__).parent
#     for finder, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
#         if module_name.startswith("_"):
#             continue
#         try:
#             mod = importlib.import_module(f"unsupervised.topics.{module_name}")
#             display = getattr(mod, "DISPLAY_NAME", module_name.replace("_", " ").title())
#             topics[display] = {
#                 "icon":        getattr(mod, "ICON",        "📖"),
#                 "subtitle":    getattr(mod, "SUBTITLE",    ""),
#                 "theory":      getattr(mod, "THEORY",      "_Coming soon._"),
#                 "operations":  getattr(mod, "OPERATIONS",  {}),
#                 "visual_html": getattr(mod, "VISUAL_HTML", ""),
#             }
#         except Exception as e:
#             print(f"[WARN] Could not load {module_name}: {e}")
#     return topics
