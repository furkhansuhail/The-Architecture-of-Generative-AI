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
        module_name = f"05_ML_Foundation_Learning_&_Optimization.topics.{py_file.stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            import warnings
            warnings.warn(f"[05_ML_Foundation_Learning_&_Optimization.topics] Could not import '{module_name}': {e}", stacklevel=2)
            continue
        if not hasattr(mod, "get_content"):
            continue
        try:
            data = mod.get_content()
        except Exception as e:
            import warnings
            warnings.warn(f"[05_ML_Foundation_Learning_&_Optimization.topics] get_content() failed in '{module_name}': {e}", stacklevel=2)
            continue
        key = data.get("display_name", py_file.stem)
        topics[key] = data
    return topics
