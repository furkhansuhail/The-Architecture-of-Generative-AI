"""
Topics Package
==============
Auto-discovers all topic modules in this directory.
Each module must expose a `get_topic_data()` function that returns a dict.

Expected dict structure:
{
    "icon":      str,          # emoji icon
    "subtitle":  str,          # one-line description shown under the title
    "theory":    str,          # markdown / HTML content for the Theory tab
    "visual_html": str,        # full HTML string for the Visual Breakdown tab
    "operations": {            # Step-by-Step tab
        "Step Name": {
            "description": str,
            "code":        str,
            "language":    str,  # "python" | "bash" | etc.
        }
    }
}
"""

import importlib
import pkgutil
from pathlib import Path


def get_all_topics() -> dict:
    """
    Auto-discover and load all topic modules in this package.
    Modules are loaded in alphabetical order (use numeric prefixes to control order).
    Returns a dict keyed by the topic display name.
    """
    topics = {}
    package_dir = Path(__file__).parent

    for module_info in sorted(pkgutil.iter_modules([str(package_dir)])):
        name = module_info.name

        # Skip private modules and the template
        if name.startswith("_") or name == "topic_template":
            continue

        try:
            module = importlib.import_module(f"topics.{name}")
            if hasattr(module, "get_topic_data"):
                data = module.get_topic_data()
                display_name = data.get("display_name", name.replace("_", " ").title())
                topics[display_name] = data
        except Exception as e:
            # Gracefully skip broken modules and show a placeholder
            display_name = name.replace("_", " ").title()
            topics[display_name] = {
                "icon": "⚠️",
                "subtitle": f"Module failed to load: {e}",
                "theory": f"**Error loading module `{name}`:** `{e}`",
                "visual_html": "",
                "operations": {},
            }

    return topics
