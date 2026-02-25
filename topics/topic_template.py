"""
TOPIC TEMPLATE
==============
Copy this file and rename it:  NN_topic_name.py
Fill in each section. The app will auto-discover this module.

Naming convention:
  01_tokenization_embeddings.py
  02_language_modeling.py
  ...
"""

# ── Display name (shown in sidebar and as page title) ──────────────────────
TOPIC_NAME   = "Topic Name"
DISPLAY_NAME = "Topic Name"
ICON         = "📖"
SUBTITLE     = "One-line description of what this topic covers"

# ── Theory ─────────────────────────────────────────────────────────────────
THEORY = """
## Overview
Write a thorough explanation of the concept here. Use markdown freely.

## Why It Matters
Explain the motivation and context.

## Core Concepts
Break down the individual ideas.

### Subtopic 1
...

### Subtopic 2
...

## Mathematical Foundation
Use LaTeX if needed:

$$y = f(x)$$

## Key Takeaways
- Bullet 1
- Bullet 2
- Bullet 3
"""

# ── Visual HTML ─────────────────────────────────────────────────────────────
# Return a full HTML string. It will be rendered in an iframe.
# Import from Required_Images/ or write inline.
VISUAL_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { background: #0d0d0d; color: #e0e0e0; font-family: monospace;
         display: flex; justify-content: center; align-items: center;
         height: 100vh; margin: 0; }
  .placeholder { text-align: center; opacity: 0.5; }
  .placeholder h2 { font-size: 2rem; }
</style>
</head>
<body>
<div class="placeholder">
  <h2>📐 Visual Breakdown</h2>
  <p>HTML diagram coming soon for this topic.</p>
</div>
</body>
</html>
"""

# ── Step-by-Step Operations ─────────────────────────────────────────────────
OPERATIONS = {
    "Step 1: Example": {
        "description": "Brief description of what this step demonstrates.",
        "language": "python",
        "code": """
# Example code — runnable in subprocess
print("Hello from Step 1!")
""".strip(),
    },
    "Step 2: Example": {
        "description": "Next step description.",
        "language": "python",
        "code": """
print("Hello from Step 2!")
""".strip(),
    },
}

# ── Entry point called by topics/__init__.py ────────────────────────────────
def get_topic_data() -> dict:
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  VISUAL_HTML,
        "operations":   OPERATIONS,
    }
