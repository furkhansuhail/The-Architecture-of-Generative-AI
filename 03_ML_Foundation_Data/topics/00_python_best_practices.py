"""
Professional Python — Style, Structure, and Engineering Excellence
==================================================================

Writing Python professionally means more than making code that runs.
It means writing code that is correct, readable, maintainable, testable,
and performant — code that a colleague can understand without a guided tour,
that survives production load, and that can be changed safely six months later.

This module covers the authoritative standards and practical techniques
used by professional Python engineers: PEP 8, type annotations, documentation,
error handling, testing, performance, and software design patterns.

Topics covered:
  1. Style and Readability — PEP 8, naming, layout, and the Zen of Python
  2. Functions and Classes — signatures, SOLID principles, design patterns
  3. Robustness — exceptions, type hints, logging, and defensive coding
  4. Performance and Profiling — algorithmic choices, NumPy, caching, profilers
"""

TOPIC_NAME   = "Professional Python"
DISPLAY_NAME = "Professional Python"
ICON         = "🐍"
SUBTITLE     = "Style, Structure, Error Handling, Performance, and Engineering Best Practices"

THEORY = """
##### PART 1 — THE PROFESSIONAL MINDSET

### Code Is Read More Than Written

The primary audience for your code is not the Python interpreter — it is
the human who reads it next, which is often future-you. Donald Knuth put
it plainly: "Programs are meant to be read by humans and only incidentally
for computers to execute."

Professional Python embodies three ideals:

    CORRECT:      it does what it claims, handles edge cases, has tests
    CLEAR:        intent is obvious without a comment; names explain purpose
    CHANGEABLE:   isolated concerns, low coupling, covered by tests

### The Zen of Python

    import this

Key principles from PEP 20:
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Readability counts.
    Errors should never pass silently.
    If the implementation is hard to explain, it's a bad idea.

These are not aesthetic preferences — they are load-bearing principles.
Violating them consistently produces codebases that rot.

---

##### PART 2 — STYLE AND NAMING (PEP 8)

### Why PEP 8 Matters

PEP 8 (Python Enhancement Proposal 8) is the style guide for Python code.
It reduces cognitive load: when everyone uses the same conventions, readers
spend effort understanding logic, not decoding style.

### Naming Conventions

    Convention              Examples                    Used for
    ─────────────────────────────────────────────────────────────────────
    snake_case              user_count, load_data()     variables, functions, methods
    PascalCase              UserAccount, HttpClient     classes
    UPPER_SNAKE_CASE        MAX_RETRIES, BASE_URL       module-level constants
    _single_leading         _internal_helper            "private" by convention
    __double_leading        __slots__, __init__         name-mangling (rare)
    _single_trailing        class_, type_               avoid keyword clash

### Layout Rules

    Indentation:    4 spaces (never tabs)
    Line length:    88 characters (Black default) or 79 (strict PEP 8)
    Blank lines:    2 between top-level definitions, 1 between methods
    Imports:        one per line, grouped: stdlib → third-party → local

    # WRONG
    import os, sys
    from os import *
    import numpy as n

    # RIGHT
    import os
    import sys
    from pathlib import Path

    import numpy as np          # third-party
    import pandas as pd

    from mypackage.utils import load_config   # local

### Expressions and Spacing

    # WRONG
    x=1+2
    if x==True: pass
    d ['key']

    # RIGHT
    x = 1 + 2
    if x:                   # not `if x == True`
    d['key']

    # Comparisons
    if value is None:       # identity test, not ==
    if value is not None:
    if not items:           # empty check, not `len(items) == 0`

---

##### PART 3 — FUNCTIONS

### Single Responsibility Principle

A function should do ONE thing. If you need "and" to describe it, split it.

    # WRONG: one function doing three things
    def process_user_data(users):
        users = [u for u in users if u['active']]
        emails = [u['email'].lower().strip() for u in users]
        send_bulk_email(emails)
        return len(emails)

    # RIGHT: separated concerns
    def filter_active_users(users: list[dict]) -> list[dict]:
        return [u for u in users if u['active']]

    def extract_emails(users: list[dict]) -> list[str]:
        return [u['email'].lower().strip() for u in users]

    def notify_users(users: list[dict]) -> int:
        active  = filter_active_users(users)
        emails  = extract_emails(active)
        send_bulk_email(emails)
        return len(emails)

### Argument Design

    # Use keyword arguments for anything non-obvious
    # WRONG
    connect(True, False, 30, 3)

    # RIGHT
    connect(use_ssl=True, verify_cert=False, timeout=30, retries=3)

    # Default arguments — use immutable defaults
    # WRONG (dangerous mutable default)
    def append_item(item, lst=[]):
        lst.append(item)
        return lst

    # RIGHT
    def append_item(item: int, lst: list | None = None) -> list:
        if lst is None:
            lst = []
        lst.append(item)
        return lst

### Return Values — Be Consistent

    # WRONG: sometimes returns value, sometimes None
    def find_user(user_id):
        if user_id in db:
            return db[user_id]
        # implicit None return

    # RIGHT: always explicit, use Optional or raise
    def find_user(user_id: int) -> dict | None:
        return db.get(user_id)

    # OR: raise if not found is part of the contract
    def get_user(user_id: int) -> dict:
        if user_id not in db:
            raise KeyError(f"User {user_id!r} not found")
        return db[user_id]

---

##### PART 4 — CLASSES AND OOP

### When to Use Classes

Use a class when:
  • You have state that is naturally grouped with behaviour
  • You need multiple instances with independent state
  • You're modelling a real-world entity

Don't use a class when a module-level function or a dataclass is clearer.

### SOLID Principles (Simplified)

    S — Single Responsibility:  one class, one reason to change
    O — Open/Closed:            extend by subclassing, not by editing
    L — Liskov Substitution:    subclasses must honour parent's contract
    I — Interface Segregation:  small focused interfaces, not fat ones
    D — Dependency Inversion:   depend on abstractions, not concretions

### Dataclasses — Prefer Over Boilerplate __init__

    from dataclasses import dataclass, field

    # WRONG: verbose manual init
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # RIGHT: dataclass generates __init__, __repr__, __eq__ automatically
    @dataclass
    class Point:
        x: float
        y: float

    @dataclass
    class Config:
        host:    str   = "localhost"
        port:    int   = 8080
        tags:    list  = field(default_factory=list)  # mutable default

### Properties — Computed Attributes

    @dataclass
    class Circle:
        radius: float

        @property
        def area(self) -> float:
            return 3.14159 * self.radius ** 2

        @property
        def diameter(self) -> float:
            return 2 * self.radius

---

##### PART 5 — TYPE ANNOTATIONS

### Why Type Hints Matter

Type hints are documentation that tools can verify. They:
  • Make function contracts explicit without reading the body
  • Enable IDE autocompletion and jump-to-definition
  • Allow static analysis (mypy, pyright) to catch bugs before runtime
  • Serve as living documentation that can't go stale

### Annotation Syntax

    from typing import Optional, Union
    from collections.abc import Callable, Sequence

    # Basic
    def greet(name: str) -> str:
        return f"Hello, {name}"

    # Optional (None or value)
    def find(key: str) -> str | None:     # Python 3.10+
        return cache.get(key)

    # Collections
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    # Callables
    def apply(fn: Callable[[int], int], data: list[int]) -> list[int]:
        return [fn(x) for x in data]

    # TypeVar for generics
    from typing import TypeVar
    T = TypeVar('T')
    def first(items: list[T]) -> T:
        return items[0]

### What NOT to Over-Annotate

    # WRONG: over-annotated, obvious from context
    x: int = 0
    name: str = "Alice"

    # RIGHT: annotate function boundaries, not obvious local variables
    x = 0
    name = "Alice"

---

##### PART 6 — DOCUMENTATION

### Docstring Standards (Google Style)

    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                    n_bins: int = 10) -> float:
        \"\"\"Compute Expected Calibration Error (ECE).

        Measures the average gap between predicted confidence and actual
        accuracy across probability bins.

        Args:
            y_true: Binary ground truth labels, shape (n,).
            y_prob: Predicted probabilities, shape (n,), in [0, 1].
            n_bins: Number of equal-width bins. Defaults to 10.

        Returns:
            ECE as a float in [0, 1]. Lower is better.

        Raises:
            ValueError: If y_true and y_prob have different lengths.

        Example:
            >>> compute_ece(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8]))
            0.067
        \"\"\"

### Comments — When and How

    # WRONG: restating the code
    x = x + 1   # increment x by 1

    # RIGHT: explaining WHY, not WHAT
    x = x + 1   # account for the fence-post offset (indices are 0-based)

    # WRONG: commented-out code left in
    # old_result = compute_v1(data)
    result = compute_v2(data)

    # RIGHT: use version control — delete dead code, never comment it out

---

##### PART 7 — ERROR HANDLING

### Exception Hierarchy

    BaseException
    └── Exception
        ├── ValueError      — bad argument value
        ├── TypeError       — wrong type
        ├── KeyError        — missing dict key
        ├── IndexError      — out-of-bounds
        ├── AttributeError  — missing attribute
        ├── FileNotFoundError
        ├── PermissionError
        └── RuntimeError    — everything else

### Professional Exception Patterns

    # WRONG: swallow all exceptions silently
    try:
        result = process(data)
    except:
        pass

    # WRONG: catch too broadly
    try:
        result = process(data)
    except Exception:
        return None

    # RIGHT: catch exactly what you expect; let the rest propagate
    try:
        result = process(data)
    except ValueError as e:
        logger.warning("Invalid data: %s", e)
        return default_value
    except FileNotFoundError:
        raise   # re-raise: caller should handle this

    # RIGHT: always use `raise ... from e` to preserve traceback
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Config file contains invalid JSON: {path}") from e

### Custom Exceptions

    class AppError(Exception):
        \"\"\"Base class for application errors.\"\"\"

    class ValidationError(AppError):
        def __init__(self, field: str, message: str):
            self.field = field
            super().__init__(f"{field}: {message}")

    class NotFoundError(AppError):
        def __init__(self, resource: str, id_: int):
            super().__init__(f"{resource} with id={id_} not found")

### Context Managers (with statements)

    # WRONG: manual resource management
    f = open("data.csv")
    try:
        data = f.read()
    finally:
        f.close()

    # RIGHT: context manager guarantees cleanup
    with open("data.csv") as f:
        data = f.read()

    # Write your own:
    from contextlib import contextmanager

    @contextmanager
    def timer(label: str):
        import time
        t0 = time.perf_counter()
        yield
        print(f"{label}: {time.perf_counter() - t0:.3f}s")

    with timer("model training"):
        model.fit(X_train, y_train)

---

##### PART 8 — TESTING

### Why Tests Are Non-Negotiable

Tests are the only way to verify that code is correct AND to verify
that changes don't break it. Code without tests is a liability.

    Unit tests:       test a single function in isolation
    Integration tests: test how components work together
    Regression tests: ensure previously fixed bugs don't reappear

### pytest — The Standard

    # test_statistics.py
    import pytest
    import numpy as np
    from mymodule import compute_mean, compute_std

    def test_mean_basic():
        assert compute_mean([1, 2, 3]) == 2.0

    def test_mean_single_element():
        assert compute_mean([5]) == 5.0

    def test_mean_raises_on_empty():
        with pytest.raises(ValueError, match="empty"):
            compute_mean([])

    def test_mean_floats():
        result = compute_mean([1.1, 2.2, 3.3])
        assert abs(result - 2.2) < 1e-10   # float comparison

    @pytest.mark.parametrize("values,expected", [
        ([1, 2, 3],    2.0),
        ([0, 0, 0],    0.0),
        ([-1, 1],      0.0),
        ([100],        100.0),
    ])
    def test_mean_parametrized(values, expected):
        assert compute_mean(values) == expected

### Test Naming Convention

    test_<function>_<scenario>_<expected_behaviour>

    test_login_with_valid_credentials_returns_token()
    test_login_with_expired_token_raises_AuthError()
    test_parse_csv_with_missing_header_raises_ValueError()

---

##### PART 9 — PROJECT STRUCTURE

### Standard Layout

    my_project/
    ├── src/
    │   └── mypackage/
    │       ├── __init__.py
    │       ├── models.py
    │       ├── utils.py
    │       └── config.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_models.py
    │   └── test_utils.py
    ├── docs/
    ├── pyproject.toml        ← replaces setup.py (PEP 517/518)
    ├── .pre-commit-config.yaml
    └── README.md

### pyproject.toml (Modern Standard)

    [project]
    name = "mypackage"
    version = "0.1.0"
    requires-python = ">=3.11"
    dependencies = ["numpy>=1.25", "pandas>=2.0"]

    [project.optional-dependencies]
    dev = ["pytest", "mypy", "ruff", "black"]

### The Toolchain: Automate Quality

    black          → opinionated auto-formatter (zero config)
    ruff           → fast linter (replaces flake8, isort, pyupgrade)
    mypy/pyright   → static type checker
    pytest         → testing framework
    pre-commit     → runs all of the above before every git commit

---

##### PART 10 — PERFORMANCE

### Know Before You Optimise

    "Premature optimisation is the root of all evil." — Knuth

    RULE 1: Make it correct first, then fast if needed.
    RULE 2: Profile before optimising — don't guess.
    RULE 3: The biggest wins come from algorithmic improvements, not micro-optimisations.

### Algorithmic Complexity First

    # O(n²) — WRONG for large n
    def has_duplicate(items):
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                if items[i] == items[j]:
                    return True
        return False

    # O(n) — RIGHT: use a set
    def has_duplicate(items):
        return len(items) != len(set(items))

### Python-Specific Performance Wins

    # List comprehensions > for-loops with append
    squares = [x**2 for x in range(1000)]          # faster
    squares = []
    for x in range(1000): squares.append(x**2)     # slower

    # Use built-in functions (implemented in C)
    total = sum(values)                # faster than a for-loop
    maximum = max(values)

    # Generator expressions for large sequences (memory-efficient)
    total = sum(x**2 for x in range(10_000_000))   # no list in memory

    # str.join() for string concatenation
    result = ", ".join(names)          # O(n) — RIGHT
    result = ""
    for n in names: result += n + ", " # O(n²) — WRONG

    # functools.lru_cache for expensive pure functions
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def fibonacci(n: int) -> int:
        if n < 2: return n
        return fibonacci(n-1) + fibonacci(n-2)

### NumPy — Vectorise, Don't Loop

    import numpy as np

    # WRONG: Python loop over array
    result = []
    for x in data:
        result.append(x * 2 + 1)

    # RIGHT: vectorised operation (10-100× faster)
    result = data * 2 + 1

    # WRONG: row-by-row matrix operation
    for i in range(len(X)):
        predictions[i] = np.dot(X[i], weights) + bias

    # RIGHT: batch matrix multiply
    predictions = X @ weights + bias
"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 1: Style, Naming, and Documentation
# ─────────────────────────────────────────────────────────────────────────────
OP1_CODE = r'''import numpy as np
import re
import time

print("=" * 65)
print("  PROFESSIONAL PYTHON — STYLE, NAMING, AND DOCUMENTATION")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — PEP 8 naming convention audit")
print("━" * 65)
print()

def audit_naming_convention(name: str, context: str) -> tuple[bool, str]:
    """
    Check a Python identifier against PEP 8 naming conventions.

    Args:
        name:    The identifier to check.
        context: 'function', 'class', 'constant', or 'variable'.

    Returns:
        (is_correct, message) tuple.
    """
    patterns = {
        'function':  (r'^[a-z][a-z0-9_]*$',  'snake_case'),
        'variable':  (r'^[a-z_][a-z0-9_]*$', 'snake_case'),
        'class':     (r'^[A-Z][a-zA-Z0-9]*$', 'PascalCase'),
        'constant':  (r'^[A-Z][A-Z0-9_]*$',   'UPPER_SNAKE_CASE'),
    }
    if context not in patterns:
        return False, f"Unknown context: {context!r}"

    pattern, style = patterns[context]
    ok = bool(re.match(pattern + '$', name))
    if ok:
        return True, f"✅ {name!r} is valid {style} for a {context}"
    else:
        return False, f"❌ {name!r} should be {style} for a {context}"

# Run the audit on good and bad examples
examples = [
    # (name,            context,    should_pass)
    ("compute_mean",    "function", True),
    ("ComputeMean",     "function", False),
    ("UserAccount",     "class",    True),
    ("user_account",    "class",    False),
    ("MAX_RETRIES",     "constant", True),
    ("maxRetries",      "constant", False),
    ("user_count",      "variable", True),
    ("UserCount",       "variable", False),
]

print("  Naming convention validation:")
print(f"  {'Identifier':<20} | {'Context':<10} | {'Result'}")
print(f"  {'─'*60}")
all_pass = True
for name, context, expected in examples:
    ok, msg = audit_naming_convention(name, context)
    status  = "✅" if ok == expected else "❌ UNEXPECTED"
    if ok != expected: all_pass = False
    print(f"  {name:<20} | {context:<10} | {msg[3:]}")

print()
print(f"  Audit self-check: {'✅ All results as expected' if all_pass else '❌ Unexpected results'}")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Docstring quality analyser")
print("━" * 65)
print()


def analyse_docstring(func_source: str) -> dict:
    """
    Parse a function's source and score its docstring quality.

    Checks for: presence, summary line, Args section, Returns section,
    Raises section, and inline examples.
    """
    lines   = func_source.strip().split('\n')
    # Find docstring content
    in_doc  = False
    doc_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') and not in_doc:
            in_doc = True
            rest = stripped[3:]
            if rest.endswith('"""') and len(rest) > 3:
                doc_lines.append(rest[:-3])
                break
            doc_lines.append(rest)
            continue
        if in_doc:
            if stripped.endswith('"""'):
                doc_lines.append(stripped[:-3])
                break
            doc_lines.append(stripped)

    doc_text = '\n'.join(doc_lines)

    checks = {
        'has_docstring':   bool(doc_text.strip()),
        'has_summary':     bool(doc_text.strip().split('\n')[0].strip()) if doc_text.strip() else False,
        'has_args':        'Args:' in doc_text or 'Parameters:' in doc_text,
        'has_returns':     'Returns:' in doc_text or 'Return:' in doc_text,
        'has_raises':      'Raises:' in doc_text,
        'has_example':     'Example' in doc_text or '>>>' in doc_text,
    }
    score   = sum(checks.values())
    maximum = len(checks)
    return {'checks': checks, 'score': score, 'max': maximum,
            'grade': 'A' if score >= 5 else ('B' if score >= 4 else ('C' if score >= 3 else 'D'))}


# Test functions at different quality levels
def well_documented(values: list[float]) -> float:
    """Compute the arithmetic mean of a list of numbers.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        The arithmetic mean as a float.

    Raises:
        ValueError: If values is empty.

    Example:
        >>> well_documented([1.0, 2.0, 3.0])
        2.0
    """
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def partially_documented(values):
    """Compute mean."""
    return sum(values) / len(values)


def undocumented(values):
    return sum(values) / len(values)


# Embed source explicitly — inspect.getsource() fails in exec'd code
func_sources = {
    "well_documented": """def well_documented(values: list[float]) -> float:
    \"\"\"Compute the arithmetic mean of a list of numbers.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        The arithmetic mean as a float.

    Raises:
        ValueError: If values is empty.

    Example:
        >>> well_documented([1.0, 2.0, 3.0])
        2.0
    \"\"\"
    pass""",
    "partially_documented": """def partially_documented(values):
    \"\"\"Compute mean.\"\"\"
    pass""",
    "undocumented": """def undocumented(values):
    pass""",
}

print("  Docstring quality assessment:")
print()
for fname, source in func_sources.items():
    result = analyse_docstring(source)
    grade  = result['grade']
    score  = result['score']
    maxs   = result['max']
    emoji  = "🟢" if grade == 'A' else ("🟡" if grade == 'B' else ("🟠" if grade == 'C' else "🔴"))
    print(f"  {emoji} {fname:<28}  Score: {score}/{maxs}  Grade: {grade}")
    for check, passed in result['checks'].items():
        status = "✅" if passed else "❌"
        print(f"      {status} {check}")
    print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Line length and complexity analyser")
print("━" * 65)
print()


def analyse_function_complexity(source: str) -> dict:
    """
    Estimate cyclomatic complexity from source code.

    Cyclomatic complexity = 1 + number of branching points
    (if, elif, for, while, except, and, or, assert, comprehension).
    """
    branch_keywords = ['if ', 'elif ', 'for ', 'while ', 'except',
                       ' and ', ' or ', 'assert ']
    complexity = 1
    long_lines  = []
    lines = source.split('\n')
    for i, line in enumerate(lines, 1):
        for kw in branch_keywords:
            complexity += line.count(kw)
        if len(line.rstrip()) > 88:
            long_lines.append((i, len(line.rstrip()), line.rstrip()[:60]))

    level = ('Low ✅' if complexity <= 5 else
             'Moderate ⚠️' if complexity <= 10 else
             'High ❌ — consider refactoring')
    return {'complexity': complexity, 'level': level, 'long_lines': long_lines}


# Example functions with varying complexity
def simple_function(x: float) -> float:
    """Square root, clamped to zero for negatives."""
    return max(0.0, x) ** 0.5


def moderate_complexity(data: list[dict], threshold: float = 0.5) -> list[dict]:
    """Filter and transform records above a threshold."""
    results = []
    for record in data:
        score = record.get('score', 0.0)
        if score > threshold and record.get('active', False):
            if 'name' in record and 'id' in record:
                results.append({
                    'id':    record['id'],
                    'name':  record['name'].strip().title(),
                    'score': round(score, 3),
                })
    return results


def high_complexity(text: str, rules: dict) -> str:
    """Apply multiple transformation rules to text."""
    result = text
    for rule_name, pattern in rules.items():
        if rule_name.startswith('replace'):
            if isinstance(pattern, tuple) and len(pattern) == 2:
                result = result.replace(pattern[0], pattern[1])
            elif isinstance(pattern, str):
                result = re.sub(pattern, '', result)
        elif rule_name.startswith('strip') and isinstance(pattern, str):
            if pattern == 'leading':
                result = result.lstrip()
            elif pattern == 'trailing':
                result = result.rstrip()
            else:
                result = result.strip(pattern)
        elif rule_name == 'lower':
            result = result.lower()
    return result


func_complexity_sources = {
    "simple_function": """def simple_function(x: float) -> float:
    return max(0.0, x) ** 0.5""",
    "moderate_complexity": """def moderate_complexity(data, threshold=0.5):
    results = []
    for record in data:
        score = record.get('score', 0.0)
        if score > threshold and record.get('active', False):
            if 'name' in record and 'id' in record:
                results.append({'id': record['id'], 'name': record['name'].strip().title(),
                                 'score': round(score, 3)})
    return results""",
    "high_complexity": """def high_complexity(text, rules):
    result = text
    for rule_name, pattern in rules.items():
        if rule_name.startswith('replace'):
            if isinstance(pattern, tuple) and len(pattern) == 2:
                result = result.replace(pattern[0], pattern[1])
            elif isinstance(pattern, str):
                result = re.sub(pattern, \'\', result)
        elif rule_name.startswith('strip') and isinstance(pattern, str):
            if pattern == 'leading':
                result = result.lstrip()
            elif pattern == 'trailing':
                result = result.rstrip()
            else:
                result = result.strip(pattern)
        elif rule_name == 'lower':
            result = result.lower()
    return result""",
}

print("  Cyclomatic complexity analysis:")
print(f"  {'Function':<28} | {'Complexity':>12} | {'Level'}")
print(f"  {'─'*65}")
for fname, source in func_complexity_sources.items():
    result = analyse_function_complexity(source)
    print(f"  {fname:<28} | {result['complexity']:>12} | {result['level']}")
    if result['long_lines']:
        for lineno, length, preview in result['long_lines'][:2]:
            print(f"    ⚠️  Line {lineno} ({length} chars): {preview}...")
print()
print("  Guideline: Keep functions below complexity 10.")
print("  Above 10: hard to test, hard to reason about, error-prone.")
print()
print("  PEP 8 line length: 79 chars (strict) or 88 chars (Black default).")
print("  Long lines are harder to read and diff in code review.")
print()
'''


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 2: Functions, Classes, Error Handling, and Type Hints
# ─────────────────────────────────────────────────────────────────────────────
OP2_CODE = r'''import numpy as np
import time
import math
from dataclasses import dataclass, field
from typing import Callable

print("=" * 65)
print("  PROFESSIONAL PYTHON — FUNCTIONS, CLASSES, AND ROBUSTNESS")
print("=" * 65)
print()

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Mutable default argument trap and its fix")
print("━" * 65)
print()

print("  THE MUTABLE DEFAULT TRAP:")
print()
print("  def broken(item, lst=[]):  ← list evaluated ONCE at definition time")
print("      lst.append(item)")
print("      return lst")
print()
print("  # The SAME list is shared across all calls without explicit lst!")
print("  broken(1) → [1]")
print("  broken(2) → [1, 2]  ← UNEXPECTED!")
print()

def broken_append(item, lst=[]):
    lst.append(item)
    return lst

def safe_append(item: int, lst: list | None = None) -> list:
    """Append item to lst, creating a new list if none provided."""
    if lst is None:
        lst = []
    lst.append(item)
    return lst

r1 = broken_append(1)
r2 = broken_append(2)
r3 = safe_append(1)
r4 = safe_append(2)

print(f"  broken_append(1): {r1}")
print(f"  broken_append(2): {r2}  ← BUG: contains both values!")
print()
print(f"  safe_append(1):   {r3}")
print(f"  safe_append(2):   {r4}  ✅ independent lists")
print()
print("  RULE: Never use mutable objects (list, dict, set) as default arguments.")
print("        Use None as sentinel and create inside the function body.")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Dataclasses vs manual classes")
print("━" * 65)
print()

# Manual approach — verbose
class PointManual:
    def __init__(self, x: float, y: float, label: str = ""):
        self.x     = x
        self.y     = y
        self.label = label

    def __repr__(self) -> str:
        return f"PointManual(x={self.x}, y={self.y}, label={self.label!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, PointManual): return NotImplemented
        return (self.x, self.y, self.label) == (other.x, other.y, other.label)

    @property
    def distance_from_origin(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)


# Dataclass approach — concise, correct
@dataclass
class Point:
    x:     float
    y:     float
    label: str = ""

    @property
    def distance_from_origin(self) -> float:
        """Euclidean distance from the origin."""
        return math.sqrt(self.x**2 + self.y**2)

    def translate(self, dx: float, dy: float) -> "Point":
        """Return a new Point shifted by (dx, dy)."""
        return Point(self.x + dx, self.y + dy, self.label)


# Frozen dataclass — immutable (hashable, safe in sets/dicts)
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float


p1 = Point(3.0, 4.0, "A")
p2 = Point(3.0, 4.0, "A")
p3 = p1.translate(1.0, 0.0)

print("  @dataclass generates __init__, __repr__, __eq__ automatically:")
print(f"  Point(3, 4, 'A') repr: {p1}")
print(f"  Equality:         p1 == p2 → {p1 == p2}  ✅")
print(f"  Translate:        p1.translate(1,0) → {p3}")
print(f"  Distance:         {p1.distance_from_origin:.4f}")
print()

# Frozen
ip1 = ImmutablePoint(1.0, 2.0)
try:
    ip1.x = 99.0  # type: ignore
    print("  UNEXPECTED: mutation succeeded")
except Exception as e:
    print(f"  frozen=True prevents mutation: {type(e).__name__} ✅")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Exception handling patterns")
print("━" * 65)
print()


class ConfigError(Exception):
    """Raised when application configuration is invalid."""


class DataValidationError(Exception):
    """Raised when input data fails validation checks."""
    def __init__(self, field: str, value, message: str):
        self.field   = field
        self.value   = value
        super().__init__(f"Field '{field}' (value={value!r}): {message}")


def validate_learning_rate(lr: float) -> float:
    """
    Validate and return a learning rate for neural network training.

    Args:
        lr: Learning rate to validate.

    Returns:
        The validated learning rate.

    Raises:
        DataValidationError: If lr is outside the valid range (0, 1].
        TypeError: If lr is not a numeric type.
    """
    if not isinstance(lr, (int, float)):
        raise TypeError(f"Learning rate must be numeric, got {type(lr).__name__!r}")
    if not (0 < lr <= 1.0):
        raise DataValidationError(
            field="learning_rate",
            value=lr,
            message=f"Must be in (0, 1], got {lr}"
        )
    return float(lr)


def safe_divide(a: float, b: float) -> float | None:
    """
    Divide a by b, returning None instead of raising on division by zero.

    This is appropriate when zero division is an EXPECTED, RECOVERABLE
    condition. If it indicates a programming error, let it propagate.
    """
    try:
        return a / b
    except ZeroDivisionError:
        return None


# Demonstrate exception patterns
print("  Custom exception hierarchy:")
test_cases = [
    (0.001,  "valid small"),
    (0.1,    "valid standard"),
    (1.0,    "valid boundary"),
    (0.0,    "invalid: zero"),
    (1.1,    "invalid: above 1"),
    (-0.01,  "invalid: negative"),
    ("0.1",  "invalid: string"),
]

for lr, desc in test_cases:
    try:
        result = validate_learning_rate(lr)
        print(f"  ✅ lr={lr!r:<8} ({desc}): valid → {result}")
    except DataValidationError as e:
        print(f"  ❌ lr={lr!r:<8} ({desc}): {e}")
    except TypeError as e:
        print(f"  ❌ lr={lr!r:<8} ({desc}): {e}")

print()
print("  safe_divide patterns:")
print(f"    10 / 2   = {safe_divide(10, 2)}")
print(f"    10 / 0   = {safe_divide(10, 0)}  (None instead of crash)")
print()
print("  RULE: Catch specific exceptions. Never use bare 'except:'.")
print("  RULE: Use 'raise X from e' to preserve original traceback.")
print("  RULE: Raise early, recover late — validate inputs at the boundary.")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Context managers and resource safety")
print("━" * 65)
print()

from contextlib import contextmanager

@contextmanager
def timer(label: str):
    """Context manager that times a block and prints elapsed time."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"  ⏱  {label}: {elapsed*1000:.2f}ms")


@contextmanager
def temporary_seed(seed: int):
    """Temporarily set numpy random seed, restoring original state after."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


print("  Context manager: timer")
with timer("Matrix multiply (1000×1000)"):
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    C = A @ B

print()
print("  Context manager: temporary_seed (reproducibility without side effects)")
with temporary_seed(42):
    sample_a = np.random.randn(3)

with temporary_seed(42):
    sample_b = np.random.randn(3)

identical = np.allclose(sample_a, sample_b)
print(f"  Seed=42 run 1: {sample_a.round(4)}")
print(f"  Seed=42 run 2: {sample_b.round(4)}")
print(f"  Both identical: {identical} ✅")
print()
print("  RULE: Use 'with' statements for ALL resources: files, locks, DB connections.")
print("  RULE: Write context managers for setup/teardown that belongs together.")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — Type hint patterns")
print("━" * 65)
print()

print("  Demonstrating key type annotation patterns:")
print()

# Show the patterns clearly
patterns = [
    ("Basic",          "def greet(name: str) -> str"),
    ("Optional",       "def find(key: str) -> str | None"),
    ("Union (3.10+)",  "def parse(v: int | str | None) -> float"),
    ("List/Dict",      "def mean(vals: list[float]) -> float"),
    ("Callable",       "def apply(fn: Callable[[int], int], xs: list[int])"),
    ("TypeVar",        "T = TypeVar('T')  # generic function"),
    ("Literal",        "from typing import Literal; Mode = Literal['r','w']"),
    ("ClassVar",       "class Cfg: MAX: ClassVar[int] = 100"),
    ("dataclass",      "@dataclass  class Point: x: float; y: float"),
]

for category, example in patterns:
    print(f"  {category:<18}: {example}")

print()
print("  Run 'mypy myfile.py' or 'pyright' to get type errors before runtime.")
print()
print("  TYPE HINT RULES:")
print("    ✅ DO annotate all public function signatures")
print("    ✅ DO annotate class attributes in __init__ or with dataclass")
print("    ✅ DO use | None instead of Optional[] (Python 3.10+)")
print("    ❌ DON'T annotate obvious local variables (x: int = 0)")
print("    ❌ DON'T use Any — it disables type checking for that variable")
print()
'''


# ─────────────────────────────────────────────────────────────────────────────
# OPERATION 3: Performance, Testing, and the Professional Toolchain
# ─────────────────────────────────────────────────────────────────────────────
OP3_CODE = r'''import numpy as np
import time
import timeit
import functools
import sys

print("=" * 65)
print("  PROFESSIONAL PYTHON — PERFORMANCE, TESTING, AND TOOLCHAIN")
print("=" * 65)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Algorithmic complexity: the biggest win")
print("━" * 65)
print()

print("  Finding duplicates — O(n²) vs O(n):")
print()

def has_duplicate_slow(items: list) -> bool:
    """O(n²) — check every pair."""
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return True
    return False


def has_duplicate_fast(items: list) -> bool:
    """O(n) — use a hash set."""
    return len(items) != len(set(items))


# Time both on increasing sizes
print(f"  {'N':>8} | {'O(n²) ms':>12} | {'O(n) ms':>10} | {'Speedup':>8}")
print(f"  {'─'*50}")

for n in [100, 1_000, 5_000, 10_000]:
    data = list(range(n))   # no duplicates — worst case for slow version

    t_slow = None
    if n <= 5_000:          # skip O(n²) for very large n
        t_slow = timeit.timeit(lambda: has_duplicate_slow(data), number=10) / 10 * 1000

    t_fast = timeit.timeit(lambda: has_duplicate_fast(data), number=100) / 100 * 1000

    if t_slow:
        speedup = t_slow / t_fast
        print(f"  {n:>8} | {t_slow:>12.3f} | {t_fast:>10.4f} | {speedup:>7.0f}×")
    else:
        print(f"  {n:>8} | {'(too slow)':>12} | {t_fast:>10.4f} | {'—':>8}")

print()
print("  O(n²) becomes 10,000× slower when n grows 10×.")
print("  O(n)  grows only proportionally.")
print()
print("  BEFORE micro-optimising, always ask: can the algorithm be better?")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Python micro-optimisations with benchmarks")
print("━" * 65)
print()

N = 100_000
data = list(range(N))
data_arr = np.arange(N, dtype=float)

benchmarks = []

# 1. List comprehension vs for-loop append
t1 = timeit.timeit(lambda: [x**2 for x in range(1000)], number=1000) / 1000 * 1e6
t2 = timeit.timeit(
    "r=[]\nfor x in range(1000):\n r.append(x**2)",
    number=1000, globals={}) / 1000 * 1e6
benchmarks.append(("List comprehension",    "[x**2 for x in range(1000)]",    t1, t2))

# 2. sum() vs manual loop
t3 = timeit.timeit(lambda: sum(data),          number=200) / 200 * 1e3
t4 = timeit.timeit(
    "t=0\nfor x in data:\n t+=x",
    number=200, globals={'data': data}) / 200 * 1e3
benchmarks.append(("Built-in sum()",         "sum(data) vs loop",              t3, t4, "ms"))

# 3. str.join() vs concatenation
words = ["word"] * 1000
t5 = timeit.timeit(lambda: ", ".join(words), number=5000) / 5000 * 1e6
t6 = timeit.timeit(
    "r=''\nfor w in words:\n r+=w+', '",
    number=5000, globals={'words': words[:100]}) / 5000 * 1e6
benchmarks.append(("str.join()",            "', '.join(words) vs += loop",    t5, t6))

# 4. numpy vectorise vs Python loop
t7 = timeit.timeit(lambda: (data_arr * 2 + 1).sum(), number=500) / 500 * 1e3
t8 = timeit.timeit(
    "s=0\nfor x in arr:\n s+=x*2+1",
    number=50, globals={'arr': list(data_arr[:10000])}) / 50 * 1e3
benchmarks.append(("NumPy vectorise",       "arr*2+1 vs loop",                t7, t8, "ms"))

print(f"  {'Technique':<22} | {'Fast impl':>28} | {'Fast µs/ms':>12} | {'Slow µs/ms':>12} | {'Speedup':>8}")
print(f"  {'─'*95}")
for row in benchmarks:
    name, desc, fast, slow = row[0], row[1], row[2], row[3]
    unit = row[4] if len(row) > 4 else 'µs'
    speedup = slow / fast
    print(f"  {name:<22} | {desc:>28} | {fast:>10.2f}{unit} | {slow:>10.2f}{unit} | {speedup:>7.1f}×")

print()
print("  Key rules:")
print("    1. List comprehensions > for-loops with append (fewer overhead calls)")
print("    2. Built-in sum/max/min  are implemented in C — always faster")
print("    3. str.join() is O(n),   += concatenation is O(n²)")
print("    4. NumPy on arrays: 10-100× faster via BLAS/LAPACK routines")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Caching and memoisation")
print("━" * 65)
print()

# lru_cache for expensive pure functions
@functools.lru_cache(maxsize=None)
def fib_cached(n: int) -> int:
    """Fibonacci with memoisation — O(n) total work."""
    if n < 2:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)


def fib_uncached(n: int) -> int:
    """Naive Fibonacci — O(2^n) without memo."""
    if n < 2:
        return n
    return fib_uncached(n - 1) + fib_uncached(n - 2)


print("  lru_cache (memoisation) on Fibonacci:")
print()
print(f"  {'n':>4} | {'Cached µs':>12} | {'Uncached µs':>13} | {'Speedup':>9} | {'Result':>12}")
print(f"  {'─'*60}")

for n in [10, 20, 30, 35]:
    t_c = timeit.timeit(lambda: fib_cached(n), number=10000) / 10000 * 1e6
    fib_cached.cache_clear()

    if n <= 30:
        t_u = timeit.timeit(lambda: fib_uncached(n), number=10) / 10 * 1e6
        speedup = f"{t_u/t_c:>8.0f}×"
    else:
        t_u = None
        speedup = "    (∞)"

    result = fib_cached(n)
    t_u_str = f"{t_u:>13.1f}" if t_u else f"{'(too slow)':>13}"
    print(f"  {n:>4} | {t_c:>12.4f} | {t_u_str} | {speedup} | {result:>12}")

print()
print("  Cache info:", fib_cached.cache_info())
print()
print("  RULES for caching:")
print("    ✅ Use @lru_cache on: pure functions with expensive computation")
print("    ✅ Use @lru_cache on: functions called repeatedly with the same args")
print("    ❌ Never cache: functions with side effects (writes to DB, sends email)")
print("    ❌ Never cache: functions with mutable arguments (lists, dicts)")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Generators and memory-efficient iteration")
print("━" * 65)
print()

def sum_squares_list(n: int) -> float:
    """Build full list in memory, then sum."""
    return sum([x**2 for x in range(n)])


def sum_squares_generator(n: int) -> float:
    """Sum using a generator — no list in memory."""
    return sum(x**2 for x in range(n))


def read_large_file_bad(path: str) -> list[str]:
    """Read entire file into memory."""
    with open(path) as f:
        return f.readlines()   # all lines in RAM at once


def read_large_file_good(path: str):
    """Generator: yield one line at a time (constant memory)."""
    with open(path) as f:
        yield from f           # each line consumed, not stored


N = 5_000_000

t_list = timeit.timeit(lambda: sum_squares_list(N),      number=3) / 3
t_gen  = timeit.timeit(lambda: sum_squares_generator(N), number=3) / 3

# sys.getsizeof reports the CONTAINER size, not the elements
list_mem  = sys.getsizeof(list(range(1000)))      # 1000-element list
gen_mem   = sys.getsizeof(x for x in range(1000)) # generator object

print(f"  Sum of {N:,} squares:")
print(f"    List comprehension: {t_list:.3f}s  (list held in RAM)")
print(f"    Generator expr:     {t_gen:.3f}s  (no list in RAM)")
print()
print(f"  Memory: list of 1000 ints ≈ {list_mem} bytes  |  generator object ≈ {gen_mem} bytes")
print()
print("  Generators yield one item at a time — constant memory regardless of N.")
print("  Use generators when: processing large files, streams, or infinite sequences.")
print()

# Custom generator
def moving_average(values, window: int):
    """
    Yield the moving average of a sequence — memory-efficient.
    Processes values one at a time regardless of sequence length.
    """
    buffer = []
    for v in values:
        buffer.append(v)
        if len(buffer) > window:
            buffer.pop(0)
        if len(buffer) == window:
            yield sum(buffer) / window

data = np.random.randn(20).round(3)
ma   = list(moving_average(data, window=3))
print(f"  Moving average (window=3) of {len(data)} values:")
print(f"    Input:  {list(data[:7])} ...")
print(f"    Output: {[round(x,3) for x in ma[:5]]} ...")
print()


# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 5 — The professional toolchain")
print("━" * 65)
print()

TOOLCHAIN = """
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Tool          │ Role                        │ Command                │
  ├──────────────────────────────────────────────────────────────────────┤
  │ black         │ Auto-formatter              │ black myfile.py        │
  │ ruff          │ Linter (replaces flake8)    │ ruff check myfile.py   │
  │ mypy          │ Static type checker         │ mypy myfile.py         │
  │ pyright       │ Static type checker (MS)    │ pyright myfile.py      │
  │ pytest        │ Testing framework           │ pytest tests/          │
  │ coverage      │ Test coverage report        │ pytest --cov           │
  │ pre-commit    │ Git hook: auto-run above    │ pre-commit install     │
  │ tox           │ Test across Python versions │ tox                    │
  ├──────────────────────────────────────────────────────────────────────┤
  │ pyproject.toml│ Project metadata & config   │ PEP 517/518 standard   │
  │ venv          │ Virtual environment         │ python -m venv .venv   │
  │ pip-tools     │ Pin exact dependencies      │ pip-compile            │
  └──────────────────────────────────────────────────────────────────────┘
"""
print(TOOLCHAIN)

print("  .pre-commit-config.yaml (automate quality checks on every commit):")
print()
PRE_COMMIT = """
  repos:
    - repo: https://github.com/psf/black
      rev: 24.3.0
      hooks: [{id: black}]

    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.4.0
      hooks: [{id: ruff, args: [--fix]}]

    - repo: https://github.com/pre-commit/mirrors-mypy
      rev: v1.9.0
      hooks: [{id: mypy}]
"""
print(PRE_COMMIT)

print("  TESTING BEST PRACTICES:")
print()
TEST_PATTERNS = """
  # 1. Arrange-Act-Assert (AAA) pattern — standard test structure
  def test_mean_basic():
      # Arrange
      values = [1.0, 2.0, 3.0]
      # Act
      result = compute_mean(values)
      # Assert
      assert result == 2.0

  # 2. Parametrize to cover multiple cases concisely
  @pytest.mark.parametrize("values,expected", [
      ([1, 2, 3],    2.0),
      ([0, 0, 0],    0.0),
      ([-1, 0, 1],   0.0),
  ])
  def test_mean(values, expected):
      assert compute_mean(values) == expected

  # 3. Test exceptions explicitly
  def test_mean_raises_on_empty():
      with pytest.raises(ValueError, match="empty"):
          compute_mean([])

  # 4. Use fixtures for shared setup
  @pytest.fixture
  def trained_model():
      X, y = make_data(n=200)
      model = MyClassifier().fit(X, y)
      return model, X, y

  def test_accuracy(trained_model):
      model, X, y = trained_model
      assert model.score(X, y) > 0.80

  # 5. Mock external dependencies
  from unittest.mock import patch

  def test_send_email_called():
      with patch('mymodule.smtp_client') as mock_smtp:
          notify_user("test@example.com")
          mock_smtp.send.assert_called_once()
"""
print(TEST_PATTERNS)

print("  COVERAGE TARGET: aim for > 80% line coverage.")
print("  100% coverage does not mean bug-free; it means every line was executed.")
print("  Focus coverage on core business logic, not boilerplate.")
print()

print("━" * 65)
print("  PROFESSIONAL PYTHON — QUICK REFERENCE")
print("━" * 65)
print()

SUMMARY = """
  NAMING:  snake_case functions/vars | PascalCase classes | UPPER_SNAKE constants
  IMPORTS: stdlib → third-party → local | one per line | no star imports
  FUNCTIONS: one responsibility | explicit returns | validate inputs early
  CLASSES: use @dataclass | frozen=True for immutable | properties not getters
  ERRORS:  catch specific exceptions | raise X from e | never bare except:
  TYPES:   annotate function boundaries | use | None (not Optional) | run mypy
  TESTS:   pytest + parametrize | AAA pattern | test errors explicitly | > 80% cov
  PERF:    algorithm first | profile before optimising | NumPy for arrays
  CACHING: @lru_cache on pure functions | generators for large sequences
  TOOLS:   black + ruff + mypy + pytest + pre-commit — automate from day one
"""
print(SUMMARY)
'''


# ─────────────────────────────────────────────────────────────────────────────
OPERATIONS = {
    "1 · Style, Naming, and Documentation — PEP 8 and Docstrings": {
        "description": (
            "Practical PEP 8: naming convention audit, docstring quality scoring, "
            "and cyclomatic complexity analysis. See exactly what separates "
            "professional code from amateur code — measured, not just described."
        ),
        "language": "python",
        "code":     OP1_CODE,
    },

    "2 · Functions, Classes, and Robustness — Type Hints and Exceptions": {
        "description": (
            "The mutable default trap, @dataclass vs manual classes, frozen "
            "dataclasses, custom exception hierarchies, context managers, and "
            "type annotation patterns — all live with benchmarks and output."
        ),
        "language": "python",
        "code":     OP2_CODE,
    },

    "3 · Performance, Testing, and the Professional Toolchain": {
        "description": (
            "Algorithmic complexity, micro-benchmarks (comprehensions, builtins, "
            "str.join, NumPy), lru_cache memoisation, generator expressions, "
            "and the full toolchain: black, ruff, mypy, pytest, pre-commit."
        ),
        "language": "python",
        "code":     OP3_CODE,
    },
}


def get_content():
    return {
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }