"""
generate_api_json.py

Generates a JSON file with all public functions and classes in the traitly
package. The HTML table loads this JSON via fetch().

Usage:
    python3 generate_api_json.py

Output:
    docs/en/api/api_data.json
"""

import ast
import json
from pathlib import Path

SRC_DIR = Path("src/traitly")
JSON_FILE = Path("docs/en/api/api_data.json")

EXCLUDE_DIRS = {"shiny_app"}

def get_module_name(filepath: Path) -> str:
    parts = filepath.with_suffix("").parts
    try:
        start = parts.index("traitly")
        return ".".join(parts[start:])
    except ValueError:
        return filepath.stem


def get_first_docstring_line(node) -> str:
    if not (node.body and isinstance(node.body[0], ast.Expr)):
        return ""
    expr = node.body[0].value
    if not isinstance(expr, ast.Constant) or not isinstance(expr.value, str):
        return ""
    lines = [l.strip() for l in expr.value.strip().splitlines() if l.strip()]
    return lines[0] if lines else ""


def extract_entries(filepath: Path) -> list[dict]:
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as e:
        print(f"Could not parse {filepath}: {e}")
        return []

    module = get_module_name(filepath)
    entries = []

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("_"):
            continue
        kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
        description = get_first_docstring_line(node)
        entries.append({
            "type": kind,
            "name": name,
            "module": module,
            "description": description or "-",
        })

    return entries


def main():
    all_entries = []

    for filepath in sorted(SRC_DIR.rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in filepath.parts):
            continue
        if filepath.name.startswith("_") and filepath.name != "__init__.py":
            continue
        if filepath.name == "__init__.py":
            continue
        entries = extract_entries(filepath)
        all_entries.extend(entries)

    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(all_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. {len(all_entries)} entries written to {JSON_FILE}")


if __name__ == "__main__":
    main()
