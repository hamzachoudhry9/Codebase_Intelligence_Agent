"""
ingest/code_chunker.py - Tree-sitter code-aware chunking.

Supported languages:
  Python   .py          - tree-sitter-python (regex fallback if not installed)
  C/C++    .cpp .cc .c .h .hpp - tree-sitter-cpp (regex fallback)
  CUDA     .cu .cuh     - tree-sitter-cpp (CUDA is C++ superset)
  Markdown .md .rst .txt - heading-boundary chunking (no extra dependency)
  Generic  all others   - whole file, capped at 4000 chars

Each chunk carries rich metadata:
  {
    "file": "agent/nodes.py",
    "function": "execution_node",
    "start_line": 87,
    "end_line": 134,
    "type": "function" | "class" | "section" | "file",
    "language": "python" | "cpp" | "cuda" | "markdown" | "unknown",
    "docstring": "..."   (first 300 chars if available)
  }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator


# ────────────────────────────────────────────────────────────────────────────
# Python chunking
# ────────────────────────────────────────────────────────────────────────────

def _try_tree_sitter_python(source: str, file_path: str) -> list[dict] | None:
    """Parse Python with tree-sitter. Returns None if tree-sitter unavailable."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(source.encode())
        source_lines = source.splitlines()
        chunks: list[dict] = []

        def visit(node):
            if node.type in ("function_definition", "class_definition"):
                start = node.start_point[0]
                end = node.end_point[0]
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode() if name_node else "unknown"

                doc = ""
                body = node.child_by_field_name("body")
                if body and body.children:
                    first = body.children[0]
                    if first.type == "expression_statement" and first.children:
                        expr = first.children[0]
                        if expr.type == "string":
                            doc = expr.text.decode().strip("\"' \n")[:300]

                chunk_text = "\n".join(source_lines[start: end + 1])
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "file": file_path,
                        "function": name,
                        "start_line": start + 1,
                        "end_line": end + 1,
                        "type": "function" if node.type == "function_definition" else "class",
                        "language": "python",
                        "docstring": doc,
                    },
                })
            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return chunks if chunks else None

    except Exception:
        return None


def _regex_chunk_python(source: str, file_path: str) -> list[dict]:
    """Regex fallback - chunks at def/class boundaries."""
    lines = source.splitlines()
    chunks: list[dict] = []
    current_start = 0
    current_name = "__module__"
    current_type = "module"
    pattern = re.compile(r"^(def |class )(\w+)")

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m and i > current_start:
            text = "\n".join(lines[current_start:i]).strip()
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "file": file_path, "function": current_name,
                        "start_line": current_start + 1, "end_line": i,
                        "type": current_type, "language": "python", "docstring": "",
                    },
                })
            current_start = i
            current_name = m.group(2)
            current_type = "function" if m.group(1).startswith("def") else "class"

    text = "\n".join(lines[current_start:]).strip()
    if text:
        chunks.append({
            "text": text,
            "metadata": {
                "file": file_path, "function": current_name,
                "start_line": current_start + 1, "end_line": len(lines),
                "type": current_type, "language": "python", "docstring": "",
            },
        })
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# C / C++ / CUDA chunking
# ────────────────────────────────────────────────────────────────────────────

def _try_tree_sitter_cpp(source: str, file_path: str, language: str = "cpp") -> list[dict] | None:
    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser

        CPP_LANGUAGE = Language(tscpp.language())
        parser = Parser(CPP_LANGUAGE)
        tree = parser.parse(source.encode())
        source_lines = source.splitlines()
        chunks: list[dict] = []

        def visit(node):
            if node.type == "function_definition":
                start = node.start_point[0]
                end = node.end_point[0]
                name = "unknown"
                declarator = node.child_by_field_name("declarator")
                if declarator:
                    d = declarator
                    for _ in range(6):
                        if d.type in ("identifier", "qualified_identifier"):
                            name = d.text.decode(errors="replace")
                            break
                        found = None
                        for child in d.children:
                            if child.type in ("identifier", "qualified_identifier",
                                              "destructor_name", "operator_name"):
                                found = child
                                break
                            if child.type in ("function_declarator", "pointer_declarator",
                                              "reference_declarator"):
                                inner = child.child_by_field_name("declarator")
                                if inner:
                                    d = inner
                                break
                        if found:
                            name = found.text.decode(errors="replace")
                            break

                if end - start >= 2:
                    chunk_text = "\n".join(source_lines[start: end + 1])
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "file": file_path, "function": name,
                            "start_line": start + 1, "end_line": end + 1,
                            "type": "function", "language": language, "docstring": "",
                        },
                    })

            elif node.type in ("class_specifier", "struct_specifier"):
                start = node.start_point[0]
                end = node.end_point[0]
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode() if name_node else "unknown"
                chunk_text = "\n".join(source_lines[start: end + 1])
                if len(chunk_text) > 100:
                    chunks.append({
                        "text": chunk_text[:3000],
                        "metadata": {
                            "file": file_path, "function": name,
                            "start_line": start + 1, "end_line": end + 1,
                            "type": "class", "language": language, "docstring": "",
                        },
                    })

            for child in node.children:
                visit(child)

        visit(tree.root_node)
        return chunks if chunks else None

    except Exception:
        return None


def _regex_chunk_cpp(source: str, file_path: str, language: str = "cpp") -> list[dict]:
    """Regex fallback for C/C++/CUDA - heuristic function-boundary detection."""
    lines = source.splitlines()
    chunks: list[dict] = []
    current_start = 0
    current_name = "file_header"

    func_pattern = re.compile(
        r"^(?:__global__|__device__|__host__|static\s+|inline\s+|virtual\s+)*"
        r"[\w:<>*&\s]+\s+(\w+)\s*\([^;]*\)\s*(?:const\s*)?(?:\{|$)"
    )

    for i, line in enumerate(lines):
        m = func_pattern.match(line)
        if m and i > current_start + 3:
            text = "\n".join(lines[current_start:i]).strip()
            if len(text) > 50:
                chunks.append({
                    "text": text[:3000],
                    "metadata": {
                        "file": file_path, "function": current_name,
                        "start_line": current_start + 1, "end_line": i,
                        "type": "function", "language": language, "docstring": "",
                    },
                })
            current_start = i
            current_name = m.group(1)

    text = "\n".join(lines[current_start:]).strip()
    if len(text) > 50:
        chunks.append({
            "text": text[:3000],
            "metadata": {
                "file": file_path, "function": current_name,
                "start_line": current_start + 1, "end_line": len(lines),
                "type": "function", "language": language, "docstring": "",
            },
        })
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# Markdown / text chunking
# ────────────────────────────────────────────────────────────────────────────

def chunk_markdown_file(file_path: str, repo_root: str = ".") -> list[dict]:
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    try:
        rel_path = str(path.relative_to(repo_root))
    except ValueError:
        rel_path = str(path)

    lines = source.splitlines()
    chunks: list[dict] = []
    current_heading = "Introduction"
    current_lines: list[str] = []
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)")

    for line in lines:
        m = heading_pattern.match(line)
        if m:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if len(text) > 50:
                    chunks.append({
                        "text": text,
                        "metadata": {
                            "file": rel_path, "function": current_heading,
                            "type": "section", "language": "markdown",
                            "start_line": 0, "end_line": 0, "docstring": "",
                        },
                    })
            current_heading = m.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if len(text) > 50:
            chunks.append({
                "text": text,
                "metadata": {
                    "file": rel_path, "function": current_heading,
                    "type": "section", "language": "markdown",
                    "start_line": 0, "end_line": 0, "docstring": "",
                },
            })
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def chunk_python_file(file_path: str, repo_root: str = ".") -> list[dict]:
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    try:
        rel_path = str(path.relative_to(repo_root))
    except ValueError:
        rel_path = str(path)
    chunks = _try_tree_sitter_python(source, rel_path)
    if chunks is None:
        chunks = _regex_chunk_python(source, rel_path)
    return [c for c in chunks if len(c["text"].strip()) > 50]


def chunk_cpp_file(file_path: str, repo_root: str = ".") -> list[dict]:
    path = Path(file_path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    try:
        rel_path = str(path.relative_to(repo_root))
    except ValueError:
        rel_path = str(path)
    language = "cuda" if path.suffix.lower() in (".cu", ".cuh") else "cpp"
    chunks = _try_tree_sitter_cpp(source, rel_path, language)
    if chunks is None:
        chunks = _regex_chunk_cpp(source, rel_path, language)
    return [c for c in chunks if len(c["text"].strip()) > 50]


def chunk_file(file_path: str, repo_root: str = ".") -> list[dict]:
    """Route a file to the correct chunker based on extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".py":
        return chunk_python_file(file_path, repo_root)

    if ext in (".cpp", ".cc", ".c", ".h", ".hpp", ".cu", ".cuh"):
        return chunk_cpp_file(file_path, repo_root)

    if ext in (".md", ".rst", ".txt"):
        return chunk_markdown_file(file_path, repo_root)

    # Generic - whole file, capped
    try:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < 50:
            return []
        try:
            rel_path = str(path.relative_to(repo_root))
        except ValueError:
            rel_path = str(path)
        return [{
            "text": text[:4000],
            "metadata": {
                "file": rel_path, "function": path.name,
                "type": "file", "language": "unknown",
                "start_line": 1, "end_line": len(text.splitlines()), "docstring": "",
            },
        }]
    except Exception:
        return []


def iter_repo_files(
    repo_root: str,
    extensions: tuple = (
        ".py", ".md", ".rst", ".txt",
        ".cpp", ".cc", ".c", ".h", ".hpp",
        ".cu", ".cuh",
    ),
) -> Iterator[str]:
    """Walk a directory tree, yielding files with the given extensions.
    Skips common non-source directories (.git, venv, chroma_db, etc.)."""
    skip_dirs = {
        ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
        "chroma_db", ".pytest_cache", "dist", "build", ".eggs",
        "site-packages", ".tox", ".mypy_cache",
    }
    root = Path(repo_root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            if not any(skip in path.parts for skip in skip_dirs):
                yield str(path)
