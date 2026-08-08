"""Corpus walking and best-effort text extraction.

Format handlers are intentionally dumb and lazy-imported: a routing layer only
needs enough text to place a file in the graph, not a faithful rendering.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

TEXT_EXT = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".scala", ".sql",
    ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
}
JSON_EXT = {".json", ".jsonl", ".ndjson"}
HTML_EXT = {".html", ".htm", ".xml"}
PDF_EXT = {".pdf"}
DOCX_EXT = {".docx"}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".tox", ".idea", ".next",
}

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Document:
    path: str
    text: str
    size: int = 0
    # Structural neighbours discovered at read time (imports, links, siblings).
    links: List[str] = field(default_factory=list)


def _read_bytes(path: str, limit: int) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(limit)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "ignore")


def _flatten_json(obj, out: List[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _flatten_json(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj[:200]:
            _flatten_json(v, out, depth + 1)
    elif isinstance(obj, str):
        out.append(obj)
    elif obj is not None:
        out.append(str(obj))


def extract_text(path: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """Return plain text for a file, or None if the format is unsupported."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in TEXT_EXT or ext == "":
            return _decode(_read_bytes(path, max_bytes))
        if ext in JSON_EXT:
            raw = _decode(_read_bytes(path, max_bytes))
            parts: List[str] = []
            try:
                _flatten_json(json.loads(raw), parts)
            except json.JSONDecodeError:
                for line in raw.splitlines()[:2000]:
                    try:
                        _flatten_json(json.loads(line), parts)
                    except json.JSONDecodeError:
                        parts.append(line)
            return " ".join(parts)
        if ext in HTML_EXT:
            return _TAG_RE.sub(" ", _decode(_read_bytes(path, max_bytes)))
        if ext in PDF_EXT:
            try:
                from pypdf import PdfReader  # optional extra
            except ImportError:
                return None
            reader = PdfReader(path)
            return "\n".join((p.extract_text() or "") for p in reader.pages[:50])
        if ext in DOCX_EXT:
            try:
                import docx  # optional extra
            except ImportError:
                return None
            return "\n".join(p.text for p in docx.Document(path).paragraphs)
    except (OSError, ValueError):
        return None
    return None


_LINK_RE = re.compile(
    r"""(?:\]\(([^)\s]+)\))"""            # markdown links
    r"""|(?:^\s*(?:from|import)\s+([\w.]+))"""  # python imports
    r"""|(?:require\(['"]([^'"]+)['"]\))"""      # js require
    r"""|(?:from\s+['"]([^'"]+)['"])""",         # js import
    re.MULTILINE,
)


def extract_links(text: str, limit: int = 64) -> List[str]:
    """Cheap structural edges: markdown links and import statements."""
    out: List[str] = []
    for m in _LINK_RE.finditer(text):
        target = next((g for g in m.groups() if g), None)
        if target and not target.startswith(("http://", "https://", "#")):
            out.append(target)
            if len(out) >= limit:
                break
    return out


def walk(
    root: str,
    include_ext: Optional[Sequence[str]] = None,
    max_bytes: int = 2_000_000,
    follow_links: bool = True,
) -> Iterator[Document]:
    """Yield Documents under `root`, skipping vendor/VCS directories."""
    allow = set(e.lower() for e in include_ext) if include_ext else None
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if allow is not None and ext not in allow:
                continue
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            text = extract_text(path, max_bytes=max_bytes)
            if not text or not text.strip():
                continue
            links = extract_links(text) if follow_links else []
            yield Document(path=path, text=text, size=size, links=links)
