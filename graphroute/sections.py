"""Split books into sections.

Routing at file granularity is the right call for a source tree, where a file
is already a unit of meaning. It is the wrong call for a 2,000-page reference
book, which is one file containing four hundred unrelated subjects: route to
it and you have narrowed the corpus to "the book", which the reader knew.

This module turns each PDF into one record per *section*, so the rest of the
stack — index, graph, communities, router — operates unchanged at a granularity
where a hit means something. A section carries the breadcrumb needed to tell
the reader where the text came from.

Three tiers, tried in order, because real PDFs vary enormously in quality:

1. **Outline / bookmarks.** The publisher already did the work. Use it.
2. **Heading detection.** Numbered headings, `Chapter N`, short ALL-CAPS lines.
   Heuristic and deliberately simple — it is a fallback, not a parser.
3. **Fixed page windows.** No structure at all, so impose some. Breadcrumbs
   degrade honestly to `Book › pp.120–128` rather than inventing a title.

Nothing here calls a model. pypdf is an optional dependency (`[rich]` extra).
"""

from __future__ import annotations

import hashlib
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

BREADCRUMB_SEP = " › "

WINDOW_PAGES = 8
"""Tier-3 window size. Big enough to hold a whole subject, small enough that
the breadcrumb still points somewhere useful."""

MIN_SECTION_CHARS = 200
"""Sections shorter than this are merged into the previous one. A heading with
two lines under it is a heading, not a section, and indexing it separately
gives the router a document whose only content is its own title."""

FIRST_PARA_CHARS = 800

# Heading shapes for tier 2. Ordered by how much we trust them.
_NUMBERED_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(\S.*)$")
_CHAPTER_RE = re.compile(r"^\s*(?:chapter|part|appendix)\s+([\dIVXLC]+)\b[.:]?\s*(.*)$", re.I)
_ALLCAPS_RE = re.compile(r"^\s*([A-Z][A-Z0-9 ,'&/\-]{3,60})\s*$")

_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


@dataclass
class Section:
    """One addressable chunk of a book.

    `page_start` / `page_end` are 1-based and inclusive, matching what a reader
    sees printed on the page rather than a zero-based index into the file.
    """

    book_path: str
    breadcrumb: str
    page_start: int
    page_end: int
    title: str
    first_para: str
    full_text: str
    index: int = 0
    """Position of this section within its book. Part of the document id."""
    page_offsets: List[int] = field(default_factory=list)
    """Character offset in `full_text` at which each covered page begins, so a
    passage found later can be attributed to the page it actually came from
    instead of to the section's whole range."""
    tier: str = "outline"
    """Which extraction tier produced this: outline | headings | windows."""
    also_titles: List[str] = field(default_factory=list)
    """Other headings that start inside this section's page span. Populated when
    several outline entries share a page and were folded into one section, so
    the headings the breadcrumb cannot show are still available to the embedder
    rather than thrown away."""

    @property
    def doc_path(self) -> str:
        """Identity of this section as a document in the lexical index."""
        return f"{self.book_path}#s{self.index}"

    def page_of(self, offset: int) -> int:
        """1-based page containing character `offset` of `full_text`."""
        if not self.page_offsets:
            return self.page_start
        i = bisect_right(self.page_offsets, offset) - 1
        return self.page_start + max(0, i)

    def to_dict(self) -> dict:
        return {
            "book_path": self.book_path,
            "breadcrumb": self.breadcrumb,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "title": self.title,
            "first_para": self.first_para,
            "full_text": self.full_text,
            "index": self.index,
            "page_offsets": self.page_offsets,
            "tier": self.tier,
            "also_titles": self.also_titles,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(**d)


# ------------------------------------------------------------------ helpers


def _clean(text: str) -> str:
    return _BLANKS_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()


def _first_para(body: str, limit: int = FIRST_PARA_CHARS) -> str:
    """First `limit` characters of real prose, heading line excluded."""
    body = body.strip()
    if not body:
        return ""
    # Drop the heading itself: it is already in `title` and repeating it in the
    # embedded text just makes every section look like its own table of
    # contents entry.
    lines = body.split("\n")
    if lines and len(lines[0]) < 100:
        body = "\n".join(lines[1:]).strip() or body
    out = " ".join(body.split())
    return out[:limit]


def book_title(path: str, reader=None) -> str:
    """Title from PDF metadata, falling back to a tidied filename."""
    if reader is not None:
        try:
            meta = reader.metadata
            if meta and meta.title and meta.title.strip():
                return meta.title.strip()
        except Exception:
            pass
    stem = os.path.splitext(os.path.basename(path))[0]
    return _WS_RE.sub(" ", stem.replace("_", " ").replace("-", " ")).strip() or stem


def _join_pages(
    pages: Sequence[str], start: int, end: int
) -> Tuple[str, List[int]]:
    """Concatenate pages `[start, end)`, returning the text and page offsets."""
    parts: List[str] = []
    offsets: List[int] = []
    total = 0
    for p in range(start, end):
        offsets.append(total)
        chunk = pages[p].strip() + "\n\n"
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts).strip(), offsets


def _drop_duplicate_text(sections: List[Section]) -> List[Section]:
    """Drop sections whose text an earlier section already carries verbatim.

    A backstop under the tier-specific splitting, not a substitute for it: a
    malformed outline whose destinations run backwards, a heading repeated by
    the extractor, or a book with genuinely identical pages all produce two
    documents with one document's content. Duplicates are not harmless — each
    copy adds to every one of its terms' document frequency, which is the
    number idf and the router's selectivity test are computed from.

    The first occurrence wins, so the surviving copy keeps the breadcrumb of
    whichever section the splitter reached first.
    """
    out: List[Section] = []
    seen = set()
    for sec in sections:
        # Digest rather than the text itself: the set is per-book, but a
        # 2,000-page book is a lot of string to hold a second copy of.
        key = hashlib.blake2b(
            " ".join(sec.full_text.split()).encode(), digest_size=16
        ).digest()
        if key in seen:
            continue
        seen.add(key)
        out.append(sec)
    return out


def _merge_short(sections: List[Section]) -> List[Section]:
    """Fold sections below `MIN_SECTION_CHARS` into their predecessor.

    A stub section is almost always a heading whose body landed on the next
    page, or a running header the outline pointed at. Merging keeps the text
    with its neighbour rather than creating a document that is all title.
    """
    out: List[Section] = []
    for sec in sections:
        if out and len(sec.full_text) < MIN_SECTION_CHARS:
            prev = out[-1]
            joiner = "\n\n"
            shift = len(prev.full_text) + len(joiner)
            prev.full_text = prev.full_text + joiner + sec.full_text
            prev.page_offsets = prev.page_offsets + [
                shift + o for o in (sec.page_offsets or [0])
            ]
            prev.page_end = max(prev.page_end, sec.page_end)
            if not prev.first_para:
                prev.first_para = sec.first_para
            prev.also_titles = prev.also_titles + [sec.title] + sec.also_titles
            continue
        out.append(sec)
    for i, sec in enumerate(out):
        sec.index = i
    return out


# ------------------------------------------------------------------- tier 1


def _outline_entries(reader) -> List[Tuple[int, str, int]]:
    """Flatten a pypdf outline into `(depth, title, page_index)`, in order.

    pypdf represents children as a nested list following their parent, so depth
    is the list nesting level. Destinations that cannot be resolved to a page
    are dropped: a bookmark pointing nowhere is worse than no bookmark, because
    it would split a section at an arbitrary place.
    """
    entries: List[Tuple[int, str, int]] = []

    def visit(nodes, depth: int) -> None:
        for node in nodes:
            if isinstance(node, list):
                visit(node, depth + 1)
                continue
            name = str(getattr(node, "title", "") or "").strip()
            try:
                page = reader.get_destination_page_number(node)
            except Exception:
                page = None
            if name and page is not None and page >= 0:
                entries.append((depth, name, int(page)))

    try:
        outline = reader.outline
    except Exception:
        return []
    visit(outline or [], 0)
    return entries


def _sections_from_outline(
    reader, pages: Sequence[str], path: str, title: str
) -> List[Section]:
    """One section per outline entry, spanning up to the next entry.

    Every entry gets a span, containers included, and the spans are disjoint:
    a chapter that has sub-sections keeps only the pages before its first
    sub-section — its preamble — while a chapter that has none runs to the next
    chapter. Taking leaves alone would be tidier and would silently drop the
    text between a chapter heading and its first sub-heading, which on a real
    book is the passage that says what the chapter is about.

    Entries that land on the *same* page are one span, not several. Page is the
    finest resolution a bookmark destination gives us, so a heading and the two
    sub-headings printed below it cannot be told apart, and giving each of them
    the page produced three sections with byte-identical text — a third of a
    real book's index, all of it redundant, inflating df, spending routing
    budget and returning the same paragraph three times under three
    breadcrumbs. They are folded into one section instead; see `also_titles`.
    """
    entries = _outline_entries(reader)
    if not entries:
        return []

    # Breadcrumb ancestry: the nearest preceding entry at each shallower depth.
    ancestors_of: List[List[str]] = []
    stack: List[Tuple[int, str]] = []
    for depth, name, _page in entries:
        while stack and stack[-1][0] >= depth:
            stack.pop()
        ancestors_of.append([n for _d, n in stack])
        stack.append((depth, name))

    # Runs of entries sharing a destination page. The first entry of a run owns
    # the span and names it — it is the heading a reader turning to that page
    # sees first — and the rest ride along in `also_titles`.
    runs: List[List[int]] = []
    for i, (_depth, _name, page) in enumerate(entries):
        if runs and entries[runs[-1][0]][2] == page:
            runs[-1].append(i)
        else:
            runs.append([i])

    n_pages = len(pages)
    sections: List[Section] = []
    for r, run in enumerate(runs):
        head = run[0]
        _depth, name, page = entries[head]
        start = max(0, min(page, n_pages - 1))
        nxt = entries[runs[r + 1][0]][2] if r + 1 < len(runs) else n_pages
        # At least one page, for an outline whose destinations run backwards.
        end = max(start + 1, min(nxt, n_pages))
        text, offsets = _join_pages(pages, start, end)
        if not text:
            continue
        crumbs = [title] + ancestors_of[head] + [name]
        sections.append(
            Section(
                book_path=path,
                breadcrumb=BREADCRUMB_SEP.join(dict.fromkeys(crumbs)),
                page_start=start + 1,
                page_end=end,
                title=name,
                first_para=_first_para(text),
                full_text=text,
                page_offsets=offsets,
                tier="outline",
                also_titles=[entries[i][1] for i in run[1:]],
            )
        )
    return sections


# ------------------------------------------------------------------- tier 2


def _classify_heading(line: str) -> Optional[Tuple[int, str]]:
    """`(level, title)` if the line looks like a heading, else None.

    Level 1 is a chapter, level 2+ a subsection. Deliberately crude: the cost
    of a false positive is one extra split, the cost of a false negative is a
    section that runs long, and neither is fatal.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None

    m = _CHAPTER_RE.match(stripped)
    if m:
        return 1, stripped

    m = _NUMBERED_RE.match(stripped)
    if m:
        number, rest = m.group(1), m.group(2)
        if len(rest) > 2 and not rest[0].islower():
            return 1 + number.count("."), stripped

    m = _ALLCAPS_RE.match(stripped)
    if m and len(stripped.split()) <= 10 and any(c.isalpha() for c in stripped):
        return 1, stripped.title()
    return None


def _sections_from_headings(
    pages: Sequence[str], path: str, title: str
) -> List[Section]:
    # Flatten to lines, remembering which page each came from, so a section
    # spanning a page break still reports both pages.
    lines: List[Tuple[str, int]] = []
    for p, page_text in enumerate(pages):
        for line in page_text.splitlines():
            lines.append((line, p))

    marks: List[Tuple[int, int, str, int]] = []   # line no, level, title, page
    for i, (line, page) in enumerate(lines):
        hit = _classify_heading(line)
        if hit:
            marks.append((i, hit[0], hit[1], page))
    if not marks:
        return []

    sections: List[Section] = []
    chapter: Optional[str] = None
    for j, (line_no, level, name, page) in enumerate(marks):
        end_line = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        body_lines = lines[line_no:end_line]
        if not body_lines:
            continue
        text = _clean("\n".join(l for l, _ in body_lines))
        if not text:
            continue

        if level == 1:
            chapter = name
            crumbs = [title, name]
        else:
            crumbs = [title] + ([chapter] if chapter else []) + [name]

        page_end = body_lines[-1][1]
        # Offsets are per covered page: recompute where each page starts inside
        # the joined text.
        offsets, seen, total = [], -1, 0
        for line, lp in body_lines:
            if lp != seen:
                offsets.append(total)
                seen = lp
            total += len(line) + 1

        sections.append(
            Section(
                book_path=path,
                breadcrumb=BREADCRUMB_SEP.join(dict.fromkeys(crumbs)),
                page_start=page + 1,
                page_end=page_end + 1,
                title=name,
                first_para=_first_para(text),
                full_text=text,
                page_offsets=offsets,
                tier="headings",
            )
        )
    return sections


# ------------------------------------------------------------------- tier 3


def _sections_from_windows(
    pages: Sequence[str], path: str, title: str, window: int = WINDOW_PAGES
) -> List[Section]:
    sections: List[Section] = []
    for start in range(0, len(pages), window):
        end = min(start + window, len(pages))
        text, offsets = _join_pages(pages, start, end)
        if not text:
            continue
        label = f"pp.{start + 1}–{end}"
        sections.append(
            Section(
                book_path=path,
                breadcrumb=f"{title}{BREADCRUMB_SEP}{label}",
                page_start=start + 1,
                page_end=end,
                title=label,
                first_para=_first_para(text),
                full_text=text,
                page_offsets=offsets,
                tier="windows",
            )
        )
    return sections


# --------------------------------------------------------------------- api


def extract_sections(pdf_path: str, window: int = WINDOW_PAGES) -> List[Section]:
    """Split one PDF into sections, best structure available.

    Raises ImportError if pypdf is not installed — a caller scanning a mixed
    directory should catch it and skip, but a caller who asked for this one
    file deserves to know why it produced nothing.
    """
    try:
        from pypdf import PdfReader  # optional extra
    except ImportError as exc:  # pragma: no cover - exercised via the pipeline
        raise ImportError(
            "PDF sectioning needs pypdf. Install it with: pip install 'graphroute[rich]'"
        ) from exc

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    if not any(p.strip() for p in pages):
        return []

    title = book_title(pdf_path, reader)
    sections = _sections_from_outline(reader, pages, pdf_path, title)
    if not sections:
        sections = _sections_from_headings(pages, pdf_path, title)
    if not sections:
        sections = _sections_from_windows(pages, pdf_path, title, window)

    sections = _merge_short(_drop_duplicate_text(sections))
    return sections


def section_from_text(path: str, text: str, index: int = 0) -> Section:
    """Wrap a non-paginated file as a single section.

    Lets a corpus mix books and ordinary files without the pipeline needing two
    code paths: everything downstream sees sections.
    """
    title = book_title(path)
    return Section(
        book_path=path,
        breadcrumb=title,
        page_start=1,
        page_end=1,
        title=title,
        first_para=_first_para(text),
        full_text=text,
        index=index,
        page_offsets=[0],
        tier="file",
    )


def collect_sections(
    source: str,
    include_ext: Optional[Sequence[str]] = None,
    max_bytes: int = 2_000_000,
    window: int = WINDOW_PAGES,
    on_error=None,
) -> List[Section]:
    """Sections for every readable file under `source` (or for one file).

    PDFs are split; anything else the ingest layer can read becomes a single
    section. Files that fail to parse are skipped, not fatal — a corpus of a
    thousand books should not be held hostage by one corrupt download.
    """
    from .ingest import extract_text

    sections: List[Section] = []
    for path in _iter_files(source, include_ext):
        try:
            if path.lower().endswith(".pdf"):
                found = extract_sections(path, window=window)
            else:
                text = extract_text(path, max_bytes=max_bytes)
                found = (
                    [section_from_text(path, text)] if text and text.strip() else []
                )
        except Exception as exc:                      # noqa: BLE001 - see docstring
            if on_error is not None:
                on_error(path, exc)
            continue
        for i, sec in enumerate(found):
            sec.index = i
        sections.extend(found)
    return sections


def _iter_files(source: str, include_ext: Optional[Sequence[str]]) -> Iterator[str]:
    from .ingest import SKIP_DIRS

    allow = set(e.lower() for e in include_ext) if include_ext else None
    if os.path.isfile(source):
        yield source
        return
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        ]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if allow is not None and os.path.splitext(name)[1].lower() not in allow:
                continue
            yield os.path.join(dirpath, name)
