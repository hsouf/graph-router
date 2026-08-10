"""A minimal PDF writer, so the demo and the tests can build books.

pypdf reads PDFs and can add bookmarks, but it will not lay out text, and the
section extractor is worthless if it is only ever tested against text that was
never inside a real PDF. This writes the smallest valid file that pypdf's
`extract_text` will read back: one Helvetica content stream per page.

Demo and test support — not part of the package, and not a PDF library. It
knows exactly enough to produce a file the extractor can be trusted against.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

PAGE_W, PAGE_H = 612, 792
LEADING = 14
TOP = 730
LEFT = 64


def _escape(s: str) -> str:
    return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: Sequence[str]) -> bytes:
    out = ["BT", "/F1 11 Tf", f"{LEFT} {TOP} Td", f"{LEADING} TL"]
    for i, line in enumerate(lines):
        # Tj draws at the current point; T* advances one leading first.
        if i:
            out.append("T*")
        out.append(f"({_escape(line)}) Tj")
    out.append("ET")
    return "\n".join(out).encode("latin-1", "replace")


def write_pdf(path: str, pages: Sequence[Sequence[str]], title: Optional[str] = None):
    """Write `pages` (a list of lists of text lines) as a PDF at `path`."""
    objects: List[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)          # 1-based object number

    n_pages = len(pages)
    catalog_num = 1
    pages_num = 2
    objects.append(b"")              # placeholder for catalog
    objects.append(b"")              # placeholder for /Pages
    font_num = add(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding "
        b"/WinAnsiEncoding >>"
    )

    kids: List[int] = []
    for lines in pages:
        stream = _content_stream(lines)
        content_num = add(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        page_num = add(
            f"<< /Type /Page /Parent {pages_num} 0 R /MediaBox "
            f"[0 0 {PAGE_W} {PAGE_H}] /Contents {content_num} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>".encode()
        )
        kids.append(page_num)

    info_num = None
    if title:
        info_num = add(f"<< /Title ({_escape(title)}) >>".encode())

    objects[pages_num - 1] = (
        "<< /Type /Pages /Kids [".encode()
        + " ".join(f"{k} 0 R" for k in kids).encode()
        + f"] /Count {n_pages} >>".encode()
    )
    objects[catalog_num - 1] = f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode()

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, body in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(buf)
    buf += f"xref\n0 {len(objects) + 1}\n".encode()
    buf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        buf += f"{off:010d} 00000 n \n".encode()
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog_num} 0 R"
    if info_num:
        trailer += f" /Info {info_num} 0 R"
    trailer += " >>"
    buf += f"trailer\n{trailer}\nstartxref\n{xref_at}\n%%EOF\n".encode()

    with open(path, "wb") as fh:
        fh.write(bytes(buf))
    return path


def write_pdf_with_outline(
    path: str,
    pages: Sequence[Sequence[str]],
    outline: Sequence[Tuple[str, int, int]],
    title: Optional[str] = None,
):
    """As `write_pdf`, plus bookmarks.

    `outline` is a sequence of `(title, page_index, level)` with level 0 at the
    top; children are attached to the most recent shallower entry.
    """
    from pypdf import PdfReader, PdfWriter

    write_pdf(path, pages, title=title)
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.append_pages_from_reader(reader)
    if title:
        writer.add_metadata({"/Title": title})

    parents: dict = {}
    for name, page_index, level in outline:
        parent = parents.get(level - 1) if level else None
        ref = writer.add_outline_item(name, page_index, parent=parent)
        parents[level] = ref
        for deeper in [k for k in parents if k > level]:
            parents.pop(deeper)

    with open(path, "wb") as fh:
        writer.write(fh)
    return path
