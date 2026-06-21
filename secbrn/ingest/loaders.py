"""Stage 1 — Load. Source-specific loaders → a common raw ``Document``.

Each loader preserves provenance (uri, title, created_at) and discovers explicit
links (markdown ``[[wikilinks]]`` / hyperlinks) which Stage 5 promotes to
high-confidence edges. PDFs keep page boundaries; transcripts keep turns.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from secbrn.models import Document, utcnow_iso

WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
MD_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _new_id() -> str:
    return uuid.uuid4().hex


def _file_uri(p: Path) -> str:
    """A valid file:// URI even for relative paths (Path.as_uri requires absolute)."""
    try:
        return p.resolve().as_uri()
    except Exception:
        return "file://" + str(p.resolve())


# ── Markdown / notes ───────────────────────────────────────────────────────────
def load_markdown(path: str | Path) -> Document:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    title = _md_title(text) or p.stem
    links = WIKILINK.findall(text) + [
        m for m in MD_LINK.findall(text) if not m.startswith("#")
    ]
    return Document(
        id=_new_id(),
        source_type="markdown",
        uri=_file_uri(p),
        title=title,
        raw_text=text,
        created_at=utcnow_iso(),
        links=sorted(set(links)),
        meta={"path": str(p)},
    )


def _md_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


# ── PDF ─────────────────────────────────────────────────────────────────────────
def load_pdf(path: str | Path) -> Document:
    """Extract text with per-page markers ``[[PAGE n]]`` so chunking keeps page spans."""
    p = Path(path)
    try:
        import fitz  # pymupdf
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pymupdf required for PDF loading: pip install pymupdf") from e

    doc = fitz.open(str(p))
    parts: list[str] = []
    for i, page in enumerate(doc, start=1):
        parts.append(f"[[PAGE {i}]]\n{page.get_text()}")
    text = "\n".join(parts)
    title = (doc.metadata or {}).get("title") or p.stem
    doc.close()
    return Document(
        id=_new_id(),
        source_type="pdf",
        uri=_file_uri(p),
        title=title,
        raw_text=text,
        created_at=utcnow_iso(),
        meta={"path": str(p), "pages": len(parts)},
    )


# ── Web ──────────────────────────────────────────────────────────────────────────
def load_web(url: str, *, html: str | None = None) -> Document:
    """Fetch + readability-extract clean article text.

    ``html`` can be supplied directly (e.g. from a clipper or test) to avoid network.
    """
    if html is None:  # pragma: no cover - network
        import httpx

        html = httpx.get(url, timeout=30, follow_redirects=True).text
    title, text, links = _readability(html, base_url=url)
    return Document(
        id=_new_id(),
        source_type="web",
        uri=url,
        title=title or url,
        raw_text=text,
        created_at=utcnow_iso(),
        links=links,
        meta={"url": url},
    )


def _readability(html: str, base_url: str = "") -> tuple[str, str, list[str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    # Prefer <article>, fall back to <main>, then body.
    root = soup.find("article") or soup.find("main") or soup.body or soup
    parts: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "p", "li"]):
        t = el.get_text(" ", strip=True)
        if t:
            prefix = "# " if el.name == "h1" else "## " if el.name in ("h2", "h3") else ""
            parts.append(prefix + t)
    links = [a.get("href") for a in root.find_all("a", href=True)]
    links = [l for l in links if l and l.startswith("http")]
    return title, "\n\n".join(parts), sorted(set(links))


# ── Transcript ───────────────────────────────────────────────────────────────────
_TURN = re.compile(r"^(?:\[(?P<ts>[^\]]+)\]\s*)?(?P<speaker>[A-Za-z0-9 _.-]{1,40}):\s*(?P<text>.*)$")


def load_transcript(path: str | Path) -> Document:
    """Parse role-tagged turns. Supports ``Speaker: text`` and ``[timestamp] Speaker: text``."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    turns: list[str] = []
    n = 0
    cur_speaker = None
    buf: list[str] = []

    def flush():
        nonlocal n, buf, cur_speaker
        if cur_speaker is not None and buf:
            n += 1
            turns.append(f"[[TURN {n}]] {cur_speaker}: " + " ".join(buf).strip())
        buf = []

    for line in raw.splitlines():
        m = _TURN.match(line.strip())
        if m:
            flush()
            cur_speaker = m.group("speaker").strip()
            buf = [m.group("text").strip()]
        elif line.strip():
            buf.append(line.strip())
    flush()
    text = "\n".join(turns) if turns else raw
    return Document(
        id=_new_id(),
        source_type="transcript",
        uri=_file_uri(p),
        title=p.stem,
        raw_text=text,
        created_at=utcnow_iso(),
        meta={"path": str(p), "turns": n},
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────────
_EXT = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_markdown,
    ".pdf": load_pdf,
}


def load_path(path: str | Path) -> Document:
    """Load a single file, dispatching on extension. Transcripts: ``*.transcript.txt``."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".transcript.txt") or name.endswith(".vtt") or name.endswith(".srt"):
        return load_transcript(p)
    loader = _EXT.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {p.suffix} ({p})")
    return loader(p)


def iter_folder(folder: str | Path) -> list[Path]:
    """All ingestible files under a folder (recursive)."""
    root = Path(folder)
    exts = {".md", ".markdown", ".txt", ".pdf"}
    return sorted(q for q in root.rglob("*") if q.is_file() and q.suffix.lower() in exts)
