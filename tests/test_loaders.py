"""Stage 1 — loader tests."""

from __future__ import annotations

from pathlib import Path

from secbrn.ingest.loaders import load_markdown, load_transcript, load_web

FIX = Path(__file__).parent / "fixtures"


def test_markdown_title_and_links():
    doc = load_markdown(FIX / "retrieval.md")
    assert doc.source_type == "markdown"
    assert doc.title == "Retrieval Notes"
    assert "Rerankers" in doc.links  # [[wikilink]] captured


def test_transcript_turns():
    doc = load_transcript(FIX / "meeting.transcript.txt")
    assert doc.source_type == "transcript"
    assert "[[TURN 1]]" in doc.raw_text
    assert doc.meta["turns"] == 3
    assert "Alice:" in doc.raw_text


def test_web_readability_strips_chrome():
    html = (FIX / "article.html").read_text(encoding="utf-8")
    doc = load_web("https://example.com/graphrag", html=html)
    assert doc.title == "GraphRAG Overview"
    assert "copyright" not in doc.raw_text       # footer dropped
    assert "home about" not in doc.raw_text       # nav dropped
    assert "GraphRAG" in doc.raw_text
    assert any("example.com" in l for l in doc.links)


def test_relative_path_gets_valid_file_uri(tmp_path, monkeypatch):
    # Regression: Path.as_uri() throws on relative paths; loaders must resolve first.
    d = tmp_path / "notes"
    d.mkdir()
    (d / "n.md").write_text("# N\nhello", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    doc = load_markdown(Path("notes/n.md"))  # relative path
    assert doc.uri.startswith("file://")
    assert doc.title == "N"
