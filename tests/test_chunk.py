from app.ingest.chunk import count_tokens, chunk_filing, split_text


def test_split_text_respects_max_tokens_and_overlap() -> None:
    text = " ".join(f"token{i}" for i in range(1200))
    chunks = split_text(text, max_tokens=100, overlap=20)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 100 for chunk in chunks)
    assert any(word in chunks[0] and word in chunks[1] for word in chunks[0].split()[-30:])


def test_chunk_filing_drops_tiny_chunks() -> None:
    assert chunk_filing("8-K", "<html><body><p>too small</p></body></html>") == []


def test_html_to_text_removes_hidden_xbrl_but_keeps_visible_facts() -> None:
    html = """
    <html>
      <body>
        <ix:header>
          <ix:hidden>taxonomy noise context false 2025 FY</ix:hidden>
          <xbrli:context>more metadata</xbrli:context>
        </ix:header>
        <p>Revenue was <ix:nonfraction>100</ix:nonfraction> million.</p>
      </body>
    </html>
    """

    chunks = chunk_filing("8-K", html + (" real filing prose" * 80))
    text = " ".join(chunk.content for chunk in chunks)

    assert "taxonomy noise" not in text
    assert "more metadata" not in text
    assert "Revenue was 100 million" in text
