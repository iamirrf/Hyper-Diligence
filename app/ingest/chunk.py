import re
import warnings
from dataclasses import dataclass

import tiktoken
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

MAX_TOKENS = 800
OVERLAP_TOKENS = 150
MIN_TOKENS = 50
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class TextChunk:
    section: str
    chunk_index: int
    content: str
    token_count: int


def html_to_text(html: str) -> str:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(_is_hidden_xbrl_tag):
        tag.decompose()
    for tag in soup.find_all(_is_inline_xbrl_fact_tag):
        tag.unwrap()
    for table in soup.find_all("table"):
        table.replace_with(" ".join(table.get_text(" ", strip=True).split()))
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_filing(form: str, html: str) -> list[TextChunk]:
    text = html_to_text(html)
    sections = split_10k_sections(text) if form == "10-K" else [("8-K", text)]
    chunks: list[TextChunk] = []
    index = 0
    for section, section_text in sections:
        for content in split_text(section_text):
            token_count = count_tokens(content)
            if token_count < MIN_TOKENS:
                continue
            chunks.append(TextChunk(section=section, chunk_index=index, content=content, token_count=token_count))
            index += 1
    return chunks


def split_10k_sections(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?i)\b(Item\s+\d{1,2}[A-Z]?\.)\s*", text))
    if not matches:
        return [("10-K", text)]

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if count_tokens(preamble) >= MIN_TOKENS:
        sections.append(("10-K", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = re.sub(r"\s+", " ", match.group(1)).strip()
        body = text[start:end].strip()
        if body:
            sections.append((heading, body))
    return sections


def split_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    token_ids = _ENCODER.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    stride = max_tokens - overlap
    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk = _ENCODER.decode(token_ids[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(token_ids):
            break
        start += stride
    return chunks


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def _is_hidden_xbrl_tag(tag: object) -> bool:
    name = getattr(tag, "name", None)
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    if lowered in {"ix:header", "ix:hidden", "ix:references", "ix:resources", "xbrli:context", "xbrli:unit"}:
        return True
    style = getattr(tag, "attrs", {}).get("style", "")
    return isinstance(style, str) and "display:none" in style.replace(" ", "").lower()


def _is_inline_xbrl_fact_tag(tag: object) -> bool:
    name = getattr(tag, "name", None)
    return isinstance(name, str) and name.lower() in {"ix:nonnumeric", "ix:nonfraction", "ix:continuation"}
