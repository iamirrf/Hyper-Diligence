import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}"
CACHE_DIR = Path("data")
TICKER_CACHE = CACHE_DIR / "company_tickers.json"
REQUEST_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class TargetFiling:
    accession: str
    form: str
    filed: date
    primary_document: str
    source_url: str
    items: str | None = None


def ticker_to_cik(ticker: str) -> int:
    """A local ticker cache avoids repeatedly touching SEC's reference file."""

    records = _load_company_tickers()
    normalized = ticker.upper()
    for record in records:
        if str(record["ticker"]).upper() == normalized:
            return int(record["cik_str"])
    raise ValueError(f"Ticker not found in SEC company_tickers.json: {ticker}")


def list_target_filings(cik: int) -> list[TargetFiling]:
    response = _get_json(SUBMISSIONS_URL.format(cik=cik))
    recent = response.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])
    items = recent.get("items", [])

    filings: list[TargetFiling] = []
    for index, form in enumerate(forms):
        accession = str(accessions[index])
        primary_doc = str(primary_docs[index])
        accession_no_dashes = accession.replace("-", "")
        item_text = _safe_index(items, index)
        filings.append(
            TargetFiling(
                accession=accession,
                form=str(form),
                filed=date.fromisoformat(str(filing_dates[index])),
                primary_document=primary_doc,
                source_url=ARCHIVE_URL.format(
                    cik=cik,
                    accession_no_dashes=accession_no_dashes,
                    primary_doc=primary_doc,
                ),
                items=item_text,
            )
        )

    annuals = [filing for filing in filings if filing.form == "10-K"][:2]
    eight_ks = [filing for filing in filings if filing.form == "8-K"]
    earnings_8ks = [filing for filing in eight_ks if filing.items and "2.02" in filing.items]
    selected_8ks = earnings_8ks[:2] if len(earnings_8ks) >= 2 else eight_ks[:2]
    return annuals + selected_8ks


def fetch_primary_document(cik: int, accession: str, primary_doc: str) -> str:
    accession_no_dashes = accession.replace("-", "")
    url = ARCHIVE_URL.format(cik=cik, accession_no_dashes=accession_no_dashes, primary_doc=primary_doc)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = _get(url)
            if response.status_code == 200:
                return response.text
            last_error = RuntimeError(f"EDGAR returned {response.status_code} for {url}")
            logger.warning("edgar_non_200", extra={"url": url, "status_code": response.status_code, "attempt": attempt})
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("edgar_request_failed", extra={"url": url, "attempt": attempt, "error": str(exc)})
    raise RuntimeError(f"Unable to fetch EDGAR document {url}") from last_error


def _load_company_tickers() -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not TICKER_CACHE.exists():
        response = _get(COMPANY_TICKERS_URL)
        response.raise_for_status()
        TICKER_CACHE.write_text(response.text, encoding="utf-8")
    raw = json.loads(TICKER_CACHE.read_text(encoding="utf-8"))
    return list(raw.values())


def _get_json(url: str) -> dict[str, Any]:
    response = _get(url)
    response.raise_for_status()
    return response.json()


def _get(url: str) -> httpx.Response:
    settings = get_settings()
    time.sleep(REQUEST_DELAY_SECONDS)
    return httpx.get(url, headers={"User-Agent": settings.edgar_user_agent}, timeout=30.0, follow_redirects=True)


def _safe_index(items: list[Any], index: int) -> str | None:
    try:
        value = items[index]
    except (IndexError, TypeError):
        return None
    if value is None:
        return None
    return str(value)
