from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT: Path = Path(__file__).resolve().parent
TEX_PATH: Path = ROOT / "paper.tex"
BIB_PATH: Path = ROOT / "references.bib"
OUT_DIR: Path = ROOT / "citation_audit"
LOCAL_BIB_DIR: Path = OUT_DIR / "bibtex" / "local"
RESOLVED_BIB_DIR: Path = OUT_DIR / "bibtex" / "resolved"
PAPER_DIR: Path = OUT_DIR / "papers"
REPORT_PATH: Path = OUT_DIR / "citation_audit.md"
CSV_PATH: Path = OUT_DIR / "citation_audit.csv"
JSON_PATH: Path = OUT_DIR / "citation_audit.json"
USER_AGENT: str = "bispectrum-citation-audit/0.1"

CITE_PATTERN: re.Pattern[str] = re.compile(
    r"\\(?:cite|citet|citep|citealp|citeauthor|citeyear|autocite|textcite|parencite|footcite|Citep|Citet|citeyearpar)\*?\{([^}]+)\}"
)

MANUAL_RECORDS: dict[str, dict[str, str | int | None]] = {
    "bonev2023spherical": {
        "source": "manual",
        "title": "Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere",
        "year": 2023,
        "venue": "Proceedings of the 40th International Conference on Machine Learning",
        "doi": None,
        "landing_url": "https://proceedings.mlr.press/v202/bonev23a.html",
        "pdf_url": "https://proceedings.mlr.press/v202/bonev23a/bonev23a.pdf",
    },
    "cesa2022program": {
        "source": "manual",
        "title": "A Program to Build E(N)-Equivariant Steerable CNNs",
        "year": 2022,
        "venue": "International Conference on Learning Representations",
        "doi": None,
        "landing_url": "https://openreview.net/forum?id=WE4qe9xlnQw",
        "pdf_url": "https://openreview.net/pdf?id=WE4qe9xlnQw",
    },
    "oreiller2022bispectral": {
        "source": "manual",
        "title": "Robust Multi-Organ Nucleus Segmentation Using a Locally Rotation Invariant Bispectral U-Net",
        "year": 2022,
        "venue": "Medical Imaging with Deep Learning",
        "doi": None,
        "landing_url": "https://proceedings.mlr.press/v172/oreiller22a.html",
        "pdf_url": "https://proceedings.mlr.press/v172/oreiller22a/oreiller22a.pdf",
    },
    "loshchilov2019adamw": {
        "source": "manual",
        "title": "Decoupled Weight Decay Regularization",
        "year": 2019,
        "venue": "International Conference on Learning Representations",
        "doi": None,
        "landing_url": "https://openreview.net/forum?id=Bkg6RiCqY7",
        "pdf_url": "https://openreview.net/pdf?id=Bkg6RiCqY7",
    },
    "zhemchuzhnikov2025equilopo": {
        "source": "manual",
        "title": "On the Fourier Analysis in the SO(3) Space: the EquiLoPO Network",
        "year": 2025,
        "venue": "International Conference on Learning Representations",
        "doi": None,
        "landing_url": "https://openreview.net/forum?id=LvTSvdiSwG",
        "pdf_url": "https://openreview.net/pdf?id=LvTSvdiSwG",
    },
}

MANUAL_ASSESSMENT_OVERRIDES: dict[str, str] = {}

MANUAL_NOTES: dict[str, str] = {}


@dataclass(slots=True)
class CitationMention:
    line_number: int
    snippet: str


@dataclass(slots=True)
class BibEntry:
    key: str
    entry_type: str
    raw: str
    fields: dict[str, str]

    @property
    def title(self) -> str:
        return self.fields.get("title", "")

    @property
    def year(self) -> int | None:
        raw_year: str = self.fields.get("year", "").strip()
        return int(raw_year) if raw_year.isdigit() else None

    @property
    def doi(self) -> str | None:
        value: str = self.fields.get("doi", "").strip()
        return value or None

    @property
    def url(self) -> str | None:
        value: str = self.fields.get("url", "").strip()
        return value or None

    @property
    def first_author_surname(self) -> str | None:
        authors: str = self.fields.get("author", "")
        if not authors:
            return None
        first_author: str = authors.split(" and ")[0].strip()
        if "," in first_author:
            return first_author.split(",", 1)[0].strip()
        parts: list[str] = first_author.split()
        return parts[-1] if parts else None

    @property
    def venue(self) -> str:
        for field_name in ("journal", "booktitle", "school"):
            value: str = self.fields.get(field_name, "").strip()
            if value:
                return value
        return ""


@dataclass(slots=True)
class RemoteRecord:
    source: str
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    landing_url: str | None = None
    pdf_url: str | None = None
    match_score: float | None = None


@dataclass(slots=True)
class AuditRow:
    key: str
    category: str
    mention_count: int
    local_entry_type: str
    local_title: str
    local_year: int | None
    local_venue: str
    local_doi: str | None
    local_url: str | None
    resolved_source: str | None
    resolved_title: str | None
    resolved_year: int | None
    resolved_venue: str | None
    resolved_doi: str | None
    title_match_score: float | None
    bibtex_source: str
    paper_status: str
    paper_path: str | None
    assessment: str
    notes: list[str] = field(default_factory=list)
    mentions: list[CitationMention] = field(default_factory=list)


def normalize_text(value: str) -> str:
    normalized: str = unicodedata.normalize("NFKD", value)
    ascii_only: str = normalized.encode("ascii", "ignore").decode("ascii")
    lowered: str = ascii_only.casefold()
    return re.sub(r"[^a-z0-9]+", "", lowered)


def plain_text_from_bib(value: str) -> str:
    plain: str = value
    replacements: dict[str, str] = {
        "{": "",
        "}": "",
        "\\&": "&",
        "\\_": "_",
        "~": " ",
        "$": "",
    }
    for old, new in replacements.items():
        plain = plain.replace(old, new)
    plain = re.sub(r"\\[a-zA-Z]+\s*", "", plain)
    plain = re.sub(r"\s+", " ", plain)
    return plain.strip()


def similarity_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def read_url(
    url: str,
    *,
    accept: str | None = None,
    timeout: int = 30,
) -> bytes:
    headers: dict[str, str] = {"User-Agent": USER_AGENT}
    if accept is not None:
        headers["Accept"] = accept
    request: Request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, *, accept: str | None = None, timeout: int = 30) -> str | None:
    try:
        return read_url(url, accept=accept, timeout=timeout).decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError):
        return None


def fetch_json(url: str, *, timeout: int = 30) -> dict[str, Any] | None:
    payload: str | None = fetch_text(url, accept="application/json", timeout=timeout)
    if payload is None:
        return None
    try:
        data: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data


def manual_record_for_key(key: str) -> RemoteRecord | None:
    payload: dict[str, str | int | None] | None = MANUAL_RECORDS.get(key)
    if payload is None:
        return None
    return RemoteRecord(
        source=str(payload["source"]),
        title=str(payload["title"]) if payload["title"] is not None else None,
        year=int(payload["year"]) if payload["year"] is not None else None,
        venue=str(payload["venue"]) if payload["venue"] is not None else None,
        doi=str(payload["doi"]) if payload["doi"] is not None else None,
        landing_url=str(payload["landing_url"]) if payload["landing_url"] is not None else None,
        pdf_url=str(payload["pdf_url"]) if payload["pdf_url"] is not None else None,
        match_score=1.0,
    )


def extract_citations(tex_text: str) -> dict[str, list[CitationMention]]:
    lines: list[str] = tex_text.splitlines()
    mentions: dict[str, list[CitationMention]] = {}
    for line_number, line in enumerate(lines, start=1):
        collapsed: str = " ".join(line.strip().split())
        for match in CITE_PATTERN.finditer(line):
            for raw_key in match.group(1).split(","):
                key: str = raw_key.strip()
                if not key:
                    continue
                mentions.setdefault(key, []).append(
                    CitationMention(line_number=line_number, snippet=collapsed)
                )
    return mentions


def split_bib_entries(bib_text: str) -> list[str]:
    entries: list[str] = []
    index: int = 0
    while True:
        start: int = bib_text.find("@", index)
        if start == -1:
            return entries
        brace_start: int = bib_text.find("{", start)
        if brace_start == -1:
            return entries
        depth: int = 0
        in_quotes: bool = False
        escape_next: bool = False
        for cursor in range(brace_start, len(bib_text)):
            char: str = bib_text[cursor]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_quotes = not in_quotes
                continue
            if in_quotes:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    entries.append(bib_text[start : cursor + 1].strip())
                    index = cursor + 1
                    break
        else:
            return entries


def parse_bib_value(chunk: str, start_index: int) -> tuple[str, int]:
    opener: str = chunk[start_index]
    if opener == "{":
        depth: int = 0
        cursor: int = start_index
        while cursor < len(chunk):
            char: str = chunk[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return chunk[start_index + 1 : cursor], cursor + 1
            cursor += 1
        return chunk[start_index + 1 :], len(chunk)
    if opener == '"':
        cursor = start_index + 1
        escaped: bool = False
        while cursor < len(chunk):
            char = chunk[cursor]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return chunk[start_index + 1 : cursor], cursor + 1
            cursor += 1
        return chunk[start_index + 1 :], len(chunk)
    cursor = start_index
    while cursor < len(chunk) and chunk[cursor] not in ",\n":
        cursor += 1
    return chunk[start_index:cursor].strip(), cursor


def parse_bib_entry(raw_entry: str) -> BibEntry:
    header_match: re.Match[str] | None = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", raw_entry, re.S)
    if header_match is None:
        raise ValueError(f"Could not parse BibTeX header: {raw_entry[:80]}")
    entry_type: str = header_match.group(1).strip().lower()
    key: str = header_match.group(2).strip()
    body: str = raw_entry[header_match.end() :].rstrip().rstrip("}").strip()
    fields: dict[str, str] = {}
    cursor: int = 0
    while cursor < len(body):
        while cursor < len(body) and body[cursor] in " \t\r\n,":
            cursor += 1
        if cursor >= len(body):
            break
        equals_index: int = body.find("=", cursor)
        if equals_index == -1:
            break
        field_name: str = body[cursor:equals_index].strip().lower()
        cursor = equals_index + 1
        while cursor < len(body) and body[cursor].isspace():
            cursor += 1
        if cursor >= len(body):
            break
        field_value, cursor = parse_bib_value(body, cursor)
        fields[field_name] = " ".join(field_value.strip().split())
        while cursor < len(body) and body[cursor] not in "\n,":
            cursor += 1
        if cursor < len(body) and body[cursor] == ",":
            cursor += 1
    return BibEntry(key=key, entry_type=entry_type, raw=raw_entry + "\n", fields=fields)


def load_bib_entries(bib_text: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}
    for raw_entry in split_bib_entries(bib_text):
        entry: BibEntry = parse_bib_entry(raw_entry)
        entries[entry.key] = entry
    return entries


def crossref_record_from_item(item: dict[str, Any], score: float) -> RemoteRecord:
    titles: list[str] = item.get("title") or []
    container_titles: list[str] = item.get("container-title") or []
    year_parts: list[list[int]] = (
        item.get("published-print", {}).get("date-parts")
        or item.get("published-online", {}).get("date-parts")
        or item.get("issued", {}).get("date-parts")
        or []
    )
    year: int | None = year_parts[0][0] if year_parts and year_parts[0] else None
    doi: str | None = item.get("DOI")
    landing_url: str | None = f"https://doi.org/{doi}" if doi else item.get("URL")
    return RemoteRecord(
        source="crossref",
        title=titles[0] if titles else None,
        year=year,
        venue=container_titles[0] if container_titles else None,
        doi=doi,
        landing_url=landing_url,
        match_score=score,
    )


def search_crossref(title: str, first_author_surname: str | None, year: int | None) -> RemoteRecord | None:
    query: str = quote(title)
    url: str = f"https://api.crossref.org/works?query.bibliographic={query}&rows=5"
    payload: dict[str, Any] | None = fetch_json(url)
    if payload is None:
        return None
    items: list[dict[str, Any]] = payload.get("message", {}).get("items", [])
    best_record: RemoteRecord | None = None
    best_score: float = 0.0
    for item in items:
        candidate_titles: list[str] = item.get("title") or []
        if not candidate_titles:
            continue
        candidate_title: str = candidate_titles[0]
        score: float = similarity_score(title, candidate_title)
        authors: list[dict[str, Any]] = item.get("author") or []
        if first_author_surname and authors:
            candidate_surname: str = str(authors[0].get("family", "")).strip()
            if normalize_text(candidate_surname) == normalize_text(first_author_surname):
                score += 0.05
        candidate_year_parts: list[list[int]] = (
            item.get("published-print", {}).get("date-parts")
            or item.get("published-online", {}).get("date-parts")
            or item.get("issued", {}).get("date-parts")
            or []
        )
        candidate_year: int | None = (
            candidate_year_parts[0][0] if candidate_year_parts and candidate_year_parts[0] else None
        )
        if year is not None and candidate_year is not None:
            if abs(year - candidate_year) == 0:
                score += 0.05
            elif abs(year - candidate_year) > 1:
                score -= 0.15
        if score > best_score:
            best_score = score
            best_record = crossref_record_from_item(item, score)
    if best_record is None or best_score < 0.72:
        return None
    return best_record


def openalex_record_from_result(result: dict[str, Any], score: float) -> RemoteRecord:
    primary_location: dict[str, Any] = result.get("primary_location") or {}
    best_oa_location: dict[str, Any] = result.get("best_oa_location") or {}
    pdf_url: str | None = (
        best_oa_location.get("pdf_url")
        or primary_location.get("pdf_url")
        or result.get("open_access", {}).get("oa_url")
    )
    landing_url: str | None = (
        best_oa_location.get("landing_page_url")
        or primary_location.get("landing_page_url")
        or result.get("doi")
    )
    publication_year: int | None = result.get("publication_year")
    venue_name: str | None = None
    if primary_location.get("source"):
        venue_name = primary_location["source"].get("display_name")
    return RemoteRecord(
        source="openalex",
        title=result.get("display_name"),
        year=publication_year,
        venue=venue_name,
        doi=(result.get("doi") or "").removeprefix("https://doi.org/") or None,
        landing_url=landing_url,
        pdf_url=pdf_url,
        match_score=score,
    )


def search_openalex(title: str, first_author_surname: str | None, year: int | None) -> RemoteRecord | None:
    url: str = f"https://api.openalex.org/works?search={quote(title)}&per_page=5"
    payload: dict[str, Any] | None = fetch_json(url)
    if payload is None:
        return None
    results: list[dict[str, Any]] = payload.get("results", [])
    best_record: RemoteRecord | None = None
    best_score: float = 0.0
    for result in results:
        candidate_title: str = str(result.get("display_name", "")).strip()
        if not candidate_title:
            continue
        score: float = similarity_score(title, candidate_title)
        authorships: list[dict[str, Any]] = result.get("authorships") or []
        if first_author_surname and authorships:
            first_author: dict[str, Any] = authorships[0].get("author") or {}
            display_name: str = str(first_author.get("display_name", "")).strip()
            candidate_surname: str = display_name.split()[-1] if display_name else ""
            if normalize_text(candidate_surname) == normalize_text(first_author_surname):
                score += 0.05
        candidate_year: int | None = result.get("publication_year")
        if year is not None and candidate_year is not None:
            if abs(year - candidate_year) == 0:
                score += 0.05
            elif abs(year - candidate_year) > 1:
                score -= 0.15
        if score > best_score:
            best_score = score
            best_record = openalex_record_from_result(result, score)
    if best_record is None or best_score < 0.72:
        return None
    return best_record


def enrich_with_openalex(record: RemoteRecord, title: str) -> RemoteRecord:
    if record.pdf_url and record.landing_url:
        return record
    openalex_record: RemoteRecord | None = None
    if record.doi:
        url: str = f"https://api.openalex.org/works?filter=doi:https://doi.org/{quote(record.doi, safe='')}&per_page=1"
        payload: dict[str, Any] | None = fetch_json(url)
        results: list[dict[str, Any]] = payload.get("results", []) if payload else []
        if results:
            openalex_record = openalex_record_from_result(results[0], similarity_score(title, results[0].get("display_name", "")))
    if openalex_record is None:
        openalex_record = search_openalex(title, None, record.year)
    if openalex_record is None:
        return record
    if record.doi and not openalex_record.doi:
        openalex_record.doi = record.doi
    if record.match_score and (openalex_record.match_score or 0.0) < record.match_score:
        openalex_record.match_score = record.match_score
    if record.source == "crossref":
        if openalex_record.title is None:
            openalex_record.title = record.title
        if openalex_record.year is None:
            openalex_record.year = record.year
        if openalex_record.venue is None:
            openalex_record.venue = record.venue
    return openalex_record


def extract_arxiv_id(entry: BibEntry) -> str | None:
    candidates: list[str] = []
    if entry.url:
        candidates.append(entry.url)
    for field_name in ("journal", "booktitle", "note"):
        value: str = entry.fields.get(field_name, "")
        if value:
            candidates.append(value)
    for candidate in candidates:
        match: re.Match[str] | None = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", candidate, re.I)
        if match is not None:
            return match.group(1)
        match = re.search(r"arXiv:([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", candidate, re.I)
        if match is not None:
            return match.group(1)
    return None


def resolve_remote_record(entry: BibEntry) -> RemoteRecord | None:
    manual_record: RemoteRecord | None = manual_record_for_key(entry.key)
    if manual_record is not None:
        return manual_record
    search_title: str = plain_text_from_bib(entry.title)
    if entry.doi:
        landing_url: str = f"https://doi.org/{entry.doi}"
        return enrich_with_openalex(
            RemoteRecord(
                source="doi",
                title=search_title,
                year=entry.year,
                venue=entry.venue,
                doi=entry.doi,
                landing_url=landing_url,
                match_score=1.0,
            ),
            search_title,
        )
    arxiv_id: str | None = extract_arxiv_id(entry)
    if arxiv_id is not None:
        return RemoteRecord(
            source="arxiv",
            title=search_title,
            year=entry.year,
            venue=entry.venue,
            doi=entry.doi,
            landing_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            match_score=1.0,
        )
    openalex_record: RemoteRecord | None = search_openalex(
        search_title,
        entry.first_author_surname,
        entry.year,
    )
    crossref_record: RemoteRecord | None = search_crossref(
        search_title,
        entry.first_author_surname,
        entry.year,
    )
    if openalex_record is not None and crossref_record is not None:
        return openalex_record if (openalex_record.match_score or 0.0) >= (crossref_record.match_score or 0.0) else enrich_with_openalex(crossref_record, search_title)
    if openalex_record is not None:
        return openalex_record
    if crossref_record is not None:
        return enrich_with_openalex(crossref_record, search_title)
    return None


def download_resolved_bibtex(entry: BibEntry, record: RemoteRecord | None) -> tuple[str, str]:
    local_path: Path = LOCAL_BIB_DIR / f"{entry.key}.bib"
    local_path.write_text(entry.raw, encoding="utf-8")
    if record is not None and record.doi:
        bibtex_text: str | None = fetch_text(
            f"https://doi.org/{record.doi}",
            accept="application/x-bibtex",
        )
        if bibtex_text:
            resolved_path: Path = RESOLVED_BIB_DIR / f"{entry.key}.bib"
            resolved_path.write_text(bibtex_text.strip() + "\n", encoding="utf-8")
            return "remote-doi", str(resolved_path.relative_to(ROOT))
    resolved_path = RESOLVED_BIB_DIR / f"{entry.key}.bib"
    resolved_path.write_text(entry.raw, encoding="utf-8")
    return "local-fallback", str(resolved_path.relative_to(ROOT))


def download_paper(entry: BibEntry, record: RemoteRecord | None) -> tuple[str, str | None]:
    candidate_pdf_urls: list[str] = []
    candidate_links: list[str] = []
    if record is not None:
        if record.pdf_url:
            candidate_pdf_urls.append(record.pdf_url)
        if record.landing_url:
            candidate_links.append(record.landing_url)
    if entry.url:
        candidate_links.append(entry.url)
    arxiv_id: str | None = extract_arxiv_id(entry)
    if arxiv_id is not None:
        candidate_pdf_urls.insert(0, f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        candidate_links.insert(0, f"https://arxiv.org/abs/{arxiv_id}")
    seen: set[str] = set()
    for pdf_url in candidate_pdf_urls:
        if pdf_url in seen:
            continue
        seen.add(pdf_url)
        try:
            payload: bytes = read_url(pdf_url, timeout=45)
        except (HTTPError, URLError, TimeoutError):
            continue
        if not payload.startswith(b"%PDF"):
            continue
        paper_path: Path = PAPER_DIR / f"{entry.key}.pdf"
        paper_path.write_bytes(payload)
        return "downloaded", str(paper_path.relative_to(ROOT))
    for link in candidate_links:
        if link in seen:
            continue
        seen.add(link)
        link_path: Path = PAPER_DIR / f"{entry.key}.link.txt"
        link_path.write_text(link + "\n", encoding="utf-8")
        return "link-only", str(link_path.relative_to(ROOT))
    return "unavailable", None


def summarize_category(mentions: list[CitationMention]) -> str:
    combined: str = " ".join(mention.snippet for mention in mentions).casefold()
    if any(token in combined for token in ("patchcamelyon", "organmnist", "camelyon", "dataset", "contains", "consists of")):
        return "dataset"
    if any(token in combined for token in ("pytorch", "numpy", "torch_harmonics")):
        return "software"
    if any(token in combined for token in ("adamw", "backbone", "trained with", "reference numbers", "follows")):
        return "experimental"
    if any(token in combined for token in ("introduced", "applied", "explore", "use iterated", "provides", "extend this")):
        return "related-work"
    return "theory"


def build_assessment(entry: BibEntry, record: RemoteRecord | None, paper_status: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    title_score: float | None = None
    if record is not None and record.title:
        title_score = similarity_score(entry.title, record.title)
    assessment: str = "good"
    if record is None:
        assessment = "check"
        notes.append("No confident live metadata match found.")
    elif title_score is not None and title_score < 0.9:
        assessment = "check"
        notes.append("Resolved title differs materially from the local BibTeX title.")
    if record is not None and entry.year and record.year and abs(entry.year - record.year) > 1:
        assessment = "check"
        notes.append(f"Year mismatch: local {entry.year}, resolved {record.year}.")
    resolved_venue: str = record.venue if record is not None and record.venue is not None else ""
    venue_text: str = f"{entry.venue} {resolved_venue}".casefold()
    if entry.entry_type == "phdthesis":
        if assessment == "good":
            assessment = "acceptable-but-weaker"
        notes.append("Foundational thesis citation rather than an archival paper.")
    if "arxiv preprint" in venue_text or (record is not None and record.source == "arxiv"):
        if assessment == "good":
            assessment = "acceptable-but-weaker"
        notes.append("Preprint citation; archival version may be preferable if one exists.")
    if "workshop" in venue_text:
        if assessment == "good":
            assessment = "acceptable-but-weaker"
        notes.append("Workshop citation rather than a main archival venue.")
    if entry.doi is None and (record is None or record.doi is None):
        notes.append("No DOI found.")
    if paper_status == "downloaded":
        notes.append("Open PDF downloaded.")
    elif paper_status == "link-only":
        notes.append("Saved landing link only; no open PDF downloaded.")
    else:
        notes.append("No paper URL resolved.")
    manual_note: str | None = MANUAL_NOTES.get(entry.key)
    if manual_note is not None:
        notes.append(manual_note)
    manual_assessment: str | None = MANUAL_ASSESSMENT_OVERRIDES.get(entry.key)
    if manual_assessment is not None:
        assessment = manual_assessment
    if not notes:
        notes.append("Metadata matches and assets resolved cleanly.")
    return assessment, notes


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def render_report(rows: list[AuditRow]) -> str:
    downloaded_count: int = sum(1 for row in rows if row.paper_status == "downloaded")
    link_only_count: int = sum(1 for row in rows if row.paper_status == "link-only")
    stronger_count: int = sum(1 for row in rows if row.assessment == "good")
    weaker_count: int = sum(1 for row in rows if row.assessment == "acceptable-but-weaker")
    check_count: int = sum(1 for row in rows if row.assessment == "check")
    lines: list[str] = [
        "# Citation Audit",
        "",
        f"- Total cited keys: {len(rows)}",
        f"- Assessments: {stronger_count} good, {weaker_count} acceptable-but-weaker, {check_count} check",
        f"- Paper assets: {downloaded_count} PDFs downloaded, {link_only_count} link-only, {len(rows) - downloaded_count - link_only_count} unavailable",
        "",
        "| Key | Category | Mentions | Assessment | Resolved source | BibTeX | Paper | Notes |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        notes: str = "; ".join(row.notes[:2])
        resolved_source: str = row.resolved_source or "unresolved"
        paper_cell: str = row.paper_status if row.paper_path is None else f"{row.paper_status} (`{row.paper_path}`)"
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row.key),
                    markdown_escape(row.category),
                    str(row.mention_count),
                    markdown_escape(row.assessment),
                    markdown_escape(resolved_source),
                    markdown_escape(row.bibtex_source),
                    markdown_escape(paper_cell),
                    markdown_escape(notes),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for row in rows:
        lines.append(f"### `{row.key}`")
        lines.append("")
        lines.append(f"- Local title: {row.local_title}")
        lines.append(f"- Resolved title: {row.resolved_title or 'unresolved'}")
        lines.append(f"- Assessment: {row.assessment}")
        lines.append(f"- BibTeX source: {row.bibtex_source}")
        lines.append(f"- Paper asset: {row.paper_status}{f' (`{row.paper_path}`)' if row.paper_path else ''}")
        lines.append(f"- Notes: {'; '.join(row.notes)}")
        if row.mentions:
            lines.append("- Citation contexts:")
            for mention in row.mentions[:3]:
                lines.append(f"  - L{mention.line_number}: {mention.snippet}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_csv(rows: list[AuditRow]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "key",
                "category",
                "mention_count",
                "assessment",
                "local_entry_type",
                "local_title",
                "local_year",
                "local_venue",
                "resolved_source",
                "resolved_title",
                "resolved_year",
                "resolved_venue",
                "resolved_doi",
                "title_match_score",
                "bibtex_source",
                "paper_status",
                "paper_path",
                "notes",
                "mention_lines",
                "mention_snippets",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.key,
                    row.category,
                    row.mention_count,
                    row.assessment,
                    row.local_entry_type,
                    row.local_title,
                    row.local_year,
                    row.local_venue,
                    row.resolved_source,
                    row.resolved_title,
                    row.resolved_year,
                    row.resolved_venue,
                    row.resolved_doi,
                    row.title_match_score,
                    row.bibtex_source,
                    row.paper_status,
                    row.paper_path,
                    " | ".join(row.notes),
                    ",".join(str(mention.line_number) for mention in row.mentions),
                    " || ".join(mention.snippet for mention in row.mentions),
                ]
            )


def write_json(rows: list[AuditRow]) -> None:
    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        payload: dict[str, Any] = asdict(row)
        serialized_rows.append(payload)
    JSON_PATH.write_text(json.dumps(serialized_rows, indent=2), encoding="utf-8")


def ensure_output_dirs() -> None:
    for directory in (LOCAL_BIB_DIR, RESOLVED_BIB_DIR, PAPER_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def audit_citations() -> list[AuditRow]:
    tex_text: str = TEX_PATH.read_text(encoding="utf-8")
    bib_text: str = BIB_PATH.read_text(encoding="utf-8")
    mentions_by_key: dict[str, list[CitationMention]] = extract_citations(tex_text)
    bib_entries: dict[str, BibEntry] = load_bib_entries(bib_text)
    rows: list[AuditRow] = []
    for key in sorted(mentions_by_key):
        if key not in bib_entries:
            raise KeyError(f"Citation key `{key}` is used in paper.tex but missing from references.bib")
        entry: BibEntry = bib_entries[key]
        mentions: list[CitationMention] = mentions_by_key[key]
        record: RemoteRecord | None = resolve_remote_record(entry)
        bibtex_source, _ = download_resolved_bibtex(entry, record)
        paper_status, paper_path = download_paper(entry, record)
        assessment, notes = build_assessment(entry, record, paper_status)
        resolved_title: str | None = record.title if record else None
        title_match_score: float | None = (
            similarity_score(entry.title, resolved_title) if resolved_title else None
        )
        rows.append(
            AuditRow(
                key=key,
                category=summarize_category(mentions),
                mention_count=len(mentions),
                local_entry_type=entry.entry_type,
                local_title=entry.title,
                local_year=entry.year,
                local_venue=entry.venue,
                local_doi=entry.doi,
                local_url=entry.url,
                resolved_source=record.source if record else None,
                resolved_title=resolved_title,
                resolved_year=record.year if record else None,
                resolved_venue=record.venue if record else None,
                resolved_doi=record.doi if record else None,
                title_match_score=title_match_score,
                bibtex_source=bibtex_source,
                paper_status=paper_status,
                paper_path=paper_path,
                assessment=assessment,
                notes=notes,
                mentions=mentions,
            )
        )
        print(f"[audit] {key}: {assessment}, paper={paper_status}, bib={bibtex_source}")
    return rows


def main() -> None:
    ensure_output_dirs()
    rows: list[AuditRow] = audit_citations()
    REPORT_PATH.write_text(render_report(rows), encoding="utf-8")
    write_csv(rows)
    write_json(rows)
    print(f"[done] wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"[done] wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"[done] wrote {JSON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
