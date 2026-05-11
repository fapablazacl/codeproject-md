#!/usr/bin/env python3
"""
crawl_sections.py

Discover CodeProject article snapshots in selected KB sections via the
Wayback Machine CDX API, pick the newest valid snapshot per article, and
write a CSV index used by orchestrate.py.

Output CSV columns:
    article_id, section, slug, title, original_url,
    snapshot_timestamp, wayback_url

Usage:
    python crawl_sections.py --sections sections.txt --output snapshots.csv
    python crawl_sections.py --section KB/dialog --section KB/shell \\
        --output snapshots.csv --concurrency 4

Dependencies: httpx
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

import httpx

from wayback_to_md import (
    USER_AGENT,
    extract_title,
    normalize_original,
    slug_from_url,
    with_flag,
)
from bs4 import BeautifulSoup


CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
MIN_BODY_BYTES = 4096

REJECT_BASENAMES = {
    "default.aspx",
    "index.aspx",
    "list.aspx",
    "articlelist.aspx",
    "section.aspx",
}


@dataclass
class SnapshotRow:
    section: str
    slug: str
    title: str
    original_url: str
    snapshot_timestamp: str
    wayback_url: str


@dataclass
class ExistingIndex:
    max_id: int
    seen_originals: set[str]
    slug_taken_by_section: dict[str, set[str]]
    header_present: bool


# ---------- CDX ----------


async def cdx_prefix_query(
    client: httpx.AsyncClient,
    section: str,
    *,
    max_retries: int = 4,
) -> list[tuple[str, str]]:
    """Return raw [(timestamp, original)] rows for codeproject.com/<section>/*."""
    params = [
        ("url", f"codeproject.com/{section}/"),
        ("matchType", "prefix"),
        ("filter", "statuscode:200"),
        ("filter", "mimetype:text/html"),
        ("collapse", "urlkey"),
        ("output", "json"),
        ("fl", "timestamp,original"),
        ("sort", "reverse"),
    ]
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            r = await client.get(CDX_ENDPOINT, params=params, timeout=120)
        except httpx.HTTPError as e:
            if attempt == max_retries - 1:
                print(f"  ! CDX network error: {section}: {e}", file=sys.stderr)
                return []
            await asyncio.sleep(backoff ** attempt + random.random())
            continue

        if r.status_code in (429, 502, 503, 504):
            if attempt == max_retries - 1:
                print(f"  ! CDX HTTP {r.status_code}: {section}", file=sys.stderr)
                return []
            await asyncio.sleep(backoff ** attempt + random.random())
            continue

        if r.status_code != 200:
            print(f"  ! CDX HTTP {r.status_code}: {section}", file=sys.stderr)
            return []

        try:
            rows = r.json()
        except ValueError:
            return []
        if not rows or len(rows) < 2:
            return []
        return [(row[0], row[1]) for row in rows[1:]]

    return []


# ---------- URL filtering ----------


def normalize_for_dedupe(original: str) -> str:
    """Canonical key for grouping: scheme-less, host lowercased, no query/fragment, no trailing slash."""
    u = original if "://" in original else "http://" + original
    p = urlparse(u)
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/").lower()
    return f"{host}{path}"


def is_article_url(original: str, section: str) -> bool:
    u = original if "://" in original else "http://" + original
    p = urlparse(u)
    host = (p.hostname or "").lower()
    if "codeproject.com" not in host:
        return False

    path = p.path
    name = unquote(Path(path).name).lower()

    if not name.endswith(".aspx"):
        return False
    if name in REJECT_BASENAMES:
        return False

    section_lc = section.strip("/").lower()
    path_lc = path.strip("/").lower()
    if path_lc == section_lc or path_lc == section_lc + "/":
        return False

    return True


def pick_newest(rows: list[tuple[str, str]]) -> dict[str, tuple[str, str]]:
    """Group rows by canonical URL, keep the entry with the highest timestamp."""
    best: dict[str, tuple[str, str]] = {}
    for ts, original in rows:
        key = normalize_for_dedupe(original)
        cur = best.get(key)
        if cur is None or ts > cur[0]:
            best[key] = (ts, original)
    return best


# ---------- Validation + title fetch ----------


async def fetch_and_title(
    client: httpx.AsyncClient,
    timestamp: str,
    original: str,
) -> Optional[str]:
    """
    GET the snapshot via id_ flag; reject if body too small.
    Return the extracted page title, or None on failure.
    """
    url = with_flag(timestamp, original, "id_")
    try:
        r = await client.get(url, timeout=60, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    if len(r.content) < MIN_BODY_BYTES:
        return None
    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return None
    return extract_title(soup)


# ---------- Slug uniqueness ----------


def unique_slug(base: str, taken: set[str]) -> str:
    if base not in taken:
        taken.add(base)
        return base
    i = 1
    while True:
        candidate = f"{base}-{i}"
        if candidate not in taken:
            taken.add(candidate)
            return candidate
        i += 1


# ---------- Main pipeline ----------


async def crawl_section(
    client: httpx.AsyncClient,
    section: str,
    sem: asyncio.Semaphore,
    max_retries: int,
    slug_taken: set[str],
    seen_originals: set[str],
) -> list[SnapshotRow]:
    print(f"[section] {section}: querying CDX...")
    raw_rows = await cdx_prefix_query(client, section, max_retries=max_retries)
    print(f"[section] {section}: {len(raw_rows)} raw CDX rows")

    raw_rows = [(ts, orig) for ts, orig in raw_rows if is_article_url(orig, section)]
    deduped = pick_newest(raw_rows)
    pending = [
        (ts, orig) for ts, orig in deduped.values()
        if normalize_original(orig) not in seen_originals
    ]
    skipped = len(deduped) - len(pending)
    print(f"[section] {section}: {len(pending)} new articles "
          f"({skipped} already in CSV)")

    async def process(ts: str, original: str) -> Optional[SnapshotRow]:
        async with sem:
            title = await fetch_and_title(client, ts, original)
            await asyncio.sleep(0.15)
        if title is None:
            print(f"  [skip] {original} @ {ts} (invalid snapshot)")
            return None
        slug = unique_slug(slug_from_url(original), slug_taken)
        canonical = with_flag(ts, original, "")
        normalized = normalize_original(original)
        seen_originals.add(normalized)
        print(f"  [ok]   {slug}: {title}")
        return SnapshotRow(
            section=section,
            slug=slug,
            title=title,
            original_url=normalized,
            snapshot_timestamp=ts,
            wayback_url=canonical,
        )

    tasks = [process(ts, orig) for ts, orig in pending]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def crawl_all(
    sections: list[str],
    concurrency: int,
    max_retries: int,
    seen_originals: set[str],
    slug_taken_by_section: dict[str, set[str]],
    output_path: Path,
    start_id: int,
    write_header_first: bool,
) -> int:
    """Crawl each section and append its rows to output_path immediately.

    Returns total rows appended across all sections.
    """
    headers = {"User-Agent": USER_AGENT}
    sem = asyncio.Semaphore(concurrency)
    total = 0
    next_id = start_id
    write_header = write_header_first

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        for section in sections:
            slug_taken = slug_taken_by_section.setdefault(section, set())
            section_rows = await crawl_section(
                client, section, sem, max_retries,
                slug_taken=slug_taken,
                seen_originals=seen_originals,
            )
            if section_rows:
                append_csv(
                    section_rows, output_path,
                    start_id=next_id,
                    write_header=write_header,
                )
                next_id += len(section_rows)
                total += len(section_rows)
                write_header = False
                print(f"[section] {section}: wrote {len(section_rows)} rows "
                      f"(total appended {total})", flush=True)

    return total


def load_sections(args: argparse.Namespace) -> list[str]:
    sections: list[str] = list(args.section or [])
    if args.sections:
        text = args.sections.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                sections.append(s)
    seen: set[str] = set()
    deduped: list[str] = []
    for s in sections:
        s = s.strip("/")
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


CSV_HEADER = [
    "article_id", "section", "slug", "title",
    "original_url", "snapshot_timestamp", "wayback_url",
]


def load_existing(path: Path) -> ExistingIndex:
    if not path.exists() or path.stat().st_size == 0:
        return ExistingIndex(0, set(), {}, False)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return ExistingIndex(0, set(), {}, False)
        missing = set(CSV_HEADER) - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"existing CSV {path} missing columns: {sorted(missing)}"
            )

        max_id = 0
        seen: set[str] = set()
        slug_taken: dict[str, set[str]] = {}
        for row in reader:
            try:
                rid = int(row["article_id"])
            except (TypeError, ValueError):
                rid = 0
            if rid > max_id:
                max_id = rid
            seen.add(row["original_url"])
            slug_taken.setdefault(row["section"], set()).add(row["slug"])

    return ExistingIndex(max_id, seen, slug_taken, True)


def append_csv(
    rows: list[SnapshotRow],
    path: Path,
    *,
    start_id: int,
    write_header: bool,
) -> None:
    mode = "w" if write_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        for i, r in enumerate(rows, start=start_id):
            w.writerow([
                i, r.section, r.slug, r.title,
                r.original_url, r.snapshot_timestamp, r.wayback_url,
            ])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description="Crawl Wayback CDX for CodeProject KB sections and emit a snapshot CSV."
    )
    ap.add_argument("--section", action="append", default=[],
                    help="Section path under codeproject.com (e.g. KB/dialog). Repeatable.")
    ap.add_argument("--sections", type=Path, default=None,
                    help="File with one section per line (# comments allowed).")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=4)
    args = ap.parse_args()

    sections = load_sections(args)
    if not sections:
        print("no sections given (use --section or --sections)", file=sys.stderr)
        return 2

    existing = load_existing(args.output)
    if existing.header_present:
        print(f"loaded existing CSV: {existing.max_id} rows, "
              f"{len(existing.seen_originals)} known originals")

    print(f"crawling {len(sections)} section(s): {', '.join(sections)}")
    added = asyncio.run(crawl_all(
        sections,
        args.concurrency,
        args.max_retries,
        seen_originals=existing.seen_originals,
        slug_taken_by_section=existing.slug_taken_by_section,
        output_path=args.output,
        start_id=existing.max_id + 1,
        write_header_first=not existing.header_present,
    ))
    if added == 0:
        print("no new snapshots found.", file=sys.stderr)
        return 0 if existing.header_present else 1

    total = existing.max_id + added
    print(f"\nappended {added} rows -> {args.output} (total now {total})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
