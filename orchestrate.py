#!/usr/bin/env python3
"""
orchestrate.py

Drive wayback_to_md.convert() across all rows in a snapshots CSV produced by
crawl_sections.py. Skip articles whose <output-dir>/<section>/<slug>/index.md
already exists. Write a failures CSV for diagnostics.

Usage:
    python orchestrate.py snapshots.csv --output-dir output/ \\
        [--concurrency 4] [--failures failures.csv]

Dependencies: httpx (transitively, via wayback_to_md)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path
from typing import Optional

import httpx

from wayback_to_md import USER_AGENT, convert


REQUIRED_COLUMNS = {
    "article_id", "section", "slug", "title",
    "original_url", "snapshot_timestamp", "wayback_url",
}


def read_snapshots(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"empty CSV: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"CSV missing columns: {sorted(missing)}")
        return list(reader)


async def run(
    rows: list[dict[str, str]],
    output_dir: Path,
    concurrency: int,
    failures_path: Path,
) -> tuple[int, int, int]:
    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": USER_AGENT}

    ok = 0
    skipped = 0
    failed = 0
    failure_records: list[tuple[str, str, str, str]] = []

    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        async def process(row: dict[str, str]) -> None:
            nonlocal ok, skipped, failed
            section = row["section"]
            slug = row["slug"]
            target_parent = output_dir / section
            target_index = target_parent / slug / "index.md"

            if target_index.exists():
                skipped += 1
                return

            async with sem:
                try:
                    result = await convert(
                        client,
                        row["wayback_url"],
                        target_parent,
                        slug=slug,
                    )
                except Exception as e:
                    failed += 1
                    failure_records.append(
                        (row["article_id"], section, slug, f"exception: {e}")
                    )
                    print(f"  ! {section}/{slug}: exception: {e}", file=sys.stderr)
                    return
                await asyncio.sleep(0.2)

            if result.ok:
                ok += 1
                print(f"  [ok]   {section}/{slug} ({result.image_count} images)")
            else:
                failed += 1
                failure_records.append(
                    (row["article_id"], section, slug, result.error or "unknown")
                )
                print(f"  [fail] {section}/{slug}: {result.error}", file=sys.stderr)

        await asyncio.gather(*(process(r) for r in rows))

    if failure_records:
        with failures_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["article_id", "section", "slug", "reason"])
            w.writerows(failure_records)

    return ok, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert all snapshots in a CSV to markdown folders."
    )
    ap.add_argument("snapshots", type=Path, help="CSV produced by crawl_sections.py")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--failures", type=Path, default=Path("failures.csv"))
    ap.add_argument("--section", action="append", default=[],
                    help="Only process rows whose section column matches exactly. "
                         "Repeatable and/or comma-separated (e.g. --section KB/buttons,KB/dialog).")
    args = ap.parse_args()
    args.section = [s.strip() for v in args.section for s in v.split(",") if s.strip()]

    if not args.snapshots.exists():
        print(f"snapshots CSV not found: {args.snapshots}", file=sys.stderr)
        return 2

    rows = read_snapshots(args.snapshots)
    if not rows:
        print("no rows in CSV", file=sys.stderr)
        return 1

    if args.section:
        wanted = set(args.section)
        before = len(rows)
        rows = [r for r in rows if r["section"] in wanted]
        print(f"section filter {sorted(wanted)}: {len(rows)}/{before} rows")
        if not rows:
            print("no rows match the section filter", file=sys.stderr)
            return 1

    print(f"processing {len(rows)} rows -> {args.output_dir} "
          f"(concurrency={args.concurrency})")

    ok, skipped, failed = asyncio.run(
        run(rows, args.output_dir, args.concurrency, args.failures)
    )

    print()
    print(f"done. {ok} ok, {skipped} skipped, {failed} failed", end="")
    if failed:
        print(f" -> {args.failures}")
    else:
        print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
