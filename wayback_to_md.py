#!/usr/bin/env python3
"""
wayback_to_md.py

Convert a single Wayback Machine snapshot of a CodeProject article into a
local markdown file with embedded image assets.

Input  : full timestamped Wayback URL, e.g.
         https://web.archive.org/web/20090228185838/codeproject.com/KB/dialog/ownrdrwsubcls.aspx
Output : <output_dir>/<slug>/index.md   (+ images/* if the article has images)

Importable API:
    async def convert(client, wayback_url, output_dir, slug=None) -> ConvertResult

CLI usage:
    python wayback_to_md.py <wayback_url> --output-dir <dir> [--slug <slug>]

Dependencies: httpx, beautifulsoup4, lxml, markdownify, python-slugify
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, unquote

import httpx
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as html_to_md
from slugify import slugify


USER_AGENT = "codeproject-rescue/1.0 (+personal archive; one-off batch)"

WAYBACK_URL_RE = re.compile(
    r"^https?://web\.archive\.org/web/(\d{14})(id_|im_|cs_|js_|fw_)?/(.+)$"
)

ARTICLE_SELECTORS = [
    {"name": "div", "id": "contentdiv"},
    {"name": "div", "id": "ctl00_MC_AT_ArticleContent"},
    {"name": "div", "id": "ctl00_MC_Content"},
    {"name": "div", "class_": "ArticleContent"},
    {"name": "div", "id": "ArticleBody"},
    {"name": "div", "class_": "content-body"},
    {"name": "article"},
    {"name": "div", "id": "ctl00_ArtDiv"},
    {"name": "div", "id": "ctl00_ArticlePane"},
    {"name": "div", "class_": "ArticlePane"},
]

NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "iframe",
    ".advert",
    ".ads",
    ".ad",
    "#ForumTable",
    ".Frm_MainTable",
    "#ratings",
    ".RatingsCtrl",
    "#comments",
    ".comments",
    ".sidebar",
    "nav",
    ".navbar",
    ".breadcrumb",
    ".social",
    ".share",
    ".related",
    ".footer",
    "#footer",
    ".header",
    "#header",
]

IMAGE_EXT_FALLBACK = ".bin"


@dataclass
class ConvertResult:
    ok: bool
    slug: str
    output_dir: Path
    image_count: int = 0
    error: Optional[str] = None


# ---------- Wayback URL helpers ----------


def parse_wayback_url(url: str) -> tuple[str, str]:
    """
    Return (timestamp, original_url) from a Wayback URL.

    Accepts either banner form (.../<ts>/<original>) or raw form
    (.../<ts>id_/<original>); the flag is stripped.
    """
    m = WAYBACK_URL_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a valid Wayback URL: {url!r}")
    ts, _flag, original = m.group(1), m.group(2), m.group(3)
    return ts, original


def with_flag(timestamp: str, original_url: str, flag: str = "") -> str:
    """Build a Wayback URL with optional asset flag (im_, id_, cs_, js_, fw_)."""
    return f"https://web.archive.org/web/{timestamp}{flag}/{original_url}"


def normalize_original(original: str) -> str:
    """Ensure original URL has a scheme; CodeProject was http:// in the captured era."""
    if original.startswith(("http://", "https://")):
        return original
    return "http://" + original


# ---------- HTML extraction ----------


def find_article_body(soup: BeautifulSoup) -> Tag:
    """Locate the article content container via known CodeProject selectors."""
    for sel in ARTICLE_SELECTORS:
        kwargs = {k: v for k, v in sel.items() if k != "name"}
        found = soup.find(sel["name"], **kwargs)
        if found and len(found.get_text(strip=True)) > 200:
            return found

    body = soup.body or soup
    candidates = body.find_all("div")
    if not candidates:
        return body
    return max(candidates, key=lambda d: len(d.get_text(strip=True)))


def strip_noise(container: Tag) -> None:
    for selector in NOISE_SELECTORS:
        for el in container.select(selector):
            el.decompose()


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        t = soup.title.string.strip()
    else:
        h1 = soup.find("h1")
        t = h1.get_text(strip=True) if h1 else "Untitled"
    for suffix in (
        ". Free source code and programming help",
        " - CodeProject",
        " - The Code Project",
        " | CodeProject",
    ):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    for prefix in ("CodeProject: ", "The Code Project: "):
        if t.startswith(prefix):
            t = t[len(prefix):]
    return t.strip() or "Untitled"


# ---------- Image handling ----------


def guess_ext_from_content_type(ct: str) -> str:
    ct = (ct or "").split(";", 1)[0].strip().lower()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
        "image/x-icon": ".ico",
    }
    return mapping.get(ct, IMAGE_EXT_FALLBACK)


def safe_image_name(raw_url: str, taken: set[str]) -> str:
    parsed = urlparse(raw_url)
    base = unquote(Path(parsed.path).name) or "image"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if "." not in base:
        base = base + IMAGE_EXT_FALLBACK
    stem, ext = base.rsplit(".", 1)
    candidate = f"{stem}.{ext}"
    i = 1
    while candidate in taken:
        candidate = f"{stem}-{i}.{ext}"
        i += 1
    taken.add(candidate)
    return candidate


async def download_image(
    client: httpx.AsyncClient,
    timestamp: str,
    absolute_src: str,
) -> tuple[bytes, str] | None:
    """
    Try Wayback `im_` flag first, then bare timestamp, then the live URL.
    Returns (bytes, content_type) or None.
    """
    candidates = [
        with_flag(timestamp, absolute_src, "im_"),
        with_flag(timestamp, absolute_src, ""),
    ]
    for url in candidates:
        try:
            r = await client.get(url, timeout=60, follow_redirects=True)
        except httpx.HTTPError:
            continue
        if r.status_code == 200 and r.content:
            return r.content, r.headers.get("content-type", "")
    return None


async def rewrite_images(
    client: httpx.AsyncClient,
    container: Tag,
    timestamp: str,
    original_page_url: str,
    images_dir: Path,
) -> int:
    images = container.find_all("img")
    if not images:
        return 0

    taken: set[str] = set()
    saved = 0
    base_url = normalize_original(original_page_url)

    for img in images:
        src = img.get("src") or img.get("data-src")
        if not src:
            img.decompose()
            continue

        if "web.archive.org/web/" in src:
            try:
                _ts, abs_src = parse_wayback_url(src)
                abs_src = normalize_original(abs_src)
            except ValueError:
                abs_src = urljoin(base_url, src)
        else:
            abs_src = urljoin(base_url, src)

        result = await download_image(client, timestamp, abs_src)
        if not result:
            img.replace_with(f"[image missing: {abs_src}]")
            continue

        data, ct = result
        filename = safe_image_name(abs_src, taken)
        if "." not in filename or filename.endswith(IMAGE_EXT_FALLBACK):
            ext = guess_ext_from_content_type(ct)
            if ext != IMAGE_EXT_FALLBACK:
                stem = filename.rsplit(".", 1)[0]
                filename = stem + ext

        images_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / filename).write_bytes(data)

        img["src"] = f"images/{filename}"
        for attr in ("srcset", "data-src", "loading"):
            if attr in img.attrs:
                del img.attrs[attr]
        saved += 1

    return saved


# ---------- Markdown writing ----------


def yaml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def build_frontmatter(
    title: str,
    original_url: str,
    timestamp: str,
    wayback_url: str,
) -> str:
    lines = [
        "---",
        f'title: "{yaml_escape(title)}"',
        f'original_url: "{yaml_escape(original_url)}"',
        f'snapshot_timestamp: "{timestamp}"',
        f'wayback_url: "{yaml_escape(wayback_url)}"',
        "source: codeproject.com",
        "---",
        "",
    ]
    return "\n".join(lines)


def slug_from_url(original_url: str) -> str:
    parsed = urlparse(normalize_original(original_url))
    name = Path(parsed.path).name or "article"
    if name.lower().endswith(".aspx"):
        name = name[:-5]
    name = unquote(name)
    return slugify(name) or "article"


# ---------- Main convert ----------


async def convert(
    client: httpx.AsyncClient,
    wayback_url: str,
    output_dir: Path,
    slug: Optional[str] = None,
) -> ConvertResult:
    try:
        timestamp, original = parse_wayback_url(wayback_url)
    except ValueError as e:
        return ConvertResult(ok=False, slug="", output_dir=output_dir, error=str(e))

    derived_slug = slug or slug_from_url(original)
    article_dir = output_dir / derived_slug
    images_dir = article_dir / "images"

    fetch_url = with_flag(timestamp, original, "id_")
    try:
        r = await client.get(fetch_url, timeout=60, follow_redirects=True)
    except httpx.HTTPError as e:
        return ConvertResult(
            ok=False, slug=derived_slug, output_dir=output_dir,
            error=f"fetch failed: {e}",
        )
    if r.status_code != 200:
        return ConvertResult(
            ok=False, slug=derived_slug, output_dir=output_dir,
            error=f"HTTP {r.status_code} from {fetch_url}",
        )

    soup = BeautifulSoup(r.text, "lxml")
    title = extract_title(soup)
    container = find_article_body(soup)
    strip_noise(container)

    image_count = await rewrite_images(
        client, container, timestamp, original, images_dir,
    )

    body_md = html_to_md(
        str(container),
        heading_style="ATX",
        strip=["script", "style"],
    ).strip()
    body_md = re.sub(r"\n{3,}", "\n\n", body_md)

    canonical_wayback = with_flag(timestamp, original, "")
    frontmatter = build_frontmatter(
        title=title,
        original_url=normalize_original(original),
        timestamp=timestamp,
        wayback_url=canonical_wayback,
    )

    article_dir.mkdir(parents=True, exist_ok=True)
    (article_dir / "index.md").write_text(
        frontmatter + f"# {title}\n\n" + body_md + "\n",
        encoding="utf-8",
    )

    return ConvertResult(
        ok=True,
        slug=derived_slug,
        output_dir=output_dir,
        image_count=image_count,
    )


# ---------- CLI ----------


async def _run_cli(args: argparse.Namespace) -> int:
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        result = await convert(
            client,
            args.wayback_url,
            args.output_dir,
            slug=args.slug,
        )
    if result.ok:
        print(
            f"OK  {result.slug}  ({result.image_count} images) "
            f"-> {result.output_dir / result.slug}"
        )
        return 0
    print(f"FAIL  {result.slug or '<?>'}: {result.error}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert one Wayback CodeProject snapshot into a markdown folder."
    )
    ap.add_argument("wayback_url", help="Full timestamped Wayback URL")
    ap.add_argument(
        "--output-dir", type=Path, required=True,
        help="Parent directory; the article is written to <output-dir>/<slug>/",
    )
    ap.add_argument(
        "--slug", default=None,
        help="Override the derived slug (folder name)",
    )
    args = ap.parse_args()
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
