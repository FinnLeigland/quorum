"""Quality gate for the built site, run before every deploy:  python -m quorum.sitecheck [--dir _site]

If any check fails, the workflow stops before publishing, so visitors keep seeing yesterday's good site instead of a
broken one. Checks: every page exists and has a title; no leaked template syntax or Python/JS placeholders
("nan", "None", "undefined"); every internal link and asset resolves; no page is unreasonably heavy.
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

PAGES = ["index.html", "stocks.html", "debates.html", "attribution.html", "earnings.html", "trades.html",
         "method.html", "audit.html"]  # fmt: skip
MAX_PAGE_BYTES = 3_000_000
# Placeholders that mean a value went missing on the way to the page (checked in visible text, not in scripts)
LEAKS = re.compile(r"(\{\{|\{%|%\}|\}\}|\bnan\b|\bNaN\b|\bNone\b|\bundefined\b|Traceback)")
TAG = re.compile(r"<script\b.*?</script>|<style\b.*?</style>|<[^>]+>", re.S)
LINK = re.compile(r"""(?:href|src)="([^"]+)\"""")


def visible_text(html: str) -> str:
    return TAG.sub(" ", html)


def check(site: Path) -> list[str]:
    problems = []
    for name in PAGES:
        if not (site / name).exists():
            problems.append(f"missing page: {name}")
    for page in sorted(site.rglob("*.html")):
        rel = page.relative_to(site)
        html = page.read_text(errors="replace")
        if len(html.encode()) > MAX_PAGE_BYTES:
            problems.append(f"{rel}: {len(html.encode()) / 1e6:.1f} MB is too heavy")
        if not re.search(r"<title>[^<]+</title>", html):
            problems.append(f"{rel}: no <title>")
        for m in LEAKS.finditer(visible_text(html)):
            start = max(0, m.start() - 40)
            snippet = " ".join(visible_text(html)[start : m.end() + 40].split())
            problems.append(f"{rel}: '{m.group(0)}' in visible text near: {snippet}")
            break
        for url in LINK.findall(re.sub(r"<script\b.*?</script>", " ", html, flags=re.S)):
            parsed = urlparse(url)
            if parsed.scheme or url.startswith(("#", "mailto:", "//")):
                continue
            target = (page.parent / unquote(parsed.path)).resolve()
            if parsed.path and not target.exists():
                problems.append(f"{rel}: broken link {url}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quorum.sitecheck")
    p.add_argument("--dir", default="_site")
    args = p.parse_args(argv)
    problems = check(Path(args.dir))
    if problems:
        print(f"site check FAILED ({len(problems)} problems):", file=sys.stderr)
        for line in problems[:40]:
            print("  " + line, file=sys.stderr)
        return 1
    print(f"site check ok: {len(list(Path(args.dir).rglob('*.html')))} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
