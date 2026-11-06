"""Collect the URLs of the reels/posts of an Instagram account page.

Uses yt-dlp's flat-playlist extraction (lists links, downloads nothing).
The URLs are deduplicated and appended to clips_collected.txt, ready for
the ingest pipeline.

    PYTHONPATH=src python -m spanish_align.collect_urls simpsonscast
    PYTHONPATH=src python -m spanish_align.collect_urls https://www.instagram.com/simpsonscast/
    PYTHONPATH=src python -m spanish_align.collect_urls simpsonscast --limit 30
    PYTHONPATH=src python -m spanish_align.collect_urls simpsonscast --cookies-from-browser safari

Note: Instagram rate-limits anonymous listing; if you get errors or
empty results, pass --cookies-from-browser (your logged-in session
usually suffices for public accounts).
"""

import argparse
import subprocess

from .config import WORKDIR, ensure_dirs

OUTPUT = WORKDIR / "clips_collected.txt"


def collect_urls(profile: str, cookies_from: str | None,
                 limit: int | None) -> list[str]:
    target = profile if profile.startswith("http") \
        else f"https://www.instagram.com/{profile.lstrip('@')}/"
    cmd = [
        "yt-dlp", "--flat-playlist", "--no-warnings",
        "--print", "%(url)s", target,
    ]
    if cookies_from:
        cmd += ["--cookies-from-browser", cookies_from]
    if limit:
        cmd = [cmd[0], "--playlist-end", str(limit)] + cmd[1:]

    print(f"Listing posts of {target} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"yt-dlp failed:\n{res.stderr.strip()}")

    urls = sorted({u.strip() for u in res.stdout.splitlines()
                   if u.strip() and ("/reel/" in u or "/p/" in u)})
    return urls


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect IG post/reel URLs from an account page")
    p.add_argument("profile", help="account name or full URL")
    p.add_argument("--limit", type=int, help="max posts to list")
    p.add_argument("--cookies-from-browser",
                   help="browser to read Instagram cookies from (safari, chrome, firefox)")
    args = p.parse_args()

    ensure_dirs()
    urls = collect_urls(args.profile, args.cookies_from_browser, args.limit)

    # append + dedupe against what we already have
    existing = set()
    if OUTPUT.exists():
        existing = {l.strip() for l in OUTPUT.read_text().splitlines() if l.strip()}
    new = [u for u in urls if u not in existing]

    with OUTPUT.open("a", encoding="utf-8") as f:
        for u in new:
            f.write(u + "\n")

    print(f"Found {len(urls)} post URLs | new: {len(new)} | already known: "
          f"{len(urls) - len(new)}")
    print(f"Saved to {OUTPUT} — ready for:")
    print("  PYTHONPATH=src python -m spanish_align.cli ingest "
          f"--urls {OUTPUT}")


if __name__ == "__main__":
    main()
