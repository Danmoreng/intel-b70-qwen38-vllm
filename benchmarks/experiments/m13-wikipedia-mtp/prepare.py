#!/usr/bin/env python3
"""Freeze four revision-pinned German Wikipedia excerpts for a local benchmark."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ARTICLES = (
    ("Marie Curie", 270535757),
    ("Apollo 11", 268725520),
    ("Photosynthese", 268891481),
    ("Berlin", 270716683),
)
USER_AGENT = "B70WikipediaBenchmark/0.1 (https://github.com/Danmoreng/intel-b70-qwen38-vllm)"
MODEL = "Qwen3.8-27B"
TARGET_TOKENS = 4096
INSTRUCTION = (
    "Fasse den folgenden Auszug aus einem Wikipedia-Artikel auf Deutsch "
    "sachlich zusammen. Schreibe etwa 250 bis 300 Wörter in zusammenhängenden "
    "Absätzen. Nenne die wichtigsten Fakten und Zusammenhänge, bleibe beim "
    "gegebenen Text und füge keine Informationen von außerhalb hinzu."
)


class Paragraphs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.skip = 0
        self.buffer: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            self.depth += 1
        elif self.depth and tag in ("sup", "style", "script"):
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("sup", "style", "script") and self.skip:
            self.skip -= 1
        elif tag == "p" and self.depth:
            self.depth -= 1
            if not self.depth:
                text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
                self.buffer = []
                if text:
                    self.parts.append(text)

    def handle_data(self, data: str) -> None:
        if self.depth and not self.skip:
            self.buffer.append(data)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def fetch_article(oldid: int) -> str:
    url = "https://de.wikipedia.org/w/api.php?" + urlencode(
        {"action": "parse", "oldid": oldid, "prop": "text", "format": "json", "formatversion": "2"}
    )
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as response:
        parsed = json.load(response)["parse"]
    if int(parsed["revid"]) != oldid:
        raise RuntimeError(f"revision mismatch: requested {oldid}, received {parsed['revid']}")
    collector = Paragraphs()
    collector.feed(parsed["text"])
    return "\n\n".join(collector.parts)


def token_count(base: str, prompt: str) -> int:
    request = Request(
        base + "/tokenize",
        data=json.dumps({"model": MODEL, "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return int(json.load(response)["count"])


def make_prompt(base: str, title: str, article: str) -> tuple[str, int, str]:
    prefix = f"Wikipedia-Artikel: {title}\n\n"
    suffix = f"\n\nAufgabe: {INSTRUCTION}\n\nZusammenfassung:\n"

    def prompt_at(length: int) -> str:
        return prefix + article[:length] + suffix

    low, high = 0, len(article)
    while low < high:
        mid = (low + high + 1) // 2
        if token_count(base, prompt_at(mid)) <= TARGET_TOKENS:
            low = mid
        else:
            high = mid - 1
    if low == len(article):
        raise RuntimeError(f"article {title!r} is shorter than {TARGET_TOKENS} prompt tokens")
    cut = low
    # Prefer a complete sentence close to the token target.
    period = article.rfind(". ", max(0, cut - 650), cut)
    if period != -1:
        cut = period + 1
    else:
        space = article.rfind(" ", max(0, cut - 100), cut)
        if space != -1:
            cut = space
    excerpt = article[:cut].rstrip()
    prompt = prefix + excerpt + suffix
    count = token_count(base, prompt)
    if not 3900 <= count <= TARGET_TOKENS:
        raise RuntimeError(f"{title}: expected about 4K tokens, got {count}")
    return prompt, count, excerpt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for title, oldid in ARTICLES:
        article = fetch_article(oldid)
        prompt, count, excerpt = make_prompt(args.base, title, article)
        row = {
            "title": title,
            "revision_id": oldid,
            "revision_url": f"https://de.wikipedia.org/w/index.php?oldid={oldid}",
            "article_sha256": digest(article),
            "excerpt_sha256": digest(excerpt),
            "prompt_sha256": digest(prompt),
            "prompt_tokens": count,
            "prompt": prompt,
        }
        rows.append(row)
        print(f"{title}: revision {oldid}, {count} prompt tokens, {len(excerpt)} chars", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
