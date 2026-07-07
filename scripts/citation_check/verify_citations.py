#!/usr/bin/env python3
"""
verify_citations.py

Checks every entry in references.json against live sources:
  - Entries with an arxiv_id: pulls the real title/authors from the arXiv API
    and diffs them against what's claimed in the bibliography.
  - Entries with a doi: pulls metadata from the CrossRef API.
  - Everything else: does an HTTP GET and reports status code + <title> tag,
    so you can eyeball whether the page still exists and matches.

This can't be run inside the Claude sandbox (its network is locked to package
registries only, not arxiv.org / crossref.org / general web). Run it on your
own machine.

Requirements:
    pip install requests

Usage:
    python verify_citations.py                 # check everything
    python verify_citations.py --only-flagged   # just re-check the ⚠ VERIFY entries
    python verify_citations.py --out report.md  # write a markdown report
"""

import json
import re
import sys
import time
import argparse
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

import requests

ARXIV_API = "http://export.arxiv.org/api/query?id_list={}"
CROSSREF_API = "https://api.crossref.org/works/{}"
TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 1.0  # be polite to arXiv/CrossRef


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def check_arxiv(arxiv_id: str):
    """Returns dict with real title/authors, or an error string."""
    try:
        resp = requests.get(ARXIV_API.format(arxiv_id), timeout=TIMEOUT)
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {"error": "No entry returned by arXiv API (ID may not exist)."}
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        authors = [
            a.find("atom:name", ns).text
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.find("atom:published", ns).text[:10]
        return {"title": title, "authors": authors, "published": published}
    except Exception as e:
        return {"error": str(e)}


def check_doi(doi: str):
    try:
        resp = requests.get(CROSSREF_API.format(doi), timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["message"]
        title = data.get("title", [""])[0]
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in data.get("author", [])
        ]
        return {"title": title, "authors": authors}
    except Exception as e:
        return {"error": str(e)}


def check_url_liveness(url: str):
    try:
        resp = requests.get(
            url, timeout=TIMEOUT, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (citation-checker)"}
        )
        status = resp.status_code
        title_match = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
        page_title = title_match.group(1).strip() if title_match else None
        return {"status": status, "page_title": page_title}
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", default="scripts/citation_check/references.json")
    parser.add_argument("--only-flagged", action="store_true",
                         help="Only re-check entries that already carry a 'flag' field")
    parser.add_argument("--out", default=None, help="Write a markdown report to this path")
    args = parser.parse_args()

    with open(args.refs, "r", encoding="utf-8") as f:
        refs = json.load(f)

    if args.only_flagged:
        refs = [r for r in refs if r.get("flag")]

    report_lines = ["# Citation Verification Report\n"]
    problems = []

    for ref in refs:
        rid = ref["id"]
        print(f"[{rid}] Checking: {ref['claimed_title'][:60]}...")
        line = [f"## [{rid}] {ref['claimed_title']}"]
        line.append(f"- Claimed authors: {ref['claimed_authors']}")
        line.append(f"- Claimed year: {ref.get('claimed_year')}")
        line.append(f"- URL: {ref['url']}")
        if ref.get("flag"):
            line.append(f"- **Pre-flagged issue:** {ref['flag']}")

        if ref.get("arxiv_id"):
            result = check_arxiv(ref["arxiv_id"])
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if "error" in result:
                line.append(f"- ❌ arXiv lookup failed: {result['error']}")
                problems.append(rid)
            else:
                title_sim = similarity(ref["claimed_title"], result["title"])
                line.append(f"- arXiv real title: {result['title']}")
                line.append(f"- arXiv real authors: {', '.join(result['authors'])}")
                line.append(f"- arXiv published: {result['published']}")
                line.append(f"- Title similarity score: {title_sim:.2f}")
                if title_sim < 0.6:
                    line.append("- ❌ TITLE MISMATCH — this ID may point to a different paper")
                    problems.append(rid)
                else:
                    line.append("- ✅ Title matches")
        elif ref.get("doi"):
            result = check_doi(ref["doi"])
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if "error" in result:
                line.append(f"- ❌ CrossRef lookup failed: {result['error']}")
                problems.append(rid)
            else:
                title_sim = similarity(ref["claimed_title"], result["title"])
                line.append(f"- CrossRef real title: {result['title']}")
                line.append(f"- Title similarity score: {title_sim:.2f}")
                if title_sim < 0.6:
                    line.append("- ❌ TITLE MISMATCH")
                    problems.append(rid)
        else:
            result = check_url_liveness(ref["url"])
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            if "error" in result:
                line.append(f"- ❌ URL unreachable: {result['error']}")
                problems.append(rid)
            else:
                line.append(f"- HTTP status: {result['status']}")
                line.append(f"- Page <title>: {result['page_title']}")
                if result["status"] >= 400:
                    line.append("- ❌ DEAD LINK")
                    problems.append(rid)
                elif result["page_title"] and similarity(ref["claimed_title"], result["page_title"]) < 0.3:
                    line.append("- ⚠️  Page title looks unrelated to claimed title — eyeball this one")

        report_lines.append("\n".join(line) + "\n")

    report_lines.append(f"\n---\n**Summary:** {len(problems)} of {len(refs)} checked entries flagged a problem: {problems}\n")
    report = "\n".join(report_lines)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport written to {args.out}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
