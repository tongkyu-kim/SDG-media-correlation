"""Scan BigKinds page HTML/JS for news search API endpoints."""
import re
import requests

s = requests.Session()
s.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

r = s.get("https://www.bigkinds.or.kr/v2/news/search.do", timeout=20)
html = r.text

# Find all .do AJAX endpoints and JS fetch/axios calls
print("=== .do endpoints mentioned in page ===")
for m in re.finditer(r'["\'/]((?:v\d+/)?[a-zA-Z/]+\.do)["\']', html):
    ep = m.group(1)
    if any(k in ep for k in ["news", "search", "result", "api", "ajax"]):
        print(" ", ep)

print("\n=== JS src files ===")
for m in re.finditer(r'<script[^>]+src=["\']([^"\']+)["\']', html):
    src = m.group(1)
    if not src.startswith("http"):
        src = "https://www.bigkinds.or.kr" + src
    print(" ", src)

# Also look for any fetch/XMLHttpRequest URLs containing "news"
print("\n=== inline JS URLs with 'news' or 'search' ===")
for m in re.finditer(r'["\`]((?:https?://[^"\'`\s]+)?/[^"\'`\s]*(?:news|search|result|ajax)[^"\'`\s]*)["\`]', html):
    u = m.group(1)
    if len(u) < 100:
        print(" ", u)
