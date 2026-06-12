"""Quick diagnostic: probe BigKinds URL structure and CSRF."""
import re
import json
import requests

s = requests.Session()
s.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

URLS = [
    "https://www.bigkinds.or.kr",
    "https://www.bigkinds.or.kr/v2/news/search.do",
    "https://www.bigkinds.or.kr/news/search.do",
]

for url in URLS:
    try:
        r = s.get(url, timeout=15, allow_redirects=True)
        csrf = ""
        m = re.search(r'name=["\']_csrf["\']', r.text)
        if m:
            m2 = re.search(r'content=["\'](.*?)["\']', r.text[m.start():m.start()+200])
            if m2:
                csrf = m2.group(1)[:30]
        print(f"  {r.status_code}  {r.url}")
        print(f"         size={len(r.text)}  csrf={csrf or 'NOT FOUND'}")
        # Show form/meta lines mentioning csrf
        for line in r.text.splitlines():
            if "_csrf" in line.lower():
                print(f"         >> {line.strip()[:100]}")
    except Exception as e:
        print(f"  ERROR  {url}: {e}")

# Try the search endpoint with a minimal POST
print("\n--- POST to newsResult.do ---")
try:
    r2 = s.post(
        "https://www.bigkinds.or.kr/news/newsResult.do",
        data={
            "keyword": "",
            "startDate": "2024-01-02",
            "endDate": "2024-01-02",
            "filterProviderCode": "01100101",
            "sortMethod": "date",
            "resultNumber": "5",
            "startNumber": "1",
            "_csrf": "",
        },
        timeout=15,
    )
    print(f"  status: {r2.status_code}")
    print(f"  content-type: {r2.headers.get('content-type','')}")
    try:
        data = r2.json()
        print(f"  json keys: {list(data.keys())}")
        print(f"  sample: {json.dumps(data, ensure_ascii=False)[:400]}")
    except Exception:
        print(f"  raw (first 400): {r2.text[:400]}")
except Exception as e:
    print(f"  ERROR: {e}")
