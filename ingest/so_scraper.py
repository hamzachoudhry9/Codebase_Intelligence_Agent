"""
ingest/so_scraper.py - Stack Overflow Q&A scraper using Stack Exchange API v2.3
Saves results to data/stackoverflow_qa.json (cached - only scrapes once).
"""
import json, time, re
from pathlib import Path

SO_CACHE_PATH = Path("data/stackoverflow_qa.json")

SO_TAGS = [
    "python+langchain", "python+chromadb", "python+langgraph",
    "cuda+out-of-memory", "cuda+kernel", "python+llm",
    "python+rag", "python+fastapi",
]

def _strip_html(text: str) -> str:
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def scrape_stackoverflow(tags: list, questions_per_tag: int = 20) -> list:
    try:
        import requests
    except ImportError:
        print("  requests not installed")
        return []

    headers = {
        "User-Agent": "CodebaseIntelligenceAgent/1.0",
        "Accept-Encoding": "gzip, deflate",
    }
    base = "https://api.stackexchange.com/2.3"
    qa_pairs = []
    seen_ids = set()

    for tag in tags:
        print(f"  Fetching tag: {tag}", end=" ")
        try:
            resp = requests.get(
                f"{base}/questions",
                params={
                    "order": "desc", "sort": "votes",
                    "tagged": tag, "site": "stackoverflow",
                    "filter": "!nNPvSNdWme",   # includes body without auth
                    "pagesize": questions_per_tag,
                },
                headers=headers,
                timeout=20,
            )
            if resp.status_code != 200:
                print(f"status={resp.status_code}")
                continue

            data = resp.json()
            items = data.get("items", [])
            print(f"got {len(items)} questions")

            for q in items:
                qid = q.get("question_id")
                if qid in seen_ids:
                    continue
                seen_ids.add(qid)
                if not q.get("is_answered"):
                    continue

                q_title = q.get("title", "")
                q_body  = _strip_html(q.get("body", ""))[:600]

                # Fetch top answer
                ans_resp = requests.get(
                    f"{base}/questions/{qid}/answers",
                    params={
                        "order": "desc", "sort": "votes",
                        "site": "stackoverflow",
                        "filter": "!nNPvSNdWme",
                        "pagesize": 1,
                    },
                    headers=headers,
                    timeout=20,
                )
                if ans_resp.status_code != 200:
                    continue

                answers = ans_resp.json().get("items", [])
                if not answers:
                    continue

                answer_body = _strip_html(answers[0].get("body", ""))[:1000]
                if len(answer_body) < 50:
                    continue

                qa_pairs.append({
                    "question": f"{q_title}\n\n{q_body}",
                    "answer": answer_body,
                    "url": q.get("link", ""),
                    "tags": q.get("tags", []),
                    "score": q.get("score", 0),
                })
                time.sleep(0.15)  # polite rate limiting

            # Check quota
            quota = data.get("quota_remaining", 999)
            if quota < 10:
                print(f"  WARNING: SE API quota nearly exhausted ({quota} remaining)")
                break

        except Exception as e:
            print(f"  Error on {tag}: {e}")

        time.sleep(1.5)  # between tags

    print(f"  Total Q&A pairs scraped: {len(qa_pairs)}")
    return qa_pairs


def load_or_scrape(force: bool = False) -> list:
    if SO_CACHE_PATH.exists() and not force:
        data = json.loads(SO_CACHE_PATH.read_text())
        print(f"  Loaded {len(data)} cached Q&A pairs from {SO_CACHE_PATH}")
        return data
    SO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = scrape_stackoverflow(SO_TAGS)
    SO_CACHE_PATH.write_text(json.dumps(pairs, indent=2))
    return pairs


if __name__ == "__main__":
    pairs = load_or_scrape(force=True)
    print(json.dumps(pairs[:2], indent=2))
