"""Chapters 72/73: CPU-only retrieval and extractive evidence baseline.
This is not an LLM and does not claim to measure generative answer quality.
"""
from collections import Counter
from dataclasses import dataclass
import json
import math
import re


@dataclass(frozen=True)
class Document:
    id: str
    tenant: str
    device: str
    active: bool
    text: str


CORPUS = [
    Document("D1", "factory-a", "A17", False, "A17旧版手册：停机温度阈值为80摄氏度。该版本已废止。"),
    Document("D2", "factory-a", "A17", True, "A17现行手册：停机温度阈值为75摄氏度，超过阈值应停机。"),
    Document("D3", "factory-a", "A18", True, "A18现行手册：滤网维护周期为30天，每次维护后应记录日期。"),
    Document("D4", "factory-b", "A17", True, "B厂A17现行手册：停机温度阈值为90摄氏度。"),
]


def features(text):
    lowered = text.lower()
    words = re.findall(r"[a-z]+[0-9]*|[0-9]+", lowered)
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    return Counter(words + [chinese[i:i + 2] for i in range(len(chinese) - 1)])


def retrieve(query, *, tenant, k=2):
    if not tenant:
        raise PermissionError("trusted tenant scope required")
    if type(k) is not int or not 1 <= k <= 5:
        raise ValueError("k must be an integer from 1 to 5")
    device_match = re.search(r"\bA[0-9]+\b", query, re.I)
    device = device_match.group().upper() if device_match else None
    allowed = [d for d in CORPUS if d.tenant == tenant and d.active and (device is None or d.device == device)]
    if not allowed:
        return []
    bags = [features(d.text) for d in allowed]
    count = len(bags)
    df = Counter(term for bag in bags for term in bag)
    idf = {term: math.log((count + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    query_vec = {term: tf * idf[term] for term, tf in features(query).items() if term in idf}
    qnorm = math.sqrt(sum(v * v for v in query_vec.values()))
    scores = []
    for doc, bag in zip(allowed, bags):
        vec = {term: tf * idf[term] for term, tf in bag.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        dot = sum(value * vec.get(term, 0.0) for term, value in query_vec.items())
        score = dot / (qnorm * norm) if qnorm and norm else 0.0
        if score > 0:
            scores.append((score, doc))
    return sorted(scores, key=lambda pair: (-pair[0], pair[1].id))[:k]


def answer(query, *, tenant):
    hits = retrieve(query, tenant=tenant)
    if not hits:
        return {"status": "insufficient_evidence", "citations": [], "text": "未找到授权范围内的现行证据。"}
    # Deliberately extractive: no unsupported prose or hidden model call.
    return {"status": "evidence_found", "citations": [doc.id for _, doc in hits],
            "text": "\n".join(f"[{doc.id}] {doc.text}" for _, doc in hits)}


def main():
    questions = [("A17 停机 温度 阈值", "D2"), ("A18 滤网 维护 周期", "D3")]
    hit_count = 0
    for query, expected in questions:
        result = answer(query, tenant="factory-a")
        assert expected in result["citations"]
        assert "D1" not in result["citations"] and "D4" not in result["citations"]
        hit_count += 1
    missing = answer("A99 停机温度阈值", tenant="factory-a")
    assert missing["status"] == "insufficient_evidence"
    assert not retrieve("A17 停机 温度 阈值", tenant="factory-c")
    try:
        retrieve("A17", tenant="factory-a", k=True)
    except ValueError:
        pass
    else:
        raise AssertionError("bool must not silently become integer k")
    print(json.dumps({"passed": True, "retrieval_hit_at_2": hit_count / len(questions),
                      "answerable_cases": len(questions), "missing_case": missing,
                      "scope": "synthetic retrieval/extractive smoke test, not generative RAG evaluation"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
