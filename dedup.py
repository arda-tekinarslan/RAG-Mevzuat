"""Birebir dedup. Kullanım: python dedup.py"""
import json, hashlib, re, collections
from pathlib import Path

def normalize(t):
    t = t.lower()
    t = re.sub(r"[^\wçğıöşü\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

kayitlar = [json.loads(s) for s in open("corpus_manifest.jsonl", encoding="utf-8")]
gorulen = {}
sayac = collections.Counter()

for r in kayitlar:
    if r["durum"] != "accepted":
        sayac[r["red_sebebi"]] += 1
        continue

    metin = Path(r["text_path"]).read_text(encoding="utf-8")
    h = hashlib.sha256(normalize(metin).encode()).hexdigest()
    r["sha256"] = h

    if h in gorulen:
        r["durum"] = "duplicate"
        r["red_sebebi"] = "duplicate"          # ← bu satır
        r["duplicate_of"] = gorulen[h]
        sayac["duplicate"] += 1
    else:
        gorulen[h] = r["doc_id"]
        sayac["accepted"] += 1

with open("corpus_manifest.jsonl", "w", encoding="utf-8") as f:
    for r in kayitlar:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DURUM DAGILIMI")
for k, v in sayac.most_common():
    print(f"  {k:20s} {v:4d}")
print(f"\nnihai korpus: {sayac['accepted']}")