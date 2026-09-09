"""Birebir dedup (sha256). Kullanim: python pipeline/04_dedup.py

03_filter.py bu scriptten SONRA calistirilirsa buradaki duplicate isaretleri
silinir; 03 bunu tespit edip uyariyor.
"""
import json, hashlib, re, collections
from pathlib import Path
KOK = Path(__file__).parent.parent
MANIFEST = KOK / "corpus_manifest.jsonl"

def normalize(t):
    t = t.lower()
    t = re.sub(r"[^\wçğıöşü\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

kayitlar = [json.loads(s) for s in open(MANIFEST, encoding="utf-8")]
gorulen = {}
sayac = collections.Counter()

for r in kayitlar:
    if r["durum"] != "accepted":
        sayac[r["red_sebebi"]] += 1
        continue

    metin = (KOK / r["text_path"]).read_text(encoding="utf-8")
    h = hashlib.sha256(normalize(metin).encode()).hexdigest()
    r["sha256"] = h

    if h in gorulen:
        r["durum"] = "duplicate"
        r["red_sebebi"] = "duplicate"
        r["duplicate_of"] = gorulen[h]
        sayac["duplicate"] += 1
    else:
        gorulen[h] = r["doc_id"]
        sayac["accepted"] += 1

with open(MANIFEST, "w", encoding="utf-8") as f:
    for r in kayitlar:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("DURUM DAGILIMI")
for k, v in sayac.most_common():
    print(f"  {k:20s} {v:4d}")
print(f"\nnihai korpus: {sayac['accepted']}")