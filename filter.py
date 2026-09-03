"""Kabul filtresi + manifest. Kullanım: python filter.py"""
import json, collections
from pathlib import Path

MIN_KARAKTER = 3000
MIN_KAR_PER_SAYFA = 200

index = {json.loads(s)["doc_id"]: json.loads(s)
         for s in open("mevzuat_index.jsonl", encoding="utf-8")}
stats = {json.loads(s)["doc_id"]: json.loads(s)
         for s in open("extract_stats.jsonl", encoding="utf-8")}

out = open("corpus_manifest.jsonl", "w", encoding="utf-8")
sayac = collections.Counter()

for doc_id, st in stats.items():
    rec = dict(index.get(doc_id, {"doc_id": doc_id}))
    rec.update({
        "sayfa": st["sayfa"],
        "karakter": st["karakter"],
        "kar_per_sayfa": st["kar_per_sayfa"],
        "text_path": f"data/text/{doc_id}.txt",
    })

    if st["hata"]:
        rec["durum"], rec["red_sebebi"] = "rejected", "extract_hatasi"
    elif st["kar_per_sayfa"] < MIN_KAR_PER_SAYFA:
        rec["durum"], rec["red_sebebi"] = "rejected", "taranmis_pdf"
    elif st["karakter"] < MIN_KARAKTER:
        rec["durum"], rec["red_sebebi"] = "rejected", "cok_kisa"
    else:
        rec["durum"], rec["red_sebebi"] = "accepted", None

    sayac[rec["red_sebebi"] or "accepted"] += 1
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")

out.close()
print("ELEME DAGILIMI")
for k, v in sayac.most_common():
    print(f"  {k:20s} {v:4d}")
print(f"\nkabul edilen: {sayac['accepted']}")