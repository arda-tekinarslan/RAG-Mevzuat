"""
Kabul filtresi + manifest üretimi.

Kullanım: python pipeline/03_filter.py

SIRA ÖNEMLİ: bu script corpus_manifest.jsonl'i SIFIRDAN yazar. 04_dedup.py
manifest üzerine "duplicate" işareti ve sha256 ekler. Bu script 04'ten SONRA
tekrar çalıştırılırsa o işaretler silinir — script bunu tespit edip uyarıyor.
Doğru sıra: 03 -> 04 -> 05 -> 06 -> 07.
"""

import collections
import json
from pathlib import Path

KOK = Path(__file__).parent.parent
MANIFEST = KOK / "corpus_manifest.jsonl"

MIN_KARAKTER = 3000
MIN_KAR_PER_SAYFA = 200


def satirlari_oku(yol):
    with open(yol, encoding="utf-8") as f:
        return [json.loads(s) for s in f]


def main():
    index = {r["doc_id"]: r for r in satirlari_oku(KOK / "mevzuat_index.jsonl")}
    stats = {r["doc_id"]: r for r in satirlari_oku(KOK / "extract_stats.jsonl")}

    # 04_dedup daha önce koşmuş mu? Koştuysa bu script onun çıktısını siler.
    onceki_dedup = 0
    if MANIFEST.exists():
        onceki_dedup = sum(1 for r in satirlari_oku(MANIFEST)
                           if r.get("durum") == "duplicate")

    sayac = collections.Counter()
    with open(MANIFEST, "w", encoding="utf-8") as out:
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

    print("ELEME DAGILIMI")
    for k, v in sayac.most_common():
        print(f"  {k:20s} {v:4d}")
    print(f"\nkabul edilen: {sayac['accepted']}")

    if onceki_dedup:
        print(f"\nUYARI: manifest'te {onceki_dedup} 'duplicate' kaydı vardı ve "
              f"bu script onları sildi.\n       04_dedup.py'yi tekrar çalıştır.")


if __name__ == "__main__":
    main()
