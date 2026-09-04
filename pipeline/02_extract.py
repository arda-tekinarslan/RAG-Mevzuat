"""PDF -> metin. Kullanım: python extract.py"""
import json
from pathlib import Path
import pdfplumber

KOK = Path(__file__).parent.parent
RAW = KOK / "data" / "raw"
TEXT = KOK / "data" / "text"
STATS = KOK / "extract_stats.jsonl"

TEXT.mkdir(parents=True, exist_ok=True)
out = STATS.open("w", encoding="utf-8")

pdfler = sorted(RAW.glob("*.pdf"))
for i, p in enumerate(pdfler, 1):
    hedef = TEXT / f"{p.stem}.txt"
    try:
        with pdfplumber.open(p) as pdf:
            sayfalar = [s.extract_text() or "" for s in pdf.pages]
        metin = "\n".join(sayfalar)
        hedef.write_text(metin, encoding="utf-8")

        kayit = {
            "doc_id": p.stem,
            "sayfa": len(sayfalar),
            "karakter": len(metin),
            "kar_per_sayfa": round(len(metin) / max(len(sayfalar), 1)),
            "hata": None,
        }
    except Exception as e:
        kayit = {"doc_id": p.stem, "sayfa": 0, "karakter": 0,
                 "kar_per_sayfa": 0, "hata": str(e)[:200]}

    out.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    print(f"[{i}/{len(pdfler)}] {p.stem} sayfa={kayit['sayfa']} kar={kayit['karakter']}")

out.close()
print("bitti")
print("bitti")