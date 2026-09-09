"""
PDF -> metin. Kullanım: python pipeline/02_extract.py

İki çıktı üretir:
  data/text/{doc_id}.txt    - düz metin (03_filter ve 04_dedup bunu okur, format
                              eskisiyle birebir aynı: sayfalar "\n" ile birleşik)
  data/pages/{doc_id}.jsonl - SAYFA SAYFA metin, her satır bir sayfa

Sayfa yapısı neden saklanıyor:
  mevzuat.gov.tr PDF'lerinde dipnotlar sayfanın ALTINDA durur. pdfplumber
  sayfayı yukarıdan aşağı okuduğu için dipnotlar, düz metinde gövdenin tam
  ortasına düşüyor. Eskiden sayfa sınırları "\n".join(sayfalar) ile atılıyordu
  ve normalize.py dipnotu gövdeden ayırmak için satır deseni tahmin etmek
  zorunda kalıyordu — bu tahmin hem dipnotların %32'sini kaçırıyordu hem de
  bazı gövde satırlarını yanlışlıkla siliyordu.

  Sayfa sınırı elde olunca kural kesinleşiyor: bir sayfada dipnot başladıysa,
  o sayfanın geri kalanı dipnottur. Tahmin gerekmiyor.
"""

import json
from pathlib import Path

import pdfplumber

KOK = Path(__file__).parent.parent
RAW = KOK / "data" / "raw"
TEXT = KOK / "data" / "text"
PAGES = KOK / "data" / "pages"
STATS = KOK / "extract_stats.jsonl"

TEXT.mkdir(parents=True, exist_ok=True)
PAGES.mkdir(parents=True, exist_ok=True)

out = STATS.open("w", encoding="utf-8")

pdfler = sorted(RAW.glob("*.pdf"))
for i, p in enumerate(pdfler, 1):
    try:
        with pdfplumber.open(p) as pdf:
            sayfalar = [s.extract_text() or "" for s in pdf.pages]
        metin = "\n".join(sayfalar)

        (TEXT / f"{p.stem}.txt").write_text(metin, encoding="utf-8")
        with (PAGES / f"{p.stem}.jsonl").open("w", encoding="utf-8") as pf:
            for no, sayfa in enumerate(sayfalar, 1):
                pf.write(json.dumps({"sayfa": no, "metin": sayfa}, ensure_ascii=False) + "\n")

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
    print(f"[{i}/{len(pdfler)}] {p.stem} sayfa={kayit['sayfa']} kar={kayit['karakter']}", flush=True)

out.close()
print("bitti")
