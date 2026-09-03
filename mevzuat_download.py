"""
mevzuat.gov.tr toplu PDF indirme.

Kullanım:
    pip install requests datasets
    python mevzuat_download.py index      # (Tur,Tertip,No) listesi -> mevzuat_index.jsonl
    python mevzuat_download.py download   # PDF'ler -> data/raw/

Mantık:
  PDF URL kalıbı sabit:
      https://www.mevzuat.gov.tr/mevzuatmetin/{Tur}.{Tertip}.{No}.pdf
      örn 1.5.5237.pdf -> Türk Ceza Kanunu (Kanun, Tertip 5, No 5237)

  Yani tek ihtiyacımız (Tur,Tertip,No) üçlüleri. Onları HF'teki hazır
  mevzuat listesinden parse ediyoruz; arama API'sini kurcalamıyoruz.

DIKKAT:
  CB Kararı / CB Genelgesi taranmış PDF, metin katmanı yok, OCR gerekir.
  Bu script sadece Kanun ve Yönetmelik çekiyor — ikisi de born-digital.
"""

import json
import re
import sys
import time
from pathlib import Path
import urllib3
urllib3.disable_warnings()

import requests

PDF_BASE = "https://www.mevzuat.gov.tr/mevzuatmetin"
INDEX = Path("mevzuat_index.jsonl")
RAW = Path("data/raw")
UYKU = 1.0

# Hedef adet (aday). ~%15 fire bekliyoruz, kanun/yönetmelik PDF'leri temiz.
HEDEF = {"1": 300}
TUR_ADI = {"1": "kanun"}

URL_RE = re.compile(
    r"MevzuatNo=(?P<no>\d+).*?MevzuatTur=(?P<tur>\d+).*?MevzuatTertip=(?P<tertip>\d+)"
)


def index():
    """HF mevzuat listesinden (Tur,Tertip,No) üçlülerini çıkar."""
    from datasets import load_dataset

    ds = load_dataset("muhammetakkurt/mevzuat-gov-dataset", split="train")
    print(f"{len(ds)} kayıt yüklendi")

    sayac = {t: 0 for t in HEDEF}
    gorulen = set()
    out = INDEX.open("w", encoding="utf-8")

    for satir in ds:
        # URL alanının adı veri setine göre değişebilir; ilk satırı bastırıp bak.
        url = satir.get("url") or ""
        m = URL_RE.search(str(url))
        if not m:
            continue

        tur, tertip, no = m["tur"], m["tertip"], m["no"]
        if tur not in HEDEF or sayac[tur] >= HEDEF[tur]:
            continue

        doc_id = f"mevzuat_{tur}.{tertip}.{no}"
        if doc_id in gorulen:
            continue

        out.write(json.dumps({
            "doc_id": doc_id,
            "source": "mevzuat",
            "alan": TUR_ADI[tur],
            "baslik": satir.get("Kanun Adı"),
            "mevzuat_no": no,
            "mevzuat_tur": tur,
            "mevzuat_tertip": tertip,
            "pdf_url": f"{PDF_BASE}/{tur}.{tertip}.{no}.pdf",
            "kaynak_url": url,
            "lisans": "FSEK m.31 - resmi metin",
            "durum": "candidate",
        }, ensure_ascii=False) + "\n")

        gorulen.add(doc_id)
        sayac[tur] += 1

    out.close()
    print(f"index yazıldı: {dict(sayac)} -> toplam {len(gorulen)} aday")
    if sum(sayac.values()) < 250:
        print("UYARI: 250'nin altında.")


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    kayitlar = [json.loads(s) for s in INDEX.open(encoding="utf-8")]

    basarili = atlanan = hatali = 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    sess.verify = False

    for i, k in enumerate(kayitlar, 1):
        hedef = RAW / f"{k['doc_id']}.pdf"
        if hedef.exists() and hedef.stat().st_size > 0:
            atlanan += 1
            continue

        try:
            r = sess.get(k["pdf_url"], timeout=90)
            if r.status_code == 404:
                print(f"[{i}] 404 {k['doc_id']}", file=sys.stderr)
                hatali += 1
                time.sleep(UYKU)
                continue
            r.raise_for_status()

            # Kalıp tutmazsa HTML hata sayfası döner; PDF olduğunu doğrula.
            if not r.content.startswith(b"%PDF"):
                print(f"[{i}] PDF değil {k['doc_id']}", file=sys.stderr)
                hatali += 1
                time.sleep(UYKU)
                continue

            hedef.write_bytes(r.content)
            basarili += 1
            print(f"[{i}/{len(kayitlar)}] {k['doc_id']} ({len(r.content)//1024} KB)")

        except Exception as e:
            print(f"[{i}] HATA {k['doc_id']}: {e}", file=sys.stderr)
            hatali += 1

        time.sleep(UYKU)

    print(f"\nindirilen={basarili} zaten_vardi={atlanan} hata={hatali}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "index"
    {"index": index, "download": download}[komut]()