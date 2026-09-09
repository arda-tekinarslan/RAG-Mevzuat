"""
Chunking / normalize çıktılarını incelemek için debug scripti.

Kullanım:
    python tools/debug_chunks.py bloklar    # normalize edilmiş metnin bloklarına bak
    python tools/debug_chunks.py madde      # madde regex'i bloklarda ne kadar tutuyor
    python tools/debug_chunks.py chunklar   # üretilmiş chunk'lara bak
    python tools/debug_chunks.py dagilim    # tüm korpusta madde_no dağılımı
    python tools/debug_chunks.py kirlilik   # dipnot/değişiklik kirliliği ölçümü

Desenler desenler.py'den import ediliyor. Eskiden MADDE regex'i burada
kopyalanmıştı ve başında "chunk_b.py ile AYNI regex olmalı" uyarısı vardı —
o uyarıyı gerektiren durum ortadan kalktı.
"""

import json
import re
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

from desenler import MADDE_ARA, MADDE_ARAMA_SINIRI, madde_no_bul   # noqa: E402

NORM_DIR = KOK / "data" / "norm"
CHUNKS_PATH = KOK / "data" / "chunks_strategy_b.jsonl"

# Bir chunk'ta değişiklik/dipnot metni kaldığını gösteren izler
KIRLILIK = re.compile(
    r"(şeklinde değiştirilmiştir|madde metninden çıkarılmıştır|"
    r"tarihli ve \d+ sayılı Kanunun \d+)"
)


def ilk_dosya() -> Path:
    dosyalar = sorted(NORM_DIR.glob("*.txt"))
    if not dosyalar:
        sys.exit(f"{NORM_DIR} boş — önce 05_normalize.py çalıştır.")
    return dosyalar[0]


def chunklari_yukle() -> list[dict]:
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} yok — önce 06_chunk_b.py çalıştır.")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(s) for s in f]


def bloklar():
    """Normalize edilmiş metnin ilk 15 bloğunu göster."""
    f = ilk_dosya()
    print(f"Dosya: {f.name}\n")
    for b in f.read_text(encoding="utf-8").split("\n\n")[:15]:
        print(repr(b[:120]))
        print("---")


def madde():
    """Madde regex'i bloklarda ne kadar tutuyor, tutmayanlar neler."""
    f = ilk_dosya()
    print(f"Dosya: {f.name}\n")
    parcalar = [b.strip() for b in f.read_text(encoding="utf-8").split("\n\n") if b.strip()]

    bulunan = sum(1 for b in parcalar if MADDE_ARA.search(b[:MADDE_ARAMA_SINIRI]))
    print(f"Bloklarda madde no bulunan: {bulunan}/{len(parcalar)}\n")

    print("Bulunamayanlar:")
    for b in parcalar:
        if not MADDE_ARA.search(b[:MADDE_ARAMA_SINIRI]):
            print(f"  {b[:100]!r}")


def chunklar():
    """İlk dokümanın chunk'larını, madde_no'larıyla birlikte göster."""
    tum = chunklari_yukle()
    hedef = tum[0]["doc_id"]
    d = [c for c in tum if c["doc_id"] == hedef]

    print(f"{hedef}: {len(d)} chunk\n")
    for c in d[:15]:
        print(f"  madde_no={str(c['madde_no'])!r:12} len={c['char_len']:5} "
              f"| {c['text'][:70]!r}")


def dagilim():
    """Tüm korpusta madde_no bulunma oranı ve chunk uzunluk dağılımı."""
    tum = chunklari_yukle()
    madde_var = sum(1 for c in tum if c.get("madde_no"))
    gecici = sum(1 for c in tum if str(c.get("madde_no") or "").startswith("GEÇİCİ"))
    ek = sum(1 for c in tum if str(c.get("madde_no") or "").startswith("EK"))
    cok_kisa = [c for c in tum if c["char_len"] < 100]

    print(f"Toplam chunk        : {len(tum)}")
    print(f"madde_no bulunan    : {madde_var}/{len(tum)} "
          f"(%{madde_var / len(tum) * 100:.1f})")
    print(f"  bunlardan geçici  : {gecici}")
    print(f"  bunlardan ek madde: {ek}")
    print(f"100 karakterden kısa: {len(cok_kisa)} "
          f"(%{len(cok_kisa) / len(tum) * 100:.1f})")

    if cok_kisa:
        print("\nEn kısa 5 chunk:")
        for c in sorted(cok_kisa, key=lambda x: x["char_len"])[:5]:
            print(f"  len={c['char_len']:4} | {c['text'][:80]!r}")


def kirlilik():
    """Dipnot/değişiklik metni kaç chunk'a sızmış — normalize kalitesinin ölçüsü."""
    tum = chunklari_yukle()
    kirli = [c for c in tum if KIRLILIK.search(c["text"])]
    # Bir chunk'ta birden fazla madde numarası = chunking sözleşmesi ihlali
    coklu = [c for c in tum
             if len({m.group(2) for m in MADDE_ARA.finditer(c["text"])}) > 1]

    print(f"Toplam chunk                    : {len(tum)}")
    print(f"Değişiklik/dipnot metni içeren  : {len(kirli)} "
          f"(%{len(kirli) / len(tum) * 100:.1f})")
    print(f"Birden fazla madde no içeren    : {len(coklu)} "
          f"(%{len(coklu) / len(tum) * 100:.1f})")

    if kirli:
        print("\nÖrnek kirli chunk:")
        c = kirli[0]
        print(f"  {c['chunk_id']} madde_no={c['madde_no']!r}")
        print(f"  {c['text'][:220]!r}")
    if coklu:
        print("\nÖrnek çoklu-madde chunk:")
        c = coklu[0]
        print(f"  {c['chunk_id']} madde_no={c['madde_no']!r}")
        print(f"  {c['text'][:220]!r}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "dagilim"
    {"bloklar": bloklar, "madde": madde, "chunklar": chunklar,
     "dagilim": dagilim, "kirlilik": kirlilik}[komut]()
