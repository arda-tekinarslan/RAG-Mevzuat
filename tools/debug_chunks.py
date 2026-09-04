"""
Chunking / normalize çıktılarını incelemek için debug scripti.

Kullanım:
    python debug_chunks.py bloklar      # normalize edilmiş metnin bloklarına bak
    python debug_chunks.py madde        # madde regex'i bloklarda ne kadar tutuyor
    python debug_chunks.py chunklar     # üretilmiş chunk'lara bak
    python debug_chunks.py dagilim      # tüm korpusta madde_no dağılımı
"""

import json
import re
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
NORM_DIR = KOK / "data" / "norm"
CHUNKS_PATH = KOK / "data" / "chunks_strategy_b.jsonl"

# chunk_b.py ile AYNI regex olmalı — burada değiştirirsen orada da değiştir
MADDE = re.compile(r"(Madde|MADDE|GEÇİCİ MADDE|EK MADDE)\s*(\d+)")
MADDE_ARAMA_SINIRI = 150


def ilk_dosya() -> Path:
    dosyalar = sorted(NORM_DIR.glob("*.txt"))
    if not dosyalar:
        sys.exit(f"{NORM_DIR} boş — önce normalize.py çalıştır.")
    return dosyalar[0]


def bloklar():
    """Normalize edilmiş metnin ilk 15 bloğunu göster."""
    f = ilk_dosya()
    print(f"Dosya: {f.name}\n")
    parcalar = f.read_text(encoding="utf-8").split("\n\n")
    for b in parcalar[:15]:
        print(repr(b[:120]))
        print("---")


def madde():
    """Madde regex'i bloklarda ne kadar tutuyor, tutmayanlar neler."""
    f = ilk_dosya()
    print(f"Dosya: {f.name}\n")
    parcalar = [b.strip() for b in f.read_text(encoding="utf-8").split("\n\n") if b.strip()]

    bulunan = sum(1 for b in parcalar if MADDE.search(b[:MADDE_ARAMA_SINIRI]))
    print(f"Bloklarda madde no bulunan: {bulunan}/{len(parcalar)}\n")

    print("Bulunamayanlar:")
    for b in parcalar:
        if not MADDE.search(b[:MADDE_ARAMA_SINIRI]):
            print(f"  {b[:100]!r}")


def chunklar():
    """İlk dokümanın chunk'larını, madde_no'larıyla birlikte göster."""
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} yok — önce chunk_b.py çalıştır.")

    tum = [json.loads(s) for s in open(CHUNKS_PATH, encoding="utf-8")]
    hedef = tum[0]["doc_id"]
    d = [c for c in tum if c["doc_id"] == hedef]

    print(f"{hedef}: {len(d)} chunk\n")
    for c in d[:15]:
        print(f"  madde_no={c['madde_no']!r:8} len={c['char_len']:5} | {c['text'][:70]!r}")


def dagilim():
    """Tüm korpusta madde_no bulunma oranı ve chunk uzunluk dağılımı."""
    if not CHUNKS_PATH.exists():
        sys.exit(f"{CHUNKS_PATH} yok — önce chunk_b.py çalıştır.")

    tum = [json.loads(s) for s in open(CHUNKS_PATH, encoding="utf-8")]
    madde_var = sum(1 for c in tum if c.get("madde_no"))

    # çok kısa chunk'lar retrieval'da işe yaramaz, kaç tane olduğunu bil
    cok_kisa = [c for c in tum if c["char_len"] < 100]

    print(f"Toplam chunk        : {len(tum)}")
    print(f"madde_no bulunan    : {madde_var}/{len(tum)} (%{madde_var / len(tum) * 100:.1f})")
    print(f"100 karakterden kısa: {len(cok_kisa)} (%{len(cok_kisa) / len(tum) * 100:.1f})")

    if cok_kisa:
        print("\nEn kısa 5 chunk:")
        for c in sorted(cok_kisa, key=lambda x: x["char_len"])[:5]:
            print(f"  len={c['char_len']:4} | {c['text'][:80]!r}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "dagilim"
    {"bloklar": bloklar, "madde": madde, "chunklar": chunklar, "dagilim": dagilim}[komut]()