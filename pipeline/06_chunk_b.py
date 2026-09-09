"""
Strateji B: Madde sınırından chunking.

Strateji A (06_chunk_a.py) metni 800 karakterden kör kesiyordu — bir maddenin
ortasından bölmek hem retrieval'ı hem cevabın okunabilirliğini bozuyor.
Burada 05_normalize.py'nin ürettiği yapıyı kullanıyoruz: her madde kendi bloğunda.

Kullanım: python pipeline/06_chunk_b.py
"""

import json
import statistics
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

from desenler import YAPI_BLOK, madde_no_bul      # noqa: E402

# --- Ayarlar ---
MAX_CHUNK = 1200        # bir chunk en fazla bu kadar karakter (artık GERÇEKTEN)
MIN_CHUNK = 200         # bundan kısa chunk tek başına kalmasın (madde sınırı hariç)
SPLIT_OVERLAP = 100     # SADECE uzun madde bölünürken (yapay sınır attığımız tek yer)
KESIM_PENCERESI = 200   # cümle sınırı ararken geriye bakılacak karakter

MANIFEST_PATH = KOK / "corpus_manifest.jsonl"
NORM_DIR = KOK / "data" / "norm"
OUTPUT_PATH = KOK / "data" / "chunks_strategy_b.jsonl"


def _kesim_noktasi(blok: str, ideal: int) -> int:
    """
    ideal'den en fazla KESIM_PENCERESI kadar geriye giderek cümle/kelime
    sınırı bulur.

    Eskiden bölme doğrudan blok[start:start+MAX_CHUNK] idi ve kelimenin
    ortasından kesiyordu ("...nazım plân" | "yon plânlarını yapmak...").
    Hem embedding'i hem LLM'in okuduğu metni bozuyordu.
    """
    alt = max(0, ideal - KESIM_PENCERESI)
    for desen in (". ", "; ", " "):
        p = blok.rfind(desen, alt, ideal)
        if p != -1:
            return p + len(desen)
    return ideal


def uzun_bloku_bol(blok: str) -> list[str]:
    """MAX_CHUNK'tan uzun metni cümle sınırlarından parçalar. Kısaysa aynen döner."""
    if len(blok) <= MAX_CHUNK:
        return [blok]

    parcalar = []
    start = 0
    while start < len(blok):
        if len(blok) - start <= MAX_CHUNK:
            parcalar.append(blok[start:])
            break
        kes = _kesim_noktasi(blok, start + MAX_CHUNK)
        parcalar.append(blok[start:kes])
        # SPLIT_OVERLAP kadar geri sar; kes her zaman start'tan büyük olduğu
        # için ilerleme garanti (MAX_CHUNK > KESIM_PENCERESI + SPLIT_OVERLAP).
        geri = max(kes - SPLIT_OVERLAP, start + 1)
        # Geri sarma da kelime ortasına düşmesin. Kesim noktası cümle
        # sınırındaydı ama overlap ham karakter sayımıyla geriye gidiyordu,
        # dolayısıyla devam chunk'ları yarım kelimeyle başlayabiliyordu.
        bosluk = blok.find(" ", geri)
        if bosluk != -1 and bosluk + 1 < kes:
            geri = bosluk + 1
        start = geri
    return parcalar


def chunk_document_madde(text: str, doc_id: str) -> list[dict]:
    """
    Bloklardan chunk üretir. Sözleşme: bir chunk EN FAZLA BİR madde başlığı
    içerir ve MAX_CHUNK'ı aşmaz.
    """
    chunks: list[dict] = []
    tampon: list[str] = []
    tampon_madde: str | None = None
    tampon_yapi_disi = False   # tamponda yapısal başlık dışında içerik var mı
    idx = 0

    def tampon_uzunluk() -> int:
        # "\n\n" ayırıcıları da sayılır; eskiden sayılmıyordu ve uzunluk
        # kontrolü bu yüzden 2*(blok-1) karakter yanılıyordu.
        return sum(len(b) for b in tampon) + 2 * max(len(tampon) - 1, 0)

    def close_tampon() -> None:
        nonlocal tampon, tampon_madde, tampon_yapi_disi, idx
        if not tampon:
            return
        madde = tampon_madde
        # Tek çıkış noktası: her chunk buradan geçer, dolayısıyla MAX_CHUNK
        # garantisi tek yerde uygulanıyor. Eskiden yapısal başlık tampona
        # boyut kontrolsüz ekleniyordu ve 51 chunk 1200 sınırını aşıyordu
        # (en uzunu 1383) — e5'in 512 token penceresini de zorluyordu.
        for parca in uzun_bloku_bol("\n\n".join(tampon)):
            chunks.append({
                "chunk_id": f"{doc_id}_b_{idx:04d}",
                "doc_id": doc_id,
                "text": parca,
                "char_len": len(parca),
                "madde_no": madde,
            })
            idx += 1
        tampon, tampon_madde, tampon_yapi_disi = [], None, False

    for blok in text.split("\n\n"):
        blok = blok.strip()
        if not blok:
            continue

        # Yapısal başlık (BİRİNCİ BÖLÜM vb.) tek başına anlam taşımaz —
        # SONRAKİ maddeye yapışmalı. Tamponda gerçek içerik varsa önce onu
        # kapat, yoksa başlık bir önceki maddenin kuyruğuna yapışırdı.
        if YAPI_BLOK.match(blok):
            if tampon_yapi_disi:
                close_tampon()
            tampon.append(blok)
            continue

        blok_madde = madde_no_bul(blok)

        # IKI AYRI MADDEYI ASLA BIRLESTIRME.
        # Eskiden MIN_CHUNK dolmadigi icin kisa maddeler bir sonrakine
        # yapisiyordu ve metadata sadece ILK maddeyi yaziyordu. Korpusta
        # 1822 chunk (%11.2) birden fazla madde iceriyordu. Somut ornek:
        # TCK Madde 81 (87 karakter) + Madde 82 tek chunk'ta, etiketi "81" —
        # LLM'in "adam oldurme" sorusunda yanlis madde alintilamasinin nedeni.
        # 87 karakterlik dogru bir chunk, iki maddeyi karistiran 912
        # karakterlik bir chunk'tan iyidir.
        if blok_madde is not None and tampon_madde is not None:
            close_tampon()

        tampon.append(blok)
        tampon_yapi_disi = True
        if tampon_madde is None:
            tampon_madde = blok_madde

        if tampon_uzunluk() >= MIN_CHUNK:
            close_tampon()

    close_tampon()
    return chunks


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Manifest'ten sadece "accepted" dokumanlar
    accepted_docs = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("durum") == "accepted":
                accepted_docs.append(item["doc_id"])

    all_chunks = []
    for doc_id in accepted_docs:
        norm_file = NORM_DIR / f"{doc_id}.txt"
        if not norm_file.exists():
            continue
        all_chunks.extend(chunk_document_madde(
            norm_file.read_text(encoding="utf-8"), doc_id))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    if not all_chunks:
        print("Hic chunk uretilmedi - manifest veya data/norm/ bos olabilir.")
        return

    lengths = [c["char_len"] for c in all_chunks]
    sorted_lengths = sorted(lengths)
    madde_bilinen = sum(1 for c in all_chunks if c.get("madde_no"))
    asan = sum(1 for x in lengths if x > MAX_CHUNK)

    print(f"Toplam Chunk Sayisi : {len(lengths)}")
    print(f"Min Karakter        : {min(lengths)}")
    print(f"Medyan Karakter     : {statistics.median(lengths)}")
    print(f"P90 Karakter        : {sorted_lengths[int(len(sorted_lengths) * 0.90)]}")
    print(f"Max Karakter        : {max(lengths)}")
    print(f"MAX_CHUNK asan      : {asan}   (0 olmali)")
    print(f"Madde no bulunan    : {madde_bilinen}/{len(all_chunks)} "
          f"(%{madde_bilinen / len(all_chunks) * 100:.1f})")


if __name__ == "__main__":
    main()
