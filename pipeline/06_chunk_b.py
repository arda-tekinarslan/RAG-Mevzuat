"""
Strateji B: Madde sınırından chunking.

Strateji A (chunk_a.py) metni 800 karakterden kör kesiyordu — bir maddenin
ortasından bölmek, hem retrieval'ı hem cevabın okunabilirliğini bozuyor.
Burada normalize.py'nin ürettiği yapıyı kullanıyoruz: her madde kendi bloğunda.

Kullanım: python chunk_b.py
"""

import json          # chunk'ları JSONL olarak yazmak ve manifest'i okumak için
import re            # madde/yapı başlıklarını tanımak için düzenli ifadeler
import statistics    # sonunda medyan chunk uzunluğunu hesaplamak için
from pathlib import Path   # dosya yollarını işletim sisteminden bağımsız yazmak için

# --- Ayarlar (bunları değiştirip sonucu ölçebilirsin) ---
MAX_CHUNK = 1200        # bir chunk en fazla bu kadar karakter; aşan madde bölünür
MIN_CHUNK = 200         # bundan kısa chunk tek başına kalmasın, sonrakiyle birleşsin
SPLIT_OVERLAP = 100     # SADECE uzun madde bölünürken kullanılır (yapay sınır attığımız tek yer)
MADDE_ARAMA_SINIRI = 150

# --- Dosya yolları ---
KOK = Path(__file__).parent.parent
MANIFEST_PATH = KOK / "corpus_manifest.jsonl"
NORM_DIR = KOK / "data" / "norm"
OUTPUT_PATH = KOK / "data" / "chunks_strategy_b.jsonl"   # a için _a 

# --- Desenler ---
# normalize.py'de aynı desenler var; orada blokları OLUŞTURMAK için, burada TANIMAK için kullanıyoruz.
# Grup 2 (\d+) madde numarasını yakalar — metadata'ya yazacağımız değer o.
MADDE = re.compile(r"(Madde|MADDE|GEÇİCİ MADDE|EK MADDE)\s*(\d+)")

# "BİRİNCİ KISIM", "İKİNCİ BÖLÜM" gibi yapısal başlıklar.
# Bunlar tek başına anlam taşımaz — chunk yapmayıp sonraki maddeye yapıştıracağız.
YAPI = re.compile(
    r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|"
    r"DOKUZUNCU|ONUNCU|[A-ZÇĞİÖŞÜ\s]+)\s*(KİTAP|KISIM|BÖLÜM|AYIRIM)"
)


# ------------------------------------------------------------- TODO 1
def madde_no_bul(blok: str) -> str | None:
    """
    Neden lazım: chunk metadata'sına madde numarasını koyacağız. "Cevap 5237
    sayılı kanunun 81. maddesinden geldi" demek, "şu dosyadan geldi"den güçlü.

    İpucu: MADDE regex'inin 2. grubu (\\d+) madde numarasını yakalıyor.
    DİKKAT: normalize.py madde başlığını maddenin ÖNÜNE ekliyor
    ("Adam öldürme Madde 81 — ..."), yani match() bloğun başında eşleşmeyebilir.
    search() kullanırsan metnin ortasındaki "madde 5'e göre" gibi atıfları da
    yakalarsın. data/norm/*.txt içinden birkaç bloğa bakıp karar ver.
    """
    m = MADDE.search(blok[:MADDE_ARAMA_SINIRI])
    return m.group(2) if m else None


# ------------------------------------------------------------- TODO 2
def uzun_bloku_bol(blok: str) -> list[str]:
    """
    blok: MAX_CHUNK'tan uzun tek bir madde metni
    dönüş: parçalanmış metin listesi, her biri <= MAX_CHUNK

    Strateji A'daki kayan pencere mantığının aynısı — ama her metne değil,
    sadece gerçekten uzun maddelere uygulanıyor. Fark bu.

    İpucu: chunk_a.py'deki chunk_document_fixed'in döngüsünü örnek al:
        start = 0, adım = MAX_CHUNK - SPLIT_OVERLAP
        while start < len(blok): dilimle, ekle, ilerlet
    """
    parcalar = []
    adim = MAX_CHUNK - SPLIT_OVERLAP
    start = 0
    while start < len(blok):
        parca = blok[start:start + MAX_CHUNK]
        if len(parca) < SPLIT_OVERLAP and parcalar:
            break
        parcalar.append(parca)
        start += adim
    return parcalar


# ------------------------------------------------------------- TODO 3
def chunk_document_madde(text: str, doc_id: str) -> list[dict]:
    """
    text:   data/norm/{doc_id}.txt içeriği (bloklar "\\n\\n" ile ayrılmış)
    doc_id: doküman kimliği
    dönüş: chunk sözlükleri listesi:
        {"chunk_id": f"{doc_id}_b_{idx:04d}", "doc_id": doc_id,
         "text": ..., "char_len": ..., "madde_no": ... veya None}
    Akış:
      1. text'i "\\n\\n" ile bloklara ayır
      2. Blokları sırayla gez, bir "tampon" biriktir:
         - YAPI satırıysa: tek başına chunk yapma, tampona ekle
         - Normal blok/madde ise: tampona ekle
      3. Tampon MIN_CHUNK'ı geçtiyse chunk olarak kapat, tamponu sıfırla
      4. Tek blok MAX_CHUNK'tan uzunsa uzun_bloku_bol ile parçala
      5. Döngü bitince tamponda kalan varsa onu da chunk yap
    madde_no: chunk'ın İLK maddesinin numarası (birleşme olduysa ilkini yaz)..
    """

    chunks = []
    tampon = [] #Daha chunk olamamız MIN_CHUNKI geçememiş şeyler beklete yeri
    tampon_madde = None #Tampondaki ilk madde numarası
    idx = 0

    def close_tampon():
        nonlocal tampon, tampon_madde, idx
        if not tampon:
            return
        metin = "\n\n".join(tampon)

        chunks.append({
            "chunk_id": f"{doc_id}_b_{idx:04d}",
            "doc_id": doc_id,
            "text": metin,
            "char_len": len(metin),
            "madde_no": tampon_madde,
        })
        idx += 1
        tampon = []
        tampon_madde = None

    for blok in text.split("\n\n"):
        blok = blok.strip()
        if not blok:
            continue

        # Yapısal başlık (BİRİNCİ BÖLÜM vb.): tek başına chunk olmasın, tampona bekle
        if YAPI.match(blok):
            tampon.append(blok)
            continue

        if len(blok) > MAX_CHUNK:
            # Tamponda sadece başlık gibi kısa bir şey varsa, onu ayrı chunk yapma —
            # uzun bloğun önüne ekle ki '?ÜÇÜNCÜ BÖLÜM İade' gibi 17 karakterlik
            # işe yaramaz chunk'lar oluşmasın.
            tampon_metin = "\n\n".join(tampon)
            if tampon and len(tampon_metin) < MIN_CHUNK:
                blok = tampon_metin + "\n\n" + blok
                tampon = []
                tampon_madde = None
            else:
                close_tampon()

            blok_madde = madde_no_bul(blok)
            for parca in uzun_bloku_bol(blok):
                chunks.append({
                    "chunk_id": f"{doc_id}_b_{idx:04d}",
                    "doc_id": doc_id,
                    "text": parca,
                    "char_len": len(parca),
                    "madde_no": blok_madde,  # parçaların hepsi aynı maddeden
                })
                idx += 1
            continue

        tampon.append(blok)
        if tampon_madde is None:
            tampon_madde = madde_no_bul(blok)

        if sum(len(b) for b in tampon) >= MIN_CHUNK:
            close_tampon()

    close_tampon()
    return chunks


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    # Manifest'ten sadece "accepted" dokümanlar — rejected/duplicate korpusa girmez
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
        text = norm_file.read_text(encoding="utf-8")
        all_chunks.extend(chunk_document_madde(text, doc_id))
 
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            # ensure_ascii=False: Türkçe karakterler kaçış dizisi değil, doğrudan yazılsın
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
 
    # --- Ölçüm ---
    if not all_chunks:
        print("Hiç chunk üretilmedi — manifest veya data/norm/ boş olabilir.")
        return
 
    lengths = [c["char_len"] for c in all_chunks]
    sorted_lengths = sorted(lengths)
    madde_bilinen = sum(1 for c in all_chunks if c.get("madde_no"))
 
    print(f"Toplam Chunk Sayısı : {len(lengths)}")
    print(f"Min Karakter        : {min(lengths)}")
    print(f"Medyan Karakter     : {statistics.median(lengths)}")
    # P90: chunk'ların %90'ı bu uzunluğun altında. Birkaç aşırı uzun chunk
    # ortalamayı çarpıtır ama P90'ı çarpıtmaz — o yüzden ortalamadan bilgilendirici.
    print(f"P90 Karakter        : {sorted_lengths[int(len(sorted_lengths) * 0.90)]}")
    print(f"Max Karakter        : {max(lengths)}")
    print(f"Madde no bulunan    : {madde_bilinen}/{len(all_chunks)} "
          f"(%{madde_bilinen / len(all_chunks) * 100:.1f})")
 
 
if __name__ == "__main__":
    main()

   