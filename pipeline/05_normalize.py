r"""
Metin normalizasyonu: dipnot ayıklama + madde/bölüm bloklarına ayırma.

Kullanım: python pipeline/05_normalize.py
          python pipeline/05_normalize.py --rapor   (dosya yazmadan sadece ölç)

DİPNOT AYIKLAMA — neden sayfa bazlı:
  mevzuat.gov.tr PDF'lerinde dipnotlar sayfanın ALTINDA durur. pdfplumber
  sayfayı yukarıdan aşağı okuduğu için, sayfalar düz metinde birleştirilince
  dipnotlar gövde metninin tam ORTASINA düşüyor.

  Eski hâlde tek satırlık bir regex vardı: ^\d+\s+[harf]. İki ayrı kusuru vardı:

    1) Dipnot numarasından sonra harf değil TARİH geliyor
       ("66 18/6/2014 tarihli ve ..."), yani filtre çoğu dipnotta hiç tutmuyordu.
       Korpusta 2588 dipnot satırı — tüm dipnotların %32'si — bu yüzden kaçtı.
    2) Dipnotlar çok satıra sarıyor. İlk satır tutulsa bile devam satırları
       kalıyor ve birlestir() onları bir önceki madde bloğuna yapıştırıyordu.

  Sonuç: chunk'ların %12.9'u değişiklik metniyle kirliydi ve LLM bunlardan
  yürürlükten kalkmış ceza miktarları çekiyordu (TCK 142 chunk'ında
  "üç yıldan yedi" — 2014'te "beş yıldan on" olarak değişmiş hâli).

  Satır deseninden dipnotun NEREDE BİTTİĞİNİ tahmin etmek çalışmıyor: denendi,
  gövde satırlarını da siliyor. Doğru çözüm yapıyı kullanmak — 02_extract.py
  artık sayfaları ayrı ayrı saklıyor, kural da kesinleşiyor:

      Bir sayfada dipnot başladıysa, o sayfanın geri kalanı dipnottur.

  Güvenlik supabı: kesim noktasından sonra bir madde/bölüm başlığı varsa o
  satır gövdedir; kesim noktası ondan sonraki ilk dipnot satırına taşınır.
  Böylece hiçbir madde başlığı ve öncesi silinemez.

BİLİNÇLİ OLARAK YAPILMAYAN:
  Satır içi "(Değişik: ...)" / "(Mülga: ...)" ibareleri SİLİNMİYOR. "(Mülga)"
  hükmün yürürlükten kalktığını söyler — hukuken anlamlı bilgidir, atılırsa
  sistem kalkmış bir hükmü yürürlükteymiş gibi sunar. Kirlilik kaynağı bunlar
  değil, sayfa altı dipnot BLOKLARIydı.
"""

import json
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

from desenler import (                                    # noqa: E402
    DEGISIKLIK_TABLOSU,
    MADDE_SATIR,
    YAPI_SATIR,
    dipnot_basi_mi,
    yapisik_dipnot_temizle,
)

TEXT = KOK / "data" / "text"
PAGES = KOK / "data" / "pages"
DST = KOK / "data" / "norm"


def yapisal_mi(satir: str) -> bool:
    s = satir.strip()
    return bool(MADDE_SATIR.match(s) or YAPI_SATIR.match(s))


def sayfa_dipnotunu_kes(satirlar: list[str]) -> tuple[list[str], list[str]]:
    """
    Tek bir SAYFANIN satırları -> (gövde, atılan dipnotlar).

    İlk dipnot satırından sayfa sonuna kadar keser. Kesimden sonra madde/bölüm
    başlığı kalıyorsa o aday yanlıştır; bir sonraki dipnot adayına geçilir.
    Hiçbiri güvenli değilse sayfa olduğu gibi bırakılır.
    """
    adaylar = [i for i, s in enumerate(satirlar) if dipnot_basi_mi(s)]
    for kes in adaylar:
        if not any(yapisal_mi(s) for s in satirlar[kes:]):
            return satirlar[:kes], satirlar[kes:]
    return satirlar, []


def dipnot_ayikla(sayfalar: list[str]) -> tuple[list[str], list[str]]:
    """Doküman sayfaları -> (temiz gövde satırları, atılan dipnot satırları)."""
    govde, atilan = [], []
    for sayfa_no, sayfa in enumerate(sayfalar):
        satirlar = [s for s in sayfa.split("\n") if s.strip()]

        # Kanun sonundaki "değişiklik getiren mevzuat" tablosu: başlıktan
        # itibaren dokümanın geri kalanı künyedir, hüküm değil. Satırları
        # "5378 Ek Madde 1 7/7/2005" biçiminde olduğu için chunker bunları
        # madde sanıp 22-25 karakterlik çöp chunk üretiyor ve metadata'ya
        # yanlış madde_no yazıyordu.
        tablo = next((i for i, s in enumerate(satirlar)
                      if DEGISIKLIK_TABLOSU.search(s)), None)
        if tablo is not None:
            govde.extend(yapisik_dipnot_temizle(s.strip())
                         for s in satirlar[:tablo])
            # Tablo doküman sonuna kadar sürer -> kalan sayfaların tamamı da tablo
            atilan.extend(satirlar[tablo:])
            for kalan in sayfalar[sayfa_no + 1:]:
                atilan.extend(s for s in kalan.split("\n") if s.strip())
            break

        tut, at = sayfa_dipnotunu_kes(satirlar)
        govde.extend(yapisik_dipnot_temizle(s.strip()) for s in tut)
        atilan.extend(at)
    return govde, atilan


def sayfalari_oku(doc_id: str) -> tuple[list[str], bool]:
    """
    (sayfalar, sayfa_yapisi_var_mi) döndürür.

    data/pages/ yoksa düz metne düşer. O durumda sayfa sınırı bilinmediği için
    "sayfanın kalanını kes" kuralı UYGULANAMAZ — tüm doküman tek sayfa sayılsaydı
    ilk dipnottan sonrası silinirdi. Fallback'te sadece dipnot satırlarının
    kendisi atılır (eski davranışın düzeltilmiş hâli).
    """
    pj = PAGES / f"{doc_id}.jsonl"
    if pj.exists():
        return [json.loads(s)["metin"] for s in pj.open(encoding="utf-8")], True
    return (TEXT / f"{doc_id}.txt").read_text(encoding="utf-8").split("\n"), False


def kisa_baslik(s: str) -> bool:
    """Kısım/bölüm adı olabilecek satır: kısa, nokta ile bitmiyor, madde değil."""
    return len(s) < 80 and not s.endswith(".") and not MADDE_SATIR.match(s)


def birlestir(satirlar: list[str]) -> list[str]:
    out, tampon = [], []
    yapi_acik = False   # out[-1] bir YAPI bloğu ve henüz adını almadı
    for s in satirlar:
        s = s.rstrip()
        if not s:
            continue
        if YAPI_SATIR.match(s):
            # "BİRİNCİ KISIM" + "BİRİNCİ BÖLÜM" gibi ard arda yapı satırları
            if yapi_acik and not tampon:
                out[-1] = f"{out[-1]} {s}"
                continue
            if tampon:
                out.append(" ".join(tampon))
                tampon = []
            out.append(s)
            yapi_acik = True
            continue
        # "BİRİNCİ KISIM" + "Mükellefiyet" -> tek blok; ad alınca yapı bloğu kapanır
        if yapi_acik and not tampon and kisa_baslik(s):
            out[-1] = f"{out[-1]} {s}"
            yapi_acik = False
            continue
        yapi_acik = False
        if MADDE_SATIR.match(s):
            baslik = None
            # tamponun son satırı kısa ve nokta ile bitmiyorsa madde başlığıdır
            if tampon and len(tampon[-1]) < 80 and not tampon[-1].rstrip().endswith("."):
                baslik = tampon.pop()
            if tampon:
                out.append(" ".join(tampon))
            tampon = [f"{baslik} {s}" if baslik else s]
            continue
        tampon.append(s)
    if tampon:
        out.append(" ".join(tampon))
    return [x for x in out if x]


def main(rapor_modu: bool = False) -> None:
    if not rapor_modu:
        # Diğer beş pipeline scripti bunu yapıyordu, bu dosyada eksikti:
        # data/ .gitignore'da olduğu için temiz klonda FileNotFoundError veriyordu.
        DST.mkdir(parents=True, exist_ok=True)

    kayitlar = [json.loads(s) for s in open(KOK / "corpus_manifest.jsonl", encoding="utf-8")]
    n = sayfasiz = toplam = atilan_top = 0

    for r in kayitlar:
        if r["durum"] != "accepted":
            continue
        sayfalar, yapili = sayfalari_oku(r["doc_id"])
        if not yapili:
            sayfasiz += 1
            # sayfa sınırı yok -> sadece dipnot satırlarını at, blok kesme yapma
            satirlar = [s.strip() for s in sayfalar if s.strip()]
            temiz = [yapisik_dipnot_temizle(s) for s in satirlar if not dipnot_basi_mi(s)]
            atilan = [s for s in satirlar if dipnot_basi_mi(s)]
        else:
            temiz, atilan = dipnot_ayikla(sayfalar)

        toplam += len(temiz) + len(atilan)
        atilan_top += len(atilan)

        if not rapor_modu:
            (DST / f"{r['doc_id']}.txt").write_text(
                "\n\n".join(birlestir(temiz)), encoding="utf-8")
        n += 1

    print(f"{n} dosya işlendi")
    print(f"  dolu satır           : {toplam}")
    print(f"  dipnot olarak atılan : {atilan_top} (%{atilan_top / max(toplam, 1) * 100:.1f})")
    if sayfasiz:
        print(f"  UYARI: {sayfasiz} dosyada sayfa yapısı yok (data/pages/ eksik) —"
              f" 02_extract.py'yi yeniden çalıştır, dipnot temizliği zayıf kalır.")
    if not rapor_modu:
        print(f"  -> {DST}")


if __name__ == "__main__":
    main(rapor_modu="--rapor" in sys.argv)
