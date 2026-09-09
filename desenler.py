"""
Mevzuat metinlerinde kullanılan ortak düzenli ifadeler.

Bu dosya bilerek bağımsız tutuldu — sadece `re` import ediyor, ağır bir
bağımlılığı yok. Böylece hem pipeline scriptleri hem tools/ altındaki debug
scriptleri aynı desenleri kullanabiliyor.

Daha önce aynı desenler 05_normalize.py, 06_chunk_b.py ve tools/debug_chunks.py
içinde ayrı ayrı kopyalanmıştı; birini değiştirince diğerleri sessizce eskiyordu.
debug_chunks.py'nin başındaki "chunk_b.py ile AYNI regex olmalı" yorumu tam da
bu riski işaret ediyordu — artık tek kaynak var.
"""

import re

# --------------------------------------------------------------- yapısal
# normalize.py SATIR bazlı çalışır: başlık kendi satırındadır, sonuna $ gerekir.
# chunk_b.py BLOK bazlı çalışır: normalize başlığı ardındaki kısa adla
# birleştirmiştir ("BİRİNCİ BÖLÜM Amaç"), orada $ olmamalı.
# İki farklı davranış bilinçli; bu yüzden iki ayrı derlenmiş desen veriyoruz.
_YAPI = (
    r"(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|"
    r"DOKUZUNCU|ONUNCU|[A-ZÇĞİÖŞÜ\s]+)\s*(KİTAP|KISIM|BÖLÜM|AYIRIM)"
)
YAPI_SATIR = re.compile(r"^" + _YAPI + r"\s*$")
YAPI_BLOK = re.compile(r"^" + _YAPI)

# ---------------------------------------------------------------- madde
# Alternatif SIRASI önemli: "GEÇİCİ MADDE" ve "EK MADDE", "MADDE"den ÖNCE
# gelmeli. Aksi halde re alternatifleri soldan denediği için tür bilgisi
# konuma göre kaybolabilir.
_MADDE_TUR = r"(GEÇİCİ MADDE|EK MADDE|Geçici Madde|Ek Madde|MADDE|Madde)"
MADDE_SATIR = re.compile(r"^" + _MADDE_TUR + r"\s*\d+")
MADDE_ARA = re.compile(_MADDE_TUR + r"\s*(\d+)")

MADDE_ARAMA_SINIRI = 150   # blok başındaki bu kadar karakterde madde no aranır


def madde_no_bul(blok: str, sinir: int = MADDE_ARAMA_SINIRI) -> str | None:
    """
    Bloğun ilk maddesinin kimliğini döndürür: "141", "GEÇİCİ 4", "EK 2".

    Tür bilgisini KORUMAK şart: bir kanunda "Madde 4", "EK MADDE 4" ve
    "GEÇİCİ MADDE 4" aynı anda bulunabiliyor. Sadece numarayı döndürmek
    korpusta 117 gerçek çarpışma üretiyordu — hem eval'i şişiriyordu
    (geçici maddeden gelen chunk, normal madde sorusunda hit sayılıyor)
    hem de kullanıcıya yanlış madde atfı gösteriliyordu.
    """
    m = MADDE_ARA.search(blok[:sinir])
    if not m:
        return None
    tur = m.group(1).upper()          # "Geçici Madde".upper() -> "GEÇICI MADDE"
    no = m.group(2)
    if tur.startswith("GEÇ"):         # GEÇİCİ / GEÇICI ikisini de yakalar
        return "GEÇİCİ " + no
    if tur.startswith("EK"):
        return "EK " + no
    return no


# ------------------------------------------------- degisiklik tablosu
# Kanun metinlerinin SONUNDA "hangi kanun hangi maddeyi ne zaman degistirdi"
# tablosu bulunur:
#     5216 SAYILI KANUNA EK VE DEGISIKLIK GETIREN MEVZUATIN
#     YURURLUGE GIRIS TARIHINI GOSTERIR LISTE
#     5378 Ek Madde 1 7/7/2005
#     5390 Madde 6 ve Islenemeyen hukum 13/7/2005
#
# Bu satirlar hukum degil, degisiklik kunyesi. Ama icinde "Madde 6",
# "Ek Madde 1" gectigi icin chunker onlari madde sanip 22-25 karakterlik
# cop chunk'lar uretiyor ve metadata'ya yanlis madde_no yaziyordu
# ("7511 29/5/2024 Madde 2" -> madde_no=2).
#
# Baslik korpusta 219/298 dokumanda ve TEK varyantla geciyor; tablodan
# sonra hicbir dokumanda "ISLENEMEYEN HUKUM" bolumu gelmiyor, yani basliktan
# itibaren dokuman sonuna kadar kesmek icerik kaybettirmiyor.
DEGISIKLIK_TABLOSU = re.compile(r"SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN")


# --------------------------------------------------------------- dipnot
# mevzuat.gov.tr PDF'lerinde dipnotlar sayfanin ALTINDA durur:
#   "66 18/6/2014 tarihli ve 6545 sayili Kanunun 62 nci maddesiyle bu fikrada
#    yer alan "uc yildan yedi" ibaresi "bes yildan on" seklinde degistirilmistir."
#
# Bir dipnot satirini tanimak icin İKİ kosul birden aranir:
#   1) satir dipnot numarasiyla baslar  (DIPNOT_NUMARA)
#   2) satirda degisiklik/dipnot sozlugu gecer  (DIPNOT_KANIT)
#
# Tek basina (1) yetmez: kanunlarda koy/mahalle listeleri "6 Filiz",
# "7 1.Orucgazi" gibi numarali satirlar iceriyor ve eski filtre (^\d+\s+[harf])
# bunlari dipnot sanip siliyordu. Iki kosulu birden aramak bunu bitiriyor.
DIPNOT_NUMARA = re.compile(r"^\d{1,3}\s+\S")
DIPNOT_KANIT = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4}|tarihli ve|sayılı Kanun|sayılı KHK|maddesiyle|"
    r"ibaresi|şeklinde değiştir|metninden çıkarıl|yürürlükten kaldırıl|"
    r"Anayasa Mahkemesi|Mülga|Değişik|İptal|eklenmiştir|yürürlüğe girer)"
)


def dipnot_basi_mi(satir: str) -> bool:
    """Bu satır bir dipnot girdisinin ilk satırı mı? (numara + dipnot sözlüğü)"""
    s = satir.strip()
    return bool(DIPNOT_NUMARA.match(s)) and bool(DIPNOT_KANIT.search(s))


# Gövdeye yapışmış üstsimge dipnot numaraları.
# pdfplumber üstsimgeyi normal metin gibi çıkarıyor:
#   "Nitelikli hırsızlık66", "cezalandırılır.37", "suçunun;33"
YAPISIK_HARF = re.compile(r"(?<=[a-zçğıöşüA-ZÇĞİÖŞÜ])\d{1,4}(?=\s|$)")

# Noktalamaya yapisan halde, noktalamadan ONCE en az IKI harf sarti var.
# Bu, "m.31" / "s.12" gibi tek harfli kisaltma+sayi kaliplarini korur;
# lookbehind sabit genislikte kaldigi icin 'ye gerek kalmiyor.
YAPISIK_NOKT = re.compile(
    # Virgul de dahil: "avukatlar,2" gibi 179 yapisik isaret vardi.
    # Rakam+noktalama korunur ("madde 3,4" -> virgulden onceki "3" harf degil).
    r"(?<=[a-zçğıöşü”\"][a-zçğıöşü”\"][.,;:])\d{1,3}(?=\s|$)"
)


def yapisik_dipnot_temizle(satir: str) -> str:
    """Gövdeye yapışmış üstsimge dipnot numaralarını at."""
    satir = YAPISIK_NOKT.sub("", satir)
    return YAPISIK_HARF.sub("", satir)
