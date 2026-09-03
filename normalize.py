"""Metin normalizasyonu. Kullanım: python normalize.py"""
import json, re
from pathlib import Path

SRC = Path("data/text")
DST = Path("data/norm")
DST.mkdir(parents=True, exist_ok=True)

# Yapisal satirlar: bunlar kendi satirinda kalmali
YAPI = re.compile(
    r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|"
    r"DOKUZUNCU|ONUNCU|[A-ZÇĞİÖŞÜ\s]+)\s*(KİTAP|KISIM|BÖLÜM|AYIRIM)\s*$"
)
MADDE = re.compile(r"^(Madde|MADDE|GEÇİCİ MADDE|EK MADDE)\s*\d+")
DIPNOT = re.compile(r"^\d+\s+[A-ZÇĞİÖŞÜa-zçğıöşü]")   # "1 Bu Kanunun..."

def kisa_baslik(s):
    """Kisim/bolum adi olabilecek satir: kisa, nokta ile bitmiyor, madde degil."""
    return len(s) < 80 and not s.endswith(".") and not MADDE.match(s)

def birlestir(satirlar):
    out, tampon = [], []
    yapi_acik = False   # out[-1] bir YAPI blogu ve henuz adini almadi
    for s in satirlar:
        s = s.rstrip()
        if not s:
            continue
        if YAPI.match(s):
            # "BIRINCI KISIM" + "BIRINCI BOLUM" gibi ard arda yapi satirlari
            if yapi_acik and not tampon:
                out[-1] = f"{out[-1]} {s}"
                continue
            if tampon:
                out.append(" ".join(tampon)); tampon = []
            out.append(s)
            yapi_acik = True
            continue
        # "BIRINCI KISIM" + "Mukellefiyet" -> tek blok; ad alinca yapi blogu kapanir
        if yapi_acik and not tampon and kisa_baslik(s):
            out[-1] = f"{out[-1]} {s}"
            yapi_acik = False
            continue
        yapi_acik = False
        if MADDE.match(s):
            baslik = None
            # tamponun son satiri kisa ve nokta ile bitmiyorsa madde basligidir
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

kayitlar = [json.loads(s) for s in open("corpus_manifest.jsonl", encoding="utf-8")]
n = 0
for r in kayitlar:
    if r["durum"] != "accepted":
        continue
    ham = Path(r["text_path"]).read_text(encoding="utf-8")
    satirlar = [s for s in ham.split("\n") if not DIPNOT.match(s.strip())]
    metin = "\n\n".join(birlestir(satirlar))
    (DST / f"{r['doc_id']}.txt").write_text(metin, encoding="utf-8")
    n += 1

print(f"{n} dosya normalize edildi -> data/norm/")