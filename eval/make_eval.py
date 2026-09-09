"""
LLM ile eval sorusu üretimi + elle onaylama.

Kullanım:
    python eval/make_eval.py                # aday sorular üret
    python eval/make_eval.py --filtresiz    # kirli chunk'ları da havuza kat
    python eval/make_eval.py review         # adayları elle onayla -> eval_set.jsonl

METODOLOJİ UYARISI — bu eval setinin iki bilinen yanlılığı var, sunumda
kendin söyle, müdürün sorması yerine:

  1) SEÇİM YANLILIĞI: KOTU_DESENLER filtresi, değişiklik/mülga metni içeren
     chunk'ları havuzun dışında bırakıyor. Bunlar sistemin en çok zorlandığı
     chunk'lar. Yani ölçtüğün hit@k bir ÜST SINIR — "korpusun temiz kısmında"
     geçerli. Filtrenin kaç chunk elediği aşağıda raporlanıyor; --filtresiz
     ile kapatıp farkı ölçebilirsin.

  2) LEAKAGE: Sorular, cevabı içeren chunk LLM'e gösterilerek üretiliyor.
     Prompt "birebir kopyalama" diyerek bunu azaltıyor ama kaldırmıyor —
     sorunun kelime dağılımı chunk'a koşullu kalıyor. Gerçek kullanıcı
     soruları böyle davranmaz. Telafi: 10-15 soruyu chunk'lara hiç bakmadan
     elle yaz ve iki seti AYRI raporla.
"""

import json
import random
import sys
from pathlib import Path

import ollama

KOK = Path(__file__).parent.parent
CHUNKS_PATH = KOK / "data" / "chunks_strategy_b.jsonl"
CANDIDATES_PATH = KOK / "eval" / "eval_candidates.jsonl"
EVAL_PATH = KOK / "eval_set.jsonl"

MODEL_NAME = "qwen2.5:3b-instruct"
HEDEF_SORU = 80          # eleme sonrası ~50 kalmasını bekliyoruz
MIN_CHUNK_LEN = 400      # çok kısa chunk'tan iyi soru çıkmaz
SEED = 42

KOTU_DESENLER = [
    "ile ilgili olup",
    "yerine işlenmiştir",
    "yürürlükten kaldırılmıştır",
    "(Mülga",
    "(Değişik",
    "Bu Kanun hükümlerini Cumhurbaşkanı yürütür",
    "Bu Kanun yayımı tarihinde yürürlüğe girer",
]


def temiz_mi(text: str) -> bool:
    return not any(d in text for d in KOTU_DESENLER)


def build_prompt(chunk_text: str) -> str:
    return f"""Aşağıda bir Türk mevzuat metni var. Bu metinden cevaplanabilecek TEK BİR soru yaz.

KURALLAR:
- Soru, metni görmemiş bir kişinin doğal olarak soracağı türden olsun.
- Metindeki cümleleri veya nadir terimleri BİREBİR KOPYALAMA. Kendi kelimelerinle sor.
- Sayı, tarih, oran gibi spesifik değerleri soruya KOYMA — onlar cevabın parçası.
- Soru tek cümle olsun, 15 kelimeyi geçmesin.
- SADECE soruyu yaz. Açıklama, giriş cümlesi, tırnak işareti ekleme.

METİN:
{chunk_text}

SORU:"""


def uret(filtresiz: bool = False):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(s) for s in f]

    temel = [c for c in chunks
             if c["char_len"] >= MIN_CHUNK_LEN and c.get("madde_no")]
    uygun = temel if filtresiz else [c for c in temel if temiz_mi(c["text"])]

    elenen = len(temel) - len(uygun)
    print(f"{len(uygun)} uygun chunk (toplam {len(chunks)})")
    print(f"  uzunluk + madde_no filtresinden geçen: {len(temel)}")
    if filtresiz:
        print("  KOTU_DESENLER filtresi KAPALI (--filtresiz)")
    else:
        print(f"  KOTU_DESENLER filtresinin elediği : {elenen} "
              f"(%{elenen / max(len(temel), 1) * 100:.1f}) "
              f"<- seçim yanlılığı buradan geliyor")

    random.seed(SEED)
    secilen = random.sample(uygun, min(HEDEF_SORU, len(uygun)))

    with CANDIDATES_PATH.open("w", encoding="utf-8") as out:
        for i, chunk in enumerate(secilen, 1):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": build_prompt(chunk["text"])}],
                    # temperature: soru üretiminde bir miktar çeşitlilik istiyoruz,
                    # cevap üretiminden farklı olarak burada 0.3 bilinçli.
                    options={"temperature": 0.3, "num_ctx": 8192},
                )
                soru = response["message"]["content"].strip().strip('"')
            except Exception as e:
                print(f"[{i}] HATA: {e}")
                continue

            out.write(json.dumps({
                "soru": soru,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "madde_no": chunk["madde_no"],
                "kaynak_metin": chunk["text"][:300],   # review sırasında bakmak için
            }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(secilen)}] {soru}")

    print(f"\n{CANDIDATES_PATH} yazıldı. Şimdi: python eval/make_eval.py review")


def review():
    if not CANDIDATES_PATH.exists():
        sys.exit(f"{CANDIDATES_PATH} yok — önce 'python eval/make_eval.py' çalıştır.")
    with open(CANDIDATES_PATH, encoding="utf-8") as f:
        adaylar = [json.loads(s) for s in f]

    onaylanan = []
    print("Her soru için: [e]vet tut, [h]ayır ele, [d]üzelt, [q] çık ve kaydet\n")

    for i, aday in enumerate(adaylar, 1):
        print("=" * 70)
        print(f"[{i}/{len(adaylar)}] SORU: {aday['soru']}")
        print(f"  kaynak: {aday.get('baslik', aday['doc_id'])} / madde {aday['madde_no']}")
        print(f"  metin  : {aday['kaynak_metin'][:200]}...")

        while True:
            cevap = input(" > ").strip().lower()
            if cevap == "e":
                onaylanan.append(aday)
                break
            if cevap == "h":
                break
            if cevap == "d":
                yeni = input(" yeni soru ").strip()
                if yeni:
                    aday["soru"] = yeni
                    onaylanan.append(aday)
                break
            if cevap == "q":
                kaydet(onaylanan)
                return
            print("  e / h / d / q yaz")
    kaydet(onaylanan)


def kaydet(onaylanan):
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        for aday in onaylanan:
            aday.pop("kaynak_metin", None)
            f.write(json.dumps(aday, ensure_ascii=False) + "\n")
    print(f"\n{len(onaylanan)} soru onaylandı -> {EVAL_PATH}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    komut = argv[0] if argv else "uret"
    if komut == "review":
        review()
    else:
        uret(filtresiz="--filtresiz" in sys.argv)
