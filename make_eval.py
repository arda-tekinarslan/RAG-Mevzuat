import json
import random
import sys
from pathlib import Path
import ollama

CHUNKS_PATH = Path("data/chunks_strategy_b.jsonl")
CANDIDATES_PATH = Path("eval_candidates.jsonl")
EVAL_PATH = Path("eval_set.jsonl")

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

def build_prompt(chunk_text:str)->str:
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

def uret():
    chunks = [json.loads(s) for s in open(CHUNKS_PATH,encoding="utf-8")]

    uygun = [c for c in chunks
        if c["char_len"] >= MIN_CHUNK_LEN
        and c.get("madde_no")
        and temiz_mi(c["text"])]
    print(f"{len(uygun)} uygun chunk (toplam {len(chunks)})")

    random.seed(SEED)
    secilen = random.sample(uygun,min(HEDEF_SORU,len(uygun)))

    out = CANDIDATES_PATH.open("w",encoding="utf-8")
    for i,chunk in enumerate(secilen,1):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(chunk["text"])}],
                options={"temperature": 0.3},
            )
            soru = response["message"]["content"].strip().strip('"')
        except Exception as e:
            print(f"[{i}] HATA: {e}")
            continue

        kayit = {
            "soru": soru,
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "madde_no": chunk["madde_no"],
            "kaynak_metin": chunk["text"][:300],   # review sırasında bakmak için
        }
        out.write(json.dumps(kayit, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(secilen)}] {soru}")
 
    out.close()
    print(f"\n{CANDIDATES_PATH} yazıldı. Şimdi: python make_eval.py review")

def review():
    if not CANDIDATES_PATH.exists():
        sys.exit(f"{CANDIDATES_PATH} yok — önce 'python make_eval.py' çalıştır.")
    adaylar = [json.loads(s) for s in open(CANDIDATES_PATH,encoding="utf-8")]
    onaylanan = []
    print("Her soru için: [e]vet tut, [h]ayır ele, [d]üzelt, [q] çık ve kaydet\n")

    for i,aday in enumerate(adaylar,1):
        print("=" * 70)
        print(f"[{i}/{len(adaylar)}] SORU: {aday['soru']}")
        print(f"  kaynak: {aday['baslik'] if 'baslik' in aday else aday['doc_id']} / madde {aday['madde_no']}")
        print(f"  metin  : {aday['kaynak_metin'][:200]}...")

        while(True):
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
    with open(EVAL_PATH,"w",encoding="utf-8") as f:
        for aday in onaylanan:
                aday.pop("kaynak_metin",None)
                f.write(json.dumps(aday,ensure_ascii=False) +"\n")
    print(f"\n{len(onaylanan)} soru onaylandı -> {EVAL_PATH}")


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "uret"
    {"uret": uret, "review": review}[komut]()


