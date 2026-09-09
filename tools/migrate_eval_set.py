"""
eval_set.jsonl'i yeni chunking'e taşı.

Kullanım:
    python tools/migrate_eval_set.py --kontrol   # sadece raporla, yazma
    python tools/migrate_eval_set.py             # eval_set.jsonl'i güncelle

Neden gerekli:
  eval_set.jsonl ground truth'u (chunk_id, doc_id, madde_no) olarak tutuyor.
  Chunking değişince chunk_id'ler kaydı ve madde_no formatı değişti
  ("4" -> "GEÇİCİ 4" gibi). Eski ground truth'la ölçüm yapmak sessizce yanlış
  sonuç verir — sorular hâlâ geçerli ama işaret ettikleri chunk artık yok.

Nasıl:
  Sorunun üretildiği ESKİ chunk metni (yedekten) ile AYNI dokümandaki yeni
  chunk'lar karşılaştırılıp en çok örtüşen seçilir. Örtüşme kelime kümesi
  Jaccard benzerliği ile ölçülür. Düşük örtüşmeli eşleşmeler raporlanır —
  onları elle gözden geçir.
"""

import json
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
EVAL_PATH = KOK / "eval_set.jsonl"
YENI_CHUNKS = KOK / "data" / "chunks_strategy_b.jsonl"
ESKI_CHUNKS = KOK.parent / "rag-mevzuat-yedek" / "chunks_strategy_b.jsonl"

DUSUK_ORTUSME = 0.35   # bunun altındaki eşleşmeler elle bakılmalı


def kelimeler(metin):
    return set(metin.lower().split())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main(yaz):
    if not ESKI_CHUNKS.exists():
        sys.exit(f"{ESKI_CHUNKS} yok — eski chunk dosyası olmadan eşleme yapılamaz.")

    sorular = [json.loads(s) for s in open(EVAL_PATH, encoding="utf-8")]
    eski = {json.loads(s)["chunk_id"]: json.loads(s)
            for s in open(ESKI_CHUNKS, encoding="utf-8")}

    yeni_by_doc = {}
    with open(YENI_CHUNKS, encoding="utf-8") as f:
        for s in f:
            c = json.loads(s)
            yeni_by_doc.setdefault(c["doc_id"], []).append(c)

    guncel, dusuk, bulunamayan = [], [], []

    for item in sorular:
        eski_chunk = eski.get(item["chunk_id"])
        if eski_chunk is None:
            bulunamayan.append(item)
            guncel.append(item)
            continue

        hedef = kelimeler(eski_chunk["text"])
        adaylar = yeni_by_doc.get(item["doc_id"], [])
        if not adaylar:
            bulunamayan.append(item)
            guncel.append(item)
            continue

        en_iyi = max(adaylar, key=lambda c: jaccard(hedef, kelimeler(c["text"])))
        skor = jaccard(hedef, kelimeler(en_iyi["text"]))

        yeni_item = dict(item)
        yeni_item["chunk_id"] = en_iyi["chunk_id"]
        yeni_item["madde_no"] = en_iyi["madde_no"]
        yeni_item["_ortusme"] = round(skor, 3)
        guncel.append(yeni_item)

        if skor < DUSUK_ORTUSME:
            dusuk.append((item, en_iyi, skor))

    print(f"{len(sorular)} soru işlendi")
    print(f"  eşleşen              : {len(sorular) - len(bulunamayan)}")
    print(f"  eski chunk bulunamadı: {len(bulunamayan)}")
    print(f"  düşük örtüşme (<{DUSUK_ORTUSME}) : {len(dusuk)}")

    degisen_madde = [(i, g) for i, g in zip(sorular, guncel)
                     if str(i.get("madde_no")) != str(g.get("madde_no"))]
    print(f"  madde_no değişen     : {len(degisen_madde)}")
    for i, g in degisen_madde[:10]:
        print(f"    {i['doc_id']}: {i['madde_no']!r} -> {g['madde_no']!r}")

    if dusuk:
        print("\n--- ELLE BAKILMASI GEREKENLER ---")
        for item, yeni_c, skor in dusuk:
            print(f"\n  ortusme={skor:.2f}  {item['soru']}")
            print(f"    eski madde {item['madde_no']} -> yeni {yeni_c['madde_no']}")
            print(f"    yeni metin: {yeni_c['text'][:130]!r}")

    if yaz:
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            for g in guncel:
                g.pop("_ortusme", None)
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"\n-> {EVAL_PATH} güncellendi")
    else:
        print("\n(kontrol modu — dosya yazılmadı, yazmak için --kontrol'ü kaldır)")


if __name__ == "__main__":
    main(yaz="--kontrol" not in sys.argv)
