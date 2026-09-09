"""
Eski indeks (yedek) ile yeni indeksi AYNI sorularda karşılaştır.

Kullanım: python tools/karsilastir_index.py

Neden gerekli:
  Chunking düzeltmelerinden sonra hit@3/hit@5 düştü. İki sebep birbirine
  karışıyor:
    (a) chunking değişti  — maddeler artık birleştirilmiyor, chunk sayısı arttı
    (b) ground truth DÜZELDİ — 34 sorunun 6'sının etiketi yanlıştı
        (madde türü ve birleşmiş madde hataları)

  (b) tek başına skoru düşürür: yanlış etiket "kolay" hedefi işaret ediyordu.
  Bu ikisini ayırmak için ölçüm, etiketi DEĞİŞMEYEN sorularla sınırlanıyor.
  Kalan farkın tek kaynağı chunking olur.
"""

import json
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
YEDEK = KOK.parent / "rag-mevzuat-yedek"
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "eval"))

import chromadb                                        # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer   # noqa: E402

import rag_core as rc                                  # noqa: E402
from metrikler import K_DEGERLERI, dogru_mu_madde, metrikleri_hesapla  # noqa: E402


def ortak_sorular():
    """Ground truth'u DEĞİŞMEYEN sorular (eski etiket == yeni etiket)."""
    yeni = [json.loads(s) for s in open(KOK / "eval_set.jsonl", encoding="utf-8")]
    eski = [json.loads(s) for s in open(YEDEK / "eval_set.jsonl", encoding="utf-8")]
    esk_map = {e["soru"]: e for e in eski}
    ortak = []
    for y in yeni:
        e = esk_map.get(y["soru"])
        if e and str(e["madde_no"]) == str(y["madde_no"]) and e["doc_id"] == y["doc_id"]:
            ortak.append(y)
    return ortak, len(yeni)


def olc(db_dir, sorular, embed_model, cross_encoder):
    client = chromadb.PersistentClient(path=str(db_dir))
    col = client.get_collection(rc.COLLECTION_NAME)
    dense_s, rerank_s = [], []
    for item in sorular:
        adaylar = rc.dense_ara(item["soru"], col, embed_model)
        dense_s.append(adaylar)
        rerank_s.append(rc.rerank(item["soru"], adaylar, cross_encoder))
    return (metrikleri_hesapla(dense_s, sorular, dogru_mu_madde),
            metrikleri_hesapla(rerank_s, sorular, dogru_mu_madde),
            col.count())


def main():
    sorular, toplam = ortak_sorular()
    print(f"Ground truth'u degismeyen soru: {len(sorular)}/{toplam}")
    print("(degisen 6 soru DISARIDA - onlarin eski etiketi yanlisti)\n")

    embed_model = SentenceTransformer(rc.EMBED_MODEL, device="cpu")
    cross_encoder = CrossEncoder(rc.RERANK_MODEL, device="cpu")

    print("Eski indeks olculuyor (yedek)...")
    e_dense, e_rerank, e_n = olc(YEDEK / "chroma_db", sorular, embed_model, cross_encoder)
    print("Yeni indeks olculuyor...")
    y_dense, y_rerank, y_n = olc(rc.DB_DIR, sorular, embed_model, cross_encoder)

    print(f"\neski chunk sayisi: {e_n}   yeni: {y_n}\n")
    for ad, e, y in (("DENSE", e_dense, y_dense), ("DENSE + RERANK", e_rerank, y_rerank)):
        print("=" * 56)
        print(f"{ad}  (madde duzeyi, {len(sorular)} soru)")
        print("=" * 56)
        print(f"{'Metrik':<10} {'ESKI':>12} {'YENI':>12} {'Fark':>12}")
        print("-" * 56)
        for k in [f"hit@{x}" for x in K_DEGERLERI] + ["MRR"]:
            fark = y[k] - e[k]
            isaret = "+" if fark > 0 else ""
            if k == "MRR":
                print(f"{k:<10} {e[k]:>12.3f} {y[k]:>12.3f} "
                      f"{isaret + f'{fark:.3f}':>12}")
            else:
                print(f"{k:<10} {e[k] * 100:>11.1f}% {y[k] * 100:>11.1f}% "
                      f"{isaret + f'{fark * 100:.1f}':>11}%")
        print()


if __name__ == "__main__":
    main()
