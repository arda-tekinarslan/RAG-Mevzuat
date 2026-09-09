"""
Sadece dense retrieval ölçümü: hit@k ve MRR.

Kullanım: python eval/eval_retrieval.py

Rerank karşılaştırması için: python eval/eval_rerank.py

Chroma'da her chunk dört şey tutuyor: id (chunk_id), embedding (384 boyutlu
vektör), document (chunk metni), metadata (doc_id, madde_no, baslik, char_len).
"""

import json
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(Path(__file__).parent))

import rag_core as rc                                          # noqa: E402
from metrikler import (                                        # noqa: E402
    K_DEGERLERI,
    dogru_mu_chunk,
    dogru_mu_madde,
    metrikleri_hesapla,
)

EVAL_PATH = KOK / "eval_set.jsonl"
TOP_K = 10


def main():
    sorular = [json.loads(s) for s in open(EVAL_PATH, encoding="utf-8")]
    n = len(sorular)
    print(f"{n} soru yüklendi\n")

    embed_model, _, collection = rc.yukle_sistem()
    cihaz = rc.cihaz_bilgisi(embed_model)

    siralamalar, sureler = [], []
    for item in sorular:
        t0 = time.perf_counter()
        # E5 sorgu öneki rag_core'dan geliyor — indeksleme "passage: " ile
        # yapıldığı için sorgunun da birebir "query: " olması şart.
        sonuclar = rc.dense_ara(item["soru"], collection, embed_model, n=TOP_K)
        sureler.append(time.perf_counter() - t0)
        siralamalar.append(sonuclar)

    madde_m = metrikleri_hesapla(siralamalar, sorular, dogru_mu_madde)
    chunk_m = metrikleri_hesapla(siralamalar, sorular, dogru_mu_chunk)

    print("=" * 60)
    print("DENSE RETRIEVAL SONUÇLARI")
    print("=" * 60)
    print(f"{'Metrik':<10} {'madde düzeyi':>14} {'chunk düzeyi':>14}")
    print("-" * 60)
    for k in K_DEGERLERI:
        a = f"hit@{k}"
        print(f"{a:<10} {madde_m[a] * 100:>13.1f}% {chunk_m[a] * 100:>13.1f}%")
    print(f"{'MRR':<10} {madde_m['MRR']:>14.3f} {chunk_m['MRR']:>14.3f}")
    print(f"\n  Ortalama sorgu süresi: {sum(sureler) / n * 1000:.0f} ms "
          f"(cihaz: {cihaz})")

    # Bulunamayanlar: madde düzeyinde bile yakalanamayanlar
    bulunamayanlar = [
        (i, item, sonuclar[:3])
        for i, (item, sonuclar) in enumerate(zip(sorular, siralamalar), 1)
        if not any(dogru_mu_madde(s, item) for s in sonuclar)
    ]
    if bulunamayanlar:
        print(f"\n{'=' * 60}")
        print(f"BULUNAMAYANLAR ({len(bulunamayanlar)} soru)")
        print("=" * 60)
        for i, item, ilk_uc in bulunamayanlar:
            print(f"\n[{i}] {item['soru']}")
            print(f"  beklenen: {item['doc_id']} / "
                  f"{rc.madde_etiketi(item['madde_no'])}")
            print("  gelenler:")
            for s in ilk_uc:
                print(f"    - {s.get('doc_id')} / "
                      f"{rc.madde_etiketi(s.get('madde_no'))} "
                      f"({(s.get('baslik') or '')[:40]})")


if __name__ == "__main__":
    main()
