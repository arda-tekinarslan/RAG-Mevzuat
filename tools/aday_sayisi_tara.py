"""
N_CANDIDATES taraması + bulunamayan soruların dökümü.

Kullanım: python tools/aday_sayisi_tara.py

Neden:
  Madde bazlı chunking'de maddeler artık birleştirilmiyor, dolayısıyla
  chunk'lar küçüldü (medyan 838 -> 652 karakter). Küçük chunk = chunk başına
  daha az bağlam = dense retrieval'ın kuyruk recall'ü biraz düşüyor.

  Rerank, dense'in getirdiği havuzu yeniden sıralar; havuza yeni aday EKLEMEZ.
  Yani dense hit@N_CANDIDATES, rerank'in TAVANIDIR. Havuzu büyütmek tavanı
  yükseltir — bedeli cross-encoder'ın daha çok çift skorlaması (CPU'da lineer).

  Bu script tavanın nerede olduğunu ve maliyetin ne olduğunu ölçer.
"""

import json
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "eval"))

import rag_core as rc                                   # noqa: E402
from metrikler import dogru_mu_madde, metrikleri_hesapla  # noqa: E402

ADAY_SAYILARI = [10, 20, 30, 50]


def main():
    sorular = [json.loads(s) for s in open(KOK / "eval_set.jsonl", encoding="utf-8")]
    embed_model, cross_encoder, collection = rc.yukle_sistem(cihaz="cpu")

    print(f"{len(sorular)} soru · {collection.count()} chunk · cihaz cpu\n")
    print(f"{'N_CAND':<8} {'dense@N':>9} {'rr hit@1':>10} {'rr hit@3':>10} "
          f"{'rr hit@5':>10} {'rr MRR':>9} {'rerank ms':>11}")
    print("-" * 72)

    en_buyuk = max(ADAY_SAYILARI)
    # En büyük havuzu bir kez çek, alt kümeleri ondan türet — tekrar tekrar
    # sorgulamak yerine. Dense sıralama zaten deterministik.
    havuzlar = [rc.dense_ara(it["soru"], collection, embed_model, n=en_buyuk)
                for it in sorular]

    for n in ADAY_SAYILARI:
        kesit = [h[:n] for h in havuzlar]
        dense_m = metrikleri_hesapla(kesit, sorular, dogru_mu_madde)
        # dense@N: dogru madde havuzun ICINDE mi (rerank'in tavani)
        tavan = sum(1 for h, it in zip(kesit, sorular)
                    if any(dogru_mu_madde(s, it) for s in h)) / len(sorular)

        t0 = time.perf_counter()
        rr = [rc.rerank(it["soru"], h, cross_encoder)
              for it, h in zip(sorular, kesit)]
        ms = (time.perf_counter() - t0) / len(sorular) * 1000
        rr_m = metrikleri_hesapla(rr, sorular, dogru_mu_madde)

        print(f"{n:<8} {tavan * 100:>8.1f}% {rr_m['hit@1'] * 100:>9.1f}% "
              f"{rr_m['hit@3'] * 100:>9.1f}% {rr_m['hit@5'] * 100:>9.1f}% "
              f"{rr_m['MRR']:>9.3f} {ms:>10.0f}ms")
        _ = dense_m

    # Havuza HİÇ giremeyenler: bunlar chunking/embedding sorunu, rerank'in işi değil
    print(f"\n{'=' * 72}")
    print(f"En buyuk havuza (N={en_buyuk}) bile GIREMEYEN sorular")
    print("=" * 72)
    for it, h in zip(sorular, havuzlar):
        if not any(dogru_mu_madde(s, it) for s in h):
            print(f"\n  {it['soru']}")
            print(f"    beklenen: {it['doc_id']} / {rc.madde_etiketi(it['madde_no'])}")
            for s in h[:3]:
                print(f"    gelen   : {s['doc_id']} / "
                      f"{rc.madde_etiketi(s.get('madde_no'))} "
                      f"({(s.get('baslik') or '')[:44]})")


if __name__ == "__main__":
    main()
