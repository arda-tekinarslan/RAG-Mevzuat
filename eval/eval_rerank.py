"""
Dense vs dense+cross-encoder rerank karşılaştırması.

Kullanım: python eval/eval_rerank.py

Sonuçları eval/sonuclar.json'a yazar; app.py sidebar tablosunu oradan okur
(rakamlar eskiden arayüze elle yazılıydı ve ölçüm yenilenince eskiyordu).

METODOLOJİ NOTU — rerank hit@10 hakkında:
  Rerank, dense'in getirdiği AYNI N_CANDIDATES adayı yeniden sıralar; havuza
  yeni aday eklemez. Dolayısıyla rerank hit@N_CANDIDATES, dense
  hit@N_CANDIDATES'e TANIM GEREĞİ eşittir. Dense hit@10 reranker'ın TAVANIDIR.
  Tabloda ikisi de gösteriliyor; eşit çıkması hata değil, beklenen davranıştır.
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
    wilson_alt_sinir,
)

EVAL_PATH = KOK / "eval_set.jsonl"
SONUC_PATH = KOK / "eval" / "sonuclar.json"


def tablo_yaz(baslik, dense_m, rerank_m):
    print("\n" + "=" * 66)
    print(baslik)
    print("=" * 66)
    print(f"{'Metrik':<10} {'Dense':>12} {'Rerank':>12} {'Fark':>12}")
    print("-" * 66)
    for anahtar in [f"hit@{k}" for k in K_DEGERLERI] + ["MRR"]:
        d, r = dense_m[anahtar], rerank_m[anahtar]
        fark = r - d
        isaret = "+" if fark > 0 else ""
        if anahtar == "MRR":
            print(f"{anahtar:<10} {d:>12.3f} {r:>12.3f} {isaret + f'{fark:.3f}':>12}")
        else:
            print(f"{anahtar:<10} {d * 100:>11.1f}% {r * 100:>11.1f}% "
                  f"{isaret + f'{fark * 100:.1f}':>11}%")


def main():
    sorular = [json.loads(s) for s in open(EVAL_PATH, encoding="utf-8")]
    n = len(sorular)
    print(f"{n} soru yüklendi")

    print("Modeller yükleniyor...")
    embed_model, cross_encoder, collection = rc.yukle_sistem()
    cihaz = rc.cihaz_bilgisi(embed_model)
    # Süreler cihaza göre ~100x değişiyor (GPU ~50ms, CPU ~10s). Hangi koşulda
    # ölçüldüğü kayıtta olmazsa rakamlar birkaç hafta sonra yorumlanamaz olur.
    print(f"Cihaz: {cihaz}\n")

    dense_siralamalar, rerank_siralamalar = [], []
    dense_sureler, rerank_sureler = [], []

    for i, item in enumerate(sorular, 1):
        t0 = time.perf_counter()
        adaylar = rc.dense_ara(item["soru"], collection, embed_model)
        dense_sureler.append(time.perf_counter() - t0)
        dense_siralamalar.append(adaylar)

        t0 = time.perf_counter()
        rerank_siralamalar.append(rc.rerank(item["soru"], adaylar, cross_encoder))
        rerank_sureler.append(time.perf_counter() - t0)

        print(f"[{i}/{n}]", end="\r")

    dense_madde = metrikleri_hesapla(dense_siralamalar, sorular, dogru_mu_madde)
    rerank_madde = metrikleri_hesapla(rerank_siralamalar, sorular, dogru_mu_madde)
    dense_chunk = metrikleri_hesapla(dense_siralamalar, sorular, dogru_mu_chunk)
    rerank_chunk = metrikleri_hesapla(rerank_siralamalar, sorular, dogru_mu_chunk)

    tablo_yaz("MADDE DUZEYI (gevsek): doc_id + madde_no eslesmesi",
              dense_madde, rerank_madde)
    tablo_yaz("CHUNK DUZEYI (kati): chunk_id birebir eslesmesi",
              dense_chunk, rerank_chunk)

    print("-" * 66)
    d_ms = sum(dense_sureler) / n * 1000
    r_ms = sum(rerank_sureler) / n * 1000
    print(f"{'Süre':<10} {d_ms:>11.0f}ms {d_ms + r_ms:>11.0f}ms "
          f"{'+' + f'{r_ms:.0f}':>11}ms   (cihaz: {cihaz})")

    # Güven aralığı: "%100" tek başına yanıltıcı, n=34'te alt sınır ~%90.
    print("\n%95 guven araligi alt siniri (Wilson), rerank / madde duzeyi:")
    for k in K_DEGERLERI:
        oran = rerank_madde[f"hit@{k}"]
        alt = wilson_alt_sinir(round(oran * n), n)
        print(f"  hit@{k:<3}: %{oran * 100:5.1f}   (alt sinir %{alt * 100:.1f})")

    SONUC_PATH.write_text(json.dumps({
        "soru_sayisi": n,
        "cihaz": cihaz,
        "eslestirme": "madde düzeyi (doc_id + madde_no)",
        "dense": dense_madde,
        "rerank": rerank_madde,
        "dense_chunk_duzeyi": dense_chunk,
        "rerank_chunk_duzeyi": rerank_chunk,
        "dense_ms": d_ms,
        "rerank_ms": r_ms,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {SONUC_PATH}")


if __name__ == "__main__":
    main()
