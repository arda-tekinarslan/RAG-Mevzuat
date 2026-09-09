"""
Retrieval metrikleri. eval_retrieval.py ve eval_rerank.py buradan import eder
(eskiden `dogru_mu` iki dosyada kopyaydı).

İKİ DÜZEYDE ÖLÇÜM — neden:
  Ground truth (doc_id, madde_no) ile eşleştiriliyor, chunk_id ile değil.
  Bu savunulabilir bir tercih: kullanıcı için doğru olan doğru MADDEYİ bulmak,
  doğru chunk'ı değil. Ama ölçümü GEVŞETİYOR — uzun bir madde birden fazla
  chunk'a bölündüğünde bunların hepsi doğru sayılıyor ve top-10 içinde aynı
  ground truth'u karşılayan birkaç chunk bulunabiliyor.

  Eski korpusta 34 sorunun 24'ünde (%71) ground truth birden fazla chunk ile
  eşleşiyordu. Bu yüzden İKİ metrik birden raporlanıyor:
    - madde düzeyi (gevşek): doc_id + madde_no eşleşmesi
    - chunk düzeyi (katı)  : chunk_id birebir eşleşmesi
  İkisinin arası, ölçümün ne kadarının bu gevşeklikten geldiğini gösterir.
"""

K_DEGERLERI = [1, 3, 5, 10]


def dogru_mu_madde(sonuc: dict, beklenen: dict) -> bool:
    """Madde düzeyi (gevşek): aynı doküman, aynı madde."""
    return (sonuc.get("doc_id") == beklenen.get("doc_id")
            and str(sonuc.get("madde_no")) == str(beklenen.get("madde_no")))


def dogru_mu_chunk(sonuc: dict, beklenen: dict) -> bool:
    """Chunk düzeyi (katı): birebir aynı chunk."""
    return sonuc.get("chunk_id") == beklenen.get("chunk_id")


def metrikleri_hesapla(siralamalar, sorular, esitlik=dogru_mu_madde) -> dict:
    """
    siralamalar: her soru için sıralı sonuç sözlüğü listesi
    sorular    : eval_set kayıtları (aynı sırada)
    dönüş      : {"hit@1": ..., ..., "MRR": ...}  (hit'ler 0-1 aralığında oran)
    """
    hits = {k: 0 for k in K_DEGERLERI}
    rr = []

    for sonuclar, item in zip(siralamalar, sorular):
        bulunan_rank = None
        for rank, s in enumerate(sonuclar):
            if esitlik(s, item):
                bulunan_rank = rank
                break

        if bulunan_rank is None:
            rr.append(0.0)
        else:
            rr.append(1 / (bulunan_rank + 1))
            for k in K_DEGERLERI:
                if bulunan_rank < k:
                    hits[k] += 1

    n = max(len(sorular), 1)
    sonuc = {f"hit@{k}": hits[k] / n for k in K_DEGERLERI}
    sonuc["MRR"] = sum(rr) / n
    return sonuc


def wilson_alt_sinir(basari: int, n: int, z: float = 1.96) -> float:
    """
    %95 güven aralığının ALT sınırı (Wilson).

    34 soruda 34/34 "hit@5 %100" demek yanıltıcı: gerçek başarı %90 da olabilir.
    Sunumda tek bir parlak rakam yerine alt sınırı da vermek dürüst olan.
    """
    if n == 0:
        return 0.0
    p = basari / n
    payda = 1 + z * z / n
    merkez = p + z * z / (2 * n)
    kok = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (merkez - kok) / payda)
