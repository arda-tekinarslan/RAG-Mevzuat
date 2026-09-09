"""
Cross-encoder rerank eşiğini ölçerek belirle.

Kullanım: python tools/kalibre_esik.py

Ne yapar:
  - eval_set.jsonl'deki GERÇEK sorular için en iyi rerank skorunu toplar
  - korpusta karşılığı olmayan NEGATİF sorular için aynısını yapar
  - iki dağılımı ayıran eşiği önerir

Neden gerekli:
  app.py ve generate.py rerank skorunu hesaplayıp atıyordu. Skor düşükken bile
  LLM çağrılıyor ve 25-55 saniye harcanıyor; "bu bilgi elimde yok" demesi de
  tamamen modelin insafına kalıyor. Ölçülmüş bir eşikle bu davranış
  deterministik hale geliyor.

Çıkan değeri rag_core.RERANK_ESIK'e yaz. None bırakılırsa eşik kapalıdır.
"""

import json
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

import rag_core as rc                                          # noqa: E402

EVAL_PATH = KOK / "eval_set.jsonl"

# Korpusta karşılığı olmayan sorular. Mevzuat dışı konular ve mevzuata
# benziyor ama kapsam dışı olanlar karışık.
NEGATIF_SORULAR = [
    "Salatalık İspanyolcada ne demek?",
    "Fotosentez nasıl gerçekleşir?",
    "Python'da liste nasıl sıralanır?",
    "Bugün hava nasıl olacak?",
    "En iyi pizza tarifi nedir?",
    "Ay'a ilk kim ayak bastı?",
    "Basketbolda bir takımda kaç oyuncu sahada olur?",
    "Kahve makinesi nasıl temizlenir?",
    "İstanbul'dan Ankara'ya tren kaç saat sürer?",
    "Bir romanın konusu nasıl özetlenir?",
]


def en_iyi_skorlar(sorular, collection, embed_model, cross_encoder):
    skorlar = []
    for soru in sorular:
        sonuclar = rc.retrieve(soru, collection, embed_model, cross_encoder, top_n=1)
        skorlar.append(sonuclar[0]["skor"] if sonuclar else float("-inf"))
    return sorted(skorlar)


def yuzdelik(sirali, q):
    if not sirali:
        return float("nan")
    i = min(int(q * (len(sirali) - 1)), len(sirali) - 1)
    return sirali[i]


def main():
    pozitif_sorular = [json.loads(s)["soru"] for s in open(EVAL_PATH, encoding="utf-8")]
    print(f"{len(pozitif_sorular)} pozitif, {len(NEGATIF_SORULAR)} negatif soru")

    print("Modeller yükleniyor...")
    embed_model, cross_encoder, collection = rc.yukle_sistem()

    print("Pozitifler ölçülüyor...")
    poz = en_iyi_skorlar(pozitif_sorular, collection, embed_model, cross_encoder)
    print("Negatifler ölçülüyor...")
    neg = en_iyi_skorlar(NEGATIF_SORULAR, collection, embed_model, cross_encoder)

    print("\n" + "=" * 58)
    print("EN IYI RERANK SKORU DAGILIMI")
    print("=" * 58)
    print(f"{'':<10} {'min':>10} {'p10':>10} {'medyan':>10} {'max':>10}")
    for ad, d in (("pozitif", poz), ("negatif", neg)):
        print(f"{ad:<10} {d[0]:>10.2f} {yuzdelik(d, 0.10):>10.2f} "
              f"{yuzdelik(d, 0.50):>10.2f} {d[-1]:>10.2f}")

    # Öneri: negatiflerin en yükseği ile pozitiflerin en düşüğü arasında bir
    # yer. Örtüşme varsa pozitifleri korumaya öncelik ver (recall > precision:
    # yanlış "bilmiyorum" demek, geç cevap vermekten kötü).
    poz_alt, neg_ust = poz[0], neg[-1]
    if poz_alt > neg_ust:
        oneri = (poz_alt + neg_ust) / 2
        print(f"\nÖRTÜŞME YOK — temiz ayrım.")
        print(f"  negatif max = {neg_ust:.2f} < pozitif min = {poz_alt:.2f}")
        print(f"  ÖNERİLEN RERANK_ESIK = {oneri:.2f}")
    else:
        oneri = poz_alt - 0.01
        kaybedilen = sum(1 for s in poz if s < oneri)
        yakalanan = sum(1 for s in neg if s < oneri)
        print(f"\nÖRTÜŞME VAR — pozitif min ({poz_alt:.2f}) <= "
              f"negatif max ({neg_ust:.2f}).")
        print(f"  Pozitifleri korumaya öncelik veren eşik: {oneri:.2f}")
        print(f"  Bu eşikte kaybedilen pozitif: {kaybedilen}/{len(poz)}, "
              f"yakalanan negatif: {yakalanan}/{len(neg)}")

    print(f"\nrag_core.py icinde: RERANK_ESIK = {oneri:.2f}")
    print("(Negatif set sadece 10 soru — eşiği düşük tutmak güvenli tarafta kalmaktır.)")


if __name__ == "__main__":
    main()
