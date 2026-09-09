"""
Uçtan uca soru-cevap: retrieval + rerank + LLM.

Kullanım:
    python generate.py                       # gömülü test sorularını çalıştır
    python generate.py "sorunuz burada"      # tek soru

Retrieval ve prompt mantığı rag_core.py'de — app.py ile AYNI kodu kullanır.
Eskiden ikisi ayrı kopya taşıyordu ve sessizce ayrışmışlardı (E5 sorgu öneki,
TOP_N, prompt talimatı). Ayrıntı için rag_core.py başlığına bak.
"""

import sys
import time

import rag_core as rc


def sor(soru, collection, embed_model, cross_encoder, kaynak_goster=True):
    t0 = time.perf_counter()
    sonuclar = rc.retrieve(soru, collection, embed_model, cross_encoder)
    t_retrieval = time.perf_counter() - t0

    print("\n" + "=" * 70)
    print(f"SORU: {soru}")
    print("=" * 70)

    # Rerank skoru eşiğin altındaysa LLM'i hiç çağırma (RERANK_ESIK None ise kapalı)
    if rc.yetersiz_mi(sonuclar):
        print(f"\n{rc.YETERSIZ_CEVAP}\n")
        print(f"  retrieval: {t_retrieval * 1000:.0f}ms · generation atlandı "
              f"(en iyi skor {sonuclar[0]['skor']:.2f} < eşik {rc.RERANK_ESIK})")
        return rc.YETERSIZ_CEVAP

    t0 = time.perf_counter()
    cevap = rc.generate(rc.build_prompt(soru, sonuclar))
    t_generation = time.perf_counter() - t0

    print(f"\n{cevap}\n")

    if kaynak_goster:
        print("-" * 70)
        print("KAYNAKLAR:")
        for i, s in enumerate(sonuclar, 1):
            baslik = (s.get("baslik") or "")[:50]
            print(f"  [{i}] ({s['skor']:6.2f}) {baslik}, "
                  f"{rc.madde_etiketi(s.get('madde_no'))}")

    print(f"\n  retrieval: {t_retrieval * 1000:.0f}ms · "
          f"generation: {t_generation * 1000:.0f}ms")
    return cevap


def main():
    print("Sistem yükleniyor...")
    embed_model, cross_encoder, collection = rc.yukle_sistem()
    print(f"Hazır. {collection.count()} chunk yüklü. "
          f"Cihaz: {rc.cihaz_bilgisi(embed_model)}\n")

    if len(sys.argv) > 1:
        sor(" ".join(sys.argv[1:]), collection, embed_model, cross_encoder)
        return

    test_sorular = [
        "Bir hayvan başkasının taşınmazına zarar verirse zilyet ne yapabilir?",
        "Grup sigortası en az kaç kişiyle kurulabilir?",
        "Yayın lisansı almadan yayın yapanlara hangi ceza uygulanır?",
        "Hırsızlığın cezası nedir?",          # TCK 141 gelmeli, 142 değil
        "Adam öldürmenin cezası nedir?",      # TCK 81 gelmeli, 85 değil
        "Salatalık İspanyolcada ne demek?",   # negatif test — korpusta yok
    ]
    for soru in test_sorular:
        sor(soru, collection, embed_model, cross_encoder)


if __name__ == "__main__":
    main()
