"""
Retrieval + prompt kurma çekirdeği. generate.py, app.py ve eval scriptleri
BURADAN import eder.

Neden ortak modül:
  Aynı mantık daha önce generate.py ve app.py içinde ayrı ayrı duruyordu ve
  sessizce ayrışmışlardı:
    - generate.py E5 sorgu önekini "query :" yazıyordu (boşluk yanlış tarafta),
      app.py ve eval scriptleri "query: " yazıyordu. İndeksleme "passage: "
      kullandığı için generate.py'nin sorgu vektörü passage uzayından kayıyordu.
      Aynı eval setinde ölçüldü: hit@10 %100 -> %97.1, hit@5 (rerank) %100 -> %94.1.
      Yani doğru madde aday havuzuna hiç giremiyor, reranker kurtaramıyor.
    - generate.py TOP_N=3, app.py TOP_N=5 kullanıyordu.
    - generate.py'nin prompt'unda "kaynaklar alaka sırasına göre dizilidir"
      paragrafı vardı, app.py'de yoktu.
  Sonuç: hangi kod yolunda ölçüm yaptığın belirsizdi. Artık tek kaynak var.
"""

from pathlib import Path

KOK = Path(__file__).parent
DB_DIR = KOK / "data" / "chroma_db"
COLLECTION_NAME = "mevzuat_strategy_b"

EMBED_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LLM_MODEL = "qwen2.5:7b-instruct-q3_K_M"

# E5 önekleri — TEK KAYNAK. Model bu stringleri birebir bekler; "query :"
# gibi bir varyant sessizce farklı bir vektör üretir, hata vermez.
QUERY_ONEK = "query: "
PASSAGE_ONEK = "passage: "

# Dense'ten çekilen aday sayısı. Rerank bu havuzu YENİDEN SIRALAR, havuza yeni
# aday eklemez — dolayısıyla dense hit@N_CANDIDATES, rerank'in TAVANIDIR.
#
# 10 değeri tahmin değil, ölçüm (tools/aday_sayisi_tara.py, 34 soru, CPU):
#     N    dense@N   rr hit@1  rr hit@3  rr hit@5   rerank süre
#     10    97.1%      85.3%     91.2%     94.1%      786 ms
#     20    97.1%      85.3%     91.2%     94.1%     1585 ms
#     30    97.1%      85.3%     91.2%     91.2%     2591 ms
#     50   100.0%      85.3%     91.2%     91.2%     4169 ms
#
# Havuzu büyütmek hit@1 ve hit@3'ü hiç değiştirmiyor, hit@5'i DÜŞÜRÜYOR
# (fazla aday = cross-encoder'a yanlış chunk'ı öne alma fırsatı), maliyet ise
# lineer artıyor. N=50'de tavan %100 oluyor ama rerank bunu kullanamıyor:
# darboğaz dense retrieval değil, cross-encoder'ın ayırt etme gücü.
N_CANDIDATES = 10

TOP_N = 5           # LLM'e giden kaynak sayısı

# Ollama seçenekleri.
#   temperature=0.0 : bu bir çıkarım/aktarım görevi, yaratıcılık istemiyoruz;
#                     ayrıca eval'in tekrarlanabilir olması için şart.
#   num_ctx=8192    : AÇIKÇA verilmeli. Verilmezse Ollama modelin varsayılanına
#                     düşer (çoğu kurulumda 2048/4096) ve uzun prompt SESSİZCE
#                     kırpılır — kırpılan kısım baştaki TALİMATLAR olur.
#                     TOP_N=5 ile P90 senaryosunda prompt ~2650 token'a çıkıyor.
OLLAMA_SECENEKLERI = {"temperature": 0.0, "num_ctx": 8192}

# Cross-encoder skoru bu eşiğin altındaysa LLM hiç çağrılmaz — reddetme
# davranışı modelin insafına bırakılmaz ve 25-55 sn'lik generation atlanır.
#
# Değer ölçümle belirlendi (tools/kalibre_esik.py, 34 pozitif + 10 negatif soru):
#     en iyi rerank skoru      min     p10   medyan     max
#     pozitif (korpusta var)  2.19    3.32     6.18   10.84
#     negatif (korpusta yok) -6.34   -6.34    -4.01   -1.68
#
# İki dağılım ÖRTÜŞMÜYOR; eşik aradaki boşluğa konuldu. Negatif set sadece
# 10 soru olduğu için eşik bilerek pozitif minimumun (2.19) epey altında —
# yanlış "bilmiyorum" demek, geç cevap vermekten kötüdür.
# Negatif seti büyütünce yeniden kalibre et.
RERANK_ESIK: float | None = 0.25

YETERSIZ_CEVAP = "Bu bilgi elimdeki mevzuat metinlerinde yok."

# En iyi kaynak ikinciyi bu kadar geçiyorsa LLM'e SADECE onu gönder.
#
# Ölçülmüş gerekçe (generate.py test sorulari, rerank skorlari):
#   "Bir hayvan baskasinin tasinmazina zarar verirse..."
#       [1] Turk Borclar Kanunu  m.68   7.93   <- dogru
#       [2] Turk Ceza Kanunu     m.141 -2.17
#   LLM, prompt'ta "[1] en alakalidir" yazmasina ragmen [2]'yi secti ve
#   iki maddeyi birbirine karistiran uydurma bir cevap yazdi. 3-bit
#   kuantize 7B modelden siralama disiplini beklemek yerine, alakasiz
#   secenegi hic gostermemek daha guvenilir.
#
# Esik 3.0 bilinçli olarak YUKSEK: fark kucukken (hirsizlik 141 vs 144:
# 0.20, oldurme 81 vs 85: 1.12) hangi maddenin dogru oldugu gercekten
# belirsizdir; orada tum kaynaklar gonderilmeye devam eder ki dogru madde
# elenmesin. Bu kural yalnizca "acik ara" durumlari kapatir.
BASKIN_FARK = 3.0


def madde_etiketi(madde_no):
    """'141' -> 'Madde 141', 'GEÇİCİ 4' -> 'Geçici Madde 4', 'EK 2' -> 'Ek Madde 2'."""
    if not madde_no:
        return ""
    parcalar = str(madde_no).split(" ", 1)
    if len(parcalar) == 2 and parcalar[0] == "GEÇİCİ":
        return "Geçici Madde " + parcalar[1]
    if len(parcalar) == 2 and parcalar[0] == "EK":
        return "Ek Madde " + parcalar[1]
    return "Madde " + str(madde_no)


def yukle_sistem(cihaz=None):
    """(embed_model, cross_encoder, collection) döndürür."""
    import chromadb
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed_model = SentenceTransformer(EMBED_MODEL, device=cihaz)
    cross_encoder = CrossEncoder(RERANK_MODEL, device=cihaz)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return embed_model, cross_encoder, collection


def cihaz_bilgisi(embed_model):
    """Ölçüm çıktılarına yazmak için: süreler CPU'da mı GPU'da mı ölçüldü."""
    try:
        return str(next(embed_model.parameters()).device)
    except Exception:
        return "bilinmiyor"


def dense_ara(soru, collection, embed_model, n=N_CANDIDATES):
    """Sadece dense retrieval — eval scriptleri rerank'siz ölçüm için kullanır."""
    q_vec = embed_model.encode(QUERY_ONEK + soru, normalize_embeddings=True)
    sonuc = collection.query(query_embeddings=[q_vec.tolist()], n_results=n)
    return [
        {
            "chunk_id": cid,
            "text": doc,
            "doc_id": meta.get("doc_id"),
            "madde_no": meta.get("madde_no"),
            "baslik": meta.get("baslik", ""),
        }
        for cid, doc, meta in zip(
            sonuc["ids"][0], sonuc["documents"][0], sonuc["metadatas"][0]
        )
    ]


def rerank(soru, adaylar, cross_encoder, top_n=None):
    """Cross-encoder ile yeniden sırala, skoru 'skor' alanına yaz."""
    import numpy as np

    if not adaylar:
        return []
    skorlar = cross_encoder.predict([(soru, a["text"]) for a in adaylar])
    sirali = np.argsort(skorlar)[::-1]
    if top_n is not None:
        sirali = sirali[:top_n]
    return [dict(adaylar[j], skor=float(skorlar[j])) for j in sirali]


def retrieve(soru, collection, embed_model, cross_encoder, top_n=TOP_N):
    """
    Uçtan uca retrieval: dense aday + cross-encoder rerank.

    Dönüş: skoruna göre sıralı sözlük listesi — her biri
    {chunk_id, text, doc_id, madde_no, baslik, skor}.
    Eskiden üç paralel liste döndürülüyordu; tek liste indeks kaymasını
    yapısal olarak imkânsız kılıyor.
    """
    adaylar = dense_ara(soru, collection, embed_model)
    return rerank(soru, adaylar, cross_encoder, top_n)


def yetersiz_mi(sonuclar):
    """
    En iyi rerank skoru eşiğin altındaysa True — LLM'i hiç çağırma.

    Reddetme davranışını modelin insafına bırakmak yerine deterministik hale
    getirir ve 25-55 saniyelik generation'ı atlar. RERANK_ESIK None ise kapalı.
    """
    if RERANK_ESIK is None or not sonuclar:
        return False
    return sonuclar[0].get("skor", 0.0) < RERANK_ESIK


def prompt_kaynaklari(sonuclar):
    """
    LLM'e gidecek kaynakları seç. En iyi kaynak ikinciyi BASKIN_FARK kadar
    geçiyorsa yalnızca onu döndürür; aksi halde hepsini.

    Arayüz yine tüm kaynakları gösterir — kullanıcı neyin bulunduğunu görsün.
    Bu sadece LLM'in seçim yapabileceği kümeyi daraltır.
    """
    if len(sonuclar) < 2:
        return sonuclar
    if sonuclar[0].get("skor", 0.0) - sonuclar[1].get("skor", 0.0) >= BASKIN_FARK:
        return sonuclar[:1]
    return sonuclar


def build_prompt(soru, sonuclar):
    sonuclar = prompt_kaynaklari(sonuclar)
    parcalar = []
    for i, s in enumerate(sonuclar, 1):
        etiket = "[" + str(i) + "] " + str(s.get("baslik", ""))
        madde = madde_etiketi(s.get("madde_no"))
        if madde:
            etiket += ", " + madde
        parcalar.append(etiket + "\n" + s["text"])
    context = "\n\n---\n\n".join(parcalar)

    return TALIMAT + "\n\nKAYNAKLAR:\n" + context + "\n\nSORU: " + soru + "\n"


# Prompt talimatı ayrı sabit: iki kod yolunda da AYNI metnin gittiğinden emin
# olmak için. Eskiden app.py ve generate.py farklı talimat taşıyordu.
TALIMAT = """Sen Türk mevzuatı konusunda uzman bir asistansın. Sana numaralı kaynak metinler ve bir soru veriliyor.

ÖNCE ŞUNU KONTROL ET: Aşağıdaki kaynaklardan en az biri soruyu cevaplıyor mu?

Cevaplamıyorsa — kaynaklar soruyla ilgisizse, konu tamamen farklıysa, ya da
soru mevzuat dışı bir konuysa — SADECE şu cümleyi yaz, başka hiçbir şey ekleme:
"Bu bilgi elimdeki mevzuat metinlerinde yok."

Cevaplıyorsa iki adımda ilerle:

ADIM 1 — SEÇİM: Soruyu tam olarak hangi kaynak cevaplıyor? Tek satır yaz,
sadece kaynak numarası ve madde numarası. Örnek:
SEÇİM: [1] Türk Ceza Kanunu, Madde 141

DİKKAT — BENZER HÜKÜMLER: Mevzuatta birbirine çok yakın maddeler bulunur
(ör. "kasten öldürme" ile "taksirle öldürme", "hırsızlık" ile "nitelikli
hırsızlık"). Soruda "taksirle", "nitelikli", "teşebbüs", "ihmal" gibi bir
niteleyici YOKSA, TEMEL hükmü seç — nitelikli veya özel hâlini değil.
Kaynaklar alaka sırasına göre dizilidir; [1] en alakalı olandır.

ADIM 2 — CEVAP: Sadece ADIM 1'de seçtiğin kaynağı kullanarak cevapla.
- Kaynaklarda GEÇMEYEN hiçbir sayı, tarih, kurum adı veya hüküm yazma.
- Kendi hukuk bilgini, genel kültürünü veya tahminini KULLANMA.
- Cevabı kısa tut, hükmü sade bir dille açıkla.
- Her cümlenin sonunda kaynağı (Kanun adı, Madde X) biçiminde yaz.

Örnek cevap:
SEÇİM: [3] Türk Ticaret Kanunu, Madde 1496
CEVAP: Grup sigortası en az on kişiyle kurulabilir (Türk Ticaret Kanunu, Madde 1496). Bu kişilerin aynı işverene bağlı olması gerekmez (Türk Ticaret Kanunu, Madde 1496)."""


def generate(prompt):
    import ollama
    yanit = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=OLLAMA_SECENEKLERI,
    )
    return yanit["message"]["content"]


def generate_stream(prompt):
    """Cevabı parça parça üretir — kullanıcı 30 saniye boş ekrana bakmasın."""
    import ollama
    for parca in ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options=OLLAMA_SECENEKLERI,
        stream=True,
    ):
        yield parca["message"]["content"]
