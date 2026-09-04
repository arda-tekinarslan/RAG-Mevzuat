import sys #sistem çi bilgilere erişebilmek için
import time
from pathlib import Path

import chromadb
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer,CrossEncoder

KOK = Path(__file__).parent
DB_DIR = KOK / "data" / "chroma_db"
COLLECTION_NAME = "mevzuat_strategy_b"
EMBED_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LLM_MODEL = "qwen2.5:7b-instruct-q3_K_M"

N_CANDIDATES = 10
TOP_N = 5


def retrieve(soru,collectino,embed_model,cross_encoder):
    q_vec = embed_model.encode(f"query:{soru}",normalize_embeddings=True)
    sonuc = collectino.query(query_embeddings=[q_vec.tolist()],n_results=N_CANDIDATES) #En iyi 10 eşleşme
    metadatalar = sonuc["metadatas"][0]
    documents = sonuc["documents"][0]

    pairs = [(soru,doc) for doc in documents] #Her eşleşme pairini birlikte sokmak için
    skorlar = cross_encoder.predict(pairs)
    sirali = np.argsort(skorlar)[::-1][:TOP_N]

    return([documents[j] for j in sirali],
           [metadatalar[j] for j in sirali],
           [float(skorlar[j]) for j in sirali])

def build_prompt(soru:str,metinler:list[str],metadatalar:list[dict]) -> str:
    parcalar = []

    for i,(metin,meta) in enumerate(zip(metinler,metadatalar),1):
        baslik = meta.get("baslik","")
        madde = meta.get("madde_no","")
        etiket = f"[{i}] {baslik}"
        if madde:
            etiket += f", Madde {madde}"
        parcalar.append(f"{etiket}\n{metin}")
    context = "\n\n---\n\n.".join(parcalar)

    return f"""Sen Türk mevzuatı konusunda uzman bir asistansın. Sana numaralı kaynak metinler ve bir soru veriliyor.

ÖNCE ŞUNU KONTROL ET: Aşağıdaki kaynaklardan en az biri soruyu cevaplıyor mu?

Cevaplamıyorsa — kaynaklar soruyla ilgisizse, konu tamamen farklıysa, ya da
soru mevzuat dışı bir konuysa — SADECE şu cümleyi yaz, başka hiçbir şey ekleme:
"Bu bilgi elimdeki mevzuat metinlerinde yok."

Cevaplıyorsa, şu kurallara uyarak cevapla:
- Kaynaklarda GEÇMEYEN hiçbir sayı, tarih, kurum adı veya hüküm yazma.
- Kendi hukuk bilgini, genel kültürünü veya tahminini KULLANMA.
- Cevabı kısa tut, hükmü sade bir dille açıkla.
- Her cümlenin sonunda hangi kaynaktan geldiğini yaz.

Kaynak gösterme örneği:
"Grup sigortası en az on kişiyle kurulabilir [3]. Bu kişilerin aynı işverene bağlı olması gerekmez [3]."

KAYNAKLAR:
{context}

SORU: {soru}

CEVAP:"""

def generate(prompt:str)->str:
    response = ollama.chat(model=LLM_MODEL,
        messages=[{"role":"user","content":prompt}],
        options={"temperatures":0.2}
        )
    return response["message"]["content"]

def sor(soru,collection,embed_model,cross_encoder,kaynak_goster=True):
    t0 = time.perf_counter()
    metinler,metadatalar,skorlar = retrieve(soru,collection,embed_model,cross_encoder)
    t_retrieval = time.perf_counter() - t0

    t0 = time.perf_counter()
    cevap = generate(build_prompt(soru,metinler,metadatalar))
    t_generation = time.perf_counter() -t0

    print(f"\n{'=' * 70}")
    print(f"SORU: {soru}")
    print("=" * 70)
    print(f"\n{cevap}\n")

    if kaynak_goster:
        print("-" * 70)
        print("KAYNAKLAR:")
        for i, (meta, skor) in enumerate(zip(metadatalar, skorlar), 1):
            baslik = meta.get("baslik", "")[:50]
            madde = meta.get("madde_no", "?")
            print(f"  [{i}] ({skor:6.2f}) {baslik}, Madde {madde}")
 
    print(f"\n  retrieval: {t_retrieval * 1000:.0f}ms · generation: {t_generation * 1000:.0f}ms")
    return cevap

def main():
    print("Sistem yükleniyor...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    cross_encoder = CrossEncoder(RERANK_MODEL)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Hazır. {collection.count()} chunk yüklü.\n")
 
    if len(sys.argv) > 1:
        sor(" ".join(sys.argv[1:]), collection, embed_model, cross_encoder)
        return
 
    test_sorular = [
        "Bir hayvan başkasının taşınmazına zarar verirse zilyet ne yapabilir?",
        "Grup sigortası en az kaç kişiyle kurulabilir?",
        "Yayın lisansı almadan yayın yapanlara hangi ceza uygulanır?",
        "Salatalık İspanyolcada ne demek?",   # negatif test — korpusta yok
    ]
    for soru in test_sorular:
        sor(soru, collection, embed_model, cross_encoder)
 
 
if __name__ == "__main__":
    main()