import json
import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder

EVAL_PATH =Path("eval_set.jsonl")
DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "mevzuat_strategy_b"
MODEL_NAME = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

N_CANDIDATES= 10
K_DEGERLERI = [1,3,5,10]

def dogru_mu(meta:dict,beklenen:dict)->bool:
    return (meta.get("doc_id") == beklenen.get("doc_id")) and str(meta.get("madde_no")) == str(beklenen.get("madde_no"))


def metrikleri_hesapla(siralamalar:list[list[dict]], sorular:list[dict])->dict:
    hits = {k:0 for k in K_DEGERLERI}
    rr = []

    for metadatalar,item in zip(siralamalar,sorular): #ziplemek her soru için sıralı olan metadata listesini eşleştiriyor
        bulunan_rank = None
        for rank,meta in enumerate(metadatalar):
            if dogru_mu(meta,item):
                bulunan_rank = rank
                break

        if bulunan_rank is None:
            rr.append(0.0)
        else:
            rr.append(1 / (bulunan_rank + 1))
            for k in K_DEGERLERI:
                if bulunan_rank < k:
                    hits[k] += 1
    n = len(sorular)
    sonuc = {f"hit@{k}": hits[k] / n for k in K_DEGERLERI}
    sonuc["MRR"] = sum(rr) / n
    return sonuc

def main():
    sorular = [json.loads(s) for s in open(EVAL_PATH,encoding="utf-8")]
    print(f"{len(sorular)} soru yüklendi")
 
    print("Modeller yükleniyor...")
    embed_model = SentenceTransformer(MODEL_NAME)
    cross_encoder = CrossEncoder(RERANK_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    dense_siralamalar = []
    rerank_siralamalar = []
    dense_sureler = []
    rerank_sureler = []

    for i,item in enumerate(sorular,1):
        t0 = time.perf_counter()
        q_vec = embed_model.encode(f"query: {item['soru']}",normalize_embeddings=True)

        sonuc = collection.query(query_embeddings=[q_vec.tolist()],n_results=N_CANDIDATES)
        dense_sureler.append(time.perf_counter() -t0)

        metadatalar = sonuc["metadatas"][0]
        documents = sonuc["documents"][0]
        dense_siralamalar.append(metadatalar)


        #cross-encoder rerank
        t0 = time.perf_counter()

        pairs = [(item["soru"],doc) for doc in documents]
        skorlar = cross_encoder.predict(pairs)

        sirali_idx = np.argsort(skorlar)[::-1]
        rerank_siralamalar.append([metadatalar[j] for j in sirali_idx])
        rerank_sureler.append(time.perf_counter() - t0)

        print(f"[{i}/{len(sorular)}]",end="\r")

    dense_m = metrikleri_hesapla(dense_siralamalar, sorular)
    rerank_m = metrikleri_hesapla(rerank_siralamalar, sorular)
    
    n = len(sorular)
    print("\n" + "=" * 66)
    print("DENSE vs DENSE + CROSS-ENCODER RERANK")
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
    
    print("-" * 66)
    d_ms = sum(dense_sureler) / n * 1000
    r_ms = sum(rerank_sureler) / n * 1000
    print(f"{'Süre':<10} {d_ms:>11.0f}ms {d_ms + r_ms:>11.0f}ms "
        f"{'+' + f'{r_ms:.0f}':>11}ms")
 
 
if __name__ == "__main__":
    main() 

