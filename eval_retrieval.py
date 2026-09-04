import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

EVAL_PATH = Path("eval_set.jsonl")
DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "mevzuat_strategy_b"#her chunk dört şey id (chunk_id), embedding (384 boyutlu vektör), document (chunk metni), metadata (doc_id, madde_no, baslik, char_len)
MODEL_NAME = "intfloat/multilingual-e5-small"

TOP_K = 10

K_DEGERLERI = [1,3,5,10]

def dogru_mu(sonuc_meta: dict, beklenen: dict) -> bool:#sonuç beklenen maddeyle eşleşiyor mu
    return (sonuc_meta.get("doc_id") == beklenen["doc_id"]
            and str(sonuc_meta.get("madde_no")) == str(beklenen["madde_no"]))


def main(): 
    sorular = [json.loads(s) for s in open(EVAL_PATH,encoding="utf-8")]
    print(f"{len(sorular)} soru yüklendi\n")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    hits = {k:0 for k in K_DEGERLERI}
    reciprocal_ranks = []
    bulunamayanlar = []
    sureler = []

    for i,item in enumerate(sorular,1):
        t0 = time.perf_counter()

        q_vec = model.encode(f"query: {item['soru']}",normalize_embeddings=True)

        sonuc = collection.query(query_embeddings=[q_vec.tolist()],n_results=TOP_K) #Soruya göre bize en iyi 10 chunkı getiriyor
        sureler.append(time.perf_counter() - t0)

        metadatalar = sonuc["metadatas"][0] #iç içe liste bu yüzden 0 ile direk listeyi alıyoruz matrisin ilk satırını almak gibi

        bulunan_rank = None
        for rank,meta in enumerate(metadatalar):
            if dogru_mu(meta,item):
                bulunan_rank = rank
                break
        if bulunan_rank is None:
            reciprocal_ranks.append(0.0)
            bulunamayanlar.append((i,item,metadatalar[:3]))
        else:
            reciprocal_ranks.append(1/(bulunan_rank + 1))
            for k in K_DEGERLERI:
                if bulunan_rank < k:
                    hits[k] += 1
    n = len(sorular)
    print("=" * 60)
    print("RETRIEVAL SONUÇLARI")
    print("=" * 60)
    for k in K_DEGERLERI:
        print(f"  hit@{k:<3}: {hits[k]:3}/{n}  (%{hits[k] / n * 100:.1f})")
    print(f"  MRR   : {sum(reciprocal_ranks) / n:.3f}")
    print(f"\n  Ortalama sorgu süresi: {sum(sureler) / n * 1000:.0f} ms")
 
    if bulunamayanlar:
        print(f"\n{'=' * 60}")
        print(f"BULUNAMAYANLAR ({len(bulunamayanlar)} soru)")
        print("=" * 60)
        for i, item, ilk_uc in bulunamayanlar:
            print(f"\n[{i}] {item['soru']}")
            print(f"  beklenen: {item['doc_id']} / madde {item['madde_no']}")
            print(f"  gelenler:")
            for m in ilk_uc:
                print(f"    - {m.get('doc_id')} / madde {m.get('madde_no')} "
                      f"({m.get('baslik', '')[:40]})")
 
 
if __name__ == "__main__":
    main()