"""
Chunk'ları embedle ve ChromaDB'ye indeksle.

Kullanım: python pipeline/07_index_chroma.py

Model adı, koleksiyon adı ve E5 önekleri rag_core.py'den geliyor — indeksleme
"passage: " kullanırken sorgu tarafının "query: " kullandığından emin olmanın
tek güvenli yolu ikisini aynı yerden okumak.
"""

import json
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

import chromadb                                               # noqa: E402
from sentence_transformers import SentenceTransformer         # noqa: E402

import rag_core as rc                                         # noqa: E402

CHUNKS_PATH = KOK / "data" / "chunks_strategy_b.jsonl"
MANIFEST_PATH = KOK / "corpus_manifest.jsonl"

# Embedding batch boyutu. CPU'da 256 DAHA YAVAŞ: her batch, içindeki en uzun
# diziye kadar padding'lenir, kısa chunk'lar boşuna hesaplanır. Ölçüm (512
# chunk, bu korpus, 6 thread):
#     batch_size= 32 -> 20.8 chunk/sn
#     batch_size= 64 -> 18.0 chunk/sn
#     batch_size=256 -> 14.7 chunk/sn
EMBED_BATCH = 32

# Chroma'ya yazma batch'i. Embedding'den bağımsız — burada padding derdi yok,
# büyük batch daha az SQLite işlemi demek.
YAZMA_BATCH = 1000


def main():
    start_time = time.time()
    rc.DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. Model yükleniyor: {rc.EMBED_MODEL}...")
    model = SentenceTransformer(rc.EMBED_MODEL)
    print(f"   Cihaz: {rc.cihaz_bilgisi(model)}")

    print(f"2. Chunk verisi okunuyor: {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    total_chunks = len(chunks)
    print(f"   Toplam yüklenecek chunk: {total_chunks}")

    # Manifest'ten doc_id -> kanun adı eşlemesi.
    # Retrieval sonucunda "mevzuat_1.5.5216" yerine "Büyükşehir Belediyesi
    # Kanunu" gösterebilmek için — hem arayüzde hem LLM'e giden prompt'ta.
    print(f"   Başlıklar okunuyor: {MANIFEST_PATH}...")
    basliklar = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            basliklar[r["doc_id"]] = r.get("baslik") or ""

    print("3. ChromaDB başlatılıyor...")
    client = chromadb.PersistentClient(path=str(rc.DB_DIR))

    # Varsa eski koleksiyonu silip temiz başla — chunking/metadata değişince
    # eski kayıtlarla karışmasın (ilk projede tam bu yüzden index bozulmuştu)
    try:
        client.delete_collection(name=rc.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=rc.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # E5 normalize vektörler için cosine
    )

    # E5'in beklediği format: dokümanlar "passage: ", sorgular "query: ".
    # Önek SADECE embedding için; saklanan metin öneksiz, yoksa LLM'e
    # giderken gürültü olur.
    texts_to_embed = [rc.PASSAGE_ONEK + item["text"] for item in chunks]

    # TÜM metinler TEK encode() çağrısında veriliyor. sentence-transformers
    # kendi içinde uzunluğa göre sıralayıp batch'liyor; dilim dilim çağırınca
    # bu sıralama sadece dilim içinde çalışıyor ve kısa chunk'lar uzunlarla
    # aynı batch'e düşüp boşuna padding'leniyordu.
    print(f"4. Embedding üretiliyor ({total_chunks} chunk, "
          f"batch={EMBED_BATCH})...")
    t0 = time.time()
    embeddings = model.encode(
        texts_to_embed,
        batch_size=EMBED_BATCH,
        normalize_embeddings=True,   # vektör uzunluğu 1 -> cosine == dot product
        show_progress_bar=True,
    )
    t_embed = time.time() - t0
    print(f"   Embedding bitti: {t_embed:.0f} sn "
          f"({total_chunks / max(t_embed, 1e-9):.1f} chunk/sn)")

    print("5. Chroma'ya yazılıyor...")
    t0 = time.time()
    for i in range(0, total_chunks, YAZMA_BATCH):
        batch = chunks[i:i + YAZMA_BATCH]

        # madde_no: Chroma metadata değeri olarak None kabul etmiyor -> ""
        # Değer "141" olabildiği gibi "GEÇİCİ 4" / "EK 2" de olabilir; madde
        # türü korunuyor çünkü aynı kanunda Madde 4, Ek Madde 4 ve Geçici
        # Madde 4 aynı anda bulunabiliyor.
        collection.add(
            ids=[item["chunk_id"] for item in batch],
            embeddings=embeddings[i:i + YAZMA_BATCH].tolist(),
            documents=[item["text"] for item in batch],
            metadatas=[{
                "doc_id": item["doc_id"],
                "char_len": item["char_len"],
                "madde_no": item.get("madde_no") or "",
                "baslik": basliklar.get(item["doc_id"], ""),
            } for item in batch],
        )  # Bu dört liste Chroma içinde aynı indekslere hizalanır

        progress = min(i + YAZMA_BATCH, total_chunks)
        print(f"   Yazılan: {progress}/{total_chunks} "
              f"(%{progress / total_chunks * 100:.1f})", end="\r")
    print(f"\n   Yazma bitti: {time.time() - t0:.0f} sn")

    elapsed = time.time() - start_time
    print(f"\nİşlem tamamlandı! Geçen süre: {elapsed:.2f} saniye")
    print(f"Koleksiyondaki kayıt sayısı: {collection.count()}")


if __name__ == "__main__":
    main()
