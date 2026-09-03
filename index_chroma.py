import json
import time
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks_strategy_b.jsonl")
MANIFEST_PATH = Path("corpus_manifest.jsonl")
DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "mevzuat_strategy_b"
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 256  # RAM ve GPU verimliliği için batch boyutu


def main():
    start_time = time.time()
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"1. Model yükleniyor: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"2. Chunk verisi okunuyor: {CHUNKS_PATH}...")
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line)) #Her line bir json nesnesi bu kod bunu python dicte dönüştürür
    total_chunks = len(chunks)
    print(f"   Toplam yüklenecek chunk: {total_chunks}")

    # Manifest'ten doc_id -> kanun adı eşlemesi.
    # Retrieval sonucunda "mevzuat_1.5.5216" yerine "Büyükşehir Belediyesi Kanunu"
    # gösterebilmek için — hem arayüzde hem LLM'e giden prompt'ta okunaklı olur.
    print(f"   Başlıklar okunuyor: {MANIFEST_PATH}...")
    basliklar = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            basliklar[r["doc_id"]] = r.get("baslik") or ""

    print("3. ChromaDB başlatılıyor...")
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Varsa eski koleksiyonu silip temiz başla — chunking/metadata değişince
    # eski kayıtlarla karışmasın (ilk projede tam bu yüzden index bozulmuştu)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # E5 normalize vektörler için cosine
    )

    print("4. Embedding üretimi ve Chroma'ya ekleme başlıyor...")
    for i in range(0, total_chunks, BATCH_SIZE): #0 dan chunkın toplam bouyutuna kadar 256 lık chunklar olarak göndeririz VRAM e
        batch = chunks[i:i + BATCH_SIZE]

        # E5'in beklediği format: dokümanlar "passage: ", sorgular "query: " sadece embedding ederken
        texts_to_embed = [f"passage: {item['text']}" for item in batch] #json içinde öyle ayrılmışlar objeler
        ids = [item["chunk_id"] for item in batch]
        # Saklanan metin ÖNEKSİZ — önek sadece embedding için, LLM'e giderken gürültü olur
        documents = [item["text"] for item in batch]

        # madde_no: Chroma metadata değeri olarak None kabul etmiyor, "" ile değiştiriyoruz
        metadatas = [{
            "doc_id": item["doc_id"],
            "char_len": item["char_len"],
            "madde_no": item.get("madde_no") or "",
            "baslik": basliklar.get(item["doc_id"], ""),
        } for item in batch]

        # normalize_embeddings=True: vektör uzunluğu 1'e sabitlenir,
        # cosine similarity ile dot product aynı sonucu verir
        embeddings = model.encode(
            texts_to_embed,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ) 

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        ) #Bu dört liste Chromadb içinde aynı indekslere hizalanır

        progress = min(i + BATCH_SIZE, total_chunks)
        print(f"   İşlenen: {progress}/{total_chunks} (%{progress / total_chunks * 100:.1f})", end="\r")

    elapsed = time.time() - start_time
    print(f"\nİşlem tamamlandı! Geçen süre: {elapsed:.2f} saniye")
    print(f"Koleksiyondaki kayıt sayısı: {collection.count()}")


if __name__ == "__main__":
    main()