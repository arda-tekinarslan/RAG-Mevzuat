import json
from pathlib import Path
import statistics

CHUNK_SIZE = 800
OVERLAP = 100
STEP = CHUNK_SIZE - OVERLAP  # 700 karakter kaydırma
KOK = Path(__file__).parent.parent
MANIFEST_PATH = KOK / "corpus_manifest.jsonl"
NORM_DIR = KOK / "data" / "norm"
OUTPUT_PATH = KOK / "data" / "chunks_strategy_a.jsonl"   # a için _a

def chunk_document_fixed(text: str, doc_id: str) -> list[dict]:
    chunks = []
    chunk_idx = 0
    text_len = len(text)
    
    if text_len == 0:
        return []

    # Metin tek chunk'tan kısaysa doğrudan al
    if text_len <= CHUNK_SIZE:
        return [{
            "chunk_id": f"{doc_id}_a_{chunk_idx:04d}",
            "doc_id": doc_id,
            "text": text,
            "char_len": text_len
        }]

    # Sabit kayan pencere (sliding window)
    for start in range(0, text_len, STEP):
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        
        # Son parça 100 karakterin altındaysa ayrı chunk yapma (zaten overlap içinde var)
        if len(chunk_text) < OVERLAP and chunks:
            break
            
        chunks.append({
            "chunk_id": f"{doc_id}_a_{chunk_idx:04d}",
            "doc_id": doc_id,
            "text": chunk_text,
            "char_len": len(chunk_text)
        })
        chunk_idx += 1
        
        if end >= text_len:
            break

    return chunks

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    accepted_docs = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("durum") == "accepted":
                accepted_docs.append(item["doc_id"])

    all_chunks = []
    for doc_id in accepted_docs:
        norm_file = NORM_DIR / f"{doc_id}.txt"
        if not norm_file.exists():
            continue
        text = norm_file.read_text(encoding="utf-8")
        all_chunks.extend(chunk_document_fixed(text, doc_id))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [c["char_len"] for c in all_chunks]
    sorted_lengths = sorted(lengths)

    print(f"Toplam Chunk Sayısı : {len(lengths)}")
    print(f"Min Karakter        : {min(lengths)}")
    print(f"Medyan Karakter     : {statistics.median(lengths)}")
    print(f"P90 Karakter        : {sorted_lengths[int(len(sorted_lengths) * 0.90)]}")
    print(f"Max Karakter        : {max(lengths)}")

if __name__ == "__main__":
    main()