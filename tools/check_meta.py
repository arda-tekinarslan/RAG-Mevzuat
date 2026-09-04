import chromadb
from pathlib import Path
KOK = Path(__file__).parent.parent
client = chromadb.PersistentClient(path=str(KOK / "data" / "chroma_db"))
col = client.get_collection("mevzuat_strategy_b")
print(col.peek(limit=2)["metadatas"])
