import chromadb

client = chromadb.PersistentClient(path="data/chroma_db")
col = client.get_collection("mevzuat_strategy_b")
print(col.peek(limit=2)["metadatas"])
