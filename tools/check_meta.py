"""
Chroma koleksiyonunun sağlık kontrolü.

Kullanım: python tools/check_meta.py
"""

import collections
import sys
from pathlib import Path

KOK = Path(__file__).parent.parent
sys.path.insert(0, str(KOK))

import chromadb                                    # noqa: E402

import rag_core as rc                              # noqa: E402

client = chromadb.PersistentClient(path=str(rc.DB_DIR))
col = client.get_collection(rc.COLLECTION_NAME)

print(f"koleksiyon : {rc.COLLECTION_NAME}")
print(f"kayıt      : {col.count()}")

ornek = col.peek(limit=3)
print("\n--- örnek metadata ---")
for cid, meta in zip(ornek["ids"], ornek["metadatas"]):
    print(f"  {cid}")
    print(f"    doc_id={meta.get('doc_id')} madde_no={meta.get('madde_no')!r} "
          f"char_len={meta.get('char_len')}")
    print(f"    baslik={(meta.get('baslik') or '')[:60]}")

# Metadata bütünlüğü: baslik boş kalırsa prompt'ta "[1] , Madde 141" gibi
# kimliksiz etiket oluşur ve LLM kaynakları birbirinden ayırt edemez.
metalar = col.get(include=["metadatas"])["metadatas"]
bos_baslik = sum(1 for m in metalar if not m.get("baslik"))
bos_madde = sum(1 for m in metalar if not m.get("madde_no"))


def tur_adi(m):
    v = str(m.get("madde_no") or "")
    if not v:
        return "YOK"
    return v.split(" ")[0] if " " in v else "NORMAL"


tur = collections.Counter(tur_adi(m) for m in metalar)

print(f"\n--- bütünlük ({len(metalar)} kayıt) ---")
print(f"  başlık boş   : {bos_baslik}   (0 olmalı)")
print(f"  madde_no boş : {bos_madde}")
print(f"  madde türleri: {dict(tur)}")
