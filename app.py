"""
Mevzuat RAG Asistanı — Streamlit arayüzü.

Kurulum:
    pip install streamlit

Çalıştırma (proje kökünden):
    streamlit run app.py
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import chromadb
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder

KOK = Path(__file__).parent
DB_DIR = KOK / "data" / "chroma_db"
COLLECTION_NAME = "mevzuat_strategy_b"
EMBED_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
LLM_MODEL = "qwen2.5:7b-instruct-q3_K_M"

N_CANDIDATES = 10
TOP_N = 5

st.set_page_config(page_title="Mevzuat RAG Asistanı", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Sistem yükleniyor (ilk açılışta biraz sürer)...")
def load_system():
    # Embedding ve rerank CPU'da — VRAM'in tamamı LLM'e kalsın
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    cross_encoder = CrossEncoder(RERANK_MODEL, device="cpu")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    return embed_model, cross_encoder, collection


def retrieve(soru, collection, embed_model, cross_encoder):
    q_vec = embed_model.encode(f"query: {soru}", normalize_embeddings=True)
    sonuc = collection.query(query_embeddings=[q_vec.tolist()], n_results=N_CANDIDATES)
    metadatalar = sonuc["metadatas"][0]
    documents = sonuc["documents"][0]

    pairs = [(soru, doc) for doc in documents]
    skorlar = cross_encoder.predict(pairs)
    sirali = np.argsort(skorlar)[::-1][:TOP_N]

    return ([documents[j] for j in sirali],
            [metadatalar[j] for j in sirali],
            [float(skorlar[j]) for j in sirali])


def build_prompt(soru, metinler, metadatalar):
    parcalar = []
    for i, (metin, meta) in enumerate(zip(metinler, metadatalar), 1):
        etiket = f"[{i}] {meta.get('baslik', '')}"
        if meta.get("madde_no"):
            etiket += f", Madde {meta['madde_no']}"
        parcalar.append(f"{etiket}\n{metin}")
    context = "\n\n---\n\n".join(parcalar)

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


def generate_stream(prompt):
    """Cevabı parça parça üretir — kullanıcı 30 saniye boş ekrana bakmasın."""
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
        stream=True,
    )
    for parca in stream:
        yield parca["message"]["content"]


# ---------------------------------------------------------------- arayüz
embed_model, cross_encoder, collection = load_system()

st.title("⚖️ Mevzuat RAG Asistanı")
st.caption(
    f"{collection.count():,} chunk · madde bazlı chunking · "
    f"dense retrieval + cross-encoder rerank · {LLM_MODEL}"
)

with st.sidebar:
    st.subheader("Sistem")
    st.markdown(f"""
- **Korpus:** mevzuat.gov.tr kanun metinleri
- **Chunking:** madde sınırı (strateji B)
- **Vektör DB:** ChromaDB, HNSW/cosine
- **Embedding:** `multilingual-e5-small`
- **Rerank:** cross-encoder (mmarco)
- **LLM:** `{LLM_MODEL}`
""")
    st.subheader("Retrieval başarımı")
    st.markdown("""
| Metrik | Dense | + Rerank |
|---|---|---|
| hit@1 | %64.7 | **%82.4** |
| hit@3 | %88.2 | **%97.1** |
| hit@5 | %91.2 | **%100** |
| MRR | 0.779 | **0.898** |
""")
    st.caption("34 soruluk eval seti üzerinde ölçüldü.")

soru = st.text_input(
    "Sorunuz:",
    placeholder="Örn: Yayın lisansı almadan yayın yapanlara hangi ceza uygulanır?",
)
sor_button = st.button("Sor", type="primary")

if sor_button and soru.strip():
    with st.spinner("Mevzuat taranıyor..."):
        t0 = time.perf_counter()
        metinler, metadatalar, skorlar = retrieve(soru, collection, embed_model, cross_encoder)
        t_retrieval = time.perf_counter() - t0

    st.subheader("Cevap")
    t0 = time.perf_counter()
    st.write_stream(generate_stream(build_prompt(soru, metinler, metadatalar)))
    t_generation = time.perf_counter() - t0

    st.caption(f"retrieval {t_retrieval * 1000:.0f} ms · generation {t_generation:.1f} sn")

    with st.expander(f"Kullanılan kaynaklar ({TOP_N} madde)"):
        for i, (meta, skor, metin) in enumerate(zip(metadatalar, skorlar, metinler), 1):
            baslik = meta.get("baslik", "—")
            madde = meta.get("madde_no", "—")
            st.markdown(f"**[{i}]** {baslik} — **Madde {madde}**  ·  rerank skoru `{skor:.2f}`")
            st.text(metin[:600] + ("..." if len(metin) > 600 else ""))
            st.divider()

elif sor_button:
    st.warning("Lütfen bir soru yazın.")