"""
Mevzuat RAG Asistanı — Streamlit arayüzü.

Kurulum:
    pip install -r requirements.txt

Çalıştırma (proje kökünden):
    streamlit run app.py

Retrieval ve prompt mantığı rag_core.py'de — generate.py ile AYNI kodu kullanır.
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import streamlit as st

import rag_core as rc

SONUC_PATH = Path(__file__).parent / "eval" / "sonuclar.json"

st.set_page_config(page_title="Mevzuat RAG Asistanı", page_icon="⚖️", layout="centered")


@st.cache_resource(show_spinner="Sistem yükleniyor (ilk açılışta biraz sürer)...")
def load_system():
    # Embedding ve rerank CPU'da — VRAM'in tamamı LLM'e kalsın (GTX 1650 Ti, 4GB)
    return rc.yukle_sistem(cihaz="cpu")


def basarim_tablosu():
    """
    Eval sonuçlarını eval/sonuclar.json'dan okur.

    Rakamlar eskiden bu dosyaya elle yazılıydı; eval yeniden koşturulunca
    arayüzdeki tablo sessizce eskiyordu. Artık ölçümün kendisi kaynak.
    """
    if not SONUC_PATH.exists():
        return None
    try:
        return json.loads(SONUC_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


embed_model, cross_encoder, collection = load_system()

st.title("⚖️ Mevzuat RAG Asistanı")
st.caption(
    f"{collection.count():,} chunk · madde bazlı chunking · "
    f"dense retrieval + cross-encoder rerank · {rc.LLM_MODEL}"
)

with st.sidebar:
    st.subheader("Sistem")
    st.markdown(f"""
- **Korpus:** mevzuat.gov.tr kanun metinleri
- **Chunking:** madde sınırı (strateji B)
- **Vektör DB:** ChromaDB, HNSW/cosine
- **Embedding:** `{rc.EMBED_MODEL.split('/')[-1]}`
- **Rerank:** cross-encoder (mmarco)
- **LLM:** `{rc.LLM_MODEL}`
""")

    st.subheader("Retrieval başarımı")
    sonuc = basarim_tablosu()
    if sonuc:
        satirlar = ["| Metrik | Dense | + Rerank |", "|---|---|---|"]
        for k in ["hit@1", "hit@3", "hit@5", "hit@10"]:
            if k in sonuc["dense"]:
                satirlar.append(
                    f"| {k} | %{sonuc['dense'][k] * 100:.1f} | "
                    f"**%{sonuc['rerank'][k] * 100:.1f}** |")
        satirlar.append(
            f"| MRR | {sonuc['dense']['MRR']:.3f} | **{sonuc['rerank']['MRR']:.3f}** |")
        st.markdown("\n".join(satirlar))
        st.caption(
            f"{sonuc['soru_sayisi']} soruluk eval seti · "
            f"eşleştirme: {sonuc.get('eslestirme', 'madde düzeyi')} · "
            f"cihaz: {sonuc.get('cihaz', '?')}")
    else:
        st.caption("Ölçüm yok — `python eval/eval_rerank.py` çalıştır.")

soru = st.text_input(
    "Sorunuz:",
    placeholder="Örn: Yayın lisansı almadan yayın yapanlara hangi ceza uygulanır?",
)
sor_button = st.button("Sor", type="primary")

if sor_button and soru.strip():
    with st.spinner("Mevzuat taranıyor..."):
        t0 = time.perf_counter()
        sonuclar = rc.retrieve(soru, collection, embed_model, cross_encoder)
        t_retrieval = time.perf_counter() - t0

    st.subheader("Cevap")

    if rc.yetersiz_mi(sonuclar):
        # Rerank skoru eşiğin altında — LLM'i hiç çağırma
        st.write(rc.YETERSIZ_CEVAP)
        st.caption(f"retrieval {t_retrieval * 1000:.0f} ms · generation atlandı "
                   f"(en iyi skor {sonuclar[0]['skor']:.2f} < eşik {rc.RERANK_ESIK})")
    else:
        t0 = time.perf_counter()
        st.write_stream(rc.generate_stream(rc.build_prompt(soru, sonuclar)))
        t_generation = time.perf_counter() - t0
        st.caption(f"retrieval {t_retrieval * 1000:.0f} ms · "
                   f"generation {t_generation:.1f} sn")

    # LLM'e giden alt küme: en iyi kaynak ikinciyi belirgin geçiyorsa sadece o
    llm_kaynaklari = {id(s) for s in rc.prompt_kaynaklari(sonuclar)}

    with st.expander(f"Bulunan kaynaklar ({len(sonuclar)} madde, "
                     f"{len(llm_kaynaklari)} tanesi LLM'e gitti)"):
        for i, s in enumerate(sonuclar, 1):
            baslik = s.get("baslik") or "—"
            madde = rc.madde_etiketi(s.get("madde_no")) or "—"
            isaret = " ← LLM'e gitti" if id(s) in llm_kaynaklari else ""
            st.markdown(f"**[{i}]** {baslik} — **{madde}**  ·  "
                        f"rerank skoru `{s['skor']:.2f}`{isaret}")
            st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
            st.divider()

elif sor_button:
    st.warning("Lütfen bir soru yazın.")
