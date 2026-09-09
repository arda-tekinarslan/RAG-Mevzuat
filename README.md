# Mevzuat RAG

mevzuat.gov.tr'den alınan Türk kanun metinleri üzerinde soru-cevap sistemi.
Dense retrieval + cross-encoder rerank + yerel LLM.

> Hangi dosyanın ne yaptığı ve akıştaki yeri: [`MIMARI.md`](MIMARI.md)
> Mimari kararların ölçülmüş gerekçeleri: [`kararlar.md`](kararlar.md)

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU notu:** `requirements.txt`'teki `torch` PyPI'nin varsayılan **CPU**
> tekerleğidir. Bu ortamda `torch.cuda.is_available()` **False** — GTX 1650 Ti
> kullanılmıyor ve buradaki tüm süreler CPU değeridir (indeksleme ~19 dk).
> GPU için: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

LLM için [Ollama](https://ollama.com) gerekli:

```bash
ollama pull qwen2.5:7b-instruct-q3_K_M
ollama pull qwen2.5:3b-instruct
```

## Pipeline

**Sıra önemli.** Her adım bir öncekinin çıktısını okur.

| # | Script | Girdi | Çıktı |
|---|---|---|---|
| 1 | `pipeline/01_download.py index` | HF mevzuat listesi | `mevzuat_index.jsonl` |
| 1 | `pipeline/01_download.py download` | `mevzuat_index.jsonl` | `data/raw/*.pdf` |
| 2 | `pipeline/02_extract.py` | `data/raw/` | `data/text/`, `data/pages/`, `extract_stats.jsonl` |
| 3 | `pipeline/03_filter.py` | stats + index | `corpus_manifest.jsonl` |
| 4 | `pipeline/04_dedup.py` | manifest | manifest (duplicate işaretli) |
| 5 | `pipeline/05_normalize.py` | `data/pages/` | `data/norm/` |
| 6 | `pipeline/06_chunk_b.py` | `data/norm/` | `data/chunks_strategy_b.jsonl` |
| 7 | `pipeline/07_index_chroma.py` | chunks | `data/chroma_db/` |

> `03_filter.py`, manifest'i sıfırdan yazar ve `04_dedup.py`'nin eklediği
> duplicate işaretlerini siler. 03'ü tekrar koşturursan 04'ü de koştur —
> script bunu tespit edip uyarıyor.

`06_chunk_a.py` (sabit 800 karakter pencere) sadece **karşılaştırma** için
duruyor; indekslenen strateji B'dir.

## Çalıştırma

```bash
streamlit run app.py                          # arayüz
python generate.py "hırsızlığın cezası nedir" # komut satırı
```

## Ölçüm

```bash
python eval/make_eval.py            # aday soru üret (LLM)
python eval/make_eval.py review     # elle onayla -> eval_set.jsonl
python eval/eval_retrieval.py       # sadece dense
python eval/eval_rerank.py          # dense vs dense+rerank -> eval/sonuclar.json
python tools/kalibre_esik.py        # rerank eşiğini kalibre et
```

`eval/sonuclar.json`'u `app.py` sidebar'ı okur — arayüzdeki tablo ölçümle
birlikte güncellenir, elle yazılmaz.

### Debug

```bash
python tools/debug_chunks.py dagilim    # madde_no dağılımı, kısa chunk'lar
python tools/debug_chunks.py kirlilik   # dipnot sızıntısı + çoklu-madde ihlali
python tools/check_meta.py              # Chroma metadata bütünlüğü
python tools/aday_sayisi_tara.py        # N_CANDIDATES taraması (rerank tavanı)
python tools/karsilastir_index.py       # eski (yedek) vs yeni indeks A/B
python tools/migrate_eval_set.py --kontrol  # chunking değişince ground truth taşı
```

> `karsilastir_index.py` ve `migrate_eval_set.py`, `../rag-mevzuat-yedek/`
> altındaki eski chunk/indeks yedeğini okur. Chunking'i değiştirmeden önce
> `data/chroma_db`, `data/chunks_strategy_b.jsonl` ve `eval_set.jsonl`
> yedeklenmezse bu karşılaştırma yapılamaz.

## Mimari kararlar

| Bileşen | Seçim | Gerekçe |
|---|---|---|
| Vektör DB | ChromaDB, PersistentClient, HNSW/cosine | Tek makinede kurulum gerektirmiyor; E5 normalize vektör ürettiği için cosine |
| Embedding | `intfloat/multilingual-e5-small` (384 boyut) | Türkçe destekli, 4GB VRAM'e sığıyor; `passage:`/`query:` öneki şart |
| Rerank | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Çok dilli; hit@1'i belirgin yükseltiyor |
| LLM | Ollama, `qwen2.5:7b-instruct-q3_K_M` | GTX 1650 Ti / 4GB VRAM kısıtı |
| Chunking | Madde sınırı, MAX 1200 / MIN 200 karakter | Bir chunk = en fazla bir madde |

Ayrıntılı karar günlüğü: [`kararlar.md`](kararlar.md)

### Neden `rag_core.py`

Retrieval, prompt ve LLM parametreleri tek modülde. Daha önce `generate.py` ve
`app.py` ayrı kopya taşıyordu ve sessizce ayrışmışlardı — E5 sorgu öneki,
`TOP_N` ve prompt talimatı üç yerde farklıydı. Aynı desenler için
[`desenler.py`](desenler.py), metrikler için `eval/metrikler.py`.

### Chunking sözleşmesi

`06_chunk_b.py` şunları garanti eder:

- Bir chunk **en fazla bir madde başlığı** içerir. Kısa maddeler bir sonrakiyle
  birleştirilmez — 87 karakterlik doğru bir chunk, iki maddeyi karıştıran
  912 karakterlik bir chunk'tan iyidir.
- Hiçbir chunk `MAX_CHUNK`'ı aşmaz (tek çıkış noktası `close_tampon`).
- Uzun maddeler cümle sınırından bölünür, kelime ortasından değil.
- `madde_no` madde **türünü korur**: `"141"`, `"GEÇİCİ 4"`, `"EK 2"`. Bir
  kanunda `Madde 4`, `Ek Madde 4` ve `Geçici Madde 4` aynı anda bulunabiliyor.

### Dipnot temizliği

mevzuat.gov.tr PDF'lerinde dipnotlar sayfa altındadır; pdfplumber sayfayı
yukarıdan aşağı okuduğu için düz metinde gövdenin ortasına düşerler ve
yürürlükten kalkmış ceza miktarları taşırlar.

`02_extract.py` sayfaları ayrı saklar (`data/pages/`), `05_normalize.py` de
şu kuralı uygular: **bir sayfada dipnot başladıysa o sayfanın kalanı
dipnottur.** Satır deseninden dipnotun nerede bittiğini tahmin etmek
denendi ve gövde metnini de siliyordu — yapı kullanmak tahminden güvenli.

Satır içi `(Değişik: ...)` / `(Mülga: ...)` ibareleri **bilinçli olarak
silinmiyor**: "(Mülga)" hükmün yürürlükten kalktığını söyler, atılırsa sistem
kalkmış bir hükmü yürürlükteymiş gibi sunar.

### Yanlış kaynak seçimi — kısmen çözüldü

Uçtan uca testte iki AYRI kök neden ölçüldü:

**1. LLM sıralamayı yok sayıyor** (çözüldü). "Bir hayvan başkasının
taşınmazına zarar verirse" sorusunda retrieval doğru maddeyi 7.93 skorla
1. sıraya koydu, model −2.17 skorlu 2. kaynağı seçip uydurma cevap yazdı.
Prompt'ta "[1] en alakalıdır" yazması yetmedi. Çözüm `rag_core.BASKIN_FARK`:
en iyi kaynak ikinciyi 3.0 puandan fazla geçiyorsa LLM'e **sadece o**
gönderiliyor — alakasız seçenek modele hiç gösterilmiyor.

**2. Reranker temel hüküm yerine nitelikli hâli öne alıyor** (AÇIK).

| Soru | reranker 1. | reranker 2. | doğru olan |
|---|---|---|---|
| "Hırsızlığın cezası nedir?" | TCK 144 (1.68) | TCK 141 (1.48) | **141** |
| "Adam öldürmenin cezası nedir?" | TCK 85 (1.32) | TCK 81 (0.20) | **81** |

Fark 0.20 ve 1.12 — belirsizlik bölgesi, `BASKIN_FARK` burada bilerek
devreye girmiyor (girseydi doğru maddeyi elerdi). Bu bir prompt sorunu
değil: `mmarco-mMiniLMv2` Türkçe hukuk metniyle eğitilmedi ve
"nitelikli hırsızlık ... cezalandırılır" sorgu terimlerini daha çok
tekrarladığı için öne çıkıyor. Çözüm yolları: daha güçlü/alan-uyarlı
reranker, BM25 füzyonu, ya da "soruda niteleyici yoksa düşük madde
numarasını tercih et" tarzı alan kuralı. Hangisinin işe yaradığını
söyleyebilmek için önce **generation eval'i** gerekiyor.

## Bilinen sınırlar

- **Incremental indeksleme yok.** `07_index_chroma.py` koleksiyonu sıfırlıyor.
  Doğru çözüm `upsert` + içerik hash'i; ama `chunk_id` doküman içi sıraya
  bağlı (`_b_0042`), bir maddeye ekleme yapılınca sonraki tüm id'ler kayar.
  Gerçek incremental için id'nin içerik bağlantılı olması gerekir.
- **BM25/hybrid yok.** Türkçe sondan eklemeli olduğu için düz BM25 zayıf kalır
  ("hırsızlığın" ≠ "hırsızlık"); gövdeleme + RRF füzyonu gerekiyor.
- **Generation eval'i yok.** Sadece retrieval ölçülüyor. Prompt artık atıfı
  `(Kanun adı, Madde X)` biçiminde istiyor — cevaptan madde numarasını regex'le
  çekip ground truth ile karşılaştırmak bunu ucuza çözer.
- **Eval seti n=34 ve iki yanlılığı var** (seçim yanlılığı + leakage);
  ayrıntı `eval/make_eval.py` başlığında.

## Lisans / kaynak

Metinler mevzuat.gov.tr'den, FSEK m.31 kapsamında resmî metin.
