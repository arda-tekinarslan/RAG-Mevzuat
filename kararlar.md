# Kararlar

## Korpus
- Kaynak: mevzuat.gov.tr, Kanun (MevzuatTur=1)
- Lisans dayanağı: FSEK m.31 — resmî metinler serbest
- Doküman listesi: HF muhammetakkurt/mevzuat-gov-dataset (907 kayıt, hepsi Kanun)
- PDF kalıbı: mevzuatmetin/{Tur}.{Tertip}.{No}.pdf
- Hedef 500'den 250'ye indirildi; Wikipedia gürültü katmanıyla telafi edilecek

## Teknik engeller
- SSL: site eksik ara sertifika sunuyor -> verify=False (bilinçli, kamu sitesi, telifsiz içerik)
- User-Agent olmadan sunucu yanıt gövdesi göndermiyor, timeout -> tarayıcı UA eklendi
- MevzuatTur=8 PDF yerine 200 ile HTML dönüyor -> %PDF byte kontrolü şart

## Metin çıkarma
- pdfplumber (teşhis için iyi, toplu iş için yavaş; pymupdf 5-10x hızlı)

## Kabul filtresi
- min 3000 karakter, min 200 karakter/sayfa
- 298 aday -> 279 kabul, 19 "çok kısa", 0 taranmış PDF
- Birebir dedup: 0 duplike

## Korpus istatistikleri
- karakter: min 449, medyan 17.337, max 960.576
- karakter/sayfa: min 449, medyan 2.262
- Uzunluk dağılımı çok geniş (max/medyan ≈ 55) -> chunking'de doküman başına
  chunk sayısı dengesizliği beklenmeli

  Normalizasyon: madde/kısım/bölüm/ayırım başlıkları birleştirildi (13705 blok,
1245'i <50 karakter — kısa öksüz parçalar, chunking'de madde-birimi gruplama
ile otomatik çözülüyor, ayrıca normalize edilmedi).

---

## Metin temizliği (revizyon)

Dipnot ayıklama satır deseninden SAYFA yapısına taşındı. Gerekçe: dipnotlar
PDF'te sayfa altındadır, düz metinde gövdenin ortasına düşerler. Satır
deseninden "dipnot nerede bitiyor" tahmin etmek denendi ve gövde metnini de
siliyordu (`"başvurabilir. Bakanlık, bu başvuruyu uygun görürse..."` gibi).
02_extract.py artık sayfaları ayrı saklıyor; kural kesinleşti:
**bir sayfada dipnot başladıysa o sayfanın kalanı dipnottur.**

Dipnot tanıma İKİ koşul birden arıyor (numara + değişiklik sözlüğü). Tek
koşul yetmiyor: köy/mahalle listeleri `"6 Filiz"`, `"7 1.Oruçgazi"` gibi
numaralı satırlar içeriyor ve eski filtre (`^\d+\s+[harf]`) bunları siliyordu.

Deseni genişletmek DENENDİ ve REDDEDİLDİ: 42 satır daha yakalıyor ama 41'i
gerçek gövde metni (`"657 sayılı Devlet Memurları Kanununun 4 üncü
maddesinin..."` gibi madde içi yasal atıflar). Kalan %0.7 kirlilik, gövde
metni silme riskinden iyi.

Kanun sonundaki "değişiklik getiren mevzuat" tablosu kesiliyor (219/298
dokümanda, tek varyant). Satırları `"5378 Ek Madde 1 7/7/2005"` biçiminde
olduğu için chunker onları madde sanıp çöp chunk üretiyor ve metadata'ya
sahte `madde_no` yazıyordu. Tablodan sonra hiçbir dokümanda "İŞLENEMEYEN
HÜKÜM" gelmiyor -> içerik kaybı yok (yedeğe karşı doğrulandı: 0 madde kaybı).

Satır içi `(Değişik: ...)` / `(Mülga: ...)` ibareleri BİLİNÇLİ olarak
korunuyor — "(Mülga)" hükmün yürürlükten kalktığını söyler, silinirse sistem
kalkmış bir hükmü yürürlükteymiş gibi sunar.

| Ölçüm | önce | sonra |
|---|---|---|
| Dipnot/değişiklik metni içeren chunk | %12.9 | %0.7 |
| Birden fazla madde içeren chunk | %11.2 | %0.3 |
| MAX_CHUNK (1200) aşan chunk | 51 | 0 |
| Madde kaybı (yedeğe karşı) | — | 0 |

## Chunking sözleşmesi

Bir chunk EN FAZLA BİR madde başlığı içerir. Kısa maddeler bir sonrakiyle
birleştirilmez. Somut gerekçe: TCK Madde 81 (87 karakter) MIN_CHUNK'ı
dolduramadığı için Madde 82 ile aynı chunk'a giriyordu ve metadata `81`
yazıyordu — LLM'in "adam öldürme" sorusunda yanlış madde alıntılamasının
nedeni buydu. 87 karakterlik doğru bir chunk, iki maddeyi karıştıran 912
karakterlik bir chunk'tan iyidir.

`madde_no` madde TÜRÜNÜ korur (`"141"`, `"GEÇİCİ 4"`, `"EK 2"`). Bir kanunda
`Madde 4`, `Ek Madde 4`, `Geçici Madde 4` aynı anda bulunabiliyor; sadece
numara tutmak korpusta 117 gerçek çarpışma üretiyordu.

## N_CANDIDATES neden 10

Rerank, dense'in havuzunu yeniden sıralar; havuza aday EKLEMEZ. Yani dense
hit@N_CANDIDATES reranker'ın tavanıdır. Ölçüm (34 soru, CPU):

| N | dense@N | rr hit@1 | rr hit@3 | rr hit@5 | rerank süre |
|---|---|---|---|---|---|
| 10 | %97.1 | %85.3 | %91.2 | %94.1 | 786 ms |
| 20 | %97.1 | %85.3 | %91.2 | %94.1 | 1585 ms |
| 30 | %97.1 | %85.3 | %91.2 | %91.2 | 2591 ms |
| 50 | %100 | %85.3 | %91.2 | %91.2 | 4169 ms |

Havuzu büyütmek hit@1/hit@3'ü hiç değiştirmiyor, hit@5'i düşürüyor, maliyet
lineer artıyor. N=50'de tavan %100 — yani her sorunun doğru maddesi dense ile
erişilebiliyor. **Darboğaz dense retrieval değil, cross-encoder'ın ayırt etme
gücü.** İyileştirme çabası oraya yönelmeli (daha güçlü reranker, ya da
BM25 füzyonu).

## Eval metodolojisi

Ground truth `(doc_id, madde_no)` ile eşleşiyor, `chunk_id` ile değil. Bu
"doğru MADDEYİ bulduk mu" sorusunu ölçer — savunulabilir ama GEVŞEK: 34
sorunun 24'ünde ground truth birden fazla chunk ile eşleşiyor. Bu yüzden iki
metrik birden raporlanıyor (madde düzeyi + chunk düzeyi).

Chunking düzeltmeleri sonrası eski/yeni indeks, etiketi DEĞİŞMEYEN 28 soruda
karşılaştırıldı (`tools/karsilastir_index.py`). Farklar 1-2 soru
mertebesinde, n=28'de gürültü sınırının içinde: **chunking düzeltmeleri
retrieval metriklerini ölçülebilir biçimde değiştirmedi.** Kazanç metrikte
değil — yanlış kaynak atfı, eskimiş ceza miktarı sızıntısı ve iki maddeyi
karıştıran chunk'lar ortadan kalktı; bunları mevcut metrikler ölçmüyor.

Eski eval setinin 6/34 etiketi YANLIŞTI (4'ü madde türü, 2'si birleşmiş
madde). Eski %100'ler kısmen bu yanlış etiketlerin üzerine kuruluydu; eski ve
yeni rakamlar doğrudan karşılaştırılamaz.

## Yanlış kaynak seçimi — iki ayrı kök neden

Uçtan uca testte (6 soru) ayrıştırıldı:

**(a) LLM sıralamayı yok sayıyor.** "Hayvan başkasının taşınmazına zarar
verirse" sorusunda retrieval doğru maddeyi (Borçlar 68) 7.93 skorla 1. sıraya
koydu; model −2.17 skorlu TCK 141'i seçip iki maddeyi karıştıran uydurma bir
cevap yazdı. Prompt'ta "[1] en alakalıdır" yazması YETMEDİ.

Çözüm `BASKIN_FARK = 3.0`: en iyi kaynak ikinciyi 3 puandan fazla geçiyorsa
LLM'e sadece o gönderiliyor. 3-bit kuantize 7B modelden sıralama disiplini
beklemek yerine alakasız seçeneği hiç göstermemek. Yan fayda: prompt küçülünce
generation 46.9 sn -> 14.8 sn.

**(b) Reranker temel hüküm yerine nitelikli hâli öne alıyor.** AÇIK SORUN.

| Soru | reranker 1. | reranker 2. | doğru |
|---|---|---|---|
| "Hırsızlığın cezası nedir?" | TCK 144 (1.68) | TCK 141 (1.48) | 141 |
| "Adam öldürmenin cezası nedir?" | TCK 85 (1.32) | TCK 81 (0.20) | 81 |

Fark 0.20 ve 1.12 — belirsizlik bölgesi. BASKIN_FARK burada bilerek devreye
girmiyor; girseydi doğru maddeyi elerdi. Prompt sorunu değil: mmarco reranker
Türkçe hukuk metniyle eğitilmedi. Çözüm adayları: alan-uyarlı reranker, BM25
füzyonu, "niteleyici yoksa düşük madde numarası" kuralı. Hangisinin işe
yaradığını söylemek için önce generation eval'i gerekiyor.

Test sonucu: 6 sorunun 3'ü doğruydu -> 4'ü doğru. Kalan 2 hata (b) sınıfı.

## Donanım

venv'de `torch==2.14.0+cpu` kurulu -> `torch.cuda.is_available()` False.
GTX 1650 Ti KULLANILMIYOR; tüm süreler CPU değeridir. İndeksleme 18-19 dk
(embedding ~1075 sn, Chroma yazma 28 sn — yazma darboğaz değil).

Embedding batch boyutu 256'dan 32'ye indirildi: CPU'da her batch içindeki en
uzun diziye kadar padding'lenir, büyük batch israf demektir.
Ölçüm: 32 -> 20.8 chunk/sn, 64 -> 18.0, 256 -> 14.7.