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