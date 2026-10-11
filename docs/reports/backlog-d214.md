# Backlog D2.14 — aramanın güveni ve katalog başına eşik (11 Ekim 2026–)

Soru: "bu istek için katalogda tool yok" kararı taşınabilir mi? Faz 2 Hafta 5 ham kosinüs eşiğinin taşınmadığını
ölçmüştü (AUROC 0,88 / 0,76, cevaplı isteklerin ortalama en iyi skoru MCP-Zero'da 0,71, LiveMCPBench'te 0,51).
D2.14'ün yolu eşiği kalibre etmek: katalog için yazılmış isteklerin en iyi skorları bir dağılım kurar,
`confidence` bir aramanın kendi en iyi skorunun o dağılımdaki yeri, `--min-confidence Q` de alt dilimi geri
çevirir. Kalibrasyon AUROC'u artırmıyor; yalnız eşiği kataloglar arasında taşınabilir kılıyor.

**Kural (koşulardan önce, `backlog.md`):** eşik %5'lik dilime konur; kabul için MCP-Zero ve LiveMCPBench'in
cevaplı isteklerinde geri çevirme ≤ %7 olmalı. Yakalama (altın sunucusu gizlenmiş "ikiz" istekler) raporlanır,
kabul ölçütü değil.

## Sonuç (tek cümle)

MCP-Zero'da eşik devrini geçti: isteklerin bir yarısında kalibre edilen %5 bandı, hiç görmediği öteki yarıda
%5,3'ünü geri çeviriyor (kural ≤ %7) ve ikizlerin %70,9'unu yakalıyor; gerçek test olan LiveMCPBench (model
yazımı kalibrasyon, insan yazımı 94 görev) GB10'da kuyrukta.

## 1. MCP-Zero: aynı katalog, görülmemiş istekler

`toolrank data split data/mcp_zero_server` ile istekler iki eşit yarıya ayrıldı (seed 0, aynı tool'lar): a
kalibrasyon, b doğrulama. İlk aşama `v0.2` bf16 (8097, head'siz), servis talimatı, `documentation` +
`instruct_query`.

| | n | en iyi skor p5 | medyan | Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| a (kalibrasyon) | 1.396 | 0,576 | 0,723 | %96,8 |
| b (doğrulama) | 1.396 | 0,574 | 0,723 | %96,4 |

a'nın bandı b'nin isteklerine uygulandığında (`scripts/confidence_accept.py`):

| band | b'de geri çevirme | b'nin ikizlerinde yakalama |
| --- | ---: | ---: |
| `--min-confidence 0.01` | %0,4 | %36,3 |
| **`--min-confidence 0.05`** | **%5,3** | **%70,9** |
| `--min-confidence 0.1` | %11,9 | %82,0 |

- **Devir tutuyor.** İki yarının dağılımı neredeyse aynı (p5 0,576'ya karşı 0,574), bu yüzden bir yarıda seçilen
  eşik ötekinde de beklenen payı geri çeviriyor: %5,3, kuralın izin verdiği %7'nin altında.
- **Bu zayıf bir test.** MCP-Zero'nun istekleri de `gen-queries`'in kullandığı modelle (Qwen3-8B) yazıldı, yani
  kalibrasyon ve sınanan istekler aynı kalemden. Planın "gerçek test LiveMCPBench" notu bu yüzden duruyor.
- Yakalama kabul ölçütü değil ama yönü anlamlı: %5'lik band cevabı katalogdan çıkarılmış isteklerin üçte ikisini
  yakalıyor, %1'lik band üçte birini. Ham eşikle (Hafta 5) MCP-Zero'da 0,45 eşiği cevaplıların %0,8'ini geri
  çevirip %23'ünü yakalıyordu; kalibre edilmiş %5 bandı aynı katalogda çok daha iyi bir takas sunuyor.

## 2. LiveMCPBench: gerçek test (kuyrukta)

Kataloğuna 200 istek üretilecek (8093'teki sohbet modeli, `data gen-queries --exclude
data/livemcpbench_server`), kalibrasyon onlardan kurulacak ve band 94 insan yazımı göreve uygulanacak
(`data/d214_chain.sh`, 11 Eki gecesi Q8_0 koşularının ardından). Beklenti belirsiz: LiveMCPBench'in görevleri
daha uzun ve çok adımlı, cevaplı isteklerinin en iyi skorları Hafta 5'te 0,51 ortalamayla MCP-Zero'nun 0,71'inin
çok altındaydı; model yazımı isteklerle kurulan bir dağılımın onları nereye koyacağı ölçülecek.

## Komutlar

```bash
# GB10, ~/toolrank; v0.2 bf16 on 8097, vectors from the embedding cache
FLAGS="--emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --tool-format documentation"
uv run toolrank data split data/mcp_zero_server --out-a data/mz_calib_a --out-b data/mz_calib_b
uv run toolrank calibrate --data data/mcp_zero_server --requests data/mz_calib_a $FLAGS   # --dry-run to look first
uv run python scripts/confidence_accept.py --check data/mz_calib_b --out results/confaccept_mcpzero_b.json -- \
  --data data/mcp_zero_server --scorer dense $FLAGS --query-format instruct_query --with-inst
```

## Sapmalar ve açıklamalar

- `calibrate --dry-run` eşiği yalnız kendisine verilen isteklerin dağılımında raporluyor; başka bir setin
  isteklerini o eşiğe vurmak için `scripts/confidence_accept.py` yazıldı (bu oturumda).
- Bölme `data split`'in seed'iyle, yani tool'lar iki yarıda aynı: ölçülen şey eşiğin istekler arasında taşınması,
  kataloglar arasında değil. Kataloglar arası taşıma D2.15'in (katalog onboarding'i) konusu.
- Kalibrasyon dosyası ilk aşamanın adını, heads sha16'sını ve talimatı anahtar alıyor; biri değişirse eski kayıt
  kullanılmaz. Bu koşuda heads yok (`v0.2` head'siz servis ediliyor).
