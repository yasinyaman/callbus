# Backlog D2.14 ve D2.1 — aramanın güveni, katalog başına eşik ve "tool yok" kapısı (11–12 Ekim 2026)

Soru: "bu istek için katalogda tool yok" kararı taşınabilir mi? Faz 2 Hafta 5 ham kosinüs eşiğinin taşınmadığını
ölçmüştü (AUROC 0,88 / 0,76, cevaplı isteklerin ortalama en iyi skoru MCP-Zero'da 0,71, LiveMCPBench'te 0,51).
D2.14'ün yolu eşiği kalibre etmek: katalog için yazılmış isteklerin en iyi skorları bir dağılım kurar,
`confidence` bir aramanın kendi en iyi skorunun o dağılımdaki yeri, `--min-confidence Q` de alt dilimi geri
çevirir. Kalibrasyon AUROC'u artırmıyor; yalnız eşiği kataloglar arasında taşınabilir kılıyor.

**Kural (koşulardan önce, `backlog.md`):** eşik %5'lik dilime konur; kabul için MCP-Zero ve LiveMCPBench'in
cevaplı isteklerinde geri çevirme ≤ %7 olmalı. Yakalama (altın sunucusu gizlenmiş "ikiz" istekler) raporlanır,
kabul ölçütü değil.

## Sonuç (tek cümle)

Kalibrasyon ölçeği kendi istek dağılımı içinde doğru taşıyor (MCP-Zero: bir yarıda seçilen %5 bandı öteki
yarının %5,3'ünü geri çeviriyor, kural ≤ %7) ama dağılımlar arasında taşımıyor: model yazımı isteklerden
kurulan band LiveMCPBench'in insan yazımı 94 görevinin **%54,3'ünü** geri çeviriyor (`--styles situation` ile
%21,3), yani **kabul ölçütü gerçek testte tutmadı** ve `--min-confidence` varsayılansız kalıyor; sıradaki
yol kalibrasyonu kurulumun kendi günlüğünden kurmak (`callbus calibrate --from-log`, bu oturumda yazıldı).
İkinci aşamanın kapıyı kurması da (D2.1) kuralı geçmedi: MCP-Zero'da çok iyi (reranker AUROC 0,976, kosinüs
0,944) ama LiveMCPBench'te hiç (0,760 / 0,772), yani kapı kataloğa bağlı ve `--cut-threshold` varsayılansız
kalıyor.

## 1. MCP-Zero: aynı katalog, görülmemiş istekler

`callbus data split data/mcp_zero_server` ile istekler iki eşit yarıya ayrıldı (seed 0, aynı tool'lar): a
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

## 2. LiveMCPBench: gerçek test — **kural tutmadı**

Kataloğuna üç ayrı istek seti üretildi (8093'teki Qwen3-8B, `--exclude data/livemcpbench_server`), her biriyle
ayrı bir kalibrasyon kuruldu ve bandı 94 insan yazımı göreve uygulandı.

| kalibrasyon | n | kendi p5 / medyanı | 94 gerçek görevde %5 bandının geri çevirdiği | yakalama |
| --- | ---: | ---: | ---: | ---: |
| varsayılan stiller (task, step, goal) | 192 | 0,498 / 0,689 | **%54,3** | %89,4 |
| `--styles situation` | 199 | 0,420 / 0,601 | **%21,3** | %55,3 |
| `--tools-per-request 2` | 132 | 0,487 / 0,640 | **%53,2** | %88,3 |
| *gerçek görevlerin kendisi* | 94 | *0,340 / 0,492* | — | — |

Kural ≤ %7 istiyordu; en iyi kol %21,3'te kaldı, yani **D2.14'ün kabulü LiveMCPBench'te tutmadı.**

- **Sebep stil değil, zorluk farkı.** Üretilmiş istekler bir tool için yazıldığı için o tool'un metniyle sözcük
  sözcük örtüşüyor: medyan en iyi skor 0,689, gerçek görevlerde 0,492. Dağılım 0,2 kayınca o dağılımın %5'lik
  dilimi gerçek trafiğin yarısının üstüne düşüyor. `situation` stili aradaki farkın yarısını kapatıyor (medyan
  0,601) ama yetmiyor; iki tool'lu görevler hiç yardım etmiyor, çünkü onlar da tool'lara bakarak yazılıyor.
- **Katalog içi devir yine de doğru** (1. bölüm): aynı kalemden çıkan isteklerde band %5,3 geri çeviriyor. Yani
  kalibrasyon ölçeği doğru taşıyor, ama yalnız kendi istek dağılımı için.
- **Ürün sonucu:** `--min-confidence` varsayılansız kalıyor (bugün de öyle) ve `docs/guides/serve.md` artık
  "her katalogda cevaplı isteklerin %5'i" demiyor; "kalibrasyonun kurulduğu isteklerin %5'i" diyor, bu ölçümle
  birlikte.
- **Sıradaki öneri:** kalibrasyonu kurulumun kendi trafiğinden kurmak. Kullanım günlüğü istek metnini tutmuyor
  ama vektörünü embedding cache'inden geri getirebiliyoruz (`learn.state_vectors` bunu zaten yapıyor), yani
  `callbus calibrate --from-log` gerçek isteklerin dağılımını ölçebilir. Üretilmiş istekler o zaman yalnız
  soğuk başlangıç için kalır — ve bandın anlamı kurulumun kendi trafiğine bağlanır.

## 3. D2.1: ikinci aşama "tool yok"u daha iyi görüyor mu? — kataloğa bağlı

Aynı ürün sorusunun öteki yarısı (D2.1): karar ham skorla değil, isteği adaylarla birlikte okuyan ikinci aşamayla
verilse ayrım düzelir mi? `scripts/gate_second_stage.py` her istek için iki kısa liste kuruyor — tam katalogdan
ilk 20 (cevaplı) ve altın tool'ların sunucuları çıkarılmış katalogdan ilk 20 (cevapsız) — ikisini de
Qwen3-Reranker-8B'ye veriyor ve en yüksek skorun ayrımını ölçüyor. Listwise bir "hiçbiri" ilk listeden
maskelenemediği için ikinci liste baştan kuruluyor.

| ölçü | MCP-Zero (500 istek) | LiveMCPBench (94 görev) |
| --- | ---: | ---: |
| en iyi kosinüs: cevaplı / cevapsız ortalama | 0,721 / 0,545 | 0,527 / 0,417 |
| **en iyi kosinüs AUROC** | **0,944** | **0,772** |
| reranker: cevaplı / cevapsız ortalama | 0,983 / 0,334 | 0,772 / 0,438 |
| **reranker AUROC** | **0,976** | **0,760** |
| reranker eşiği 0,2: geri çevirme / yakalama | %0,4 / %52,6 | %8,5 / %41,5 |
| reranker eşiği 0,5 | %1,4 / %69,4 | %19,2 / %55,3 |
| reranker eşiği 0,9 | %2,8 / %88,6 | %44,7 / %80,9 |

Kural (koşudan önce): iki sette de AUROC ≥ 0,90, MCP-Zero'da seçilen eşik LiveMCPBench'in cevaplı isteklerinin
≤ %5'ini geri çevirip ≥ %50'sini yakalamalı. **İkisi de tutmadı:** LiveMCPBench'te AUROC 0,760 ve MCP-Zero'da
seçilen 0,2 eşiği orada isteklerin %8,5'ini geri çevirip yalnız %41,5'ini yakalıyor. **`--cut-threshold`
varsayılansız kalıyor.**

Yine de iki şey öğrenildi:

- **Kapı MCP-Zero kalıbında gerçekten çalışıyor.** Kısa, tek tool'luk ajan istekleriyle reranker eşiği
  isteklerin %1,4'ünü geri çevirip cevapsızların %69'unu yakalıyor. Böyle bir katalogda kapı açılabilir —
  ama kararı kurulumun kendi ölçümü vermeli, varsayılan değil.
- **Güçlenen ilk aşama kapıyı da güçlendirdi.** Aynı ayrım v0.1 head'leriyle AUROC 0,882'ydi (Hafta 5),
  `v0.2`'de 0,944. Reranker bunun üstüne 0,03 katıyor ve istek başına bir ikinci aşama çağrısı istiyor:
  kapı için ilk aşama neredeyse yeterli, ikinci aşama yalnız zaten yoldaysa anlamlı.
- **LiveMCPBench'te hiçbir yol açılmıyor.** Çok adımlı görevlerin skorları cevaplıyken de düşük; ne kosinüs
  ne reranker ayırıyor. 2. bölümün bulgusuyla aynı kök: o görevler katalog metnine benzemiyor.

## Komutlar

```bash
# GB10, ~/callbus; v0.2 bf16 on 8097, vectors from the embedding cache
FLAGS="--emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --tool-format documentation"
uv run callbus data split data/mcp_zero_server --out-a data/mz_calib_a --out-b data/mz_calib_b
uv run callbus calibrate --data data/mcp_zero_server --requests data/mz_calib_a $FLAGS   # --dry-run to look first
uv run python scripts/confidence_accept.py --check data/mz_calib_b --out results/confaccept_mcpzero_b.json -- \
  --data data/mcp_zero_server --scorer dense $FLAGS --query-format instruct_query --with-inst

# D2.1, the second stage as the gate (scripts/gate_second_stage.py; 8095 serves Qwen3-Reranker-8B)
uv run python scripts/gate_second_stage.py --rerank-url http://127.0.0.1:8095/v1 --rerank-model qwen3-reranker \
  --depth 20 --max-chars 3000 --rerank-workers 8 --limit 500 --out results/gate_mcpzero.json -- \
  --data data/mcp_zero_server --scorer dense $FLAGS --query-format instruct_query --with-inst
# ... and --data data/livemcpbench_server (94 tasks, no --limit) into results/gate_livemcp.json

# a calibration from the traffic a server answered, instead of generated requests (this session)
uv run callbus calibrate --data data/mytools --from-log [--since 2026-10-01] [--tenant NAME] $FLAGS

# the LiveMCPBench arms (data/d214_chain.sh and data/d214_styles.sh, untracked): the chat model up and down
docker compose -f deploy/spark/compose.yaml --profile gen up -d qwen3-8b-chat   # 8093
uv run callbus data gen-queries --data data/livemcpbench_server --out data/calib_livemcp --n 200 \
  --exclude data/livemcpbench_server --gen-workers 32        # and --styles situation / --tools-per-request 2
docker compose -f deploy/spark/compose.yaml --profile gen stop qwen3-8b-chat
uv run callbus calibrate --data data/livemcpbench_server --requests data/calib_livemcp $FLAGS
uv run python scripts/confidence_accept.py --check data/livemcpbench_server \
  --out results/confaccept_livemcp.json -- --data data/livemcpbench_server --scorer dense $FLAGS \
  --query-format instruct_query --with-inst
```

## Sapmalar ve açıklamalar

- `calibrate --dry-run` eşiği yalnız kendisine verilen isteklerin dağılımında raporluyor; başka bir setin
  isteklerini o eşiğe vurmak için `scripts/confidence_accept.py` yazıldı (bu oturumda).
- Bölme `data split`'in seed'iyle, yani tool'lar iki yarıda aynı: ölçülen şey eşiğin istekler arasında taşınması,
  kataloglar arasında değil. Kataloglar arası taşıma D2.15'in (katalog onboarding'i) konusu.
- Kalibrasyon dosyası ilk aşamanın adını, heads sha16'sını ve talimatı anahtar alıyor; biri değişirse eski kayıt
  kullanılmaz. Bu koşuda heads yok (`v0.2` head'siz servis ediliyor).
