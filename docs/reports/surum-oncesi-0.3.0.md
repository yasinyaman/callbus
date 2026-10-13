# 0.3.0 öncesi gerçek dünya testi (13 Ekim 2026)

Sürümden önce sorulan şey: **kullanıcının kuracağı yığın, bizim ölçtüğümüz yığın gibi çalışıyor mu?** Bugüne
kadarki uçtan uca koşular v0.1 head'leriyle ve temel modelle yapılmıştı; 0.3.0'ın varsayılanı head'siz
`toolrank-emb-v0.2`. Altı test o yığınla koşuldu: sunucu (MCP + REST + günlük + soğuk başlangıç), gerçek HTTP
trafiğiyle öğrenme döngüsü, çerçeve entegrasyonları, platform kancaları, **yayımlanmış** imaj ve canlı Claude.

Ortam: Mac'te `toolrank serve` ve istemciler, embedding'ler GB10'un 8097'sinden (`v0.2` bf16, head'siz),
katalog `data/w3` (1.862 tool, 5 kaynak: time, everything, swagger-petstore, stripe, github).

## Sonuç (tek cümle)

Altı test iki yığında da (bf16 ve gönderilen FP8) geçti: sunucu soğuk başlangıçta 134 sn'de hazır oldu ve o süre boyunca anahtar kelimeyle
cevap verdi, gerçek MCP çağrıları çalıştı, 120 isteklik gerçek trafik günlükten 124 çift verdi, `learn`'ün kapısı
öğrenilecek bir şey olmadığını görüp hiçbir şey yayımlamadı, `calibrate --from-log` bandı gerçek trafikten kurdu
ve sonraki aramalar `confidence` taşıdı, dört çerçeve ve iki platform kancası doğru tool'u buldu, yayımlanan
imaj sekiz kontrolden geçti, ve canlı Claude koşusunda `inline` beta'sı ilk kez denenip klasik yola göre bir tur
ve 3,7 MB katalog tasarruf ettiği ölçüldü. Tek açık: Helm chart'ının gerçek kurulumu (Mac'te Docker kapalı) ve
OpenAI tarafının canlı koşusu.

## 1. `toolrank serve`, uçtan uca (`scripts/serve_e2e.py`)

| ölçü | değer |
| --- | ---: |
| HTTP ayağa kalktı | 1,02 sn |
| MCP `tools/list` cevap verdi | 1,32 sn |
| semantik indeks hazır (soğuk, 1.862 tool, Tailscale üzerinden) | 134,4 sn |
| indeks hazırken REST arama (uçtan uca / sunucu içi) | 3,8 / 2,6 ms |
| aynı isteğin tekrarı (stdio, cache'ten) | 3,4 ms |
| gerçek MCP çağrısı (`time/get_current_time`, ilk çağrı sunucuyu başlatıyor) | 5.060 ms |
| stdio `initialize` | 1,05 sn |

- **İndeks kurulurken sunucu boş dönmüyor:** BM25 vekili `mode: lexical` ile cevap verdi, `call_tool` çalıştı,
  `/v1/rank` 503 verdi — tasarlandığı gibi. İkinci başlatma indeksi anlık buldu ("embedded 0, kept 1862").
- **Guard'lar:** token'sız arama 401, yabancı `Host` 421, web sayfası Origin'i 403.
- **Gerçek çağrılar:** Tokyo saati doğru döndü, `everything/echo` çalıştı, petstore yerel vekilden cevap verdi,
  Stripe'ın POST'u reddedildi (`--allow-write` yok).
- **Günlük:** 127 olay (53 MCP arama, 70 MCP çağrı, 1 REST arama, 3 REST çağrı), istek metni yazılmadı
  (`query_text_logged: false`), çağrılar aramalara bağlandı, anahtar dosyası 0o600.

## 2. Döngü, gerçek HTTP trafiğiyle (`scripts/loop_e2e.py`, bu oturumda yazıldı)

Sunucu ayakta tutuldu, `data/dev_w3`'ün 120 isteği `POST /v1/search`'e oynatıldı ve her birinin ilk tool'u
`POST /v1/feedback` ile "istemci çağırdı" diye bildirildi (dışarıya hiçbir çağrı gitmedi). Sonra günlük üstünde
`learn`, `calibrate --from-log` ve `ab`:

| adım | sonuç |
| --- | --- |
| trafik | 120 istek, 120 bildirilen çağrı, 0 boş liste |
| `learn --dry-run` | 733 olay → 125 istek, **124 kullanılabilir çift** (1 zayıf pozitif, 656 negatif), 99 eğitim / 25 dev |
| `learn` (CPU) | epoch 0'da `log.Recall@5` = 1,000; hiçbir epoch başlangıcı geçmedi, **yayımlanmadı** |
| `calibrate --from-log` | 124 istek; en iyi skor p5 0,510, medyan 0,679; %5 bandı ikizlerin %100'ünü yakalıyor |
| `ab --dry-run` | "no candidate" (doğru: `learn` aday yazmadı) |
| arama + band | `time/convert_time` ilk sırada, `confidence` 0,379 döndü |

- **Kapı doğru çalıştı.** Bu sürücünün ajanı her zaman ilk tool'u çağırıyor, yani günlük "sıralama zaten
  doğruydu" diyor; `learn` de buna bakıp hiçbir şey yayımlamadı. Öğrenmenin kazancı bu testin konusu değil
  (onu `scripts/learn_sim.sh` ölçüyor), burada ölçülen şey döngünün gerçek bir sunucuda baştan sona yürümesi.
- **`--from-log`'un süzgeçleri görünür çalışıyor:** 87 arama vektörsüz (eski şema ya da anahtar kelime cevabı),
  81 arama başka modelin, 22 çağrı bağsız — hepsi sayılıp dışarıda bırakıldı.
- Bu, `calibrate --from-log`'un (12 Eki) gerçek bir günlükteki ilk koşusu.

## 3. Çerçeve entegrasyonları (`scripts/frameworks_e2e.py`)

| entegrasyon | sonuç |
| --- | --- |
| langgraph-bigtool | 1.862 tool'luk registry 0,01 sn'de kuruldu, görevde doğru tool'lar getirildi ve çağrıldı |
| LangChain middleware (`ToolrankToolSelector`) | ajanın 1.862 tool'u daraltıldı (0,57 sn), doğru tool gösterildi |
| LlamaIndex (`ToolrankToolRetriever`) | 4 tool önerildi, `time/get_current_time` çağrıldı, 1 arama |
| LiteLLM filtresi | istek başına 120 tool süzüldü, `stripe__GetAccount` kullanıldı; MCP gateway yolu da çalıştı |

Dokuz çağrının yedisi aramasına bağlı (`link: search_id`), ikisi istemci bağıyla.

## 4. Platform kancaları, çevrimdışı (`scripts/platforms_e2e.py`)

Katalog 1.862 tool, API adları tekil (109'u kısaltılıp hash'lenmiş), Anthropic'in göreceği gövde 3,67 MB,
JSON Schema 2020-12'ye takılan şema yok. Üç görevde doğru tool'lar ilk sıralarda (17,9 / 1,1 / 180,6 ms),
petstore çağrısı yerel vekilden `ok`, Stripe'ın POST'u `refused`.

## 5. Yayımlanan imaj (`scripts/container_smoke.py`, GB10/arm64)

`ghcr.io/yasinyaman/toolrank:0.2.0` — kullanıcının gerçekten çektiği imaj, GPU'suz, sahte embedding sunucusuyla:
sürüm `toolrank 0.2.0`, ingest + ısıtma, `/healthz`, localhost'ta arama, yabancı Host 421, MCP `initialize` 200,
`node/npx/uvx` yerinde, scorer hazır. Sekiz kontrolün hepsi `ok`.

## 6. Canlı Claude koşusu (13 Eki, kullanıcının anahtarıyla)

`claude-opus-5-5`, `data/w3` (1.862 tool) servis edilirken, `examples/anthropic_tool_reference.py`. Aynı görev
iki yolla: katalogu `defer_loading` ile gönderen klasik yol ve **ilk kez canlı denenen** `inline` beta'sı
(`Toolbox(inline=True)`, katalog hiç gönderilmiyor, bulunan tool'lar değerle ekleniyor; `--prefetch` görevi ilk
turdan önce arıyor).

| yol | tur | istek başına katalog | girdi token | çıktı token | cevap |
| --- | ---: | ---: | ---: | ---: | --- |
| `inline` + `prefetch` | 2 | 0,0 MB | 2.331 | 132 | doğru |
| klasik (1.862 tool deferred) | 3 | 3,7 MB | 3.140 | 207 | doğru |
| `inline` + `prefetch`, iki tool'lu görev | 3 | 0,0 MB | 2.732 | 239 | doğru |

- **İnline yol bir turu ve kablodaki 3,7 MB'ı birden kaldırıyor.** `prefetch` görevi önceden aradığı için modelin
  kendi `search_tools` turu kalkıyor; girdi token'ı da %26 düşüyor. Faz 1 Hafta 4'ün "istemci tarafı arama bir
  fazladan tur" notu bu yolla kapanıyor.
- **Çağrılar gerçek:** `time/get_current_time` 918 ms (ilk çağrı MCP sunucusunu başlatıyor), ikinci görevde 8,9
  ms ve `everything/echo` 1.570 ms; cevaplar doğru (Tokyo'da 14 Eki 01:42–01:43, JST).
- **İki tool'lu görevde konuşma ortasında tool ekleme (`additions()`) çalıştı:** arama ikisini birden döndürdü,
  model sırayla çağırdı.
- İlk denemede çağrı `refused` döndü, çünkü sunucuyu `--config` olmadan başlatmıştım (MCP sunucularını
  başlatacak yapılandırma yok): ürün hatası değil, kurulum hatası — ama `--config`'siz bir sunucunun yalnız
  arama yapabildiğini canlı olarak da gösteriyor.
- Maliyet: üç koşu toplamda 8.203 girdi + 578 çıktı token'ı.

## 7. Aynı altı test, gönderilen yığınla: FP8 (13 Eki akşamı)

1.–6. bölümler `v0.2`'yi bf16 olarak kullandı; imajların varsayılanı FP8. Yani "gönderdiğimiz yığın test edildi"
demek için hepsi FP8'de tekrarlandı: GB10'da `--profile lora-fp8` (8098, `qwen3-emb-lora-fp8`, ağırlık **7,64
GiB**, bf16'nın 14,11'ine karşı), aynı katalog, aynı komutlar.

| test | bf16 | FP8 |
| --- | --- | --- |
| `serve` hazır (soğuk, 1.862 tool) | 134,4 sn | 154,9 sn |
| HTTP / MCP listesi | 1,02 / 1,32 sn | 0,82 / 1,09 sn |
| REST arama (uçtan uca / sunucu içi) | 3,8 / 2,6 ms | 2,3 / 0,9 ms |
| yedi çağrı yolu (ok, 404, POST reddi, kötü argüman, bilinmeyen tool) | hepsi beklendiği gibi | hepsi beklendiği gibi |
| stdio `initialize` / tekrar arama | 1,05 sn / 3,4 ms | 0,61 sn / 3,2 ms |
| döngü: 120 istek → çift | 124 | 124 |
| günlükten kalibrasyon: %5 bandı | 0,510 | 0,513 |
| sonraki aramanın `confidence`'ı | 0,379 | 0,371 |
| aynı aramanın ilk tool'u / skoru | `time/convert_time` 0,6543 | `time/convert_time` 0,6457 |
| çerçeveler (bigtool, LangChain, LlamaIndex, LiteLLM) | dördü doğru | dördü doğru |
| platform kancaları (katalog, 3 arama, 2 çağrı) | geçti | geçti |
| canlı Claude, inline + prefetch | 2 tur, 2.331 girdi token | 2 tur, 2.331 girdi token, cevap doğru |

- **Kuantizasyon yolu değiştirmiyor.** Skorlar 0,009, güven 0,008 oynuyor; sıralamalar, çağrı sonuçları ve
  canlı cevap aynı. Soğuk başlangıçtaki 20 sn'lik fark Tailscale üzerinden gömmenin oynaması (FP8 sunucu
  tarafında daha hızlı), kalite farkı değil.
- **Günlükteki model süzgeci canlı doğrulandı:** FP8 kalibrasyonu kurulurken aynı günlükteki 265 bf16 araması
  `searches_of_other_models` olarak dışarıda bırakıldı — iki backbone'un vektörleri karışmadı.

## 8. İki imaj yeniden kuruldu (0.2.0'dan beri girdileri değişmiş)

Yayımlanan `0.2.0` imajı iki değişikliği taşımıyor: Dependabot'un uv 0.12.21 → 0.12.23 bump'ı (12 Eki merge) ve
`dev` ekstrasının PEP 735 grubuna taşınması. İkisi de imajın kurulum satırlarına dokunuyor, o yüzden ikisi de
güncel ağaçtan yeniden kuruldu ve duman testine girdi (GB10/arm64, GPU'suz):

- `toolrank:dev` — sekiz kontrol `ok` (sürüm, ingest + ısıtma, sağlık, arama, yabancı Host 421, MCP, node/npx/uvx
  — `uvx 0.12.23` görüldü, scorer).
- `toolrank-vllm:dev` (paket imajı, sahte vLLM içinde) — on iki kontrol `ok`: **servis edilen ad
  `toolrank-emb-v0.2-fp8`**, yani FP8 varsayılanı imajın kendisinde teyit edildi; ayrıca vLLM root, toolrank
  kendi kullanıcısıyla, bind mount sahibini koruyor, `/data` toolrank'e geçiyor.

Yirmi kontrolün hepsi geçti, sıfır hata. Bu, sürüm öncesi paketin asıl keşfiydi: imajların yeniden kurulması
gerektiği ancak bu kontrolle görüldü.

## Açık kalanlar

- **Helm chart'ının gerçek kurulumu** (`scripts/helm_smoke.py`, kind): Mac'te Docker kapalı, GB10'da k3s sudo
  parolası istiyor. 0.2.0'da 10/10 geçmişti ve chart'ta o günden beri yalnız sürüm alanları değişecek; yine de
  sürümden önce bir kez koşulmalı. Docker'ı açman yeterli.
- **OpenAI tarafının canlı koşusu:** `platforms_e2e --live` ve `examples/openai_client_tool_search.py`
  (Faz 1 Hafta 4'te iade görevi TPM sınırına takılmıştı; D2.4'ün kısaltılmış şemalarıyla yeniden koşulacak).
  Claude tarafı 6. bölümde koşuldu; OpenAI anahtarı ve onayı duruyor.
- ~~Bu testler v0.2'yi bf16 olarak kullandı~~: 7. bölümde altısı da FP8'de tekrarlandı.

## Komutlar

```bash
export GB10=$(ssh -G gb10 | awk '/^hostname /{print $2}')   # the address stays out of tracked files
uv run python scripts/serve_e2e.py --data data/w3 --emb-url http://$GB10:8097/v1 \
  --emb-model qwen3-emb-lora --heads none --out results/serve_e2e_release.json
uv run python scripts/loop_e2e.py --data data/w3 --requests data/dev_w3 --n 120 \
  --emb-url http://$GB10:8097/v1 --emb-model qwen3-emb-lora --out results/loop_e2e.json
uv run toolrank learn --data data/w3 --emb-url http://$GB10:8097/v1 --emb-model qwen3-emb-lora \
  --init none --device cpu            # the loop's own learn, with the gate
uv run python scripts/frameworks_e2e.py --data data/w3 --emb-url http://$GB10:8097/v1 \
  --emb-model qwen3-emb-lora --heads none --out results/frameworks_e2e_release.json
uv run python scripts/platforms_e2e.py --data data/w3 --emb-url http://$GB10:8097/v1 \
  --emb-model qwen3-emb-lora --heads none --out results/platforms_e2e_release.json
ssh gb10 'cd ~/toolrank && uv run python scripts/container_smoke.py ghcr.io/yasinyaman/toolrank:0.2.0 --version 0.2.0'

# the live Claude runs: serve with a config that can start the MCP servers, then the example
printf '{"mcpServers":{"time":{"command":"uvx","args":["mcp-server-time"]},"everything":{"command":"npx","args":["-y","@modelcontextprotocol/server-everything"]}}}' > /tmp/w3_config.json
TOOLRANK_API_KEY=$(python3 -c 'import secrets;print(secrets.token_urlsafe(16))') toolrank serve --data data/w3 \
  --config /tmp/w3_config.json --emb-url http://$GB10:8097/v1 --emb-model qwen3-emb-lora --clm-ckpt none --port 8765
uv run --extra anthropic python examples/anthropic_tool_reference.py "What time is it in Tokyo right now?" \
  --inline --prefetch --yes          # the beta; drop --inline --prefetch for the deferred path

# the shipped stack: the same six against FP8 (GB10; the reranker and Ollama stopped to make room)
TOOLRANK_LORA=$HOME/toolrank/data/lora/qwen3-emb-lora-20k/merged \
  docker compose -f deploy/spark/compose.yaml --profile lora-fp8 up -d qwen3-emb-lora-fp8   # 8098
#   then every command above with --emb-url http://$GB10:8098/v1 --emb-model qwen3-emb-lora-fp8
# both images from the current tree, with their smoke tests (data/image_rebuild.sh, untracked)
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t toolrank:dev .
uv run python scripts/container_smoke.py toolrank:dev --version 0.2.0
docker build -f deploy/docker/Dockerfile.vllm --build-context heads=dist/heads -t toolrank-vllm:dev .
uv run python scripts/container_smoke.py toolrank-vllm:dev --bundle
```
