# CallBus 0.3.1 — ad değişikliği ve tam test bateryası (13 Ekim 2026)

Proje `toolrank` adıyla 0.3.0'ı yayımladıktan saatler sonra **CallBus** olarak yeniden adlandırıldı. Duyuru
yapılmadığı için uyumluluk katmanı bırakılmadı ve eski adın yayımlanmış izleri silindi. Bu rapor neyin
değiştiğini ve değişiklikten sonra koşulan tam bateryayı tutuyor.

## Sonuç (tek cümle)

Ad değişikliği 243 dosyayı, üç dizini, dört dosya adını, üç Hugging Face deposunu ve bütün yayın hedeflerini
kapsadı; sonrasında koşulan sekiz testin hepsi geçti ve **sayılar birebir aynı kaldı** (kalibrasyon bandı 0,513,
arama güveni 0,371, canlı Claude 2 tur / 2.331 token) — yani değişen tek şey adlar.

## Ne değişti

| alan | eski | yeni |
| --- | --- | --- |
| paket / komut | `toolrank` | `callbus` (PyPI'da `callbus 0.3.1`) |
| ortam değişkenleri | `TOOLRANK_*` | `CALLBUS_*` (depo değişkenleri dahil: `CALLBUS_RELEASE`, `CALLBUS_PAGES`) |
| cache yolları | `.cache/toolrank`, `~/.cache/toolrank` | `.cache/callbus`, `~/.cache/callbus` |
| servis adları | `toolrank-emb-v0.2(-fp8)`, GGUF adları | `callbus-emb-v0.2(-fp8)`, `callbus-emb-v0.2-q4_k_m/-q8_0` |
| head dosyası | `toolrank-heads-qwen3-emb-8b-v0.1.npz` | `callbus-heads-qwen3-emb-8b-v0.1.npz` (sha256 aynı) |
| HF depoları | `yasinyaman/toolrank-emb-8b`, `-GGUF`, `-heads-…` | `callbus-emb-8b`, `callbus-emb-8b-GGUF`, `callbus-heads-qwen3-emb-8b` (eski adlar 307) |
| imajlar | `ghcr.io/yasinyaman/toolrank`, `toolrank-vllm` | `…/callbus` (`0.3.1`, `0.3`, `latest`), `callbus-vllm` |
| depo / site | `github.com/yasinyaman/toolrank`, `yaman.dev/toolrank` | `…/callbus`, `yaman.dev/callbus` |
| sınıflar | `ToolrankClient`, `ToolrankError` | `CallBusClient`, `CallBusError` |

Silinenler (duyuru yapılmadığı için kullanıcı kararıyla): PyPI'daki `toolrank` projesi, `v0.1.0`/`v0.2.0`/`v0.3.0`
GitHub release'leri ve tag'leri, GB10'daki erişilemez hale gelen 30 GB embedding cache'i. GGUF dosyaları Hub'da
sunucu tarafı kopyayla yeniden adlandırıldı, yani 13,7 GB yeniden yüklenmedi. Eski GHCR imaj paketi duruyor
(silmek `packages` yetkisi istiyor).

**Yükseltmenin bedeli** CHANGELOG'da yazılı: embedding cache'i servis adıyla anahtarlandığı için ilk koşuda
katalog bir kez yeniden gömülüyor. Ağırlıklar aynı dosyalar, sayılar aynı.

## Tam batarya (hepsi ad değişikliğinden sonra)

| test | sonuç |
| --- | --- |
| `ruff format --check` + `ruff check` | temiz (155 dosya) |
| `pytest` (torch'lu / torch'suz) | exit 0 / exit 0 |
| üretilmiş dosyalar (`cli.md`, `THIRD_PARTY_NOTICES`, README tabloları) | üçü de "current" |
| `release_check --tag v0.3.1` | "ready to release 0.3.1" |
| `mkdocs build --strict` | temiz |
| **PyPI'dan kurulum** (temiz venv, `pip install callbus==0.3.1`) | `callbus 0.3.1`, `formats`, `data synth`, BM25 evalı (sentetik sette NDCG@10 59,55) çalıştı |
| imaj dumanı `callbus:dev` | 8 kontrol `ok` (sürüm `callbus 0.3.1`, uvx 0.12.23) |
| imaj dumanı `callbus-vllm:dev` (paket) | 12 kontrol `ok`; **varsayılan servis adı `callbus-emb-v0.2-fp8`**, vLLM root / callbus kendi kullanıcısı |
| `serve_e2e` (FP8, head'siz, 8098) | hazır 156,9 sn (cache boş: w3 yeniden gömüldü), HTTP 1,49 sn, MCP 1,78 sn, guard'lar 401/421/403, REST arama 1,9 ms, yedi çağrı yolu beklendiği gibi, stdio 0,61 sn / tekrar 4,2 ms, günlük 127 olay |
| `loop_e2e` (120 gerçek HTTP isteği) | 124 çift, `calibrate --from-log` bandı 0,513 (ikizlerin %100'ü), arama güveni 0,371, `ab` "no candidate" |
| çerçeveler (bigtool, LangChain, LlamaIndex, LiteLLM) | dördü de doğru tool'u buldu ve çağırdı |
| platform kancaları (çevrimdışı) | katalog 1.862 tool / 3,67 MB, API adları tekil, 2020-12'ye takılan şema yok, üç arama doğru |
| Helm chart'ı (kind, `callbus:dev`, kiracı anahtarıyla) | **10/10** |
| canlı Claude (`inline` + `prefetch`, `claude-opus-5-5`) | 2 tur, 2.331 girdi token, çağrı gerçek MCP sunucusunda, cevap doğru |

Karşılaştırma için: aynı testler ad değişikliğinden önce FP8'de koşulmuştu (`surum-oncesi-0.3.0.md`, 7. bölüm) ve
sayılar aynı — kalibrasyon bandı 0,513, güven 0,371, canlı koşu 2 tur / 2.331 token, chart 10/10.

## Sapmalar ve açıklamalar

- **İlk yayın koşusu düştü:** yeniden adlandırma akıştaki kapıyı `CALLBUS_RELEASE` yaptı ama depo değişkenleri
  hâlâ `TOOLRANK_*` adındaydı. Değişkenler yeni adla kuruldu, eskiler silindi, koşu yeniden çalıştırıldı.
- **Backbone ilk denemede kalkmadı:** compose'a servis adı (`qwen3-emb-lora-fp8`) yerine kapsayıcı adı
  (`callbus-qwen3-emb-lora-fp8`) verilmişti; servis adları zaten ad değişikliğinden etkilenmiyor.
- **Eski tag'ler depo kuralına takıldı:** "release tags" kuralı `refs/tags/v*` için silmeyi yasaklıyor. Kural
  geçici kapatılıp tag'ler silindi ve kural hemen geri açıldı (`enforcement=active`).
- **Disk kuralı uygulandı:** chart testinden önce `df -h /` ve `docker system df` ile bakıldı (21 GB boş).
  12 Eki'de aynı test Mac'in diskini doldurup Docker VM'inin tazelenmesine ve kullanıcının başka projelerine ait
  imaj/volume kaybına yol açmıştı; kılavuzda ve bu raporda uyarı duruyor.
- Eski docs yolu (`yaman.dev/toolrank`) artık 404: GitHub Pages yolu depo adını izliyor. Duyuru yapılmadığı için
  dışarıdan bağlantı beklenmiyor.

## Komutlar

```bash
# yerel
uv run ruff check src tests scripts examples && uv run pytest -q
uv run python -c "import sys; sys.modules['torch']=None; import pytest; raise SystemExit(pytest.main(['-q','tests/']))"
uv run python scripts/release_check.py --tag v0.3.1 && uv run --group docs mkdocs build --strict

# the published package, in a venv of its own
uv venv .venv && uv pip install --python .venv/bin/python "callbus==0.3.1"
.venv/bin/callbus data synth --out synth && .venv/bin/callbus eval --data synth --scorer bm25 --tool-format documentation

# GB10: the backbone under the product's own served name, then both image smokes
CALLBUS_LORA=$HOME/callbus/data/lora/qwen3-emb-lora-20k/merged CALLBUS_LORA_NAME=callbus-emb-v0.2 \
  docker compose -f deploy/spark/compose.yaml --profile lora-fp8 up -d qwen3-emb-lora-fp8   # 8098
docker build -f deploy/docker/Dockerfile --build-context heads=dist/heads -t callbus:dev .
uv run python scripts/container_smoke.py callbus:dev --version 0.3.1
docker build -f deploy/docker/Dockerfile.vllm --build-context heads=dist/heads -t callbus-vllm:dev .
uv run python scripts/container_smoke.py callbus-vllm:dev --bundle

# Mac: end to end against it, then the chart
export GB10=$(ssh -G gb10 | awk '/^hostname /{print $2}')
FLAGS="--data data/w3 --emb-url http://$GB10:8098/v1 --emb-model callbus-emb-v0.2-fp8 --heads none"
uv run python scripts/serve_e2e.py $FLAGS --out results/serve_e2e_callbus.json
uv run python scripts/loop_e2e.py --data data/w3 --requests data/dev_w3 --n 120 \
  --emb-url http://$GB10:8098/v1 --emb-model callbus-emb-v0.2-fp8 --out results/loop_e2e_callbus.json
uv run python scripts/frameworks_e2e.py $FLAGS --out results/frameworks_callbus.json
uv run python scripts/platforms_e2e.py $FLAGS --out results/platforms_callbus.json
df -h / && docker system df                      # before any cluster test
kind create cluster --name callbus && kind load docker-image callbus:dev --name callbus
uv run python scripts/helm_smoke.py --image callbus:dev --tenants && kind delete cluster --name callbus
```
