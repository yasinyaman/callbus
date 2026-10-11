# Ölçüm tabanı — varsayılan backbone'da (11 Ekim 2026)

GB10 sırası 4. Bugüne kadarki bütün `learn` / `ab` sonuçları, adaptive K marjı, hybrid ve sunucu oyu temel model
(`qwen3-emb`) ve v0.1 head'leri üzerinde ölçülmüştü; varsayılan backbone 2 Ekim'de head'siz `v0.2` oldu. Soru:
bu dört karar `v0.2`'de de aynı mı?

## Sonuç (tek cümle)

Öğrenme döngüsü `v0.2`'de de kazandırıyor ama daha az (ayrılmış isteklerde +1,88 ve +1,55; v0.1'de +3,03), yani
önceden yazılan kural tuttu ve `learn` gecelik komutta kalıyor; adaptive K marjı 0,2'de kalıyor (hiçbir marj
kuralı geçmedi); hybrid artık her sette kaybettiriyor (v0.1'de yalnız MCP-Zero'da kazanıyordu) ve sunucu oyu
berabere düştü (v0.1'de MCP-Zero top-1'e +1,1) — ikisi de isteğe bağlı kalıyor, belgeleri düzeltildi.

## Ölçüler, birimler ve yön

`NDCG@10 ↑ %`, `Recall@10 ↑ %`, `cat-macro ↑ %` README'deki gibi, talimatlı (w/ inst), `documentation` +
`instruct_query`. `ayrılmış`: `learn_sim`'in hiç servis etmediği %30 (trafik %70). `yeni`: ayrılmış sorgunun
altın tool'larının hiçbiri trafikte çağrılmamış. `K ↓`: adaptive K'nın gösterdiği tool sayısı,
`Recall@cut ↑ %` / `Comp@cut ↑ %` o listenin bulduğu altın tool payı ve tamlığı. Başlangıç = `v0.2` tek başına
(head'siz, dense); öğrenilen = günlükten eğitilen head'ler (kimlikten başlayarak, `--init none`).

## 1. Öğrenme döngüsü (`learn`) — kural tuttu

`scripts/learn_sim.sh`, `HEADS=none`, trafiğin tamamı (`SIZES=0`), head eğitimi CPU'da. Bölmeler Hafta 4'ün
bölmeleriyle aynı (aynı seed): ToolRet 5.573 trafik / 2.388 ayrılmış / 1.061 yeni, MCP-Zero 1.954 / 838 / 829.
Kural (`karar-modelleri.md`, koşudan önce): ayrılmış isteklerde başlangıcı ≥ 1 puan geçmezse `learn` `v0.2` için
gecelik komuttan çıkar ve D2.15'in head kolu düşer.

| katalog | set | n | başlangıç | öğrenilen | fark |
| --- | --- | ---: | ---: | ---: | ---: |
| ToolRet | ayrılmış | 2.388 | 59,61 | **61,49** | **+1,88** |
| ToolRet | ayrılmış, cat-macro | 3 kat. | 54,90 | 55,90 | +1,00 |
| ToolRet | yeni | 1.061 | 69,12 | 70,16 | +1,04 |
| ToolRet | yeni, cat-macro | 3 kat. | 60,63 | 59,93 | −0,70 |
| ToolRet | LiveMCPBench (unutma) | 94 | 55,74 | 54,28 | −1,46 |
| ToolRet | MCP-Zero (unutma) | 2.792 | 93,67 | 93,68 | +0,01 |
| MCP-Zero | ayrılmış | 838 | 94,28 | **95,83** | **+1,55** |
| MCP-Zero | yeni | 829 | 94,32 | 95,90 | +1,58 |
| MCP-Zero | LiveMCPBench (unutma) | 94 | 55,74 | 58,72 | +2,98 |
| MCP-Zero | ToolRet (unutma) | 7.961 | 58,90 | 56,42 | −2,48 |
| MCP-Zero | ToolRet, cat-macro | 35 görev | 54,36 | 52,74 | −1,62 |

Kural iki katalogda da geçildi: **`learn` `v0.2` için gecelik komutta kalıyor, D2.15'in head kolu duruyor.**
Üç şey v0.1'den farklı:

- **Kazanç küçüldü.** v0.1'de aynı ToolRet bölmesinde 54,20 → 57,23 (+3,03) idi; `v0.2` zaten 59,61'den
  başlıyor ve +1,88 alıyor. Güçlenen ilk aşama öğrenmeye daha az yer bırakıyor.
- **Yeni tool'lara giden istekler de kazanıyor.** v0.1'de hiç oynamıyordu (63,67 → 63,83); burada ToolRet'te
  +1,04 (ama kategori ortalaması −0,70) ve MCP-Zero'da +1,58. MCP-Zero'da her ayrılmış istek yeni bir tool için
  yazıldığı için bu beklenen yön.
- **Yan etki büyüdü.** v0.1'de trafiğin kapsamadığı setler ya duruyor ya kazanıyordu (LiveMCPBench +2,1,
  MCP-Zero +0,4, ToolRet −0,5). `v0.2`'de yön trafiğe bağlı: ToolRet trafiğiyle LiveMCPBench −1,46;
  MCP-Zero trafiğiyle ToolRet −2,48 ama LiveMCPBench +2,98. Bu tam olarak `--replay`'in (genel çiftleri
  batch'lere karıştırmak) yapmak için yazıldığı iş; sıradaki ölçüm o.

`learn`'ün kendi kapısı (dev seti olarak guard benchmark, `--max-drop`) iki koşuda da tutuyordu: ToolRet
koşusunda MCP-Zero `dev.NDCG@10` 93,67 → 93,68, MCP-Zero koşusunda LiveMCPBench 55,74 → 58,72. Yani kapı,
yukarıdaki yan etkiyi görmüyor: guard hangi setse o korunuyor, öteki korunmuyor.

## 2. Adaptive K marjı — 0,2'de kalıyor

`scripts/adaptive_k_sweep.py`, aynı sıralama üstünde dört marj. Kural: marj yalnız aynı ya da daha düşük K'de
tamlık ≥ 1 puan artarsa değişir.

| set | kesim | K ↓ | Recall@cut | Comp@cut |
| --- | --- | ---: | ---: | ---: |
| ToolRet | sabit top-10 | 10,00 | 69,54 | 60,04 |
| ToolRet | marj 0,15 | 7,32 | 67,87 | 57,38 |
| ToolRet | **marj 0,2 (varsayılan)** | 8,33 | 69,06 | 59,28 |
| ToolRet | marj 0,3 | 9,50 | 69,51 | 60,01 |
| LiveMCPBench | marj 0,2 | 7,71 | 60,44 | 36,17 |
| LiveMCPBench | marj 0,3 | 9,03 | 63,07 | 39,36 |
| MCP-Zero | marj 0,2 | 5,51 | 97,78 | 97,67 |
| MCP-Zero | marj 0,3 | 8,66 | 97,90 | 97,85 |

0,3 ToolRet'te tamlığa +0,73 katıyor ama K'yı 8,33'ten 9,50'ye çıkarıyor: kural geçilmedi, **varsayılan 0,2
kalıyor.** Marj hâlâ işini yapıyor — MCP-Zero'da sabit top-10'un tamlığından 0,22 puan geride, gösterilen tool
sayısı neredeyse yarısı (5,51'e karşı 10).

## 3. Hybrid (RRF) — artık her sette kaybettiriyor

| set | `v0.2` | + hybrid | fark |
| --- | ---: | ---: | ---: |
| ToolRet NDCG@10 | 58,90 | 47,88 | −11,02 |
| ToolRet cat-macro | 54,36 | 43,60 | −10,76 |
| LiveMCPBench NDCG@10 | 55,74 | 41,26 | −14,48 |
| MCP-Zero NDCG@10 | 93,67 | 93,07 | −0,60 |
| MCP-Zero top-1 | 88,57 | 87,21 | −1,36 |

v0.1 head'lerinde hybrid'in tek kazandığı set MCP-Zero'ydu (79,87 → 84,31, agent'ın yazdığı iki satırlık
istekler BM25'e yarıyordu). `v0.2` o isteklerde zaten 88,57 top-1 yapıyor ve karışıma giren BM25 kolu artık
sıralamayı bozuyor. CLAUDE.md'deki cümle buna göre düzeltildi; hybrid opt-in kalıyor (kimse için varsayılan
değildi).

## 4. Sunucu oyu (`--server-weight 0.2`) — berabere

| set | `v0.2` | + srv 0,2 | fark |
| --- | ---: | ---: | ---: |
| ToolRet NDCG@10 / cat-macro | 58,90 / 54,36 | 58,92 / 54,15 | +0,02 / −0,21 |
| LiveMCPBench NDCG@10 / cat-macro | 55,74 / 55,30 | 55,92 / 55,78 | +0,18 / +0,48 |
| MCP-Zero NDCG@10 / top-1 | 93,67 / 88,57 | 93,77 / 88,57 | +0,10 / 0,00 |

v0.1'de MCP-Zero top-1'e +1,11 katıyordu; `v0.2`'de hiçbir şey katmıyor. Sunucu özeti kataloğun hangi
sunucusunun konu olduğunu söylüyor, `v0.2` bunu tool metninden zaten çıkarıyor. İsteğe bağlı kalıyor,
`docs/guides/serve.md` iki yığının sayılarını da yazıyor.

## Komutlar

```bash
# GB10, ~/toolrank (data/v02_loop_chain.sh, untracked; vectors all cached, 6 minutes end to end)
export HEADS=none EMB_URL=http://127.0.0.1:8097/v1 EMB_MODEL=qwen3-emb-lora UV=~/.local/bin/uv
NAME=toolret_v02 TAG=v02 SIZES=0 LEARN="--device cpu" bash scripts/learn_sim.sh
BENCH=data/mcp_zero_server NAME=mcpzero_v02 TAG=v02 GUARD=data/livemcpbench_server \
  EVALS="data/toolret data/livemcpbench_server" SIZES=0 LEARN="--device cpu" bash scripts/learn_sim.sh
V02="--emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --emb-batch 128
  --tool-format documentation --query-format instruct_query --with-inst"
for d in toolret livemcpbench_server mcp_zero_server; do
  $UV run python scripts/adaptive_k_sweep.py --margins 0.1,0.15,0.2,0.3 -- --data data/$d --scorer dense $V02
  $UV run toolrank eval --data data/$d --scorer dense $V02 --hybrid --out results/v02base_${d}_hybrid.json
  $UV run toolrank eval --data data/$d --scorer dense $V02 --server-weight 0.2 --out results/v02base_${d}_srv.json
done   # mcp_zero_server also --ks 1,5,10,20
# Mac, after scp 'gb10:toolrank/results/{sim_*_v02_*,v02base_*}.json' results/:
uv run toolrank compare results/sim_toolret_v02_base_heldout.json results/sim_toolret_v02_n0_v02_heldout.json
uv run toolrank compare results/lora_toolret_lora.json results/v02base_toolret_hybrid.json results/v02base_toolret_srv.json
```

## Ortam

GB10, 11 Eki 19:46–19:52. `v0.2` bf16 8097'de (NGC imajı), reranker 8095 ve temel model 8091 de ayakta, başka
bir projenin `llama-server`'ı GPU'yu paylaşıyor. Bütün vektörler embedding cache'inden geldi (`encoder tokens
0`), head eğitimi CPU'da epoch başına 7–8 sn (ToolRet) ve 2 sn (MCP-Zero). Gecikme sütunları bu yüzden
anlamsız; kalite sayıları yükten bağımsız.

## Sapmalar ve açıklamalar

- **Eşli test yok.** `learn_sim.sh` sorgu düzeyinde satır (`--runs-out`) yazmıyor, bu yüzden +1,88 ve +1,55
  için p değeri verilemiyor. Ayrılmış setler 2.388 ve 838 sorgu; Dalga 3'ün ölçtüğü seed gürültüsü (cat-macro'da
  1,5 puan) eğitim seed'inin oynaması, bu değil. Kuralın eşiği (1 puan) koşudan önce yazılıydı ve iki koşu da
  onu açıkça geçti; yine de gürültü tabanı (GB10 sırası 4'ün üçüncü maddesi, qwen3-emb head'leriyle 5 seed)
  ölçülmedi — "±X puan" kuralları o ölçümü bekliyor.
- **`data/sim` silinmişti.** 11 Ekim'in disk temizliğinde v0.1 dönemi çıktıları gitti; bölmeler seed'den
  yeniden üretildi ve Hafta 4'ün sayılarıyla aynı çıktı (5.573 / 2.388), yani karşılaştırma aynı bölme üstünde.
- **Trafik simülasyonu kusursuz bir ajan.** `learn_sim`'in ajanı gösterilen altın tool'ları çağırıyor,
  gürültü yalnız `--noise` ile giriyor (bu koşularda 0). Gerçek trafiğin tekrar oranı ve hata payı farklı
  olacak; Hafta 4'ün notu burada da geçerli.

## Sonraki adımlar

- `--replay` aynı bölmede ölçülsün (dakikalar): MCP-Zero trafiğiyle ToolRet'in −2,48'ini ve ToolRet trafiğiyle
  LiveMCPBench'in −1,46'sını kapatıyor mu? Kapatıyorsa gecelik komuta girer (CLAUDE.md'deki not o ölçüme bağlı).
- Head gürültü tabanı: aynı tarifle 5 seed (GB10 sırası 4), bundan sonraki "±X puan" kuralları için.
- Kullanım hafızası (`backlog-memory.md`) ile `learn`'ün aynı bölmede karşılaştırması ve ikisinin birlikte:
  hafıza aynı ToolRet bölmesinde +3,48 almıştı, `learn` burada +1,88.
