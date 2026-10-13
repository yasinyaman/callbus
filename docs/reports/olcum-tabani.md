# Ölçüm tabanı — varsayılan backbone'da (11 Ekim 2026)

GB10 sırası 4. Bugüne kadarki bütün `learn` / `ab` sonuçları, adaptive K marjı, hybrid ve sunucu oyu temel model
(`qwen3-emb`) ve v0.1 head'leri üzerinde ölçülmüştü; varsayılan backbone 2 Ekim'de head'siz `v0.2` oldu. Soru:
bu dört karar `v0.2`'de de aynı mı?

## Sonuç (tek cümle)

Öğrenme döngüsü `v0.2`'de de kazandırıyor ama daha az (ayrılmış isteklerde beş eğitim seed'inin ortalaması +2,43
micro / +1,99 cat-macro, MCP-Zero kataloğuyla +1,55; v0.1'de +3,03), yani önceden yazılan kural tuttu ve `learn`
gecelik komutta kalıyor; yanında bıraktığı 1,5–3 puanlık kaybı `--replay`
kapatıyor, ama ancak 10.000 çiftle (varsayılan 1.000 hiçbir şey yapmıyor, o yüzden varsayılan değişti); eğitim
seed'i tek başına ayrılmış NDCG@10'u ±0,5 oynattığı için bundan sonraki farklar 2·SD'yi (micro 1,1, cat-macro 1,6)
geçmek zorunda; adaptive K
marjı 0,2'de kalıyor (hiçbir marj kuralı geçmedi); hybrid artık her sette kaybettiriyor (v0.1'de yalnız
MCP-Zero'da kazanıyordu) ve sunucu oyu berabere düştü (v0.1'de MCP-Zero top-1'e +1,1) — ikisi de isteğe bağlı
kalıyor, belgeleri düzeltildi.

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

Aşağıdaki satırlar tek bir eğitim seed'inin (0) koşusu. **6. bölümdeki gürültü tabanı sonradan ölçüldü ve
seed 0'ın beş koşunun en kötüsü olduğunu gösterdi**: ToolRet kolunda ayrılmış NDCG@10 beş seedde 61,46–62,52
(ortalama 62,04, yani +2,43) ve cat-macro 55,90–57,67 (ortalama 56,89, +1,99). Tabloyu o bantla birlikte okumak
gerekiyor; MCP-Zero kolunun bandı ölçülmedi.

| katalog | set | n | başlangıç | öğrenilen (seed 0) | fark |
| --- | --- | ---: | ---: | ---: | ---: |
| ToolRet | ayrılmış | 2.388 | 59,61 | **61,49** (5 seed ort. **62,04**) | **+1,88** (ort. +2,43) |
| ToolRet | ayrılmış, cat-macro | 3 kat. | 54,90 | 55,90 (ort. 56,89) | +1,00 (ort. +1,99) |
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

## 5. `--replay` — işe yarıyor, ama varsayılan dozda değil

Yukarıdaki yan etkinin ilacı `--replay`: genel istek–tool çiftlerinden bir örnek eğitim batch'lerine
karışıyor. Aynı bölme, aynı günlük, yalnız `learn` adımı değişti (`--replay data/toolret_train/pairs.jsonl`),
iki dozla: varsayılan 1.000 çift ve 10.000 çift.

**MCP-Zero kataloğu** (unutmanın en belirgin olduğu kol: ToolRet trafiğin hiç kapsamadığı set):

| set | başlangıç | learn | + replay 1k | + replay 10k |
| --- | ---: | ---: | ---: | ---: |
| ayrılmış (838) | 94,28 | 95,83 | 95,78 | **95,76** |
| yeni (829) | 94,32 | 95,90 | 95,84 | 95,84 |
| ToolRet (7.961) | 58,90 | 56,42 | 56,43 | **59,27** |
| ToolRet cat-macro | 54,36 | 52,74 | 52,38 | **54,40** |
| LiveMCPBench (94) | 55,74 | 58,72 | 57,68 | 56,83 |

**ToolRet kataloğu** (unutulan set LiveMCPBench):

| set | başlangıç | learn | + replay 1k | + replay 10k |
| --- | ---: | ---: | ---: | ---: |
| ayrılmış (2.388) | 59,61 | 61,49 | 62,56 | **62,49** |
| ayrılmış, cat-macro | 54,90 | 55,90 | 57,42 | 57,05 |
| yeni (1.061) | 69,12 | 70,16 | 70,49 | 70,67 |
| LiveMCPBench (94) | 55,74 | 54,28 | 54,35 | **55,15** |
| MCP-Zero (2.792) | 93,67 | 93,68 | 93,71 | 93,33 |

- **Varsayılan 1.000 çift hiçbir şey yapmıyor.** İki kolda da unutulan set yerinde kalıyor (ToolRet 56,43,
  LiveMCPBench 54,35). 208.826 çiftin binde beşi, batch'lerde günlüğün yanında görünmüyor.
- **10.000 çiftte unutma kapanıyor.** MCP-Zero kolunda ToolRet 56,42 → 59,27 (başlangıç 58,90, cat-macro da
  54,40'a dönüyor; +2,85, aşağıdaki tabandan büyük) ve kataloğun kendi kazancı duruyor (95,76). ToolRet kolunda
  LiveMCPBench 54,28 → 55,15 (başlangıç 55,74; kalan −0,59, 94 görevde bir görevin altında).
- **ToolRet kolundaki "replay fazladan kazandırıyor" okuması düştü.** Replay'li ayrılmış skorlar (62,56 ve 62,49)
  replay'siz beş seedin bandının (61,46–62,52) içinde; yani in-domain replay'in ayrılmış isteklere kattığı şey
  eğitim seed'i gürültüsünden ayırt edilemiyor. Replay'in ölçülebilir işi unutmayı onarmak.
- **Bedeli, tesadüfi kazançlar.** MCP-Zero trafiğinin LiveMCPBench'e kattığı +2,98, replay'le +1,09'a iniyor;
  ToolRet kolunda MCP-Zero 93,67 → 93,33. Yani replay head'leri backbone'a doğru geri çekiyor: unutmayı da,
  beklenmedik kazancı da kırpıyor.
- **Sızma yok.** `pairs.jsonl`'ın 208.826 isteğinden 768'i (binde 4) ToolRet'in değerlendirme sorgularıyla
  birebir aynı; 10.000'lik örnekte beklenen ~37 istek, yani ToolRet'in geri gelişi örneklemenin kendi
  sorgularını görmesinden değil. (`learn` yalnız `--dev` setinin sorduklarını atıyor; burada `--dev`
  LiveMCPBench ve MCP-Zero'ydu.)

**Karar:** `--replay` gecelik komuta giriyor ve `--replay-n`'in varsayılanı 1.000'den **10.000**'e çıkıyor
(`cli.py`; kılavuz ve CLAUDE.md da bunu yazıyor). Çiftlerin vektörleri embedding cache'inden geliyor, eksikler
bir kez gömülüyor; GB10'da hepsi önbellekteydi ve epoch süresi 3 sn'den 9 sn'ye çıktı.

## 6. Gürültü tabanı — bir "+X puan" ne kadar olmalı

Aynı tarif, aynı bölme, aynı günlük; değişen tek şey `learn`'ün eğitim seed'i (`--seed 0..4`, veri seed'i sabit).
Ayrılmış isteklerin (2.388) NDCG@10'u:

| yığın | başlangıç | 5 seedin aralığı | ortalama | SD | 2·SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| temel model + v0.1 head'leri (8091) | 54,20 | 56,65 – 57,39 | 57,11 | 0,31 | 0,62 |
| aynısı, cat-macro | 48,59 | 51,31 – 52,77 | 52,34 | 0,60 | 1,20 |
| `v0.2`, kimlik head'lerinden (8097) | 59,61 | 61,46 – 62,52 | 62,04 | 0,53 | 1,05 |
| aynısı, cat-macro | 54,90 | 55,90 – 57,67 | 56,89 | 0,78 | 1,56 |

- **Kural:** bundan sonra bu düzenekteki bir fark, ancak 2·SD'yi geçerse sayılır — micro'da 0,6 (temel model) /
  1,1 (`v0.2`), cat-macro'da 1,2 / 1,6 puan. Dalga 3'ün LoRA seed replikası (cat-macro 1,5 puan) eğitim
  verisinin değil backbone'un gürültüsüydü; bu taban head eğitiminin kendi gürültüsü ve ikisi toplanıyor.
- **Hafta 4'ün sayısı yerinde:** v0.1'de bildirilen 57,23, buradaki 56,65–57,39 bandının içinde.
- **Bugünün `learn` sonucu bu bantla daha güçlü:** `v0.2`'de ortalama +2,43 micro ve +1,99 cat-macro, ikisi de
  2·SD'nin üstünde. Ama 1. bölümün tek seedi (seed 0) bandın en altıydı; tek koşuya dayanan "+1,00 cat-macro"
  gibi bir ifade bu bantta anlamsız.
- Bant yalnız head eğitiminin seed'i için ölçüldü (ToolRet kataloğu, trafiğin tamamı). MCP-Zero kolu, `--replay`
  kolları ve farklı trafik büyüklükleri tek seed; onların bandı bu büyüklükte varsayılıyor, ölçülmedi.

## Komutlar

```bash
# GB10, ~/callbus (data/v02_loop_chain.sh, untracked; vectors all cached, 6 minutes end to end)
export HEADS=none EMB_URL=http://127.0.0.1:8097/v1 EMB_MODEL=qwen3-emb-lora UV=~/.local/bin/uv
NAME=toolret_v02 TAG=v02 SIZES=0 LEARN="--device cpu" bash scripts/learn_sim.sh
BENCH=data/mcp_zero_server NAME=mcpzero_v02 TAG=v02 GUARD=data/livemcpbench_server \
  EVALS="data/toolret data/livemcpbench_server" SIZES=0 LEARN="--device cpu" bash scripts/learn_sim.sh
V02="--emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --emb-batch 128
  --tool-format documentation --query-format instruct_query --with-inst"
for d in toolret livemcpbench_server mcp_zero_server; do
  $UV run python scripts/adaptive_k_sweep.py --margins 0.1,0.15,0.2,0.3 -- --data data/$d --scorer dense $V02
  $UV run callbus eval --data data/$d --scorer dense $V02 --hybrid --out results/v02base_${d}_hybrid.json
  $UV run callbus eval --data data/$d --scorer dense $V02 --server-weight 0.2 --out results/v02base_${d}_srv.json
done   # mcp_zero_server also --ks 1,5,10,20
# the replay arms (data/replay_chain.sh, untracked): the same logs, only the learn step changes
NAME=toolret_v02 TAG=replay SIZES=0 LEARN="--device cpu --replay data/toolret_train/pairs.jsonl" \
  bash scripts/learn_sim.sh      # and TAG=replay10k with --replay-n 10000; the same two for mcpzero_v02
# the noise floor (data/noise_floor.sh and data/noise_v02.sh, untracked): the same split and log, five
# training seeds, nothing else changed
for s in 0 1 2 3 4; do
  NAME=toolret TAG=seed$s SIZES=0 LEARN="--device cpu --seed $s" bash scripts/learn_sim.sh   # 8091 + v0.1 heads
  NAME=toolret_v02 TAG=v02seed$s SIZES=0 LEARN="--device cpu --seed $s" bash scripts/learn_sim.sh  # HEADS=none, 8097
done
# Mac, after scp 'gb10:callbus/results/{sim_*_v02_*,v02base_*,sim_toolret_n0_seed*}.json' results/:
uv run callbus compare results/sim_toolret_v02_base_heldout.json results/sim_toolret_v02_n0_v02_heldout.json
uv run callbus compare results/lora_toolret_lora.json results/v02base_toolret_hybrid.json results/v02base_toolret_srv.json
uv run callbus compare results/sim_mcpzero_v02_{base,n0_v02,n0_replay,n0_replay10k}_toolret.json
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

- `--replay`'in dozu tek bölmede iki noktada ölçüldü (1.000 ve 10.000); nerede doyduğu (3.000? 30.000?) açık.
- Bant yalnız ToolRet kataloğu ve trafiğin tamamı için var. MCP-Zero kolunun ve `--replay` kollarının kendi
  bandı, bir karar onlara dayanacaksa ölçülmeli (her biri ~13 dakika).
- LoRA tarafında bant yok: eğitim seed'i replikası yalnız bir kez koşuldu (Dalga 3), o yüzden backbone
  karşılaştırmaları hâlâ eşli test istiyor (D1.10).
- Kullanım hafızası (`backlog-memory.md`) ile `learn`'ün aynı bölmede karşılaştırması ve ikisinin birlikte:
  hafıza aynı ToolRet bölmesinde +3,48 almıştı, `learn` burada +1,88.
