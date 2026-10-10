# Backlog — Dalga 3 raporu (8 Ekim 2026–)

Dalga 3, kaliteye dönük backlog maddeleri: LoRA'lı backbone'un kaldıraçları, ölçülerek. Bu rapor maddeler
bittikçe büyüyor. İlk tur: v0.2 tarifinin seed replikası (gürültü tabanı), iki kol (belge biçimine göre dengeli
çiftler, sahte negatif marjı) ve iki seed'in ağırlık ortalaması ("soup").

## Sonuç (tek cümle)

İki kol da önceden yazılan kuralı geçmedi: biçim dengesi MCP biçimli setlerde 1,2–1,4 puan kazanıyor ama
LiveMCPBench'te 2,7 ve ToolRet cat-macro'da 1,2 kaybediyor; sahte negatif marjı tek tool'luk setlerde kazanıyor,
karar setinde kazanmıyor. `v0.2` ile replikasının ağırlık ortalaması, eğitimsiz ve 20 saniyede, ToolRet'te `v0.2`'yi
geçiyor (NDCG@10 58,90 → 59,71, cat-macro 54,36 → 54,98, ikisi de anlamlı) ve hiçbir sette 0,4 puandan fazla
kaybetmiyor; karar setinde ise `v0.2`'yle berabere (+0,30, p 0,25). Yayın kararı kullanıcıda.

## Ölçüler, birimler ve yön

`NDCG@10 ↑ %`, README tablosundaki gibi talimatlı (w/ inst), documentation + instruct_query, `--truncate 8192`.
`fark`: yüzde puanı, sağdaki model eksi referans. p: `toolrank compare --paired`'in eşli permütasyon testi;
ToolRet'in kategorileri ve cat-macro'su için aynı test, sorgular kategori ortalamalarının tarttığı gibi
ağırlıklı (`--data data/toolret`). `seçim` ve `teyit`: `dev_w3_multi2`'nin `toolrank data split` yarıları (400'er
istek, aynı tool'lar); kollar seçim yarısında seçildi, teyit yarısı hiçbir seçimde kullanılmadı.

## Ön kayıt (koşulardan önce yazıldı)

- **Kol kuralı (3 Eki, sayıları 8 Eki'de replikadan):** her kol aynı seed'li replikaya karşı okunur;
  `dev_w3_multi2` ve ToolRet cat-macro, `v0.2` ile replika arasındaki farkın iki katından fazla iyileşir (0,35 ve
  2,98 puan) ve hiçbir set 1 puandan fazla kaybetmezse yeni etiket ve servis adı alır.
- **Ağırlık ortalaması kuralı (7 Eki):** ToolRet code ve cat-macro'yu 20k'nın (`v0.2`) 1 puan altına düşürmeden
  `dev_w3_multi2`'nin teyit yarısında replika yayılımından (`v0.2` ile replikanın farkı, 0,03) fazla kazanırsa bir
  kol olarak kalır.
- Yayın kuralı ayrıca yazılmadı; D1.1'inki ("karar setinde `v0.2`'yi geçer, hiçbir sette 1 puandan fazla
  kaybetmez") aşağıda bağlam olarak uygulanıyor.

## Koşular

Hepsi 20.000 ToolRet-train çifti, rank 16, 625 adım (1 epoch), lr 1e-4, mikro-batch 16 × 2, `--data-seed 0`,
`--seed 1`, seçim `dev_w3_multi2_sel`'de; tek fark sütundaki kaldıraç. Dev eğrisi transformers'la, eğitim
sırasında (seçim yarısı, NDCG@10).

| koşu | kaldıraç | adım 0 | 300 | 600 | 625 | seçilen | süre ↓ dk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| replika (`20k-r1`) | yok | 69,55 | 81,81 | 82,92 | **83,03** | 625 | 538 |
| biçim (`20k-shape`) | `--balance shape` | 69,55 | 81,91 | **84,03** | 84,01 | 600 | 511 |
| marj (`20k-fnm`) | `--fn-margin 0.1` | 69,55 | 81,52 | **83,24** | 83,17 | 600 | 539 |
| seed soup | `v0.2` + replika adaptörleri, eşit ağırlık | — | — | — | — | — | 0,3 (CPU) |

Çiftlerin belge biçimleri: replikada ToolBench REST (template_response'lu) 8.684, ToolBench REST 4.930, fonksiyon
biçimi 5.537, fonksiyon biçimi + `required` 849; biçim kolunda dördü 5.000'er. `v0.2` ile replika aynı 20.000 çifti
kullanıyor ve ikisi de son adımı seçti: aradaki tek fark LoRA'nın başlangıcı ve batch sırası. Seçim yarısında
servis edilen modelin skoru eğitimdekiyle tutarlı (replika 83,09, biçim 84,16, marj 83,32).

## Tablo: replikaya karşı

NDCG@10 ↑ %; parantezde replikaya göre fark ve p.

| set | n | replika | `v0.2` | biçim | marj | seed soup | 60k (D1.1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_w3_multi2 (karar) | 800 | 82,77 | 82,95 (+0,17; 0,62) | 84,03 (+1,25; 0,001) | 83,07 (+0,30; 0,28) | 83,25 (+0,48; 0,05) | 83,12 (+0,35; 0,35) |
| – seçim yarısı | 400 | 83,09 | 83,40 (+0,31; 0,52) | 84,16 (+1,07; 0,06) | 83,32 (+0,24; 0,56) | 83,43 (+0,34; 0,26) | 83,11 (+0,03; 0,96) |
| – teyit yarısı | 400 | 82,46 | 82,49 (+0,04; 0,94) | 83,90 (+1,44; 0,004) | 82,81 (+0,35; 0,38) | 83,07 (+0,62; 0,10) | 83,13 (+0,67; 0,17) |
| dev_w3_multi3 | 498 | 75,46 | 75,40 (−0,06; 0,88) | 76,89 (+1,43; 0,003) | 75,54 (+0,07; 0,84) | 76,06 (+0,60; 0,03) | 74,66 (−0,80; 0,13) |
| dev_w3 | 1.000 | 92,67 | 92,49 (−0,18; 0,52) | 92,86 (+0,19; 0,57) | 93,33 (+0,66; 0,005) | 93,21 (+0,54; 0,02) | 92,91 (+0,24; 0,45) |
| dev_w3_sit | 598 | 86,99 | 87,71 (+0,72; 0,13) | 88,37 (+1,38; 0,01) | 88,42 (+1,43; <0,001) | 88,15 (+1,16; 0,001) | 87,45 (+0,46; 0,38) |
| LiveMCPBench `_server` | 94 | 56,30 | 55,74 (−0,56; 0,66) | **53,62 (−2,68; 0,02)** | 57,14 (+0,84; 0,38) | 56,09 (−0,22; 0,84) | 54,96 (−1,34; 0,27) |
| MCP-Zero `_server` | 2.792 | 92,93 | 93,67 (+0,74; <0,001) | 94,17 (+1,24; <0,001) | 93,78 (+0,85; <0,001) | 93,31 (+0,38; 0,001) | 92,14 (−0,79; <0,001) |
| ToolRet | 7.961 | 58,66 | 58,90 (+0,24; 0,16) | 59,36 (+0,70; <0,001) | 58,49 (−0,17; 0,15) | 59,71 (+1,06; <0,001) | 58,86 (+0,20; 0,20) |
| ToolRet cat-macro | 35 görev | 52,87 | 54,36 (+1,49; <0,001) | **51,72 (−1,15; 0,01)** | **52,06 (−0,81; 0,004)** | 54,98 (+2,11; <0,001) | 52,31 (−0,56; 0,13) |
| – code | 7 görev | 58,34 | 61,39 (+3,05; <0,001) | 57,85 (−0,49; 0,45) | 58,12 (−0,21; 0,58) | 61,12 (+2,79; <0,001) | 56,42 (−1,91; 0,002) |
| – web | 19 görev | 44,58 | 46,39 (+1,81; 0,001) | 43,61 (−0,97; 0,07) | 43,63 (−0,94; 0,006) | 47,12 (+2,54; <0,001) | 44,59 (+0,02; 0,98) |
| – customized | 9 görev | 55,70 | 55,29 (−0,40; 0,63) | 53,71 (−1,99; 0,07) | 54,44 (−1,26; 0,04) | 56,71 (+1,01; 0,06) | 55,92 (+0,23; 0,75) |

## Tablo: seed soup, `v0.2`'ye karşı

| set | `v0.2` | seed soup | fark | p |
| --- | ---: | ---: | ---: | ---: |
| dev_w3_multi2 (karar) | 82,95 | 83,25 | +0,30 | 0,25 |
| – seçim / teyit yarısı | 83,40 / 82,49 | 83,43 / 83,07 | +0,03 / +0,58 | 0,93 / 0,15 |
| dev_w3_multi3 | 75,40 | 76,06 | +0,66 | 0,03 |
| dev_w3 | 92,49 | 93,21 | +0,72 | 0,001 |
| dev_w3_sit | 87,71 | 88,15 | +0,43 | 0,20 |
| LiveMCPBench `_server` | 55,74 | 56,09 | +0,34 | 0,63 |
| MCP-Zero `_server` | 93,67 | 93,31 | −0,36 | 0,004 |
| ToolRet | 58,90 | 59,71 | +0,82 | <0,001 |
| ToolRet cat-macro | 54,36 | 54,98 | +0,62 | 0,02 |
| – code / web / customized | 61,39 / 46,39 / 55,29 | 61,12 / 47,12 / 56,71 | −0,27 / +0,73 / +1,41 | 0,46 / 0,07 / 0,01 |

## Karar

- **Biçim kolu kuralı geçmiyor.** Karar setinde +1,25 (eşiğin üç katı) ve teyit yarısında +1,44, ama LiveMCPBench
  −2,68 (1 puan sınırının ötesinde; 94 istek, yine de diğer dört koşunun hepsinin altında) ve cat-macro −1,15.
  Kazanç MCP biçimli setlerde (w3 dev setleri, MCP-Zero) ve ToolRet'in büyük görevlerinde (mikro +0,70); kayıp
  ToolRet'in küçük görevlerinde. Biçim karışımı bir kaldıraç, ama eşit dağıtım fazla.
- **Marj kolu kuralı geçmiyor.** Karar setinde +0,30 (eşik 0,35, p 0,28), cat-macro −0,81. Kazancı tek tool'luk
  setlerde (`dev_w3` +0,66, `dev_w3_sit` +1,43) ve MCP-Zero'da (+0,85).
- **Seed soup ağırlık ortalaması kuralını geçiyor ve bir kol olarak kalıyor:** code `v0.2`'nin 0,27 altında
  (sınır 1), cat-macro 0,62 üstünde, teyit yarısında +0,58 (eşik 0,03). İki ebeveyninin de üstünde: ToolRet
  59,71 (ebeveynler 58,90 / 58,66), cat-macro 54,98 (54,36 / 52,87), web 47,12 (46,39 / 44,58), customized 56,71
  (55,29 / 55,70); yalnız MCP-Zero'da ikisinin arasında. Seed'in getirdiği oynamayı ortalama alıyor.
- **Yayın için:** D1.1'in kuralıyla seed soup da karar setinde berabere (+0,30, p 0,25), yani o kural `v0.2`'de
  kalmayı söylüyor; ToolRet'teki kazanç ise anlamlı ama küçük (seed farkının altında; LoRA'nın kendi kazancının ve
  ikinci aşamanınkinin onda biri). **Karar (kullanıcı, 10 Eki): 0.3.0 `v0.2` ile çıkıyor; model iyileştirme sonraki
  adımlara kalıyor.** Soup bir yöntem olarak kalıyor: sonraki backbone tek bir koşu yerine birkaç seed'in ortalaması.

## Komutlar

```bash
# GB10, ~/toolrank. The selection and confirmation halves (once):
uv run toolrank data split data/dev_w3_multi2 --out-a data/dev_w3_multi2_sel --out-b data/dev_w3_multi2_conf
# the arms (data/lora/chain_lora_arms.sh, untracked): the replica's flags plus one lever each
P="--pairs data/toolret_train/pairs.jsonl --dev data/dev_w3_multi2_sel --eval data/toolret --eval data/livemcpbench_server \
  --eval data/mcp_zero_server --eval data/dev_w3 --eval data/dev_w3_sit --eval data/dev_w3_multi3 --eval data/dev_w3_multi2_conf \
  --n-train 20000 --seed 1 --data-seed 0 --check-parity --keep-all"
uv run --extra lora python scripts/lora_train.py $P --out data/lora/qwen3-emb-lora-20k-r1
uv run --extra lora python scripts/lora_train.py $P --balance shape --out data/lora/qwen3-emb-lora-20k-shape
uv run --extra lora python scripts/lora_train.py $P --fn-margin 0.1 --out data/lora/qwen3-emb-lora-20k-fnm
# the seed soup (data/lora/chain_soup.sh, untracked)
uv run --extra lora python scripts/lora_soup.py --adapter data/lora/qwen3-emb-lora-20k/adapter \
  --adapter data/lora/qwen3-emb-lora-20k-r1/adapter --out data/lora/qwen3-emb-lora-soup-seeds
# each model served on 8097 under its own name, then scored like D1.1's (docs/reports/backlog-d1.md, Komutlar):
TOOLRANK_LORA=$HOME/toolrank/data/lora/qwen3-emb-lora-20k-shape/merged TOOLRANK_LORA_NAME=qwen3-emb-lora-shape \
  docker compose -f deploy/spark/compose.yaml --profile lora up -d --force-recreate qwen3-emb-lora
E="--scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora-shape --truncate 8192 --emb-batch 128 \
  --tool-format documentation --query-format instruct_query --with-inst"
uv run toolrank eval --data data/dev_w3_multi2 $E --ks 5,10,20 --cut-margin 0.2 \
  --runs-out results/lorashape_dev_w3_multi2_runs.jsonl --out results/lorashape_dev_w3_multi2.json
# ... dev_w3_multi3 (--ks 5,10,20 --cut-margin 0.2), dev_w3 and dev_w3_sit (--ks 1,5,10), livemcpbench_server,
# mcp_zero_server (--ks 1,5,10,20), toolret; tags lorar1, lorashape, lorafnm, lorasoupseeds
# Mac, after scp 'gb10:toolrank/results/lora{r1,shape,fnm,soupseeds}_*' results/ and the halves' and ToolRet's
# queries.jsonl into data/:
uv run toolrank compare --paired results/lorar1_dev_w3_multi2_runs.jsonl results/lorashape_dev_w3_multi2_runs.jsonl
uv run toolrank compare --paired --subset data/dev_w3_multi2_conf \
  results/lorar1_dev_w3_multi2_runs.jsonl results/lorashape_dev_w3_multi2_runs.jsonl
uv run toolrank compare --paired --data data/toolret results/lorar1_toolret_runs.jsonl results/lorashape_toolret_runs.jsonl
```

## Ortam

GB10, NGC vLLM imajı (`deploy/spark/compose.yaml`'ın `lora` profili, 8097, bf16), eğitim `[lora]` ekiyle
(transformers 5.x, peft 0.21.1, torch 2.14.0+cu130). Her model kendi servis adıyla, yani embedding cache'i
ayrı (soğuk); `v0.2`'nin satırları D1.1'deki cache'ten. Kod `fbc363e`'ye kadar (kollar `9d24e0c`'yle, soup
`5739abb`'yle koştu).

## Sapmalar ve açıklamalar

- Seçimdeki dev skoru (transformers) ile servis edilen modelin skoru (vLLM) aynı yarıda 0,06–0,13 puan içinde.
- Tablolardaki p'ler sorgu örneklemesinin belirsizliğini ölçüyor, seed'inkini ölçmüyor: `v0.2` ile replika
  arasında MCP-Zero'da 0,74, cat-macro'da 1,49 puan fark var (`backlog-d1.md`, D1.1'in eki). Kural bu yüzden
  replika farkının iki katını istiyor; tek koşuluk kolların bu tabana yakın farkları karar verdirmez.
- Mac 8–9 Ekim gecesi uykuya geçti, izleme durdu; GB10'daki zincirler etkilenmedi.
- GB10'da disk %99 dolu (14 GB boş): beş birleştirilmiş model (16'şar GB). Replika, biçim ve marj kollarının
  `merged`'i ölçüldü ve adaptörlerinden `scripts/lora_soup.py` ile (tek `--adapter`) bit bit yeniden kurulabilir.

## Sonraki adımlar

- Disk (kullanıcı): ölçülmüş üç kolun `merged` klasörleri silinebilir (48 GB), adaptörleri kalır.
- Seed soup'a daha fazla seed (her biri ~9 sa): soup'lar malzeme arttıkça iyileşir ve seed oynamasını azaltır.
  Önce yayın kuralı yazılır.
- Tarif soup'u, eğitimsiz: `v0.2` + replika + biçim + marj (biçim kolunun MCP kazancını ToolRet kaybı olmadan
  almayı dener; aynı taban modelin dört ince ayarı).
- GradCache kolu (`--cache-batch 32`, ~11 sa) ve yarı yarıya biçim karışımı, aynı kuralla.
