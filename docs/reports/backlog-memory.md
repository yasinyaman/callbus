# Backlog — kullanım hafızası, simülasyon (10 Ekim 2026)

Soru: `v0.2`'ye dokunmadan, bir kurulumun daha önce gördüğü istekler sıralamayı iyileştirir mi? Fikir: bir istek bir
tool'un çağrılmasıyla bittiğinde, isteğin vektörü o tool'a bir "hafıza satırı" olarak eklenir; sonraki istek bir tool'u
kendi kosinüsüyle ve o tool'un en iyi hafıza satırının kosinüsüyle puanlar. Eğitim yok, ağırlık değişmiyor; kiracı
başına ayrı tutulabilir.

## Sonuç (tek cümle)

ToolRet'te, trafiğin %70'inden kurulan hafıza ayrılmış sorgularda NDCG@10'u 59,62'den 63,09'a çıkarıyor (+3,48,
p<0,001; cat-macro +2,86; başka bir bölmede +3,54; ajan her beş çağrıdan birini yanlış tool'a yapınca +2,31). Kazancın
tamamı trafiğin daha önce istediği tool'lara gelen sorgularda (+10,3); hiç istenmemiş tool'lara gelen sorgular 1,2–1,7
puan kaybediyor. İstekleri tekrar etmeyen setlerde (MCP-Zero, w3 dev setleri) seçilen kural hiçbir şeyi değiştirmiyor.

## Ölçüler, birimler ve yön

`NDCG@10 ↑ %`, `Recall@k ↑ %`, `Comprehensiveness@10 ↑ %` README'deki gibi, w/ inst; `K@cut ↓`: adaptive K'nın
(margin 0,2, en çok 10) gösterdiği tool sayısı, `Recall@cut ↑ %` o listenin bulduğu altın tool payı. `istenmiş`: ayrılmış
sorgunun altın tool'larından biri trafikte çağrılmış; `yeni`: hiçbiri çağrılmamış. p: `toolrank compare --paired`
(ToolRet'te kategori ağırlıklı da).

## Kurgu (koşudan önce sabitlendi)

- Backbone `v0.2` (8097'de `qwen3-emb-lora`), documentation + instruct_query, w/ inst; vektörler embedding cache'inden.
- `scripts/learn_sim.py`'nin bölmesi: altın tool'u olan sorgular seed'le karıştırılır, %70'i trafik, %30'u ayrılmış set
  (ToolRet'te 5.573 / 2.388, `learn_sim`'deki sayılar).
- Simüle ajan `learn_sim`'deki gibi: trafik sorgusu ilk aşamayla sıralanır, serve'ün adaptive K'sıyla kesilir, gösterilen
  altın tool'lar çağrılır ve her çağrı bir hafıza satırı bırakır (gösterilmeyen altın tool hiç öğrenilmez). `--noise 0.2`:
  beş çağrıdan biri gösterilen ilk yanlış tool'a gider ve o yanlış bağ da hafızaya girer.
- Kural: tool'un skoru `c + λ · max(0, m − β − c)` (`add`; λ 1'de `max(c, m − β)`, `max`), c tool'un kendi kosinüsü,
  m en iyi hafıza satırının kosinüsü. β ve λ ızgarası (β 0–0,5, λ 0,25 / 0,5 / 1) yalnız trafikte seçilir: her trafik
  sorgusu kendi satırları dışarıda bırakılarak sıralanır, NDCG@10'u en yüksek ayar alınır. Ayrılmış sorgular seçimde
  hiç kullanılmaz.

## Tablo

Ayrılmış sorgular; ilk aşama → hafızalı.

| set (bölme) | hafıza satırı / tool | seçilen kural | n | NDCG@10 | p | istenmiş: n, NDCG@10 | yeni: n, NDCG@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ToolRet (seed 0) | 6.156 / 4.274 | add β 0 λ 0,5 | 2.388 | 59,62 → **63,09** | <0,001 | 978: 61,48 → **71,74** | 1.410: 58,32 → 57,10 |
| ToolRet (seed 1) | 6.198 / 4.341 | add β 0 λ 0,5 | 2.388 | 59,49 → **63,03** | <0,001 | 1.002: 58,19 → **68,98** | 1.386: 60,44 → 58,73 |
| ToolRet (seed 0, %20 yanlış çağrı) | 6.047 / 4.362 | add β 0 λ 0,5 | 2.388 | 59,62 → **61,93** | <0,001 | 985: 62,36 → **69,76** | 1.403: 57,69 → 56,44 |
| LiveMCPBench `_server` | 110 / 90 | add β 0,05 λ 0,5 | 28 | 46,93 → 51,62 | 0,12 | 15: 36,20 → 47,78 | 13: 59,30 → 56,05 |
| MCP-Zero `_server` | 1.928 / 1.914 | add β 0,5 λ 0,25 | 838 | 94,28 → 94,28 | 1,0 | 9 | 829 |
| dev_w3_multi2 | 1.046 / 820 | max β 0,4 | 240 | 83,98 → 84,00 | 1,0 | 152: 82,70 → 82,73 | 88: aynı |
| dev_w3, dev_w3_sit | 695 / 695, 409 / 409 | max β 0,5 | 300, 179 | değişmiyor | — | 0 | hepsi |

ToolRet ayrıntısı (seed 0): P@1 52,60 → 57,24 (152 sorgu yukarı, 41 aşağı), cat-macro 54,92 → 57,78 (web +4,92,
customized +2,51, code +1,14), Comprehensiveness@10 60,26 → 63,15, K@cut 8,39 → 8,14 ve Recall@cut 69,63 → 71,77: daha
az tool gösterip daha çok doğru tool buluyor. Seed 1'de cat-macro 56,30 → 59,02.

## Yorum

- **Kazanç tekrara bağlı.** ToolRet'in görevlerinde aynı tool'a birçok istek var; hafıza orada büyük kazandırıyor
  (+10). MCP-Zero ve w3 dev setlerinde her istek ayrı bir tool için yazıldı, ayrılmış sorguların neredeyse hiçbiri
  trafiğin istediği bir tool'a gitmiyor; birini-dışarıda-bırak seçimi en temkinli kuralı alıyor ve sıralama değişmiyor.
  Yani hafıza zarar vermeden kendini kapatabiliyor, ama gerçek kazancı gerçek trafiğin tekrar oranına bağlı; bunu bu
  setler söyleyemez.
- **Bedeli yeni tool'lara giden isteklerde.** Hafızası olan tool'lar yukarı çekilince hafızası olmayan doğru tool bir
  iki sıra iniyor (ToolRet −1,2 / −1,7). Başabaş: isteklerin ~%10'undan fazlası daha önce çağrılmış tool'lara giderse
  net kazanç.
- **Gürültüye dayanıklı.** Yanlış çağrıların %20'si de hafızaya girince kazancın üçte ikisi kalıyor (+2,31).
- **Bağlam:** `learn_sim`'de v0.1 head'lerini aynı bölmeyle eğitmek ToolRet'i 54,20'den 57,23'e çıkarmıştı (istenmiş
  tool'larda +5,3, yenilerde ±0; `faz2-week4.md`). Backbone farklı olduğu için doğrudan karşılaştırma değil; `learn`'ün
  `v0.2`'deki koşusu (G4) yapılınca ikisi aynı bölmede yan yana konur.

## Komutlar

```bash
# GB10, ~/toolrank (8097 serving v0.2 as qwen3-emb-lora; vectors from .cache/toolrank)
E="--scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192 --emb-batch 128 \
  --tool-format documentation --query-format instruct_query --with-inst --cut-margin 0.2"
uv run python scripts/memory_sim.py --out results/memory_toolret_s0 -- --data data/toolret $E --ks 5,10,20
uv run python scripts/memory_sim.py --seed 1 --out results/memory_toolret_s1 -- --data data/toolret $E --ks 5,10,20
uv run python scripts/memory_sim.py --noise 0.2 --out results/memory_toolret_s0n20 -- --data data/toolret $E --ks 5,10,20
uv run python scripts/memory_sim.py --out results/memory_mcp_zero_server_s0 -- --data data/mcp_zero_server $E --ks 1,5,10,20
# ... dev_w3, dev_w3_sit (--ks 1,5,10), dev_w3_multi2, livemcpbench_server (--ks 5,10,20)
# Mac, after scp 'gb10:toolrank/results/memory_*' results/:
uv run toolrank compare --paired --data data/toolret results/memory_toolret_s0_base_runs.jsonl \
  results/memory_toolret_s0_memory_runs.jsonl
```

## Ortam

GB10, `v0.2` bf16 (NGC vLLM, 8097); bütün vektörler cache'ten, koşuların hepsi 8 dakika (CPU, eğitim yok). Kod
`44abe60`.

## Sapmalar ve açıklamalar

- Simüle trafik benchmark sorgularından: aynı görevin istekleri birbirine benziyor, gerçek trafikte tekrar oranı ve
  ifade benzerliği bilinmiyor. ToolRet'teki +3,5 bir üst sınıra yakın okunmalı.
- Ajan kusursuz (gösterilen altın tool'u hep çağırıyor) ya da %20 yanlış; gerçek ajanlar ve `tool_error` ayrımı yok.
- Seçim ızgarası küçük; tekrar etmeyen setlerde seçim "ilk aşama tek başına"yı aday saymadığı için en temkinli kuralı
  alıyor (sonuç aynı).

## Sonraki adımlar

- Yeni tool'lara giden isteklerdeki kaybı azaltmak: mutlak bir eşik (hafıza satırı ancak çok benzerse kaldırsın) ya da
  seçimde "yeni" isteklerin kaybına sınır.
- Ürünleştirme (kullanıcı kararı): `serve`'de kiracı başına hafıza (`DATA/memory/tenants/<ad>/`), kullanım günlüğünden
  `ok` çağrılarıyla (istek vektörü cache'ten, `learn.state_vectors`'ın yaptığı gibi), arka planda yenilenen; kuralı
  kurulumun kendi günlüğünde birini-dışarıda-bırakla seçer, kazanç yoksa kapalı kalır.
- `learn` (head eğitimi) ile hafızanın `v0.2`'de aynı bölmede karşılaştırması ve ikisinin birlikte.
