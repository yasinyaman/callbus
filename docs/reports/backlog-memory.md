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
Kapılı "güvenli" seçim bu kaybı −0,12'ye indiriyor, kazanç +1,77'de kalıyor (ek bölüm): kaybı ve kazancı birlikte
kısıyor ama başabaş tekrar payını %10'dan %3'e çekiyor.

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

## Ek: kapılar ve "güvenli" seçim (10 Ekim, aynı gün)

Yeni tool'lara giden isteklerdeki kayıp için üç kapı eklendi (`b6c9486`): bir hafıza satırı ancak kosinüsü mutlak bir
eşiğin üstündeyse (`row ≥ τ`, τ 0,70–0,85), ilk aşamanın o istekteki en iyi kosinüsünü bir paydan fazla geçiyorsa
(`row ≥ best tool + δ`, δ −0,05…+0,1) ya da tool ilk aşamanın ilk N'indeyse (10 / 20 / 50) sayılır. β ızgarası 0–0,5,
λ 1 / 0,5 / 0,25; toplam 216 ayar ve "hafıza kapalı". Tanı: ayrılmış ToolRet sorgularında en yakın hafıza satırının
kosinüsü istenmiş sorgularda medyan 0,74, yenilerde 0,57; 0,8 eşiğinde istenmişlerin %38'inin altın tool'unda böyle bir
satır var, yenilerin %6'sında herhangi bir satır.

İki seçim, ikisi de yalnız trafikte (birini-dışarıda-bırak): **en iyi** (NDCG@10'u en yüksek) ve **güvenli** (trafikte
başka hiçbir isteğin çağırmadığı tool'lara giden isteklerde en çok 0,2 puan kaybeden ayarlar arasında en iyisi; "hafıza
kapalı" her zaman aday). Ayrılmış sorgular, NDCG@10 farkı (hepsi p<0,001):

| bölme | seçim | kural | hepsi | istenmiş | yeni | cat-macro |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| seed 0 | en iyi | λ 0,5 β 0 | +3,48 | +10,26 | −1,22 | +2,86 |
| seed 0 | güvenli | max β 0,2, satır ≥ en iyi tool | +1,77 | +4,50 | −0,12 | +0,99 |
| seed 1 | en iyi | λ 0,5 β 0 | +3,54 | +10,79 | −1,71 | +2,72 |
| seed 1 | güvenli | λ 0,5 β 0,1, satır ≥ 0,75 | +2,18 | +5,62 | −0,31 | +1,08 |
| seed 0, %20 yanlış çağrı | en iyi | λ 0,5 β 0 | +2,31 | +7,40 | −1,25 | +1,81 |
| seed 0, %20 yanlış çağrı | güvenli | λ 0,25 β 0,1, satır ≥ en iyi tool + 0,05 | +1,25 | +3,24 | −0,14 | +0,94 |

MCP-Zero'da iki seçim de "hafıza kapalı"; `dev_w3_multi2`'de ikisi de λ 0,25 β 0,3 (+0,11); LiveMCPBench'te (28 sorgu)
en iyisi ilk 50 kapısıyla +9,49 (istenmiş +22,8, yeni −5,9), güvenlisi ilk 10 kapısıyla +1,44 (yeni −0,1).

Kapılar bedava değil: kaybı azaltırken kazancı da aynı yönde kısıyorlar. Kazanan şey oran: başabaş için isteklerin
istenmiş tool'lara giden payı en iyi seçimde ~%10, güvenli seçimde ~%3 (seed 0: 0,12 / (4,50 + 0,12)). Gerçek bir
kurulumda daha önce çağrılmış tool'lara giden isteklerin %3'ün altında kalması beklenmez, bu yüzden ürün için varsayılan
güvenli seçim; kurulum kendi günlüğünde ölçüp en iyiye geçebilir.

```bash
# the same commands as above, with the gates (memory_sim.py at b6c9486), tags g0, g1, g0n20
uv run python scripts/memory_sim.py --out results/memory_toolret_g0 -- --data data/toolret $E --ks 5,10,20
uv run toolrank compare --paired --data data/toolret results/memory_toolret_g0_base_runs.jsonl \
  results/memory_toolret_g0_memory_safe_runs.jsonl
```

## Ek: LLM'in yazdığı tool metni (10 Ekim)

Aynı soru, trafik yerine bir LLM'le: her tool için Qwen3-8B (8093, düşünme kapalı) bir arama açıklaması (ne yapar,
hangi istekleri ve durumları karşılar, ne zaman başka bir tool daha uygun; belgede olmayan yetenek ve örnek çağrı yok)
ve beş farklı üslupta istek yazdı (`scripts/enrich_sim.py`, `b6c9486`/`6b97916`). Üç varyant, hiçbiri ayarlanmadan,
`v0.2`'nin tek başına sıralamasına karşı, setin bütün sorgularında: açıklama tool metninin sonuna eklenir (`describe`);
istekler metnin sonuna eklenir (`requests_concat`, doc2query); her istek tool'un ayrı bir belge satırıdır ve tool en iyi
satırının skorunu alır (`requests_rows`, hafızanın `max`'ı yazılmış isteklerle). Yazar, sorguları yazan model olmamalı:
ToolRet'in ve LiveMCPBench'in sorgularını Qwen3-8B yazmadı (MCP-Zero'nunkini ve w3 dev setlerininkini o yazdı, onlar
dışarıda). ToolRet'te 44.453 açıklama ve 220.374 istek (50 tool dışında hepsine; tool başına 4,96), üretim 2,5 sa.

NDCG@10; parantezde `v0.2`'ye göre fark ve p.

| set | n | `v0.2` | açıklama | istekler, metne | istekler, ayrı satır |
| --- | ---: | ---: | ---: | ---: | ---: |
| ToolRet | 7.961 | 58,92 | 57,75 (−1,17; <0,001) | 58,35 (−0,58; 0,001) | **55,78 (−3,15; <0,001)** |
| ToolRet cat-macro | 35 görev | 54,43 | 52,65 (−1,78; <0,001) | 52,83 (−1,59; <0,001) | 50,26 (−4,17; <0,001) |
| – code / web / customized | | 61,39 / 46,60 / 55,29 | −2,41 / −0,39 / −2,53 | −1,98 / −0,65 / −2,15 | −2,70 / −3,04 / −6,77 |
| LiveMCPBench `_server` | 94 | 55,74 | 53,25 (−2,49; 0,04) | 56,23 (+0,49; 0,61) | 56,24 (+0,50; 0,68) |

(`v0.2`'nin ToolRet'i burada 58,92: vektörler eval'inkilerle aynı cache'ten, sıralama süreç içinde; README'nin 58,90'ıyla
fark eşitlik sıralarından.) ToolRet'te `requests_rows`'ta P@1 347 sorguda düşüyor, 75'inde yükseliyor.

**Sonuç: yazılmış metin `v0.2`'ye eğitimsiz eklenince sıralamayı bozuyor; trafikten gelen gerçek istekler ise
düzeltiyor.** Aynı mekanizma (tool'a ait istek satırları, en iyi satır) gerçek isteklerle +3,48, yazılmış isteklerle
−3,15. Olası nedenler: yazılmış istekler genel (ToolRet'te aynı işi yapan çok sayıda tool var; birine yazılan istek
ötekilere de uyuyor ve satırların en iyisi alınınca yanlış tool öne geçiyor), trafik istekleri ise ayrılmış sorgularla
aynı dağılımdan; açıklamada "ne zaman başka bir tool uygun" cümlesi başka konuların kelimelerini metne taşıyor ve
embedding olumsuzlamayı ayırt etmiyor. 3 Ekim'de incelenen Tool-DE de Qwen3-Embedding-8B'de eğitimsiz zenginleştirmenin
kazandırmadığını (46,21 → 46,23) bulmuştu; burada `v0.2`'de kaybettiriyor. Zenginleştirme ancak eğitim verisi olarak (aynı
tool'un ikinci görünümü) bir aday olarak kalıyor.

```bash
# GB10: data/enrich_toolret.sh (untracked): 8093 up (Qwen3-8B, profile gen), the reranker stopped, then
uv run python scripts/enrich_sim.py --cache data/enrich/toolret.jsonl --generate-only --gen-workers 96 -- $FLAGS
# 8093 stopped, then (all answers cached)
uv run python scripts/enrich_sim.py --cache data/enrich/toolret.jsonl --out results/enrich_toolret -- $FLAGS
#   FLAGS="--data data/toolret --scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora --truncate 8192
#     --emb-batch 128 --tool-format documentation --query-format instruct_query --with-inst"
uv run python scripts/enrich_sim.py --cache data/enrich/livemcpbench_server.jsonl \
  --out results/enrich_livemcpbench_server -- ${FLAGS/toolret/livemcpbench_server}
# Mac
uv run toolrank compare --paired --data data/toolret results/enrich_toolret_base_runs.jsonl \
  results/enrich_toolret_requests_rows_runs.jsonl
```

## Sonraki adımlar

- ~~Yeni tool'lara giden isteklerdeki kaybı azaltmak~~: yukarıdaki ek; güvenli seçim kaybı −0,1/−0,3'e indiriyor.
- Ürünleştirme (kullanıcı kararı): `serve`'de kiracı başına hafıza (`DATA/memory/tenants/<ad>/`), kullanım günlüğünden
  `ok` çağrılarıyla (istek vektörü cache'ten, `learn.state_vectors`'ın yaptığı gibi), arka planda yenilenen; kuralı
  kurulumun kendi günlüğünde birini-dışarıda-bırakla seçer, kazanç yoksa kapalı kalır.
- `learn` (head eğitimi) ile hafızanın `v0.2`'de aynı bölmede karşılaştırması ve ikisinin birlikte.
