# Backlog — Dalga 1 raporu (3 Ekim 2026–)

Dalga 1, Faz 1 kapısından önce benimsenmeyi artırmak için öne alınan backlog maddeleri: 60k LoRA kararı
(D1.1), küçük backbone'lar (D1.2), GGUF ile GPU'suz yol (D1.3), LangChain 1.x middleware'i (D1.4), toolrank
skill'i (D1.5), 0.3.0 (D1.6), Jev koltuğu ve öğrenme düzeltmeleri (D1.7–D1.9, D1.12), eşli testler (D1.10)
ve Strands rehberi (D1.11). Bu rapor maddeler bittikçe büyüyor.

## Sonuç (tek cümle)

D1.1, D1.3, D1.4, D1.5, D1.7–D1.9, D1.11, D1.12 ve D1.10'un kodu bitti; D1.2'nin dizüstü kısmı bitti, MCP-Zero
ve dev setleri GB10'u bekliyor. 0.3.0'ın taşıdığı yığın (resmi `vllm/vllm-openai:v0.30.0` imajı, FP8) NGC
0.13'teki satırları tutuyor: üç sette fark en çok 0,5 puan, vektör kosinüsü 0,997, ağırlık 7,92 GiB ve soğuk
batch-1 gecikmesi 56,1 ms (bf16'da 14,11 GiB ve 92,4 ms). 60k çiftle eğitilen LoRA, önceden yazılan kurala göre v0.3 olmuyor: karar
setinde 20k'yla berabere (NDCG@10 83,12'ye karşı 82,95, p 0,64), MCP-Zero'da 1,5 puan geride (p 0,0001); 0.3.0
`v0.2` ile çıkıyor. Hub'daki `v0.2` yerel 20k'yla bayt bayt aynı. 20k tarifinin yalnız eğitim seed'i değişmiş
tekrarı da `v0.2`'den MCP-Zero'da 0,7, ToolRet cat-macro'da 1,5 puan geride (ikisi de p 0,0001): sorgu düzeyindeki
p, seed'in getirdiği oynamayı ölçmüyor; bundan sonraki LoRA kararları bu tabana göre veriliyor.
Varsayılan backbone'un Q4_K_M GGUF'u, 4 GB'lık bir dizüstü GPU'sunda Ollama'yla vLLM'deki bf16 kadar iyi: ToolRet
NDCG@10 59,50'ye karşı 58,90, LiveMCPBench 55,75'e karşı 55,74. Bu dizüstünde yeni bir arama 0,6–0,7 sn sürüyor;
Qwen3-Embedding 0.6B ve 4B 2–14 kat hızlı ama 6–11 puan geride. `ToolrankToolSelector`, LangChain 1.x'in
`create_agent`'ına takılan bir middleware. 1.862 tool'luk katalogda iki görevin ikisinde de model yalnız 2 tool
gördü, doğru tool aralarındaydı; tur başına tek arama yapıldı ve çağrı o aramaya bağlandı. toolrank skill'iyle
yalnız kabuğu olan bir ajan aynı katalogda doğru tool'u bulup çağırdı.

## Ölçüler, birimler ve yön

`gösterilen tool ↓ adet`: bir model çağrısında modele bağlanan tool sayısı. `arama / tur ↓`: bir kullanıcı
mesajında toolrank'a gidilen arama sayısı. `doğru ↑`: görevin tool'u gösterildi ve çağrısı `ok` bitti.
`süre ↓ s`: ajan kurulumu hariç, bir görevin baştan sona süresi. `NDCG@10 ↑ %`, `Recall@k ↑ %`, `cat-macro ↑ %`:
README tablosundaki gibi, talimatlı (w/ inst). `soğuk ↓ ms`: sunucunun daha önce görmediği bir isteğin aranması,
embedding dahil, istekler tek tek (batch 1); `ılık ↓ ms`: aynı istek embedding cache'inden. `kodlama ↓`:
kataloğun ilk kez gömülmesi. `GGUF ↓ GB`: model dosyasının boyutu.

## D1.1 — 60k LoRA koşusu, v0.2'ye (20k) karşı

İkinci koşu (`qwen3-emb-lora-60k-r2`) 6 Ekim 06:33'te baştan başladı, 7 Ekim 09:44'te bitti: 60.000
ToolRet-train çifti, 1.875 adım, `--data-seed 0`, seçim `dev_w3_multi2`'de (karar setiyle aynı, kural böyle
yazıldı). Seçimdeki dev eğrisi (`dev_w3_multi2`, NDCG@10, transformers'la, eğitim sırasında):

| adım | 0 | 300 | 600 | **900** | 1.200 | 1.500 | 1.800 | 1.875 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NDCG@10 ↑ % | 70,35 | 82,99 | 83,10 | **83,13** | 83,05 | 82,21 | 82,94 | 82,85 |

En iyi adım 900 (yaklaşık 28.800 çift); sonraki 30 bin çift bu sette bir şey katmadı. Birleştirilmiş ağırlıklar
8097'de `qwen3-emb-lora-60k` adıyla, 20k'nın satırlarıyla aynı bayraklarla ölçüldü (documentation + instruct_query,
w/ inst, `--truncate 8192`); 20k aynı önbellekten yeniden puanlandı (kodlanan token 0), sayıları eskisiyle aynı.

| set | n | 20k NDCG@10 ↑ % | 60k NDCG@10 ↑ % | fark | p | 20k P@1 ↑ % | 60k P@1 ↑ % | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev_w3_multi2 (karar) | 800 | 82,95 | 83,12 | +0,17 | 0,64 | 80,75 | 80,12 | 0,65 |
| dev_w3_multi3 | 498 | 75,40 | 74,66 | −0,74 | 0,13 | 74,70 | 72,49 | 0,24 |
| dev_w3 | 1.000 | 92,49 | 92,91 | +0,42 | 0,15 | 83,50 | 84,20 | 0,40 |
| dev_w3_sit | 598 | 87,71 | 87,45 | −0,26 | 0,63 | 76,25 | 75,75 | 0,79 |
| LiveMCPBench `_server` | 94 | 55,74 | 54,96 | −0,78 | 0,50 | 58,51 | 53,19 | 0,23 |
| MCP-Zero `_server` | 2.792 | 93,67 | 92,14 | **−1,53** | 0,0001 | 88,57 | 85,89 | <0,0001 |
| ToolRet | 7.961 | 58,90 | 58,86 | −0,04 | 0,84 | 51,29 | 52,08 | 0,05 |

p: NDCG@10 için eşli permütasyon testi, P@1 için kesin işaret testi (`toolrank compare --paired`). ToolRet
cat-macro 54,36 → 52,31 (−2,05, p 0,0001; aşağıdaki ekteki ağırlıklı permütasyon testiyle).

**Karar (kural 3 Ekim'de, ilk koşudan önce yazıldı): v0.2 kalıyor.** Kural: 60k karar setinde 20k'yı geçer ve
hiçbir sette 1 puandan fazla kaybetmezse v0.3 olur. Karar setinde fark gürültü içinde (+0,17, p 0,64), MCP-Zero'da
−1,53 (p 0,0001). MCP-Zero 20k'nın seçim setiydi, yani orada 20k lehine bir yanlılık var; kural yine de değişmedi
ve 60k başka hiçbir sette anlamlı kazanmıyor. FP8 ölçümü yalnız kazanana yapılacaktı, yapılmadı. Bu yüzden
0.3.0'ın backbone'u `v0.2` (`yasinyaman/toolrank-emb-8b@v0.2`), servis adı `toolrank-emb-v0.2`.

**Ek (8 Ekim): seed replikası, gürültü tabanı.** 20k tarifi başka bir eğitim seed'iyle yeniden koşuldu
(`qwen3-emb-lora-20k-r1`: `--seed 1 --data-seed 0`, seçim `dev_w3_multi2`'nin seçim yarısında). Veri seed'i ve
düşülen çiftler (ToolRet'in 776'sı) aynı olduğundan 20 bin çift `v0.2`'ninkilerle aynı; iki koşu da son adımı (625)
seçti. Yani `v0.2` ile replika arasındaki tek fark LoRA'nın başlangıcı ve batch sırası. ToolRet kategorileri için
eşli test (`toolrank compare --paired --data data/toolret`, bu ekle geldi): sorguların NDCG@10 farkları, her sorgu
1/(görevinin boyu × kategorisinin görev sayısı) ağırlıkla, işaretler rastgele çevrilerek (10.000 permütasyon, en küçük
p 0,0001); ağırlıklı ortalamalar raporlardaki kategori sayılarını birebir veriyor.

| fark | MCP-Zero NDCG@10 | p | ToolRet cat-macro | p | code | p | web | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed: `v0.2` → replika | −0,74 | 0,0001 | −1,49 | 0,0001 | −3,05 | 0,0001 | −1,81 | 0,0006 |
| veri: replika → 60k | −0,79 | 0,0001 | −0,56 | 0,12 | −1,91 | 0,001 | +0,02 | 0,98 |
| `v0.2` → 60k (yukarıdaki) | −1,53 | 0,0001 | −2,04 | 0,0001 | −4,97 | 0,0001 | −1,80 | 0,005 |

Dev setlerinde ve LiveMCPBench'te replika `v0.2`'den ±0,7 içinde, hiçbiri anlamlı değil (`dev_w3_multi2` 82,77,
p 0,63). Sonuç: yalnız seed'i değiştirmek, MCP-Zero'da ve ToolRet'in code/web kategorilerinde 60k'nın kaybı
kadar fark yaratıyor; sorgu düzeyindeki p bu farkı yakalamıyor, çünkü seed'in getirdiği oynamayı ölçmüyor. Karar
değişmiyor (60k hiçbir sette kazanmıyor), ama "60k code kategorisini bozdu" bulgusu tek koşuyla kurulamaz:
replikaya karşı kalan −1,91 de tek seed çiftinin farkı. MCP-Zero'daki "seçim seti yanlılığı" da küçük: iki koşu da
son adımı seçti. Sonraki LoRA karşılaştırmaları (Dalga 3'ün kolları) aynı seed'li replikaya karşı yapılıyor ve
bir etki ancak bu tablodaki seed farkını aşarsa sayılıyor.

**Hub'daki `v0.2` = yerel 20k.** Hub'daki `v0.2` revizyonunun (commit `c89f13b`) dosyaları, yerel
`qwen3-emb-lora-20k/merged`'le karşılaştırıldı: `model.safetensors` SHA-256 `53789cff…7f42`, `tokenizer.json`
`83cdf8c3…eb8d`, config, tokenizer config, vocab, merges, modules ve pooling dosyalarının git blob kimlikleri
aynı. Yerel 20k bu sette servis edilip ölçüldüğü için Hub'daki revizyon da aynı vektörleri verir. Bu, servis
edip kosinüs bakmaktan daha sıkı bir kanıt. Hub'da parçalı ağırlık ya da `model.safetensors.index.json` yok.

## D1.2 ve D1.3 — dizüstünde küçük backbone'lar ve GGUF

GB10'a erişilemediği için ikisi de 4 GB'lık bir dizüstü GPU'sunda ölçüldü (RTX 3050 Ti Laptop, 14 GB RAM),
Ollama 0.35.1 ile. Modeller GGUF:

- Qwen3-Embedding 0.6B (f16, Q8_0), 4B (Q4_K_M, Q8_0) ve 8B (Q4_K_M, Q8_0): Qwen'in kendi GGUF depolarından.
- `toolrank-emb-v0.2`: Hub'daki `yasinyaman/toolrank-emb-8b@v0.2`, llama.cpp'nin dönüştürücüsüyle önce f16'ya,
  sonra Q8_0 ve Q4_K_M'ye (imatrix'siz).

Her model Ollama'ya `num_ctx 8192` ile girdi. Ollama vLLM'in `truncate_prompt_tokens`'ını yok sayıyor, uzun bir
girdiyi `num_ctx`'te kesiyor ve vLLM gibi başını tutuyor; yani 8192 token'lık sınır iki yanda aynı. Değerlendirme
README satırlarıyla aynı: dense, `documentation` + `instruct_query`, talimatlı, tam katalog üstünde top-100.
Değerlendirmeler dizüstünde koştu, Ollama'ya kendi Tailscale adresinden gidildi (`scripts/gguf_matrix.sh`).

### LiveMCPBench (94 istek, 525 tool)

| Model | GGUF ↓ GB | NDCG@10 ↑ % | Recall@5 ↑ % | Recall@10 ↑ % | 525 tool'un kodlanması ↓ dk |
| --- | ---: | ---: | ---: | ---: | ---: |
| *vLLM bf16, Qwen3-Embedding-8B* | | 53,74 | 50,82 | 61,09 | |
| *vLLM bf16, `toolrank-emb-v0.2`* | | 55,74 | 52,06 | 63,34 | |
| Qwen3-Embedding-0.6B f16 | 1,2 | 49,12 | 47,03 | 55,47 | 0,9 |
| Qwen3-Embedding-0.6B Q8_0 | 0,6 | 49,21 | 47,39 | 55,39 | 0,5 |
| Qwen3-Embedding-4B Q4_K_M | 2,5 | 48,71 | 45,79 | 57,71 | 2,8 |
| Qwen3-Embedding-4B Q8_0 | 4,3 | 49,39 | 45,63 | 56,86 | 4,4 |
| Qwen3-Embedding-8B Q4_K_M | 4,7 | 54,42 | 49,49 | 62,44 | 5,4 |
| Qwen3-Embedding-8B Q8_0 | 8,1 | 53,62 | 50,70 | 61,00 | 8,7 |
| `toolrank-emb-v0.2` Q4_K_M | 5,0 | 55,75 | 52,22 | 62,59 | 5,4 |
| `toolrank-emb-v0.2` Q8_0 | 8,7 | 55,23 | 52,17 | 63,34 | 8,8 |

### ToolRet (7.961 istek, 44.453 tool)

8B'nin ToolRet kataloğunu kodlaması bu dizüstünde 7,3 saat sürdüğü için üç model koşuldu.

| Model | NDCG@10 ↑ % | cat-macro ↑ % | Recall@10 ↑ % | Recall@20 ↑ % | 44.453 tool'un kodlanması ↓ sa |
| --- | ---: | ---: | ---: | ---: | ---: |
| *vLLM bf16, Qwen3-Embedding-8B* | 51,11 | 46,54 | 62,32 | 69,45 | — |
| *vLLM bf16, `toolrank-emb-v0.2`, GB10* | 58,90 | 54,36 | 69,54 | 75,25 | 0,4 |
| Qwen3-Embedding-0.6B f16 | 48,54 | 43,14 | 59,41 | 66,35 | 1,0 |
| Qwen3-Embedding-4B Q4_K_M | 50,41 | 44,99 | 61,80 | 68,56 | 3,6 |
| `toolrank-emb-v0.2` Q4_K_M | 59,50 | 54,51 | 69,85 | 75,67 | 7,3 |

### Yayımlanan yapılar (11 Eki 2026)

Dosyalar dizüstünden taşınmak yerine GB10'da, ağırlıkların yanında yeniden üretildi (kullanıcının kararı) ve
oradan Hub'a yüklendi. Bu sırada **llama.cpp'nin kuantizasyonunun 3 Ekim'den beri değiştiği** görüldü: aynı
tarif, aynı ağırlıklar, ama üç dosyada da metaveri 704 bayt kısa *ve* tensör yükü farklı (son 5 GB'ın sha256'sı
tutmuyor). Bu yüzden yayımlanan baytlar GB10'da Ollama ile yeniden ölçüldü — ölçülmemiş bir dosyayı kartta
ölçülmüş sayılarla sunmamak için:

| Yapı | LiveMCPBench NDCG@10 | Recall@5 | ToolRet NDCG@10 | cat-macro |
| --- | ---: | ---: | ---: | ---: |
| *vLLM bf16, `toolrank-emb-v0.2`* | 55,74 | 52,06 | 58,90 | 54,36 |
| Q4_K_M, 3 Eki yapısı (dizüstü) | 55,75 | 52,22 | 59,50 | 54,51 |
| **Q4_K_M, yayımlanan** | **56,22** | 52,22 | **59,51** | **54,53** |
| **Q8_0, yayımlanan** | 55,41 | 52,17 | **58,94** | **54,38** |

İki yapı arasındaki fark yarım puanın altında, yani kuantizasyonun değişmesi sıralamayı değiştirmiyor. Depo:
`yasinyaman/toolrank-emb-8b-GGUF@v0.2`, kart `docs/backbone/MODEL_CARD_GGUF.md`, yükleme GB10'dan
(`088af90`, kart `1cb12ac3`).

**Ama D1.3'ün "Q4_K_M önerilir" kararı değişti.** Dizüstünün erişemediği MCP-Zero (2.792 ajan isteği) GB10'da
koşulunca Q4_K_M'in bir bedeli göründü; eşli testler (`compare --paired`, aynı sorgular):

| karşılaştırma | P@1 | NDCG@10 | p (P@1 / NDCG@10) |
| --- | ---: | ---: | --- |
| MCP-Zero, Q4_K_M − bf16 | **−1,50** (9 yukarı, 51 aşağı) | −0,74 | <0,0001 / 0,0001 |
| MCP-Zero, Q8_0 − bf16 | +0,14 (11 yukarı, 7 aşağı) | +0,09 | 0,48 / 0,13 |
| MCP-Zero, Q4_K_M − Q8_0 | −1,65 | −0,83 | <0,0001 / 0,0001 |
| ToolRet, Q4_K_M − bf16 | — | +0,61 (cat-macro +0,17) | 0,0001 / 0,36 |
| ToolRet, Q8_0 − bf16 | −0,04 | +0,04 (cat-macro +0,03) | 0,82 / 0,31 (0,81) |
| `dev_w3` (tek tool), Q4_K_M − bf16 | +0,50 | +0,07 | 0,50 / 0,77 |
| `dev_w3_sit` (problem anlatan), Q4_K_M − bf16 | −1,67 | −0,66 | 0,064 / 0,068 |
| `dev_w3_multi2` (iki tool), Q4_K_M − bf16 | −1,25 | **−0,73** | 0,17 / **0,0017** |
| `dev_w3_multi3` (üç tool), Q4_K_M − bf16 | −1,61 | **−0,64** | 0,22 / **0,035** |

Yani Q8_0 ölçülen her sette bf16 seviyesinde; Q4_K_M MCP-Zero'nun iki satırlık isteklerinde top-1'de 1,5 puan
kaybediyor, w3 kataloğundaki çok tool'lu görevlerde 0,7 puan (tek tool'lu isteklerde fark yok), ToolRet'in
talimatlı ve uzun dokümantasyonlu isteklerinde ise 0,6 puan kazanıyor (kategori ortalamasında fark yok).
Kaba ağırlıkların bedeli kısa istekte ve birden çok tool gerektiren görevde çıkıyor — ajanın yazdığı istek de
kısa. **D1.3'ün kuralı ("Q8_0 her sette 1 puan içinde") ilk kez harfiyen sağlandı:** Q8_0 üç sette de bf16'dan
ayırt edilemiyor (ToolRet +0,04 p 0,31, MCP-Zero top-1 +0,14 p 0,48, LiveMCPBench −0,33 / 94 görev). Q8_0'ın
dev setleri 12 Eki 00:04'te koşmaya başladı.
**Yeni öneri: 9 GB sığıyorsa Q8_0, sığmıyorsa Q4_K_M** (kart ve `docs/guides/local.md` bunu yazıyor). Bu, 3 Eki'de
94 görevlik LiveMCPBench ve ToolRet'e bakarak verilen kararın neden yetersiz olduğunu da gösteriyor: kuantizasyon
farkını görmek için büyük ve kısa istekli bir set gerekiyordu. Q8_0'ın ToolRet satırı (D1.3'ün kuralının hiç
koşulmamış parçası) 11 Eki gecesi koşuyor.

### Gecikme, istekler tek tek

`scripts/latency.py`: LiveMCPBench'te model başına 50, ToolRet'te 100 istek.

| Model | Set | soğuk p50 ↓ ms | soğuk p95 ↓ ms | ılık p50 ↓ ms |
| --- | --- | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B f16 | LiveMCPBench | 74 | 167 | 6,0 |
| Qwen3-Embedding-0.6B Q8_0 | LiveMCPBench | 46 | 133 | 3,0 |
| Qwen3-Embedding-4B Q4_K_M | LiveMCPBench | 326 | 381 | 6,0 |
| Qwen3-Embedding-4B Q8_0 | LiveMCPBench | 412 | 509 | 3,0 |
| Qwen3-Embedding-8B Q4_K_M | LiveMCPBench | 657 | 719 | 3,0 |
| Qwen3-Embedding-8B Q8_0 | LiveMCPBench | 790 | 994 | 6,0 |
| `toolrank-emb-v0.2` Q4_K_M | LiveMCPBench | 668 | 707 | 6,0 |
| `toolrank-emb-v0.2` Q8_0 | LiveMCPBench | 780 | 948 | 3,0 |
| Qwen3-Embedding-0.6B f16 | ToolRet | 93 | 191 | 6,0 |
| Qwen3-Embedding-4B Q4_K_M | ToolRet | 275 | 388 | 9,0 |
| `toolrank-emb-v0.2` Q4_K_M | ToolRet | 571 | 694 | 13,4 |

### Ne çıktı

- **D1.3: kuantizasyon ölçülen her yerde kalite kaybettirmiyor.** `toolrank-emb-v0.2`'nin Q4_K_M'i LiveMCPBench'te
  bf16'yla aynı (+0,01), ToolRet'te 0,60 üstünde (cat-macro +0,15; code −0,13, customized +1,05, web −0,44).
  Q8_0 LiveMCPBench'te 0,51 geride, 94 isteğin gürültüsü içinde. Temel Qwen3-Embedding-8B'de de aynı: Q4_K_M
  +0,68, Q8_0 −0,12. GGUF yolu Mac ve CPU kurulumu olarak belgelendi (`docs/guides/local.md`); önerilen yapı
  Q4_K_M: daha küçük (5,0'a karşı 8,7 GB), daha hızlı (soğuk p50 668'e karşı 780 ms) ve aynı kalitede.
- **D1.2: küçük modeller varsayılanın yerini tutmuyor.** 0.6B ve 4B, LiveMCPBench'te `toolrank-emb-v0.2`
  Q4_K_M'in 6,4–7,0 puan, ToolRet'te 9,1–11,0 puan gerisinde (cat-macro 9,5–11,4). ToolRet'te temel 8B'ye
  yakınlar (0,7–2,6 puan geride): fark çoğunlukla LoRA'dan geliyor. 4B ToolRet'te 0.6B'nin 1,9 puan önünde,
  LiveMCPBench'te ikisi aynı yerde (48,7–49,4).
- **Gecikme.** Bu dizüstünde yeni bir arama 0.6B ile 46–93 ms (p95 133–191), 4B Q4_K_M ile 275–326 ms (p95
  381–388), 8B Q4_K_M ile 571–668 ms (p95 694–719) sürüyor. Daha önce görülen bir istek her modelde 3–13 ms:
  sorgu embedding cache'inden geliyor, kalan iş CPU'da sıralama. 300 ms'nin altında bir p95 isteyen bir kurulum
  için bu donanımda yalnız 0.6B yetiyor.
- **Kodlama.** ToolRet'in 44.453 tool'u dizüstünde 0.6B ile 1,0 saat, 4B Q4_K_M ile 3,6 saat, 8B Q4_K_M ile 7,3
  saat; GB10'da vLLM bf16 ile 0,4 saatti. Dakikada kabaca 740, 205 ve 102 tool. 8B Q4_K_M yüklendiğinde 7,6 GB
  tutuyor, bunun 2,6 GB'ı GPU'da, gerisi CPU'da.
- **`toolrank-vllm` imajı** dizüstünde x86_64'te de derlendi ve duman testinden geçti (sahte vLLM'le giriş
  noktası, kim neyi hangi kullanıcıyla koşturuyor); şimdiye kadar yalnız arm64'te (GB10) denenmişti.

## D1.4 — LangChain 1.x middleware'i

`toolrank.integrations.langchain.ToolrankToolSelector` (`[langchain]` ekstrası) bir `AgentMiddleware`.
`wrap_model_call` / `awrap_model_call` içinde `request.tools`'u son kullanıcı mesajına göre daraltıyor.
İki kullanım şekli var:

- **Ajanın kendi tool'ları.** Herhangi bir LangChain tool listesi `/v1/rank` ile sıralanıyor ve uyarlanabilir
  K ile kesiliyor. LangChain'in `LLMToolSelectorMiddleware`'iyle aynı yerde duruyor, ama ek bir model çağrısı
  yapmıyor.
- **toolrank kataloğu (`toolbox=`).** Katalog tool'ları sunucunun indeksinde `/v1/search` ile seçiliyor.
  `Toolbox`'ın tool'larıyla yapılan çağrılar bu aramaya `search_id` ile bağlanıyor; `toolrank learn` bu
  günlüğü okuyor. Ajanın katalog dışı tool'ları olduğu gibi gösteriliyor.

Her iki şekilde de şunlar hep gösteriliyor:

- konuşmada zaten çağrılmış tool'lar;
- `tool_choice`'un adını verdiği tool;
- `always_include`;
- sağlayıcı tool'ları (dict).

Seçim şu durumlarda yapılmıyor:

- seçilecek tool sayısı `min_tools`'u (20) geçmiyorsa;
- bir tool sağlayıcının kendi tool aramasına bırakılmışsa (`extras["defer_loading"]`);
- kullanıcı metni yoksa;
- toolrank hata verirse ya da `timeout_s` (5 sn) içinde cevap vermezse.

Süre aşılırsa seçim arka planda bitiyor ve cache'e giriyor; sonraki model çağrısı onu kullanıyor. Bir turdaki
model çağrıları aynı seçimi `ttl_s` boyunca cache'ten alıyor. Seçim işi bir havuz iş parçacığında
`contextvars.copy_context()` ile koşuyor; LangGraph thread'inin kimliği bir context değişkeni, onsuz arama
oturumsuz kalırdı.

**Uçtan uca** (`scripts/frameworks_e2e.py --only langchain`): `data/w3` kataloğu (1.862 tool), Qwen3-Embedding-8B
ve v0.1 head'leri. Model betikli: gösterilen tool'lar arasında görevin tool'u varsa onu çağırıyor, sonra cevap
veriyor. Görevlerden biri senkron, öbürü asenkron koşuldu.

| Yol | Görev | Ajanın tool'u | gösterilen tool ↓ | arama / tur ↓ | model çağrısı | doğru ↑ | süre ↓ s |
| --- | --- | ---: | ---: | ---: | ---: | :---: | ---: |
| katalog (`toolbox=`) | Tokyo'da saat (senkron) | 1.862 | 2 | 1 | 2 | ✓ | 0,83 |
| katalog (`toolbox=`) | `dog` etiketli evcil hayvanlar (asenkron) | 1.862 | 2 | 1 | 2 | ✓ | 0,04 |
| kendi tool'ları (`/v1/rank`) | Tokyo'da saat (senkron) | 120 | 120 (seçim atlandı) | — | 2 | ✓ | 60,08 |
| kendi tool'ları (`/v1/rank`) | `dog` etiketli evcil hayvanlar (asenkron) | 120 | 120 (seçim atlandı) | — | 2 | ✓ | 60,11 |

- **Katalog yolu.** Gösterilen tool'lar `time__get_current_time` + `time__convert_time` ve
  `swagger-petstore__findPets` + `addPet`. Turun ikinci model çağrısı cache'ten geldi. Kullanım günlüğünde iki
  çağrı da `link: search_id` ile LangGraph thread'inin oturumunda (`rest:langgraph:e2e-langchain-toolbox-N`).
  Ajanı 1.862 tool'la kurmak 0,55 sn sürdü.
- **Kendi tool'ları yolu, ilk koşu.** 120 tool'un LangChain'in gönderdiği biçimdeki metni cache'te yoktu ve
  embedding sunucusu (GB10) kapalıydı. Middleware tasarlandığı gibi davrandı: `/v1/rank` 30 sn'de zaman aşımına
  düştü, model her iki çağrıda da 120 tool'u gördü, görev yine doğru bitti.
- **İkinci koşu, dizüstündeki Ollama'yla** (Qwen3-Embedding-0.6B f16, head'siz: `--heads none` → `serve
  --clm-ckpt none`; `w3` önce cache'e gömüldü, 1.862 tool 642 bin token, 329 sn). İki yol da gerçek embedding'le
  çalıştı:

| Yol | Görev | Ajanın tool'u | gösterilen tool ↓ | arama / tur ↓ | doğru ↑ | süre ↓ s |
| --- | --- | ---: | ---: | ---: | :---: | ---: |
| katalog (`toolbox=`) | Tokyo'da saat (senkron) | 1.862 | 2 | 1 | ✓ | 2,61 |
| katalog (`toolbox=`) | `dog` etiketli evcil hayvanlar (asenkron) | 1.862 | 6 | 1 | ✓ | 0,39 |
| kendi tool'ları (`/v1/rank`) | Tokyo'da saat (senkron) | 120 | 2 | — | ✓ | 19,99 |
| kendi tool'ları (`/v1/rank`) | `dog` etiketli evcil hayvanlar (asenkron) | 120 | 4 | — | ✓ | 0,03 |

  - İlk sıralamanın 20 sn'si 120 tool metninin ilk kez gömülmesi; o sırada aynı GPU'da ToolRet de koşuyordu.
    Varsayılan `timeout_s` (5 sn) ile bu ilk model çağrısı bütün tool'larla gider, sonraki çağrı arka planda
    biten seçimi kullanır.
  - Kendi tool'ları yolunda çağrılar aramaya bağlanmıyor (`link: none`): `/v1/rank` günlüğe arama olarak
    yazılmıyor. `toolrank learn` için katalog yolu gerekiyor.
- **e2e betiklerinde hata.** `serve_e2e.py`, `platforms_e2e.py` ve `frameworks_e2e.py`'nin başlattığı
  `serve`'e `--emb-model` verilmiyordu. 0.2.0'dan beri sunucunun varsayılanı `toolrank-emb-v0.2` olduğundan,
  8091'e (`qwen3-emb`) yanlış model adı gidiyordu ve v0.1 head'lerinin cache'i de tutmuyordu. Üç betik artık
  `--emb-model` alıyor (varsayılan `qwen3-emb`).

## D1.5 — toolrank skill'i

`examples/skills/toolrank`, yalnız kabuğu olan ajanlar için (Claude Code skill'leri, bash-only harness'ler).
İçinde kısa bir `SKILL.md` (2.000 karakterin altında) ve yalnız standart kütüphaneyle yazılmış iki betik var:
`scripts/search.py` (`/v1/search`) ve `scripts/call.py` (`/v1/call`). Klasör tek başına kopyalanabiliyor.

- Betikler yönlendirmeyi izlemiyor; bearer token başka bir sunucuya gitmiyor.
- `call.py` çıkış kodunu ayırıyor: 1 tool'un kendi hatası, 2 toolrank'ın reddi ya da erişilemezlik.
- `--search-id` verilmezse sunucu çağrıyı aynı istemcinin son aramasına bağlıyor (User-Agent
  `toolrank-skill/1`).
- `tests/test_skill.py` betikleri ayrı süreç olarak, gerçek bir HTTP sunucusunun arkasındaki toolrank
  uygulamasına karşı koşuyor: arama, çağrı, stdin'den argüman, yazma reddi, 404, 401, yönlendirme,
  erişilemezlik.

**Elle deneme** (Mac, `data/w3`, 1.862 tool; Claude Code betikleri bir ajan gibi çalıştırdı):

- "What time is it in Tokyo right now?" araması `time/get_current_time`'ı ilk sıraya koydu (salt okunur).
- `call.py` ile `{"timezone": "Asia/Tokyo"}` saati döndürdü (çıkış 0); günlükte `link: search_id`.
- Geçersiz saat diliminde tool'un hatası ve sunucunun şema ipucu çıktıya geldi (çıkış 1); `--search-id`
  verilmediği için günlükte `link: client`.

## D1.6 — varsayılan yığının ölçümü (11 Eki 2026)

0.3.0'dan önceki soru: kullanıcının kurduğu yığın bizim ölçtüğümüz yığın mı? README'nin satırları NVIDIA'nın
NGC imajındaki vLLM 0.13'ten geliyor; Docker örnekleri ise vLLM'in resmi `vllm/vllm-openai:v0.30.0` imajını
taşıyor ve varsayılanı FP8 (`entrypoint-vllm.sh`). Ölçüm: `v0.2`'nin merged ağırlıkları o imajda FP8 olarak
8099'da (`qwen3-emb-lora-fp8-v030`), üç set NGC satırlarına karşı; ardından bf16 ile vektör uyumu, batch-1
gecikme ve ağırlık belleği. Hepsi talimatlı (w/ inst), `documentation` + `instruct_query`, head'siz.

| ölçü | NGC 0.13 bf16 (8097) | NGC 0.13 FP8 (8098) | resmi v0.30.0 FP8 (8099) |
| --- | ---: | ---: | ---: |
| ToolRet NDCG@10 | 58,90 | 59,02 | **58,95** |
| ToolRet cat-macro | 54,36 | 54,53 | **54,37** |
| LiveMCPBench NDCG@10 | 55,74 | 55,34 | **56,24** |
| MCP-Zero NDCG@10 | 93,67 | 93,26 | **93,46** |
| MCP-Zero top-1 | 88,57 | 87,71 | **88,07** |

Resmi imaj üç sette de NGC'nin bf16 ve FP8 satırlarının arasında ya da yarım puan içinde; LiveMCPBench'teki
+0,50 bir sorgunun ağırlığından küçük (94 istek). Vektör uyumu (aynı metinler, 8097 bf16 ↔ 8099 FP8):
tool'larda kosinüs ortalama 0,99739 (p1 0,99604, en kötü 0,99231), sorgularda 0,99711; 1.000 sorgunun ilk
10'unda örtüşme %95,6, ilk sırada %94,1 — Hafta 7'nin NGC içindeki FP8/bf16 karşılaştırmasıyla (0,997,
%95,4) aynı. Yani fark imajdan değil kuantizasyondan geliyor.

Gecikme ve bellek, 44.453 tool'luk ToolRet kataloğunda, istekler tek tek (`scripts/latency.py --n 100`):

| yığın | ağırlık ↓ GiB | soğuk p50 ↓ ms | soğuk p95 ↓ ms | ılık p50 ↓ ms |
| --- | ---: | ---: | ---: | ---: |
| `v0.2` bf16, NGC 0.13 (8097) | 14,11 | 92,4 | 108,5 | 7,5 |
| `v0.2` FP8, resmi v0.30.0 (8099) | 7,92 | 56,1 | 106,5 | 7,5 |

FP8 ağırlığı yarıya indiriyor, soğuk gecikmeden üçte birini alıyor. README'nin FP8 cümlesi bugüne kadar temel
modelin sayılarını (14,1 → 7,6 GiB, 98,5 → 54,9 ms; Faz 1 Hafta 6) anlatıyordu; artık varsayılan backbone'un
kendi sayıları yazıyor (`docs/results.toml`).

Leaderboard satırı (aynı oturumda, 8097): `v0.2`'nin ToolRet w/o inst koşusu — düz sorgular, talimat yok.

| model | NDCG@10 | cat-macro |
| --- | ---: | ---: |
| Qwen3-Embedding-8B (temel) | 42,73 | 35,28 |
| + v0.1 head'leri | 44,86 | 34,64 |
| `v0.2` | **46,63** | **36,41** |

ToolRet leaderboard'unun w/o inst sütununda NV-Embed-v1 35,50 ile öndeydi (Faz 1 Hafta 7); `v0.2` onu da
geçiyor. Gönderim 0.3.0'dan sonra.

Sapma: zincirin vektör uyumu adımı ilk denemede düştü. `scripts/fp8_agreement.py`'nin `--heads` varsayılanı
CLM'in referans head'lerini GPU'ya yüklüyor; GB10'da üç vLLM sunucusu ve başka bir projenin `llama-server`'ı
varken GPU belleği kalmadı (`torch.AcceleratorError: CUDA error: out of memory`). `--heads none` ile tekrar
koşuldu — `v0.2` head'siz servis edildiği için ölçümün kapsamı da bu.

## D1.7–D1.12 — Jev koltuğu, öğrenme ve eşli testler (kod)

Ölçüm değil kod maddeleri; davranışı testler sabitliyor, sayılar GB10'daki koşularla gelecek.

- **D1.12, `learn` varsayılan (v0.2) backbone'da.** Uyan head yoksa başlangıç kimlik (identity) head'leri:
  epoch 0 backbone'un kendi skoru. Kullanım günlüğü aramayı sunan modeli yazıyor, `learn` yalnız o modelin
  isteklerini kullanıyor (diğerleri sayılıp atlanıyor). Başlangıç bir sunucunun sunacağı head: önce
  `DATA/heads/current.npz` (`--tenant` ile kiracınınki), sonra uyan paketli head, yoksa kimlik. `/v1/rank`
  da aramanın head'leriyle sıralıyor.
- **D1.7, Jev koltuğu sağlayıcıya göre.** `TYPESAFE_API_KEY` yalnız `api.typesafe.ai`'ye gidiyor; başka bir
  `/systemone` ucu anahtarsız soruluyor, cevapları kendi `jev-<host>.sqlite` dosyasında (URL anahtarda).
  TypeSafe'in önbellek anahtarları değişmedi.
- **D1.8, Jev'in cevapladığı aramalar** ne `learn`'e, ne `ab` kararına, ne birlikte-kullanım tablosuna giriyor
  (MCA 2.3(b)); her biri atladığını sayıyor.
- **D1.9, `--replay`** dosyanın tamamından seed'li bir örnek (reservoir), yalnız pozitifler, `--dev`
  isteklerine dokunan çiftler atılmış.
- **D1.10, eşli testler.** `toolrank eval --runs-out` sorgu başına satır yazıyor, `toolrank compare --paired`
  iki koşuyu sorgu sorgu sınıyor (P@1 ve hit@5 için kesin işaret testi, NDCG@10 için eşli permütasyon testi).
  `finetune`, `learn` ve `scripts/lora_train.py` veri seed'ini eğitim seed'inden ayırıyor (`--data-seed`).
  D1.1–D1.3 satırlarının p değerleri GB10 koşularını bekliyor.
- **D1.11:** framework rehberinde Strands Agents bölümü (MCP üzerinden, adapter'sız).

5 Ekim'deki kod incelemesinin ölçümleri bozabilecek bulguları da bu dalgada kapandı:
- çalışan bir sunucu, aynı adla yeniden yazılan head dosyasını görmüyordu (önbellekteki hash yola göreydi);
- `scripts/learn_sim.sh` head'siz (`HEADS=none`) koşuda öğrenilen head'leri hiç puanlamıyordu;
- iki backbone da 4096 boyutlu olduğu için bir backbone'un head'leri sessizce ötekine takılabiliyordu;
- ikinci aşamalı aramalar günlüğe `model: null` yazıyor, `learn`'ün model filtresinden geçiyordu.

## Komutlar

```bash
# D1.1, GB10, ~/toolrank: the 60k run (user-started), then the chain (data/lora/chain_60k_r2.sh, untracked):
uv run --extra lora python scripts/lora_train.py --pairs data/toolret_train/pairs.jsonl --dev data/dev_w3_multi2 \
  --eval data/toolret --eval data/livemcpbench_server --eval data/mcp_zero_server --eval data/dev_w3 \
  --eval data/dev_w3_sit --eval data/dev_w3_multi3 --n-train 60000 --data-seed 0 --check-parity --keep-all \
  --out data/lora/qwen3-emb-lora-60k-r2
TOOLRANK_LORA=$HOME/toolrank/data/lora/qwen3-emb-lora-60k-r2/merged TOOLRANK_LORA_NAME=qwen3-emb-lora-60k \
  docker compose -f deploy/spark/compose.yaml --profile lora up -d --force-recreate qwen3-emb-lora
E="--scorer dense --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora-60k --truncate 8192 --emb-batch 128 \
  --tool-format documentation --query-format instruct_query --with-inst"
uv run toolrank eval --data data/dev_w3_multi2 $E --ks 5,10,20 --cut-margin 0.2 \
  --runs-out results/lora60k_dev_w3_multi2_runs.jsonl --out results/lora60k_dev_w3_multi2.json
# ... dev_w3_multi3 (--ks 5,10,20 --cut-margin 0.2), dev_w3 and dev_w3_sit (--ks 1,5,10), livemcpbench_server,
# mcp_zero_server (--ks 1,5,10,20), toolret; then the 20k as qwen3-emb-lora on 8097, the same flags, lora20k_*
# Mac, after scp 'gb10:toolrank/results/lora{20k,60k}_*' results/:
uv run toolrank compare --paired results/lora20k_mcp_zero_server_runs.jsonl results/lora60k_mcp_zero_server_runs.jsonl
# the seed replica (8 Oct), GB10: the 20k recipe with another training seed, picked on dev_w3_multi2's selection half
uv run toolrank data split data/dev_w3_multi2 --out-a data/dev_w3_multi2_sel --out-b data/dev_w3_multi2_conf
uv run --extra lora python scripts/lora_train.py --pairs data/toolret_train/pairs.jsonl --dev data/dev_w3_multi2_sel \
  --eval data/toolret --eval data/livemcpbench_server --eval data/mcp_zero_server --eval data/dev_w3 \
  --eval data/dev_w3_sit --eval data/dev_w3_multi3 --eval data/dev_w3_multi2_conf \
  --n-train 20000 --seed 1 --data-seed 0 --check-parity --keep-all --out data/lora/qwen3-emb-lora-20k-r1
# then served as qwen3-emb-lora-r1 and scored like the 60k above, lorar1_* (data/lora/chain_lora_arms.sh, untracked);
# Mac, ToolRet's categories from its queries (scp gb10:toolrank/data/toolret/queries.jsonl data/toolret/):
uv run toolrank compare --paired --data data/toolret results/lora20k_toolret_runs.jsonl results/lorar1_toolret_runs.jsonl
# the Hub's v0.2 against the local 20k: file list + LFS sha256 (HfApi.model_info(..., files_metadata=True)),
# sha256sum / git hash-object of data/lora/qwen3-emb-lora-20k/merged/*
# Mac; tool ve sorgu vektörleri data/w3/cache'ten (GB10 kapalıydı)
export GB10=<GB10'un Tailscale adresi>
uv run python scripts/frameworks_e2e.py --only langchain --data data/w3 --emb-url http://$GB10:8091/v1 \
  --heads dist/heads/toolrank-heads-qwen3-emb-8b-v0.1.npz --out results/frameworks_e2e_langchain.json
# the same with the laptop's Ollama (0.6B, no heads; w3 tool texts embedded into data/w3/cache first)
uv run python scripts/frameworks_e2e.py --only langchain --data data/w3 --emb-url http://$LAPTOP:11434/v1 \
  --emb-model qwen3-emb-0.6b-f16 --heads none --out results/frameworks_e2e_langchain_06b.json
uv run pytest tests/test_frameworks.py -k langchain
# D1.5: serve on the w3 catalogue, then the skill's scripts as an agent runs them
uv run toolrank serve --data data/w3 --config skill_trial.json --emb-url http://$GB10:8091/v1 --emb-model qwen3-emb --port 8766
cd examples/skills/toolrank && export TOOLRANK_URL=http://127.0.0.1:8766
python3 scripts/search.py "What time is it in Tokyo right now?"
python3 scripts/call.py time/get_current_time '{"timezone": "Asia/Tokyo"}' --search-id <search_id>
uv run pytest tests/test_skill.py
# D1.2 / D1.3, on the laptop from ~/toolrank (uv in ~/.local/bin, Ollama on the laptop's Tailscale address)
hf download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-f16.gguf Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir ~/models/qwen3-emb-0.6b-gguf
hf download Qwen/Qwen3-Embedding-4B-GGUF Qwen3-Embedding-4B-Q8_0.gguf Qwen3-Embedding-4B-Q4_K_M.gguf --local-dir ~/models/qwen3-emb-4b-gguf
hf download Qwen/Qwen3-Embedding-8B-GGUF Qwen3-Embedding-8B-Q8_0.gguf Qwen3-Embedding-8B-Q4_K_M.gguf --local-dir ~/models/qwen3-emb-8b-gguf
hf download yasinyaman/toolrank-emb-8b --revision v0.2 --local-dir ~/models/toolrank-emb-8b-v0.2
# the LoRA backbone as GGUF: f16, then Q8_0 and Q4_K_M, with llama.cpp's image (no imatrix)
RUN="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $HOME/models:/models ghcr.io/ggml-org/llama.cpp:full-cuda"
$RUN --convert /models/toolrank-emb-8b-v0.2 --outtype f16 --outfile /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-f16.gguf
$RUN --quantize /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-f16.gguf /models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-Q4_K_M.gguf Q4_K_M  # and Q8_0
# one Ollama model per build, each with num_ctx 8192 (names as in docs/guides/local.md)
printf 'FROM %s\nPARAMETER num_ctx 8192\n' ~/models/toolrank-emb-gguf/toolrank-emb-8b-v0.2-Q4_K_M.gguf > Modelfile
ollama create toolrank-emb-v0.2-q4_k_m -f Modelfile
export LAPTOP=<dizüstünün Tailscale adresi>; export EMB_URL=http://$LAPTOP:11434/v1 UV=~/.local/bin/uv
MODELS="qwen3-emb-0.6b-f16 qwen3-emb-0.6b-q8_0 qwen3-emb-4b-q4_k_m qwen3-emb-4b-q8_0" bash scripts/gguf_matrix.sh   # LiveMCPBench
MODELS="qwen3-emb-8b-q4_k_m toolrank-emb-v0.2-q4_k_m qwen3-emb-8b-q8_0 toolrank-emb-v0.2-q8_0" bash scripts/gguf_matrix.sh
SETS=toolret MODELS="qwen3-emb-0.6b-f16 qwen3-emb-4b-q4_k_m toolrank-emb-v0.2-q4_k_m" bash scripts/gguf_matrix.sh
$UV run python scripts/latency.py --data data/livemcpbench_server --emb-url $EMB_URL --emb-model <model> --n 50 \
  --scorer dense --truncate 8192 --tool-format documentation --query-format instruct_query   # ToolRet: --n 100

# D1.6, GB10: v0.2 in FP8 on the image the Docker examples ship, the entrypoint's own flags
docker run -d --name toolrank-v030-fp8 --gpus all --ipc host \
  -v $HOME/toolrank/data/lora/qwen3-emb-lora-20k/merged:/models/lora:ro \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -p 8099:8000 vllm/vllm-openai:v0.30.0 \
  /models/lora --served-model-name qwen3-emb-lora-fp8-v030 --runner pooling --max-model-len 8192 \
  --quantization fp8 --no-enable-chunked-prefill --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.2 --host 0.0.0.0 --port 8000
# then the three sets, the agreement and both latency runs (data/v030_chain.sh, untracked):
FLAGS="--emb-url http://127.0.0.1:8099/v1 --emb-model qwen3-emb-lora-fp8-v030 --truncate 8192 --emb-batch 128
  --tool-format documentation --query-format instruct_query --with-inst"
for d in livemcpbench_server mcp_zero_server toolret; do   # mcp_zero_server also --ks 1,5,10,20
  $UV run toolrank eval --data data/$d --scorer dense $FLAGS --out results/v030_${d}_fp8.json; done
$UV run python scripts/fp8_agreement.py --a http://127.0.0.1:8097/v1=qwen3-emb-lora \
  --b http://127.0.0.1:8099/v1=qwen3-emb-lora-fp8-v030 --truncate 8192 --tool-format documentation \
  --query-format instruct_query --heads none          # without --heads none it loads CLM's heads onto the GPU
$UV run python scripts/latency.py --data data/toolret --scorer dense --truncate 8192 \
  --tool-format documentation --n 100 --emb-url http://127.0.0.1:8099/v1 --emb-model qwen3-emb-lora-fp8-v030
$UV run python scripts/latency.py --data data/toolret --scorer dense --truncate 8192 \
  --tool-format documentation --n 100 --emb-url http://127.0.0.1:8097/v1 --emb-model qwen3-emb-lora
# the leaderboard's w/o inst row: no --with-inst, so the query format falls back to plain
$UV run toolrank eval --data data/toolret --scorer dense --emb-url http://127.0.0.1:8097/v1 \
  --emb-model qwen3-emb-lora --truncate 8192 --emb-batch 128 --tool-format documentation \
  --out results/readme_lora_toolret_qwen3emb_noinst.json
# Mac, after scp 'gb10:toolrank/results/{v030_*,readme_lora_toolret_qwen3emb_noinst}.json' results/:
uv run toolrank compare results/lora_toolret_lora.json results/lora_toolret_lora_fp8.json results/v030_toolret_fp8.json
uv run toolrank compare results/readme_toolret_qwen3emb_noinst.json results/readme_toolret_heads_noinst.json \
  results/readme_lora_toolret_qwen3emb_noinst.json
```

## Ortam

- Mac'te `toolrank serve` (`--emb-model qwen3-emb`, v0.1 head'leri), `data/w3` indeksi ve embedding cache'i sıcak.
- langchain 1.4.3, langchain-core 1.6.6, langgraph 1.2.12.
- Test paketi (D1.5 sonrası): 278 geçti, 1 atlandı; torch'suz 264 geçti, 13 atlandı.
- D1.6: GB10, 11 Eki 16:54–17:55. Aynı anda 8091 (temel model) ve 8097 (`v0.2` bf16) ayakta, reranker (8095)
  yer açmak için durduruldu; başka bir projenin `llama-server`'ı GPU'yu paylaşıyordu (ToolRet'in 44.453
  tool'u 43 dakikada kodlandı, kendi GPU'sunda 0,4 saat). Ölçümlerin hiçbiri paylaşılan GPU'dan etkilenmeyen
  büyüklükler değil: gecikme satırları bu yükle birlikte okunmalı, kalite satırları yükten bağımsız.
- D1.2 / D1.3: dizüstü NVIDIA RTX 3050 Ti Laptop (4 GB), 14 GB RAM, 20 iş parçacığı, Ubuntu 26.04 x86_64.
  Ollama 0.35.1, her modelde `num_ctx` 8192; dönüştürme `ghcr.io/ggml-org/llama.cpp:full-cuda` ile. Koşular
  3 Ekim 08:28–23:00 arası, sırayla. ToolRet'in ilk satırı koşarken dizüstünde D1.4'ün LangChain denemesi ve
  `toolrank-vllm` imajının derlemesi de çalıştı; kodlama süresini etkileyebilir, kaliteyi değil. Gecikme
  ölçümleri (22:54–23:00) tek başına koştu.

## Sapmalar ve açıklamalar

- **D1.1 ertelendi.** GB10'a 3 Ekim'de ne SSH ne 8091 üzerinden ulaşılabiliyordu (zaman aşımı); 5 Ekim
  akşamı döndü. 60k koşusu makine düştüğünde, 3 Ekim 02:11'de, 1.875 adımın 690'ında durmuş; 300. ve 600.
  adımın adaptörleri var, merge ve değerlendirme zinciri hiç çalışmadı. 600. adım yaklaşık 19.200 çift
  görmüş, 20k koşusunun 625 adımı kadar: "daha çok veri" sorusunu cevaplamıyor. Koşu kaldığı yerden devam
  edemiyor; ya baştan (yaklaşık 25 saat) ya da 0.3.0 `v0.2` ile.
- **D1.1'in birleştirilmiş klasörü.** `scripts/lora_train.py`, orijinal deponun config, tokenizer ve pooling
  dosyalarını `snapshot_download`'un döndürdüğü önbellek klasöründen bütünüyle kopyalıyordu. 1 Ekim'de önbellekte
  yalnız bu dosyalar vardı; 7 Ekim'de taban modelin ağırlıkları da önbellekteydi ve 60k'nın `merged` klasörüne
  Qwen3-Embedding-8B'nin dört parçası ve `model.safetensors.index.json`'u da kopyalandı. vLLM index'in listelediği
  parçaları yüklediği için 60k adıyla taban model servis edilecekti. Ölçümden önce fark edildi (parçaların
  SHA-256'sı önbellektekilerle aynı), dosyalar silinmeden yana taşındı ve betik yalnız listelenen dosyaları
  kopyalıyor (`069a1d0`). Değerlendirme zinciri, `dev_w3_multi2` NDCG@10 0,78'in altında kalırsa (taban model
  ~0,70) durmak üzere yazıldı; 0,8312 geldi, eğitimdeki 0,8313'le aynı.
- **GPU'lu maddeler.** D1.2 ve D1.3 GB10 yerine 4 GB'lık bir dizüstü GPU'sunda (RTX 3050 Ti Laptop, 14 GB RAM)
  yapıldı. D1.3'ün hedef donanımı zaten bu. ToolRet üç modelle koşuldu; MCP-Zero ve dev setleri dizüstünde yok,
  sorguları GB10'da üretildi. İkisi de GB10'u bekliyor.
- **D1.3'ün karar kuralı harfiyen tamamlanmadı.** Kural "Q8_0 her sette 1 puan içinde" diyor. Q8_0 yalnız
  LiveMCPBench'te ölçüldü (−0,51); ToolRet'te yalnız Q4_K_M koşuldu, MCP-Zero hiç ölçülmedi. Karar yine de
  verildi: Q4_K_M daha kayıplı kuantizasyon ve ölçüldüğü iki sette de bf16'nın gerisinde değil. MCP-Zero teyidi
  GB10'la.
- **Toplu gecikme sayıları yanıltıcı.** `toolrank eval`'in 64'lük partilerle verdiği sorgu başı p50, Ollama'nın
  modeli belleğe yüklediği süreyi de içerebiliyor; 0.6B f16 LiveMCPBench'te toplu 321 ms, tek tek 74 ms. Gecikme
  için yalnız `scripts/latency.py`'nin tek tek ölçümleri kullanıldı.
- **llama.cpp'nin kendi sunucusu.** `llama-server --embeddings` embedding modunda her girdi token'ı için bir çıktı
  satırı tutuyor, Qwen3'te token başına yaklaşık 0,6 MB; 8192 token'lık bir girdi yaklaşık 5 GB GPU belleği
  istiyor ve kesmeyi de yok sayıyor. Sığmayan uzun bir girdi sunucuyu hata döndürmek yerine çökertti. Ölçümler bu
  yüzden Ollama'yla yapıldı; Ollama belleğini planlıyor ve sığmayanı CPU'ya bırakıyor.

## Sonraki adımlar

- D1.2 ve D1.3'ün GB10 kısmı: MCP-Zero ve dev setleri, GGUF yapılarıyla ve küçük modellerle.
- GGUF'ların Hub'a yüklenmesi (`yasinyaman/toolrank-emb-8b-GGUF`), 0.3.0'la; herkese açık adım, ayrı onayla.
  Kılavuzdaki `TODO(launch)` işareti o güne kadar sürümü durduruyor.
- Apple Silicon'da Ollama ölçülmedi; kılavuz bunu söylüyor.
