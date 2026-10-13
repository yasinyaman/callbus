# callbus-emb-8b v0.2, GGUF builds

[callbus's default backbone](https://huggingface.co/yasinyaman/callbus-emb-8b) as GGUF, for
llama.cpp and Ollama: tool retrieval without a datacentre GPU. A request and each tool become
vectors, and the tools closest to the request come first. Same weights, same instruction and same
tool text as the bf16 model — only the numeric format changes.

Code, documentation and benchmarks: [github.com/yasinyaman/callbus](https://github.com/yasinyaman/callbus),
[yaman.dev/callbus](https://yaman.dev/callbus/). The bf16 weights and their card:
[`yasinyaman/callbus-emb-8b`](https://huggingface.co/yasinyaman/callbus-emb-8b) at revision `v0.2`.

| File | Size | sha256 |
| --- | ---: | --- |
| `callbus-emb-8b-v0.2-Q4_K_M.gguf` | 5.03 GB | `9f128fa253f58e686852d072bc76af7226263392bfbb38a7ad8a91e03d0754e0` |
| `callbus-emb-8b-v0.2-Q8_0.gguf` | 8.71 GB | `2a121eda4319551a9a472c511df4c5d32bd66d175097562a12e89b33a862fae5` |

| | |
| --- | --- |
| Base | `Qwen/Qwen3-Embedding-8B` (Apache-2.0), with callbus's LoRA merged in (`yasinyaman/callbus-emb-8b@v0.2`) |
| Built with | llama.cpp's `convert_hf_to_gguf.py` to f16, then `llama-quantize`; no importance matrix |
| Dimensions | 4096, last-token pooling, L2-normalise before comparing |
| Context | 8192 tokens; set it explicitly, see below |
| License | Apache-2.0; see *Training data* on the bf16 model's card |

## Serving

```bash
hf download yasinyaman/callbus-emb-8b-GGUF callbus-emb-8b-v0.2-Q4_K_M.gguf --revision v0.2 --local-dir models
printf 'FROM ./models/callbus-emb-8b-v0.2-Q4_K_M.gguf\nPARAMETER num_ctx 8192\n' > Modelfile
ollama create callbus-emb-v0.2-q4_k_m -f Modelfile

export CALLBUS_EMB_URL=http://127.0.0.1:11434/v1 CALLBUS_EMB_MODEL=callbus-emb-v0.2-q4_k_m
callbus ingest mcp --server time="uvx mcp-server-time" --out tools/ --emb-url $CALLBUS_EMB_URL
callbus serve --data tools/
```

- Requests: `Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: <request>`.
- Tools: callbus's `documentation` text; for MCP and OpenAPI tools `{"server", "name", "description",
  "inputSchema"}` as JSON.
- `PARAMETER num_ctx 8192` matters: Ollama ignores a request's truncation length and cuts at
  `num_ctx` instead, keeping the head of the text as vLLM does. Leave it at the default 2048 and long
  tool documentation loses its tail silently.
- No heads. callbus's v0.1 heads were trained on the base model and cost this one points; the
  packaged heads are switched off for these builds automatically.

## Numbers

callbus's own evaluation, with instruction, `documentation` + `instruct_query`, top 100 over the
whole corpus — the same flags as the README's rows. `bf16` is the same weights served by vLLM.

**LiveMCPBench** (94 tasks, 525 tools; one task is worth about a point)

| Build | NDCG@10 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| bf16, vLLM | 55.74 | 52.06 | 63.34 |
| **Q4_K_M** | **56.22** | 52.22 | 62.77 |
| **Q8_0** | 55.41 | 52.17 | 62.80 |

**MCP-Zero** (2,792 requests, 2,792 tools — one request per tool, agent-written, two lines each)

| Build | NDCG@10 | top-1 | Recall@5 |
| --- | ---: | ---: | ---: |
| bf16, vLLM | 93.67 | 88.57 | 96.53 |
| Q4_K_M | 92.93 | 87.07 | 96.33 |
| **Q8_0** | **93.76** | **88.72** | 96.65 |

**ToolRet** (7,961 requests, 44,453 tools)

| Build | NDCG@10 | cat-macro | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| bf16, vLLM | 58.90 | 54.36 | 69.54 | 75.25 |
| **Q4_K_M** | **59.51** | **54.53** | 69.88 | 75.61 |
| **Q8_0** | 58.94 | 54.38 | 69.62 | 75.33 |

**Which build.** Q8_0 is the one to take if 9 GB fits: it is level with bf16 on all three sets, with
nothing to separate them on paired tests (MCP-Zero top-1 +0.14, p 0.48 on 2,792 requests; ToolRet
NDCG@10 +0.04, p 0.31 on 7,961; LiveMCPBench −0.33 on 94 tasks).
Q4_K_M costs 1.5 points of top-1 there (87.07 against 88.57, 9 requests up and 51 down, p < 0.0001)
— the kind of short request an agent writes is where its coarser weights show. On ToolRet, whose
requests come with instructions and whose tools carry long documentation, Q4_K_M is instead 0.6
points above bf16 (p 0.0001) and level by category (+0.17, p 0.36). Generated tasks over a GitHub +
Stripe catalogue agree with MCP-Zero: Q4_K_M costs 0.7 NDCG@10 on tasks needing two or three tools
(p 0.002 and 0.03) and nothing on requests for a single tool (+0.07, p 0.77). So: 5 GB and a laptop,
take Q4_K_M; room for 9 GB, take Q8_0.

Qwen3-Embedding's own smaller models are a different story — 0.6B and 4B are 6–7 points behind on
LiveMCPBench and 9–11 on ToolRet
([the report](https://github.com/yasinyaman/callbus/blob/main/docs/reports/backlog-d1.md)).

Both tables are these files, scored where they were built. An earlier Q4_K_M of the same recipe,
made on 3 October with an older llama.cpp, scores the same (ToolRet 59.50, LiveMCPBench 55.75)
although its quantized payload is not byte-identical: llama.cpp's quantizer changed in between, the
ranking did not.

## What it costs on a small GPU

Measured on a laptop with a 4 GB NVIDIA RTX 3050 Ti and 14 GB of RAM, Ollama 0.35.1, one request at
a time and nothing cached:

| Build | a new search | p95 | a catalogue of 525 tools, embedded |
| --- | ---: | ---: | ---: |
| Q4_K_M | 571–668 ms | ~0.7 s | 5.4 min |
| Q8_0 | 780 ms | 948 ms | 8.8 min |

A request the server has already seen is answered from callbus's embedding cache in 3–13 ms, and a
catalogue is embedded once: `callbus ingest --emb-url …` stores the vectors and a restart reuses
them. On 4 GB these 8B builds run partly on the CPU; the guide
[Running locally](https://yaman.dev/callbus/guides/local/) has the details and the smaller models.

## Training data

The LoRA was trained on 20,000 pairs from `mangopy/ToolRet-Training-20w`, which states no license.
The bf16 model's card explains what that means and what was and was not trained on.
