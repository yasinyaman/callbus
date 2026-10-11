# toolrank-emb-8b v0.2, GGUF builds

[toolrank's default backbone](https://huggingface.co/yasinyaman/toolrank-emb-8b) as GGUF, for
llama.cpp and Ollama: tool retrieval without a datacentre GPU. A request and each tool become
vectors, and the tools closest to the request come first. Same weights, same instruction and same
tool text as the bf16 model — only the numeric format changes.

Code, documentation and benchmarks: [github.com/yasinyaman/toolrank](https://github.com/yasinyaman/toolrank),
[yaman.dev/toolrank](https://yaman.dev/toolrank/). The bf16 weights and their card:
[`yasinyaman/toolrank-emb-8b`](https://huggingface.co/yasinyaman/toolrank-emb-8b) at revision `v0.2`.

| File | Size | sha256 |
| --- | ---: | --- |
| `toolrank-emb-8b-v0.2-Q4_K_M.gguf` | 5.03 GB | `9f128fa253f58e686852d072bc76af7226263392bfbb38a7ad8a91e03d0754e0` |
| `toolrank-emb-8b-v0.2-Q8_0.gguf` | 8.71 GB | `2a121eda4319551a9a472c511df4c5d32bd66d175097562a12e89b33a862fae5` |

| | |
| --- | --- |
| Base | `Qwen/Qwen3-Embedding-8B` (Apache-2.0), with toolrank's LoRA merged in (`yasinyaman/toolrank-emb-8b@v0.2`) |
| Built with | llama.cpp's `convert_hf_to_gguf.py` to f16, then `llama-quantize`; no importance matrix |
| Dimensions | 4096, last-token pooling, L2-normalise before comparing |
| Context | 8192 tokens; set it explicitly, see below |
| License | Apache-2.0; see *Training data* on the bf16 model's card |

## Serving

```bash
hf download yasinyaman/toolrank-emb-8b-GGUF toolrank-emb-8b-v0.2-Q4_K_M.gguf --revision v0.2 --local-dir models
printf 'FROM ./models/toolrank-emb-8b-v0.2-Q4_K_M.gguf\nPARAMETER num_ctx 8192\n' > Modelfile
ollama create toolrank-emb-v0.2-q4_k_m -f Modelfile

export TOOLRANK_EMB_URL=http://127.0.0.1:11434/v1 TOOLRANK_EMB_MODEL=toolrank-emb-v0.2-q4_k_m
toolrank ingest mcp --server time="uvx mcp-server-time" --out tools/ --emb-url $TOOLRANK_EMB_URL
toolrank serve --data tools/
```

- Requests: `Instruct: Given an agent's request for a tool, retrieve the MCP tool that fulfills it.\nQuery: <request>`.
- Tools: toolrank's `documentation` text; for MCP and OpenAPI tools `{"server", "name", "description",
  "inputSchema"}` as JSON.
- `PARAMETER num_ctx 8192` matters: Ollama ignores a request's truncation length and cuts at
  `num_ctx` instead, keeping the head of the text as vLLM does. Leave it at the default 2048 and long
  tool documentation loses its tail silently.
- No heads. toolrank's v0.1 heads were trained on the base model and cost this one points; the
  packaged heads are switched off for these builds automatically.

## Numbers

toolrank's own evaluation, with instruction, `documentation` + `instruct_query`, top 100 over the
whole corpus — the same flags as the README's rows. `bf16` is the same weights served by vLLM.

**LiveMCPBench** (94 tasks, 525 tools; one task is worth about a point)

| Build | NDCG@10 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| bf16, vLLM | 55.74 | 52.06 | 63.34 |
| **Q4_K_M** | **56.22** | 52.22 | 62.77 |
| **Q8_0** | 55.41 | 52.17 | 62.80 |

**ToolRet** (7,961 requests, 44,453 tools)

| Build | NDCG@10 | cat-macro | Recall@10 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| bf16, vLLM | 58.90 | 54.36 | 69.54 | 75.25 |
| **Q4_K_M** | **59.51** | **54.53** | 69.88 | 75.61 |

So quantization costs nothing measurable here: Q4_K_M lands within half a point of bf16 on both
sets, in both directions, and Q8_0 does the same. Qwen3-Embedding's own smaller models are a
different story — 0.6B and 4B are 6–7 points behind on LiveMCPBench and 9–11 on ToolRet
([the report](https://github.com/yasinyaman/toolrank/blob/main/docs/reports/backlog-d1.md)).

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

A request the server has already seen is answered from toolrank's embedding cache in 3–13 ms, and a
catalogue is embedded once: `toolrank ingest --emb-url …` stores the vectors and a restart reuses
them. On 4 GB these 8B builds run partly on the CPU; the guide
[Running locally](https://yaman.dev/toolrank/guides/local/) has the details and the smaller models.

## Training data

The LoRA was trained on 20,000 pairs from `mangopy/ToolRet-Training-20w`, which states no license.
The bf16 model's card explains what that means and what was and was not trained on.
