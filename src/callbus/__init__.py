"""CallBus: tool retrieval for LLM agents with hundreds of tools.

The package is organised as ports and adapters:

* ``callbus.domain``       - plain dataclasses (Tool, Query, RankedList, EvalReport).
* ``callbus.ports``        - Protocols every adapter implements (Scorer, TextEncoder, VectorIndex, ...).
* ``callbus.adapters``     - BM25, dense (any OpenAI-compatible embeddings endpoint), heads, indexes,
  the MCP proxy and REST API, MCP and OpenAPI backends.
* ``callbus.ingest``       - MCP servers and OpenAPI specs -> an ingest dir (``tools.jsonl``).
* ``callbus.retriever``    - query-time retrieval over an ingest dir, shared by search and serve.
* ``callbus.client``       - a standard-library client for ``callbus serve``'s REST API.
* ``callbus.integrations`` - Claude's and OpenAI's tool search, LangChain, LangGraph, LlamaIndex,
  LiteLLM.
* ``callbus.datasets``     - benchmark converters and the on-disk JSONL format.
* ``callbus.eval``         - trec_eval-compatible metrics and the benchmark runner.
"""

from callbus.domain import EvalReport, Query, RankedList, Tool

__all__ = ["EvalReport", "Query", "RankedList", "Tool", "__version__"]
__version__ = "0.3.1"
