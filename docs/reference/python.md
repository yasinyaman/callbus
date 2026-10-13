# Python API

The client and the integrations are the stable Python surface. Everything else is reached through
the CLI.

::: callbus.client.CallBusClient

::: callbus.client.AsyncCallBusClient

::: callbus.client.CallBusError

## Claude tool search

::: callbus.integrations.anthropic.Toolbox

::: callbus.integrations.anthropic.run

## OpenAI tool search

::: callbus.integrations.openai.Toolbox

::: callbus.integrations.openai.run

## LangChain

`CallBusToolSelector` is built lazily, on first use (`langchain` is an optional extra), so the
module carries its documentation:

::: callbus.integrations.langchain
    options:
      members:
        - Selector

## LangGraph

::: callbus.integrations.langgraph.Toolbox

## LlamaIndex

::: callbus.integrations.llamaindex.CallBusToolRetriever

## LiteLLM

::: callbus.integrations.litellm.ToolFilter
