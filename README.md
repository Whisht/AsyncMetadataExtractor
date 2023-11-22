# AsyncExtractorForLlamaIndex
Custom Metadata Extractor with `httpx.AsyncClient`

## Requirements
Please note that this repo is written with `llama_index` **0.8.59** and `openai-python` **0.8.21**, but the current version are [0.9.5](https://github.com/run-llama/llama_index) and [1.3.4](https://github.com/openai/openai-python). For `llama_index`, they did a lot of modification of the `pipeline` of `MetadatExtractor`. Meanwhile, there is a huge gap between `openai 0.8.21` and `1.3.4`, especially the replication of `atiohttp` to `httpx`. So, it may conflict with the newest version. Please note that again.

Meanwhile, it is from a custom llm with an async HTTP request to my own `api`. You need to reconfig the access mechanism to the one suited for your LLM or just use the `OpenAI` llm with [`from llama_index.llms import OpenAI`](https://docs.llamaindex.ai/en/v0.8.39/examples/metadata_extraction/MetadataExtractionSEC.html).
```
llama_index == 0.8.59
openai == 0.8.21
```
