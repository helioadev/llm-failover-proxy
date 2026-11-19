# llm-failover-proxy

A lightweight FastAPI proxy that routes OpenAI-compatible LLM requests across local inference nodes (`llama.cpp` / LM Studio) and cloud APIs with transport-level failover (`429`/`5xx`/timeouts) and telemetry logging.

## Quickstart

```bash
pip install -r requirements.txt
export LOCAL_LLM_URL="[http://127.0.0.1:8080/v1/chat/completions](http://127.0.0.1:8080/v1/chat/completions)"
export OPENAI_API_KEY="sk-..." # optional cloud fallback
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
