import os
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from router import Provider, route_with_failover, telemetry

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LLM Failover Proxy",
    description="Lightweight gateway routing hybrid local (llama.cpp / LM Studio) and cloud LLM APIs.",
    version="0.1.0",
)


def load_providers() -> list[Provider]:
    providers: list[Provider] = [
        Provider(
            name="local-llama-cpp",
            url=os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8080/v1/chat/completions"),
            timeout=15.0,
        )
    ]
    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            Provider(
                name="openai-fallback",
                url="https://api.openai.com/v1/chat/completions",
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=30.0,
            )
        )
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append(
            Provider(
                name="openrouter-fallback",
                url="https://openrouter.ai/api/v1/chat/completions",
                api_key=os.environ["OPENROUTER_API_KEY"],
                timeout=30.0,
            )
        )
    return providers


class ChatRequest(BaseModel):
    model: str = Field(default="qwen2.5-7b-instruct")
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "providers": [p.name for p in load_providers()]}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    return {"total_events": len(telemetry.events), "recent_events": telemetry.events[-20:]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest) -> dict[str, Any]:
    payload = req.model_dump(exclude_none=True)
    return await route_with_failover(payload, load_providers())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
