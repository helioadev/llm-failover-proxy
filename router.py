import time
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import HTTPException

logger = logging.getLogger("llm-failover-proxy")


@dataclass
class Provider:
    name: str
    url: str
    api_key: str = ""
    timeout: float = 30.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def headers(self) -> dict[str, str]:
        base = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            base["Authorization"] = f"Bearer {self.api_key}"
        return base


class TelemetryStore:
    """In-memory telemetry store (to be backed by Postgres in production)."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, provider: str, status: str, status_code: int, latency_ms: float) -> None:
        event = {
            "timestamp": time.time(),
            "provider": provider,
            "status": status,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        }
        self.events.append(event)
        logger.info("event=%s", event)


telemetry = TelemetryStore()


async def route_with_failover(
    payload: dict[str, Any], providers: list[Provider]
) -> dict[str, Any]:
    """Forward a chat completion request across providers with transport-level failover."""
    for provider in providers:
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=provider.timeout) as client:
                res = await client.post(
                    provider.url, json=payload, headers=provider.headers
                )
                latency_ms = (time.perf_counter() - start) * 1000

                if res.status_code in (429, 500, 502, 503, 504):
                    telemetry.record(provider.name, "failover", res.status_code, latency_ms)
                    continue

                res.raise_for_status()
                telemetry.record(provider.name, "success", res.status_code, latency_ms)
                return {
                    "routed_provider": provider.name,
                    "latency_ms": round(latency_ms, 2),
                    "response": res.json(),
                }
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            telemetry.record(provider.name, f"error:{type(exc).__name__}", 0, latency_ms)
            continue

    raise HTTPException(status_code=503, detail="All configured LLM providers exhausted.")
