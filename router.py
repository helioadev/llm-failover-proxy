async def route_with_failover(payload: dict, providers: list[Provider]):
    for p in providers:
        try:
            async with httpx.AsyncClient(timeout=p.timeout) as client:
                res = await client.post(p.url, json=payload, headers=p.headers)
                if res.status_code in (429, 500, 502, 503, 504):
                    await log_event(p.name, "failover", res.status_code)
                    continue
                return res.json()
        except (httpx.TimeoutException, httpx.ConnectError):
            await log_event(p.name, "timeout", 0)
            continue
    raise HTTPException(status_code=503, detail="All providers exhausted")
