"""Show today's client-side API usage against the configured daily limits.

    python -m scripts.quota

Counts are OUR counts (app/quota.py), kept in Postgres. Compare them with https://ai.dev/rate-limit
once: if embedding N texts in one batch call moves Google's counter by N, keep EMBED_QUOTA_UNIT=text;
if it moves by 1, you may set EMBED_QUOTA_UNIT=call and ingest ~60x faster.
"""

import asyncio

from app.config import get_settings
from app.db import close_pool, init_schema, open_pool
from app.quota import quota_day, seconds_until_reset


async def usage_today(pool) -> dict[str, dict[str, int]]:
    """Return {api: {"used": n, "limit": n, "remaining": n}} for the current Pacific quota day."""
    s = get_settings()
    limits = {"embed": s.quota("embed")[2], "llm": s.quota("llm")[2]}
    async with pool.connection() as conn:
        rows = await (
            await conn.execute("SELECT api, requests FROM api_usage WHERE day=%s", (quota_day(),))
        ).fetchall()
    used = {r["api"]: r["requests"] for r in rows}
    return {
        api: {
            "used": used.get(s.usage_key(api), 0),
            "limit": lim,
            "remaining": max(lim - used.get(s.usage_key(api), 0), 0),
        }
        for api, lim in limits.items()
    }


async def main() -> None:
    """Print the usage table and the time until the quota resets (midnight Pacific)."""
    pool = await open_pool()
    await init_schema(pool)
    usage = await usage_today(pool)
    print(f"Quota day (Pacific): {quota_day()}   resets in {seconds_until_reset() / 3600:.1f} h")
    for api, u in usage.items():
        print(f"  {api:6} used {u['used']:>4} / {u['limit']:<4} remaining {u['remaining']}")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
