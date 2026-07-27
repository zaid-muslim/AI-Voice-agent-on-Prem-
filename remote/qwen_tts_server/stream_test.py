import asyncio
import time
import httpx

SERVER_URL = "http://localhost:8091/v1/audio/speech"
MODEL = "/home/nauyan/voice-agent-pipeline/models/Qwen3-TTS-0.6B-custom"  # MUST match exactly what you passed to `vllm serve`
CONCURRENCY = 8
TEXT = "This is a concurrent load test of the Qwen text to speech server."


async def one_request(client: httpx.AsyncClient, idx: int) -> dict:
    payload = {
        "model": MODEL,
        "input": TEXT,
        "voice": "vivian",
        "response_format": "wav",  # avoids the known stream+pcm duplication bug we found earlier
        "language": "English",
    }
    start = time.monotonic()
    resp = await client.post(SERVER_URL, json=payload, timeout=120.0)
    elapsed_ms = (time.monotonic() - start) * 1000
    resp.raise_for_status()
    return {"idx": idx, "elapsed_ms": elapsed_ms, "audio_bytes": len(resp.content)}


async def main():
    async with httpx.AsyncClient() as client:
        print("--- Baseline: 1 solo request ---")
        baseline = await one_request(client, -1)
        print(f"  solo: {baseline['elapsed_ms']:.1f} ms\n")

        print(f"--- {CONCURRENCY} concurrent requests, fired simultaneously ---")
        start = time.monotonic()
        results = await asyncio.gather(
            *[one_request(client, i) for i in range(CONCURRENCY)]
        )
        total_wall_ms = (time.monotonic() - start) * 1000

        for r in sorted(results, key=lambda x: x["idx"]):
            print(
                f"  request {r['idx']}: {r['elapsed_ms']:.1f} ms  ({r['audio_bytes']} bytes)"
            )

        times = [r["elapsed_ms"] for r in results]
        print(f"\nTotal wall time for all {CONCURRENCY}: {total_wall_ms:.1f} ms")
        print(
            f"Min: {min(times):.1f} ms   Max: {max(times):.1f} ms   Avg: {sum(times) / len(times):.1f} ms"
        )
        print(f"\nSolo baseline was {baseline['elapsed_ms']:.1f} ms.")
        print(
            f"If concurrency were free, {CONCURRENCY} requests would each still take ~{baseline['elapsed_ms']:.1f} ms."
        )
        print(
            f"If concurrency were fully serial, worst-case would be ~{CONCURRENCY * baseline['elapsed_ms']:.1f} ms."
        )
        print(
            "Where the actual Max falls between those two numbers tells us how well it's really batching."
        )


if __name__ == "__main__":
    asyncio.run(main())
