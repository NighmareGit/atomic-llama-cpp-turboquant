#!/usr/bin/env python3
"""Multi-turn single-slot KV fill via /v1/chat/completions (cache_prompt on)."""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request


def repetition_score(text: str) -> float:
    if len(text) < 80:
        return 0.0
    words = re.findall(r"\w+", text.lower())
    if len(words) < 20:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    return 1.0 - (len(set(trigrams)) / len(trigrams))


def chat(port: int, messages: list, max_tokens: int, timeout: int) -> dict:
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        "temperature": 0.6,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    port = int(os.environ.get("BENCH_PORT", "8081"))
    timeout = int(os.environ.get("BENCH_CURL_TIMEOUT", "900"))
    ctx = int(os.environ.get("BENCH_CTX", "8192"))
    target = int(os.environ.get("BENCH_KV_TARGET", str(max(6000, ctx - 500))))
    turn_gen = int(os.environ.get("BENCH_KV_TURN_GEN", "48"))
    final_gen = int(os.environ.get("BENCH_KV_FINAL_GEN", "128"))
    max_turns = int(os.environ.get("BENCH_KV_MAX_TURNS", "40"))
    out_path = os.environ.get("BENCH_KV_OUT", "")

    para = (
        "Distributed computing research covers consistency models, latency hiding, fault tolerance, "
        "and pipeline parallelism across heterogeneous GPUs. In llama.cpp RPC setups, ROCm and CUDA "
        "workers exchange activations through copy slots, scheduler barriers, and deferred RPC GETs. "
        "Recurrent layers such as Gated Delta Net require sequential handoff completion before the "
        "next decode step reads prior state. KV cache growth across multi-turn chats stresses graph "
        "reuse and checkpoint restore on hybrid memory models. "
    )
    filler = (para * 12) + (
        "Turn note: add one new detail about barriers, RPC defer, or KV slots. Stay factual."
    )
    messages: list[dict] = []
    rows: list[dict] = []
    prompt_tokens = 0

    for turn in range(1, max_turns + 1):
        messages.append({"role": "user", "content": filler})
        try:
            data = chat(port, messages, turn_gen, timeout)
        except urllib.error.URLError as e:
            print(f"FAIL turn={turn} curl_error={e}", file=sys.stderr)
            return 2
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        rep = repetition_score(content)
        row = {
            "turn": turn,
            "phase": "grow",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "repetition": round(rep, 3),
            "g_tps": timings.get("predicted_per_second"),
            "preview": content[:200],
        }
        rows.append(row)
        print(
            f"turn={turn} phase=grow prompt_tokens={prompt_tokens} "
            f"compl={completion_tokens} rep={rep:.2f} "
            f"G={timings.get('predicted_per_second', 0):.1f} "
            f"preview={content[:120]!r}"
        )
        messages.append({"role": "assistant", "content": content})
        if prompt_tokens >= target:
            break

    messages.append({
        "role": "user",
        "content": (
            "Summarize the themes we discussed in exactly three sentences. "
            "Mention distributed computing, pipeline parallelism, and KV cache."
        ),
    })
    try:
        data = chat(port, messages, final_gen, timeout)
    except urllib.error.URLError as e:
        print(f"FAIL final_turn curl_error={e}", file=sys.stderr)
        return 2
    choice = data.get("choices", [{}])[0]
    final_content = choice.get("message", {}).get("content", "")
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    rep = repetition_score(final_content)
    row = {
        "turn": len(rows) + 1,
        "phase": "final",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "repetition": round(rep, 3),
        "g_tps": timings.get("predicted_per_second"),
        "preview": final_content[:400],
        "content": final_content,
    }
    rows.append(row)
    print(
        f"turn=final prompt_tokens={prompt_tokens} compl={row['completion_tokens']} "
        f"rep={rep:.2f} G={timings.get('predicted_per_second', 0):.1f}"
    )

    ok = True
    min_ratio = float(os.environ.get("BENCH_KV_MIN_RATIO", "0.85"))
    if prompt_tokens < target * min_ratio:
        print(f"WARN: prompt_tokens={prompt_tokens} below target={target} (ratio={min_ratio})")
        ok = False
    if rep > 0.55:
        print(f"FAIL: final repetition={rep:.2f}")
        ok = False
    if re.search(r"(here){4,}", final_content.lower()):
        print("FAIL: Here-loop in final output")
        ok = False
    low = final_content.lower()
    for kw in ("distributed", "pipeline", "kv"):
        if kw not in low:
            print(f"WARN: final missing keyword '{kw}'")

    summary = {
        "target_prompt_tokens": target,
        "final_prompt_tokens": prompt_tokens,
        "turns": len(rows),
        "pass": ok,
        "rows": rows,
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"KV_FILL_SUMMARY pass={ok} final_prompt_tokens={prompt_tokens} turns={len(rows)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())