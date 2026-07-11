#!/usr/bin/env python3
"""D4.1 Longer Context Tests - conversation and complex reasoning."""

import json, subprocess, sys, time, os

PORT = 8083
BASE = f"http://127.0.0.1:{PORT}"
OUTDIR = "/tmp/d41-baseline/long-context"
os.makedirs(OUTDIR, exist_ok=True)

def chat(messages, max_tokens=256, timeout=300):
    """Send chat completion request and return parsed response."""
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
    })
    url = f"{BASE}/v1/chat/completions"
    cmd = ["curl", "-s", "--max-time", str(timeout), url,
           "-H", "Content-Type: application/json", "-d", payload]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10)
    try:
        data = json.loads(result.stdout)
        return data
    except json.JSONDecodeError:
        print(f"  ERROR: curl returned: {result.stdout[:200]}")
        return None

def show_result(label, data, save_as=None):
    if data is None:
        print(f"  {label}: FAILED")
        return
    t = data.get("timings", {})
    u = data.get("usage", {})
    c = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    pt = u.get("prompt_tokens", t.get("prompt_n", "?"))
    gt = u.get("completion_tokens", t.get("predicted_n", "?"))
    pp = t.get("prompt_per_second", 0)
    tg = t.get("predicted_per_second", 0)
    fr = data.get("choices", [{}])[0].get("finish_reason", "?")
    print(f"  {label}: prompt={pt} gen={gt} pp={pp:.1f} t/s tg={tg:.1f} t/s finish={fr}")
    if c:
        print(f"    response ({len(c)} chars): {c[:200]}...")
    if save_as:
        with open(os.path.join(OUTDIR, save_as), "w") as f:
            json.dump(data, f, indent=2)

print("=== Test 1: Long conversation (5-turn hypothetical) ===")
conv = [
    {"role": "user", "content": "What are the main challenges of establishing a permanent human colony on Mars? Consider radiation, gravity, psychological factors, and life support."},
]
data = chat(conv, max_tokens=128)
show_result("Turn 1", data, "conv1-turn1.json")

# Add assistant response and follow-up
if data:
    c1 = data["choices"][0]["message"]["content"]
    conv.append({"role": "assistant", "content": c1[:500]})
else:
    conv.append({"role": "assistant", "content": "Mars colonization faces challenges including radiation exposure, low gravity effects, psychological isolation, and closed-loop life support requirements."})
conv.append({"role": "user", "content": "How would you design a power system for 100 colonists that is reliable during planet-wide dust storms?"})
data = chat(conv, max_tokens=128)
show_result("Turn 2 (dust storms)", data, "conv1-turn2.json")

if data:
    c2 = data["choices"][0]["message"]["content"]
    conv.append({"role": "assistant", "content": c2[:500]})
else:
    conv.append({"role": "assistant", "content": "A robust power system would use nuclear fission for baseload and solar/battery for peak."})
conv.append({"role": "user", "content": "What computational infrastructure would be needed? Consider AI, communication delay, and resilience."})
data = chat(conv, max_tokens=128)
show_result("Turn 3 (computing)", data, "conv1-turn3.json")

print()
print("=== Test 2: Complex calculus problem ===")
data = chat([
    {"role": "user", "content": """Solve step by step: A particle moves along a line with velocity v(t) = 3t^2 - 12t + 9 m/s.
(a) Find the displacement from t=0 to t=5.
(b) Find the total distance traveled from t=0 to t=5.
Show all your work including finding when the particle changes direction."""}
], max_tokens=512, timeout=300)
show_result("Calculus", data, "math-calc.json")

print()
print("=== Test 3: Einstein's Riddle (logic puzzle) ===")
data = chat([
    {"role": "user", "content": """Solve Einstein's riddle step by step:
1. The Englishman lives in the red house.
2. The Spaniard owns the dog.
3. Coffee is drunk in the green house.
4. The Ukrainian drinks tea.
5. The green house is immediately to the right of the ivory house.
6. The Old Gold smoker owns snails.
7. Kools are smoked in the yellow house.
8. Milk is drunk in the middle house.
9. The Norwegian lives in the first house.
10. The man who smokes Chesterfields lives next to the man with the fox.
11. Kools are smoked next to the house where the horse is kept.
12. The Lucky Strike smoker drinks orange juice.
13. The Japanese smokes Parliaments.
14. The Norwegian lives next to the blue house.

Question: Who drinks water? Who owns the zebra?"""}
], max_tokens=512, timeout=300)
show_result("Einstein Riddle", data, "logic-einstein.json")

print()
print("=== Test 4: Multi-turn deepening conversation ===")
deep_conv = [
    {"role": "user", "content": "Explain how quantum computing differs from classical computing."},
]
data = chat(deep_conv, max_tokens=128)
show_result("Turn 1 (quantum basics)", data, "conv-deep1.json")

if data:
    a1 = data["choices"][0]["message"]["content"]
    deep_conv.append({"role": "assistant", "content": a1[:500]})
else:
    deep_conv.append({"role": "assistant", "content": "Quantum computing uses qubits in superposition..."})
deep_conv.append({"role": "user", "content": "What specific algorithms demonstrate quantum advantage over classical computers?"})
data = chat(deep_conv, max_tokens=128)
show_result("Turn 2 (algorithms)", data, "conv-deep2.json")

if data:
    a2 = data["choices"][0]["message"]["content"]
    deep_conv.append({"role": "assistant", "content": a2[:500]})
else:
    deep_conv.append({"role": "assistant", "content": "Shor's algorithm for factoring and Grover's for search..."})
deep_conv.append({"role": "user", "content": "What are the main engineering challenges in building fault-tolerant quantum computers?"})
data = chat(deep_conv, max_tokens=128)
show_result("Turn 3 (engineering)", data, "conv-deep3.json")

print()
print("=== Test 5: Abstract reasoning / problem deduction ===")
data = chat([
    {"role": "user", "content": """You have 12 identical-looking coins, but one is counterfeit and weighs slightly different (either heavier or lighter - you don't know which). You have a balance scale that can compare weights. Describe how to find the counterfeit coin and determine whether it's heavier or lighter in exactly 3 weighings. Explain your reasoning step by step."""}
], max_tokens=512, timeout=300)
show_result("12 coins puzzle", data, "reasoning-coins.json")

print()
print("=== GPU snapshot ===")
ts = int(time.time())
gpu = {"timestamp": ts}
if os.system("command -v rocm-smi >/dev/null") == 0:
    r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10)
    for line in r.stdout.split("\n"):
        if "GPU[0]" in line:
            parts = line.split()
            if len(parts) >= 5:
                gpu["7900_XTX"] = f"{int(parts[4])/1024/1024:.0f} MiB"
if os.system("command -v nvidia-smi >/dev/null") == 0:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    gpu["3060_Ti"] = f"{r.stdout.strip()} MiB"
with open(os.path.join(OUTDIR, "gpu-snap.json"), "w") as f:
    json.dump(gpu, f)
print(f"  GPU: {json.dumps(gpu)}")

print()
print("=== LONG CONTEXT TESTS COMPLETE ===")
print(f"Results in {OUTDIR}/")
