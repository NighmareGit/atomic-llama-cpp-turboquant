#!/usr/bin/env python3
"""Validate TSC spike bench outputs from .full.jsonl + .result files."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def load_prompts(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {p["id"]: p for p in data}


def repetition_score(text: str) -> float:
    if len(text) < 80:
        return 1.0
    words = re.findall(r"\w+", text.lower())
    if len(words) < 20:
        return 0.5
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique = len(set(trigrams))
    return 1.0 - (unique / len(trigrams))


def garble_score(text: str) -> float:
    if not text:
        return 1.0
    weird = sum(1 for ch in text if ord(ch) > 0x3000 or (ord(ch) < 32 and ch not in "\n\t"))
    return weird / max(len(text), 1)


def check_prompt(pid: str, content: str, spec: dict) -> tuple[bool, str]:
    if not pid or pid not in spec and not pid.isidentifier() and len(pid) > 40:
        return False, "parse_artifact"

    reasons: list[str] = []
    rep = repetition_score(content)
    garb = garble_score(content)
    n_tok_est = len(content.split())

    rep_limit = 0.55
    if pid in ("logic", "niche_json"):
        rep_limit = 0.85
    if rep > rep_limit:
        reasons.append(f"high_repetition={rep:.2f}")
    if garb > 0.08:
        reasons.append(f"garbled={garb:.2f}")
    if len(content.strip()) < 20:
        reasons.append("too_short")
    min_tok = spec.get("min_tokens", 0)
    if n_tok_est < min_tok * 0.35 and min_tok >= 100:
        reasons.append(f"underlength words={n_tok_est} need~{min_tok}")

    low = content.lower()
    for token in spec.get("expect", []):
        if token.lower() not in low and token not in content:
            reasons.append(f"missing:{token}")

    if pid == "math" and "391" not in content and "three hundred" not in low:
        reasons.append("wrong_math_answer")

    if pid == "niche_json":
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start < 0 or end <= start:
                reasons.append("no_json")
            else:
                obj = json.loads(content[start : end + 1])
                if not all(k in obj for k in ("model", "gpus", "kv_tokens")):
                    reasons.append("json_keys")
                else:
                    reasons = [r for r in reasons if not r.startswith("high_repetition")]
        except json.JSONDecodeError:
            reasons.append("json_parse")

    if re.search(r"(here){5,}", low):
        reasons.append("here_loop")
    if re.search(r"(\w{2,})\1\1\1", low):
        reasons.append("morpheme_loop")

    ok = len(reasons) == 0
    return ok, "; ".join(reasons) if reasons else "ok"


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <prompts.json> <arm_dir>", file=sys.stderr)
        return 2

    prompts_path = Path(sys.argv[1])
    arm_dir = Path(sys.argv[2])
    specs = load_prompts(prompts_path)

    full_files = sorted(arm_dir.glob("*.full.jsonl"))
    if not full_files and (arm_dir / "full.jsonl").is_file():
        full_files = [arm_dir / "full.jsonl"]
    if not full_files:
        result_files = sorted(arm_dir.glob("*.result")) + sorted(arm_dir.glob("result.txt"))
        if not result_files:
            print(f"no outputs in {arm_dir}")
            return 2
        full_files = result_files

    total = 0
    passed = 0
    rows: list[str] = []

    for fpath in full_files:
        arm = fpath.name.replace(".full.jsonl", "").replace(".result", "")
        if fpath.suffix == ".jsonl":
            for line in fpath.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                pid = row["prompt_id"]
                content = row.get("content", "")
                spec = specs.get(pid, {})
                ok, detail = check_prompt(pid, content, spec)
                if detail == "parse_artifact":
                    continue
                total += 1
                if ok:
                    passed += 1
                usage = row.get("usage") or {}
                comp = usage.get("completion_tokens") or usage.get("completion_tokens", "?")
                rows.append(f"{arm}|{pid}|{'PASS' if ok else 'FAIL'}|{detail}|completion_tokens={comp}|len={len(content)}")
        else:
            for line in fpath.read_text().splitlines():
                m = re.search(r"prompt=(\S+).*preview='([^']*)'", line)
                if not m:
                    continue
                pid, preview = m.group(1), m.group(2)
                spec = specs.get(pid, {})
                ok, detail = check_prompt(pid, preview, spec)
                total += 1
                if ok:
                    passed += 1
                rows.append(f"{arm}|{pid}|{'PASS' if ok else 'FAIL'}|{detail}|preview_only|len={len(preview)}")

    report = arm_dir / "validation-report.tsv"
    report.write_text("arm|prompt|verdict|detail|meta|len\n" + "\n".join(rows) + "\n")
    print(f"validation: {passed}/{total} passed -> {report}")
    for r in rows:
        if "FAIL" in r:
            print(f"  FAIL {r}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())