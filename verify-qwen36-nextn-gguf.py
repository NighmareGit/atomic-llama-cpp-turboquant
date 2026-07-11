#!/usr/bin/env python3
"""Check a GGUF for Qwen NextN tensor names (blk/layer *.nextn.*). Exit 1 if none found."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", type=Path)
    args = ap.parse_args()

    gguf_py = Path(__file__).resolve().parent.parent / "gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore

    r = GGUFReader(str(args.gguf))
    draft_like: list[str] = []
    for t in r.tensors:
        nm = t.name.decode("utf-8") if isinstance(t.name, bytes) else str(t.name)
        if ".nextn." in nm or ".mtp." in nm or nm.endswith("pre_projection.weight") or nm.endswith("post_projection.weight"):
            draft_like.append(nm)

    if not draft_like:
        print(
            "error: no NextN/MTP draft tensors found (expected '.nextn.', '.mtp.', or pre/post_projection)",
            file=sys.stderr,
        )
        return 1

    print(f"ok: found {len(draft_like)} NextN/MTP-related tensors (showing up to 12):")
    for nm in sorted(draft_like)[:12]:
        print(f"  {nm}")
    if len(draft_like) > 12:
        print(f"  ... and {len(draft_like) - 12} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
