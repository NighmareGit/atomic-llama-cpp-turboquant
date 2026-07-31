# Legacy romulus checkout review (2026-07-04)

Inventory before archiving `~/atomic-llama-cpp-turboquant` and cleaning `~/projects/atomic-llama-cpp-turboquant`.

## Paths to retire

| Path | Size | Verdict |
|------|------|---------|
| `~/atomic-llama-cpp-turboquant` | **3.5 GB** | Archive whole tree (dirty git @ `92f5b43e5`, 2088 untracked files) |
| `~/projects/atomic-llama-cpp-turboquant` | **7.3 GB** | Archive — old nested layout, not a clean git root |
| `~/docker/Atomic-Llama-Romulus-PathB/staging` | 502 MB | Rebuild from fresh deploy (ephemeral) |

## Worth salvaging (pull or tarball before archive)

| Content | Size | Why keep |
|---------|------|----------|
| `rpc-patch/patch/bench-results/rpc-server-bench/tsc-*` | ~2 MB | TSC validation matrix (seq-repair 8/8, KV fill) |
| `rpc-patch/patch/bench-results/rpc-server-bench/trace-g-*tsc*` | ~3 MB | Plus on/off bisect evidence |
| `rpc-patch/patch/bench-results/rpc-server-bench/tsc-kvfill-*.kvfill.json` | small | Multi-turn KV fill proof |

**Optional (large, low urgency):** `benches/path-b-plus/` (989 MB, 118 labels) — profiler gate history. Leave in archived tarball unless a specific label is needed.

## Safe to discard (do not salvage)

| Content | Why |
|---------|-----|
| `scripts/b6-gate-romulus-local-mtp-up.sh` (+ `.bk`) | Replaced by `scripts/romulus-local-up.sh` |
| `scripts/b6-gate-romulus-pathb-deploy.sh` | Replaced by `scripts/romulus-local-build.sh` |
| Stray root copies (`pathb-rpc-vram-preflight.py`, `b6-gate-b15-*.sh`) | Duplicates; in repo under proper paths |
| `build-rocm-mtp-test`, `build-rocm-profiler`, `build-rocm-docke` | Stale cmake trees |
| 2000+ untracked scratch bench dirs | Artifacts of in-place bisects; archived with tree if needed |

## Recommended sequence (romulus)

```bash
# 1) Salvage TSC evidence tarball (optional pull to laptop after)
bash scripts/cluster-legacy-salvage.sh

# 2) Fresh clone at canonical path
bash scripts/cluster-legacy-archive.sh   # archives both legacy paths
bash scripts/cluster-node-fresh-clone.sh

# 3) Build + run
bash scripts/romulus-local-build.sh
bash scripts/romulus-local-up.sh -c 8192   # default Qwen3.6 35B MTP
```

## Fresh canonical layout after cleanup

```
~/projects/atomic-llama-cpp-turboquant/     # single git root
~/docker/Atomic-Llama-Romulus-PathB/        # docker deploy (rebuilt)
~/archive/                                  # salvaged tarballs (optional)
```