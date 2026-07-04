# Cluster node layout (canonical)

**Read this first** for Path-B+ ops on Linux cluster nodes. Credentials: `.scratch/cluster-access.env` (gitignored).

## Canonical repo path (all Linux nodes)

```
~/projects/atomic-llama-cpp-turboquant
```

| Node | User | Repo (target) | Notes |
|------|------|---------------|-------|
| **romulus** | hunter | `~/projects/atomic-llama-cpp-turboquant` | ROCm client + local 3060 docker RPC |
| **remus** | hunter | `~/projects/atomic-llama-cpp-turboquant` | CUDA RPC worker |
| **triton** | hunter | `~/projects/atomic-llama-cpp-turboquant` | CUDA RPC :50054 / :50055 |
| **dev laptop** | hunter | `~/projects/atomic-llama-cpp-turboquant` | edit + push to gitea/github |

**Deprecated (do not use for new work):**

- `~/atomic-llama-cpp-turboquant` on romulus (legacy; dirty tree from in-place dev)
- Nested `atomic-llama-cpp-turboquant/atomic-llama-cpp-turboquant/` (historical accident)

**Windows jupiter** stays separate: `D:\projects\atomic-llama-cpp-5070ti\...`

## Build outputs (under repo root)

| Artifact | Path | Host |
|----------|------|------|
| ROCm client | `build-rocm-docker/bin/llama-server` | romulus |
| CUDA worker (native) | `build-cuda-b-bin/bin/rpc-server` | remus, triton |
| Bench artifacts | `benches/path-b-plus/<label>/` | wherever client ran |

## Docker deploy dirs (not the git repo)

Synced from `rpc-patch/deploy/`; do not edit only on remote.

| Node | Dir | Container |
|------|-----|-----------|
| romulus | `~/docker/Atomic-Llama-Romulus-PathB` | `pathb-rpc-romulus` |
| remus | `~/docker/Atomic-Llama-Remus-PathB` | `pathb-rpc-remus` |
| triton | `~/docker/Atomic-Llama-Triton-PathB` | `:50054`, `:50055` |

Docker `build.sh` clones **gitea** at build time (`http://192.168.8.108:3005/...`). Host checkout path is independent of image source.

## Git remotes (mirror policy)

Both remotes should track the same branch tip. Nodes pick the **newest reachable** tip.

| Remote | URL | Role |
|--------|-----|------|
| **github** | `https://github.com/NighmareGit/atomic-llama-cpp-turboquant.git` | **Primary** when internet is up |
| **gitea** | `http://192.168.8.108:3005/hunter/atomic-llama-cpp-turboquant.git` | **LAN mirror** when github offline or romulus gitea is the only reachable host |

Branch: `Path-B-Event-Support-Pipeline-Plus`

**Sync logic** (`rpc-patch/scripts/pathb-cluster-git-remotes.sh`):

1. `git fetch github` then `git fetch gitea` (either may fail)
2. If tips match -> reset to `github`
3. If one is ancestor of the other -> reset to the **ahead** remote
4. If diverged -> error (fix mirror manually)
5. Initial clone uses `git ls-remote` probe; prefers github when both reachable

Romulus gitea is not always powered on — nodes must not hard-depend on it.

**After dev push:** push to **both** github and gitea to keep mirrors aligned.

## Env overrides (scripts)

| Var | Default |
|-----|---------|
| `CLUSTER_REPO` / `PATHB_NODE_GIT_REPO` | `~/projects/atomic-llama-cpp-turboquant` |
| `PATHB_ROMULUS_GIT_REPO` | same |
| `PATHB_TRITON_REPO` | same |
| `LLAMA_TURBOQUANT_ROOT` | same (dev laptop) |

## Legacy cleanup (romulus)

Review: [LEGACY-ROMULUS-REVIEW-2026-07-04.md](LEGACY-ROMULUS-REVIEW-2026-07-04.md)

```bash
bash scripts/cluster-legacy-inventory.sh    # what is on disk
bash scripts/cluster-legacy-salvage.sh      # tarball TSC bench evidence (~MB)
bash scripts/cluster-legacy-archive.sh      # stop services, archive legacy paths
bash scripts/cluster-node-fresh-clone.sh    # clean clone at canonical path
```

## Romulus-local quick start (2-GPU, no cluster)

```bash
cd ~/projects/atomic-llama-cpp-turboquant
bash scripts/romulus-local-build.sh                                    # first time / after pull
bash scripts/romulus-local-up.sh -c 8192                               # default Qwen3.6 35B MTP
bash scripts/romulus-local-up.sh -m /mnt/models/other.gguf --mtp off # non-MTP override
```

Listens on `0.0.0.0:8080` by default. TSC seq-repair on by default (`GGML_PIPELINE_MULTI_BACKEND_SEQ=1` on client + RPC).

## Related docs

| Doc | Role |
|-----|------|
| [HANDOVER-CLUSTER-OPS.md](HANDOVER-CLUSTER-OPS.md) | Docker lifecycle, bench launchers |
| [HANDOVER-SESSION-2026-07-03.md](HANDOVER-SESSION-2026-07-03.md) | TSC spike state |
| [CONTEXT.md](../../docs/rpc-multi-backend-pipeline-plus/CONTEXT.md) | Domain glossary |