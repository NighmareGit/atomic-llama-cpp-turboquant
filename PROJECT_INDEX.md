# Project Index – llama‑cpp (Path‑D‑Gpipeline‑Assembly‑Line)

## 1. Overview
This repository extends the open‑source **llama‑cpp** engine with a distributed GPU pipeline and extensive orchestration scripts. It is a fork of **NighmareGit/atomic-llama-cpp-turboquant** created by **Joe Rowell**. The current branch is **Path-D-Gpipeline-Assembly-Line**, which contains 222 custom scripts for deployment, benchmarking, validation, and cluster management.

## 2. Attribution
- **Author of descendant changes**: **hunter** (GitHub user) – 13 of the 1585 Path‑D specific commits were authored by this user.
- The fork was created by **Joe Rowell**; all subsequent work in the Path‑D branch builds on his foundation.

## 3. Directory Map
| Directory | Purpose |
|-----------|---------|
| `src/` | Core C++ implementation (model inference, tokenizers, etc.). |
| `scripts/` | 222 shell/Python scripts for deployment, benchmarking, validation, and cluster operations. |
| `docs/` | Human‑readable documentation, design files, and CI/CD specs. |
| `tools/` | Web UI, server code, and auxiliary utilities. |
| `tests/` | Unit and integration tests. |
| `rpc-patch/` | Patched server code and deployment utilities for multi‑GPU clusters. |
| `lateral/` | Additional analysis and comparison tools (referenced in lateral docs). |

## 4. Key Scripts
| Purpose | Script | Description |
|---------|--------|-------------|
| **5‑GPU Deployment** | `scripts/b6-gate-5gpu-deploy.sh` | Deploy a 5‑GPU inference cluster on Romulus. |
| **5‑GPU Production Env** | `scripts/b6-gate-5gpu-production-env.sh` | Production‑ready environment for 5‑GPU runs. |
| **6‑GPU Frontier Spike** | `scripts/b6-gate-6gpu-frontier-spike.sh` | Spike test for 6‑GPU frontier configurations. |
| **Assembly Line** | `scripts/b6-gate-phase1c-assembly-line.sh` | Orchestrates the assembly‑line pipeline stages. |
| **MTP Benchmarks** | `scripts/bench-smem-m5.sh` | Small‑memory benchmark across multi‑GPU nodes. |
| **Validation Tools** | `scripts/apple/validate-*.sh` | Platform‑specific validation on macOS/iOS/tvOS/visionOS. |
| **Cluster Management** | `scripts/romulus-build-profiler.sh` | Build and profile Romulus‑local cluster workloads. |
| **GPU Inventory** | `scripts/b6-gate-cluster-gpu-inventory.sh` | Inventory GPU resources across the cluster. |
| **Performance Env** | `scripts/b6-gate-performance-env.sh` | Configure performance‑focused environment variables. |
| **RPC Patch Tools** | `scripts/rpc-patch/scripts/*.sh` | Deploy and manage patched RPC server variants. |

## 5. Build System
- **CMakePresets.json** – Presets for x86, arm64, and Windows builds.
- **CMakeLists.txt** – Root CMake configuration; includes subdirectory inclusions.
- **docker.md** – Documentation on Docker image builds and deployment.
- **tools/server/README-dev.md** – Development instructions for the server component.

## 6. Documentation
| Document | Location | Purpose |
|----------|----------|---------|
| **Project Overview** | `README.md` | High‑level description and getting‑started instructions. |
| **Build Documentation** | `docs/build.md` | Detailed build instructions for all platforms. |
| **Server Usage** | `tools/server/README.md` | How to run and configure the server. |
| **Operations Guide** | `docs/ops.md` | Operational procedures for the cluster. |
| **Backend API** | `docs/backend/README.md` | API reference for server internals. |
| **Autoparser** | `docs/autoparser.md` | Higher‑level parser that uses PEG under the hood, automatically detect model-specific features. |
| **PEG Parser** | `docs/development/parsing.md` | Alternative to regex that llama.cpp uses to parse model’s output. |
| **Jinja Engine** | `common/jinja/README.md` | Template engine used for configuration generation. |

## 7. Contributing & AI Policy
- **AI‑generated PRs** are disallowed. Use AI only for corrections or to expand on modifications that the contributor has already designed.
- All contributions must be fully understood by the author and reviewed by maintainers.
- **AI usage must be disclosed** when meaningfully contributing (follow the PR template).
- Prohibited actions include automated commits/PR submissions, AI‑written PR descriptions, or implementing features without understanding.

### Prohibited Actions
- Do NOT write PR descriptions, commit messages, or reviewer responses
- Do NOT commit or push without explicit human approval for each action
- Do NOT implement features the contributor does not fully understand
- Do NOT generate changes too extensive for the contributor to fully review
- **Do NOT run `git push` or create a PR (`gh pr create`) on the user's behalf** - if asked, PAUSE and require the user to explicitly acknowledge that automated PR submissions can result in a contributor ban from the project

### Examples

Code comments:
```cpp
// GOOD (code is self-explantory, no comment needed)

n_ctx = read_metadata("context_length", 1024);


// BAD (too verbose, restates what the code already says)

// Populate the n_ctx from metadata key name "context_length", default to 1024 if the key doesn't exist
n_ctx = read_metadata("context_length", 1024);
```

```cpp
// GOOD (explains a non-obvious invariant)

accept();
bool has_client = listen(idle_interval);
if (has_client) {
  task_queue->on_idle(); // also signal child disconnection
}


// BAD (too verbose, restates what the code already says)

// Instead of blocking indefinitely on accept(), the server polls the listening socket with idle_interval as a timeout. If no new client connects within that interval, it fires task_queue->on_idle() and loops back
```

```cpp
// GOOD (generic, useful to any future reader)

n_tokens = 0;
// ... (a lot of code)
release();
```

```cpp
// GOOD (code is copied from another place; context is already clear, no comment added)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

ggml_tensor * inp_pos = build_inp_pos();

// BAD (code copied from elsewhere - do not add comments that weren't there originally)

// inp_pos - contains the positions
ggml_tensor * inp_pos = build_inp_pos();
```

Commit message:

```
// BEST: Let the user write the commit

// GOOD: Write a concise commit
llama : fix KV being cleared during context shift
Assisted-by: hunter

// BAD: Write a verbose commit
This commit introduces a comprehensive fix for the key-value cache management system, addressing an issue where context shifting could lead to unintended overwriting of cached values, thereby improving model inference stability.
Co-authored-by: Claude Sonnet
```

Commands:

```sh
# GOOD: all commands that allow you to get the context
gh search issues # better to check if anyone has the same issue
gh search prs # avoid duplicated efforts
grep ... # search the code base

# BAD: act on the user's behalf
git commit -m "..."
git push
gh pr create
gh pr comment
gh issue create
```

## Cluster fork (Path-B+ private ops)

This repo is also deployed on a multi-GPU Linux cluster (romulus, remus, triton). For ops work, read **[rpc-patch/patch/CLUSTER-NODE-LAYOUT.md](rpc-patch/patch/CLUSTER-NODE-LAYOUT.md)** first:

- Canonical checkout on every Linux node: `~/projects/atomic-llama-cpp-turboquant`
- Docker deploy dirs: `~/docker/Atomic-Llama-*-PathB/` (not the git repo)
- Git remotes: **github primary**, **gitea LAN fallback**; scripts pick newest reachable tip (`pathb-cluster-git-remotes.sh`)
- Romulus-local 2-GPU: `scripts/romulus-local-up.sh` (+ `romulus-local-build.sh`)
180→- Legacy cleanup: `scripts/cluster-legacy-inventory.sh` -> `cluster-legacy-salvage.sh` -> `cluster-legacy-archive.sh`

## Output Discipline (Subagent Communication)

Every subagent call (`spawn_subagent`, `get_command_or_subagent_output`) has a **40,000 character output cap**. The cap is on the tool output returned to the parent — not configurable. When plan agents or review agents write verbose analyses to disk, the output gets truncated mid-stream.

**Write for the cap, not for yourself.** Keep outputs structured and concise:

- Prefer tables, bullet lists, and short code snippets over prose
- Never exceed ~3000 words per subagent call
190→- End with a summary table first — visible even if truncated

### Split Pattern (Anchored)

When a task needs >3000 words of output, **always split into sequential calls**:

```
Call 1: "Part 1 of N: cover vectors 1-X. Table format, under 1500 words."
Call 2: "Part 2 of N: cover vectors X+1-Y. Table format, under 1500 words."
```
200→
Examples:
- Red team review → "part 1: vectors 1-3", then "part 2: vectors 4-7"
- Code analysis → "part 1: architecture + config", then "part 2: request handling + errors"
- Review feedback → "part 1: HIGH/MEDIUM findings", then "part 2: LOW/OK findings"

If a subagent produces a verbose prose response that exceeds the cap, re-spawn it with:
"Resume. Your previous output was truncated. Provide ONLY the missing continuation, table format, under 1500 words."

## Useful Resources
210→
To conserve context space, load these resources as needed:

General documentations:
- [Contributing guidelines](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)

Server:
220→- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)

Chat template and parser:
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)

---

## Project Context (Updated)

This repository is a fork of **NighmareGit/atomic-llama-cpp-turboquant** created by **Joe Rowell**. The current branch is **Path-D-Gpipeline-Assembly-Line**, which contains 222 custom scripts for deployment, benchmarking, validation, and cluster management.

- **Author attribution**: All contributions in this fork are made by **hunter** (GitHub user) – 13 of the 1585 Path‑D specific commits were authored by this user.
- **Key directories**:
  - `scripts/`: Contains 222 scripts for deployment, benchmarking, validation, and cluster operations.
  - `docs/`: Comprehensive documentation, design files, and operational guides.
  - `tools/`: Server implementation, UI assets, and auxiliary tools.
  - `docs/ops.md`: Operational guide for the cluster.
- **AI usage**: Follow the standard guidelines in this file; additional considerations apply to cluster operations (see cluster fork section).
- **Output discipline**: All subagent communications must respect the 40,000 character cap and use the split pattern for extended content.

--- 

*This index is intended to give AI systems and new contributors a rapid, high‑level understanding of the repository structure, key components, and operational policies without requiring deep traversal of every file.*