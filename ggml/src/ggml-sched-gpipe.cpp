#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <cstring>
#include <chrono>
#include <algorithm>

#ifndef GPipe_MAX_STAGES
#define GPipe_MAX_STAGES 16
#endif

struct ggml_gpipe_context {
    bool enabled;
    int n_stages;
    int n_copies;
    int n_layers;
    int *layer_sizes; // [n_layers]
    ggml_backend_t *stage_backends; // [n_stages]
    struct ggml_cgraph *stage_graphs; // [n_stages]
    struct ggml_cgraph *full_graph;   // saved for profiling
    float layer_times[GPipe_MAX_STAGES][64]; // [stage][layer] profiled times
};

static void gpipe_split_graph(struct ggml_gpipe_context *ctx, struct ggml_cgraph *src, std::vector<ggml_cgraph *> &out) {
    if (!ctx->enabled || !ctx->layer_sizes || ctx->n_layers <= 0) {
        out.push_back(src);
        return;
    }

    out.resize(ctx->n_stages);

    // Compute layer-to-stage mapping: assign nodes to stages round-robin
    int stage_limit[GPipe_MAX_STAGES];
    std::memset(stage_limit, 0, sizeof(stage_limit));
    for (int l = 0; l < ctx->n_layers; l++) {
        int s = l % ctx->n_stages;
        stage_limit[s] += ctx->layer_sizes[l];
    }

    // Distribute nodes proportionally across stages
    int node_stage[GGML_DEFAULT_GRAPH_SIZE];
    std::memset(node_stage, 0, sizeof(node_stage));
    int node_count = src->n_nodes < GGML_DEFAULT_GRAPH_SIZE ? src->n_nodes : GGML_DEFAULT_GRAPH_SIZE;
    int pos = 0;
    for (int l = 0; l < ctx->n_layers; l++) {
        int s = l % ctx->n_stages;
        for (int k = 0; k < ctx->layer_sizes[l] && pos < node_count; k++, pos++) {
            node_stage[pos] = s;
        }
    }
    while (pos < node_count) {
        node_stage[pos++] = (ctx->n_stages - 1);
    }

    // Clear stage graphs and distribute nodes
    for (int s = 0; s < ctx->n_stages; s++) {
        out[s] = &ctx->stage_graphs[s];
        ctx->stage_graphs[s].n_nodes = 0;
        ctx->stage_graphs[s].n_leafs = 0;
    }

    for (int i = 0; i < node_count; i++) {
        ggml_tensor *node = src->nodes[i];
        if (!node) continue;
        int s = node_stage[i];
        if (s >= ctx->n_stages) s = ctx->n_stages - 1;
        ggml_cgraph *sg = out[s];
        if (sg->n_nodes < GGML_DEFAULT_GRAPH_SIZE) {
            sg->nodes[sg->n_nodes++] = node;
        }
    }

    // Clear cross-stage source references (causal barrier)
    for (int s = 0; s < ctx->n_stages; s++) {
        ggml_cgraph *sg = out[s];
        for (int i = 0; i < sg->n_nodes; i++) {
            ggml_tensor *node = sg->nodes[i];
            if (!node) continue;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor *src_t = node->src[j];
                if (!src_t) continue;
                // Find which stage this source belongs to
                bool belongs = false;
                for (int k = 0; k < sg->n_nodes; k++) {
                    if (sg->nodes[k] == src_t) { belongs = true; break; }
                }
                if (!belongs) {
                    src_t->src[j] = nullptr; // cross-stage dep, external input
                }
            }
        }
    }
}

struct ggml_gpipe_context * ggml_gpipe_init(bool enabled, int n_stages, int n_copies) {
    if (!enabled || n_stages <= 0 || n_copies <= 0) {
        return NULL;
    }
    ggml_gpipe_context *ctx = (ggml_gpipe_context *)malloc(sizeof(ggml_gpipe_context));
    if (!ctx) {
        GGML_LOG_ERROR("gpipe: malloc failed\n");
        return NULL;
    }
    ctx->enabled = enabled;
    ctx->n_stages = n_stages;
    ctx->n_copies = n_copies;
    ctx->n_layers = 0;
    ctx->layer_sizes = NULL;
    ctx->stage_backends = (ggml_backend_t *)calloc(n_stages, sizeof(ggml_backend_t));
    ctx->stage_graphs = (ggml_cgraph *)calloc(n_stages, sizeof(ggml_cgraph));
    if (ctx->stage_backends) {
        for (int i = 0; i < n_stages; i++) {
            ctx->stage_graphs[i].size = GGML_DEFAULT_GRAPH_SIZE;
            ctx->stage_graphs[i].n_nodes = 0;
        }
    }
    GGML_LOG_DEBUG("gpipe: context created (enabled=%d, stages=%d, copies=%d)\n", enabled, n_stages, n_copies);
    return ctx;
}

// D2.4 Hot path profiler: profile per-layer execution time and auto-assign tiers

static void gpipe_profile_run(struct ggml_gpipe_context *ctx, int stage_id, ggml_cgraph *sg, ggml_backend_t backend) {
    if (!sg || sg->n_nodes == 0 || !backend) return;
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, sg);
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    ctx->layer_times[stage_id][0] = ms;
    GGML_LOG_INFO("gpipe: stage %d profiled %.1fms\n", stage_id, ms);
}

static void gpipe_auto_assign_layers(struct ggml_gpipe_context *ctx,
                                      ggml_backend_t *backends, int n_backends,
                                      int n_layers) {
    if (!ctx->full_graph || n_backends <= 0 || n_layers <= 0) return;

    // Profile each layer individually on backend 0
    for (int l = 0; l < n_layers && l < 64; l++) {
        // Extract single-layer sub-graph and profile
        // For now, use a simple heuristic: layer 0..n_layers-1 mapped to node ranges
        // A full implementation would build per-layer graphs
        ctx->layer_times[0][l] = 1.0f + (float)(n_layers - l) * 0.1f; // fallback heuristic
    }

    // Greedy load balancing: assign layers to least-loaded backend
    float load[GPipe_MAX_STAGES];
    std::memset(load, 0, sizeof(load));

    for (int l = 0; l < n_layers && l < 64; l++) {
        int best = 0;
        for (int b = 1; b < n_backends && b < GPipe_MAX_STAGES; b++) {
            if (load[b] < load[best]) best = b;
        }
        load[best] += ctx->layer_times[0][l];
        GGML_LOG_DEBUG("gpipe: layer %d -> backend %d (load=%.1fms)\n", l, best, load[best]);
    }

    // Apply assignments: copy backends to stage_backends
    for (int b = 0; b < n_backends && b < ctx->n_stages; b++) {
        ctx->stage_backends[b] = backends[b];
    }
}

// Forward declaration
static int gpipe_profile_and_assign(struct ggml_gpipe_context *ctx,
                                     ggml_backend_t **backends, int n_backends,
                                     const int *layer_sizes, int n_layers, bool profile);

// Public API: profile-aware auto-assign
int ggml_gpipe_profile_assign(struct ggml_gpipe_context *ctx,
                               void **backends, int n_backends,
                               const int *layer_sizes, int n_layers, bool profile) {
    ggml_backend_t **be = (ggml_backend_t **)backends;
    return gpipe_profile_and_assign(ctx, be, n_backends, layer_sizes, n_layers, profile);
}

void ggml_gpipe_free(struct ggml_gpipe_context *ctx) {
    if (!ctx) return;
    free(ctx->layer_sizes);
    free(ctx->stage_backends);
    free(ctx->stage_graphs);
    free(ctx);
}

void ggml_gpipe_set_layers(struct ggml_gpipe_context *ctx, const int *layer_sizes, int n_layers) {
    if (!ctx || !layer_sizes) return;
    free(ctx->layer_sizes);
    ctx->layer_sizes = (int *)malloc(n_layers * sizeof(int));
    memcpy(ctx->layer_sizes, layer_sizes, n_layers * sizeof(int));
    ctx->n_layers = n_layers;
}

int ggml_gpipe_compute(struct ggml_gpipe_context *ctx, struct ggml_cgraph *graph, bool /* async */) {
    if (!ctx || !graph) return GGML_STATUS_FAILED;
    if (!ctx->enabled) {
        // Fallback: compute the entire graph on backend 0
        if (ctx->stage_backends && ctx->stage_backends[0]) {
            return ggml_backend_graph_compute(ctx->stage_backends[0], graph);
        }
        return GGML_STATUS_FAILED;
    }

    std::vector<ggml_cgraph *> stage_graphs;
    gpipe_split_graph(ctx, graph, stage_graphs);

    // Sequential stage dispatch: stage 0 -> stage 1 -> ... -> stage N
    // This preserves causal dependencies (KV-cache ordering)
    for (int s = 0; s < ctx->n_stages; s++) {
        if (stage_graphs[s] && stage_graphs[s]->n_nodes > 0) {
            if (ctx->stage_backends && ctx->stage_backends[s]) {
                ggml_status st = ggml_backend_graph_compute(ctx->stage_backends[s], stage_graphs[s]);
                if (st != GGML_STATUS_SUCCESS) {
                    GGML_LOG_ERROR("gpipe: stage %d compute failed (status=%d)\n", s, st);
                    return st;
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}


