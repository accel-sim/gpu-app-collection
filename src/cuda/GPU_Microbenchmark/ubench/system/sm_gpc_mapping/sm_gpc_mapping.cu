// Dumps the per-SM (GPC, TPC) mapping for the local GPU using
// NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS, then runs a small kernel that
// captures %smid from many blocks and cross-checks the result.
//
// Build:   make release  (in this directory)
// Run:     ../../../bin/sm_gpc_mapping
//
// Output is two streams: '#'-prefixed human/diagnostic lines on stdout,
// and a CSV block at the end (smid,gpcId,tpcId,observed_in_kernel).

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
using namespace std;

#include "../../../hw_def/hw_def.h"

__device__ __forceinline__ unsigned get_smid()
{
    unsigned ret;
    asm("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ unsigned get_cluster_ctarank()
{
    unsigned ret;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(ret));
    return ret;
}

// Each block writes its %smid into out[blockIdx.x]. The spin keeps the block
// resident long enough that the scheduler is forced to dispatch siblings to
// other SMs, giving us coverage of every enabled SM.
__global__ void capture_smid(unsigned *out, unsigned spin_iters)
{
    unsigned smid = get_smid();
    if (threadIdx.x == 0)
        out[blockIdx.x] = smid;

    volatile unsigned acc = 0;
    for (unsigned i = 0; i < spin_iters; ++i)
        acc += i;
    if (threadIdx.x == 0 && acc == 0xdeadbeef)
        out[blockIdx.x] = 0xffffffffu;  // unreachable; defeats DCE
}

// Cluster-aware variant: captures both %smid and %cluster_ctarank per block.
// Indexed by the linear block id derived from blockIdx + gridDim.
__global__ void capture_smid_with_cluster(unsigned *smid_out,
                                          unsigned *crank_out,
                                          unsigned spin_iters)
{
    unsigned smid = get_smid();
    unsigned crank = get_cluster_ctarank();

    if (threadIdx.x == 0) {
        unsigned linear = blockIdx.x
                        + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        smid_out[linear] = smid;
        crank_out[linear] = crank;
    }

    volatile unsigned acc = 0;
    for (unsigned i = 0; i < spin_iters; ++i)
        acc += i;
    if (threadIdx.x == 0 && acc == 0xdeadbeef)
        smid_out[0] = 0xffffffffu;
}

int main(int argc, char *argv[])
{
    initializeDeviceProp(0, argc, argv);

    printf("\n# Device Name        = %s\n", deviceProp.name);
    printf("# Compute Capability = %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("# SM_NUMBER          = %u\n", config.SM_NUMBER);
    printf("# NUM_GPCS (RM)      = %u\n", config.NUM_GPCS);
    printf("# FBP_COUNT, L2_BANKS = %u, %u\n", config.FBP_COUNT, config.L2_BANKS);

    vector<SmGpcTpcEntry> mapping = querySmToGpcMapping();
    printf("# RM smCount         = %zu\n", mapping.size());

    if (mapping.empty()) {
        fprintf(stderr,
                "# ERROR: querySmToGpcMapping returned no entries. "
                "The driver may not support NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS, "
                "or /dev/nvidiactl access was denied.\n");
        return 1;
    }

    // Build per-GPC and per-(GPC,TPC) reverse indexes.
    map<uint32_t, vector<uint32_t>> gpc_to_sms;
    map<pair<uint32_t, uint32_t>, vector<uint32_t>> gpc_tpc_to_sms;
    uint32_t max_gpc = 0, max_tpc = 0;
    for (size_t smid = 0; smid < mapping.size(); ++smid) {
        auto e = mapping[smid];
        gpc_to_sms[e.gpcId].push_back((uint32_t)smid);
        gpc_tpc_to_sms[{e.gpcId, e.tpcId}].push_back((uint32_t)smid);
        max_gpc = std::max(max_gpc, e.gpcId);
        max_tpc = std::max(max_tpc, e.tpcId);
    }
    printf("# distinct GPCs      = %zu  (max gpcId=%u, max tpcId=%u)\n",
           gpc_to_sms.size(), max_gpc, max_tpc);

    printf("# GPC histogram (gpcId : sm_count : sm_ids):\n");
    for (auto &kv : gpc_to_sms) {
        printf("#   GPC %u: %zu SMs [", kv.first, kv.second.size());
        for (size_t i = 0; i < kv.second.size(); ++i)
            printf("%s%u", i ? "," : "", kv.second[i]);
        printf("]\n");
    }

    // Sanity invariants printed loudly so failures are obvious.
    bool ok_count =
        (config.NUM_GPCS == 0) || (gpc_to_sms.size() == config.NUM_GPCS);
    bool ok_total = (mapping.size() == config.SM_NUMBER);
    printf("# CHECK distinct_gpcs(%zu) == NUM_GPCS(%u) : %s\n",
           gpc_to_sms.size(), config.NUM_GPCS, ok_count ? "OK" : "MISMATCH");
    printf("# CHECK rm_smCount(%zu)    == SM_NUMBER(%u)   : %s\n",
           mapping.size(), config.SM_NUMBER, ok_total ? "OK" : "MISMATCH");

    // Runtime %smid capture: launch enough blocks to cover every enabled SM.
    unsigned blocks = config.SM_NUMBER * 4;
    if (blocks < 64)
        blocks = 64;
    unsigned *d_out = nullptr;
    gpuErrchk(cudaMalloc(&d_out, blocks * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_out, 0xff, blocks * sizeof(unsigned)));

    // ~10us per block at ~1.6GHz keeps blocks resident long enough for the
    // scheduler to fan siblings out to other SMs without a slow run.
    capture_smid<<<blocks, 32>>>(d_out, 1u << 14);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    vector<unsigned> h_out(blocks);
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, blocks * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));
    cudaFree(d_out);

    set<unsigned> observed_smids(h_out.begin(), h_out.end());
    printf("# kernel observed %zu distinct %%smid values across %u blocks\n",
           observed_smids.size(), blocks);

    bool all_observed_in_table = true;
    for (auto smid : observed_smids) {
        if (smid >= mapping.size()) {
            printf("# WARN: kernel saw %%smid=%u but RM table size = %zu\n",
                   smid, mapping.size());
            all_observed_in_table = false;
        }
    }
    printf("# CHECK observed_smids subset_of rm_table        : %s\n",
           all_observed_in_table ? "OK" : "MISMATCH");
    printf("# CHECK observed_count(%zu) == SM_NUMBER(%u)        : %s\n",
           observed_smids.size(), config.SM_NUMBER,
           observed_smids.size() == config.SM_NUMBER ? "OK"
                                                     : "PARTIAL (scheduler did not cover all SMs)");

    // Stable CSV table.
    printf("\n# CSV: smid,gpcId,tpcId,observed_in_kernel\n");
    for (size_t smid = 0; smid < mapping.size(); ++smid) {
        bool seen = observed_smids.count((unsigned)smid) != 0;
        printf("%zu,%u,%u,%d\n", smid, mapping[smid].gpcId,
               mapping[smid].tpcId, seen ? 1 : 0);
    }

    // ============================================================
    // Cluster-shape sweep
    // ============================================================
    // For each cluster shape, launch one block per cluster slot covering as
    // many SMs as fit (clusters of size K -> floor(SM_NUMBER / K) clusters).
    // Per block we capture %smid and %cluster_ctarank, then look up gpcId/tpcId
    // via the RM-reported mapping. This shows how blocks within a cluster
    // distribute across one GPC's TPCs and how cluster_id rasterizes across
    // GPCs.
    //
    // Note: user request listed "1x8x8" as the last shape, but cluster_size=64
    // exceeds the maximum cluster size (8 portable / 16 non-portable on H100).
    // Treating it as a typo for "1x8x1" so we still cover an 8-block cluster
    // laid out along y. Edit the shapes[] array below if a different
    // interpretation is wanted.
    struct ClusterShape {
        int x, y, z;
        const char *name;
    };
    const ClusterShape shapes[] = {
        {1, 1, 1, "1x1x1"}, {1, 2, 1, "1x2x1"}, {2, 1, 1, "2x1x1"},
        {2, 2, 1, "2x2x1"}, {1, 4, 1, "1x4x1"}, {4, 1, 1, "4x1x1"},
        {2, 4, 1, "2x4x1"}, {4, 2, 1, "4x2x1"}, {8, 1, 1, "8x1x1"},
        {1, 8, 1, "1x8x1"}, // user typed "1x8x8"; size=64 > max, treating as typo
    };

    auto cluster_id_from_block = [](int bx, int by, int bz, int gx, int gy,
                                    const ClusterShape &s) {
        int cx = bx / s.x, cy = by / s.y, cz = bz / s.z;
        int ncx = gx / s.x, ncy = gy / s.y;
        return cx + ncx * (cy + ncy * cz);
    };

    for (const auto &s : shapes) {
        int csize = s.x * s.y * s.z;
        int nclusters = (int)(config.SM_NUMBER / (unsigned)csize);
        if (nclusters < 1) {
            printf("\n# === cluster shape %s skipped (cluster_size=%d > "
                   "SM_NUMBER=%u) ===\n",
                   s.name, csize, config.SM_NUMBER);
            continue;
        }

        // Lay clusters out along x: grid.x = s.x * nclusters, grid.y = s.y,
        // grid.z = s.z. This trivially satisfies gridDim.* % clusterDim.* == 0.
        dim3 grid(s.x * nclusters, s.y, s.z);
        dim3 block(32, 1, 1);
        unsigned total_blocks = grid.x * grid.y * grid.z;

        unsigned *d_smid = nullptr, *d_crank = nullptr;
        gpuErrchk(cudaMalloc(&d_smid, total_blocks * sizeof(unsigned)));
        gpuErrchk(cudaMalloc(&d_crank, total_blocks * sizeof(unsigned)));
        gpuErrchk(cudaMemset(d_smid, 0xff, total_blocks * sizeof(unsigned)));
        gpuErrchk(cudaMemset(d_crank, 0xff, total_blocks * sizeof(unsigned)));

        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = 0;

        cudaLaunchAttribute attr[1] = {};
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = s.x;
        attr[0].val.clusterDim.y = s.y;
        attr[0].val.clusterDim.z = s.z;
        cfg.attrs = attr;
        cfg.numAttrs = 1;

        cudaError_t err = cudaLaunchKernelEx(&cfg, capture_smid_with_cluster,
                                             d_smid, d_crank, 1u << 14);
        if (err != cudaSuccess) {
            printf("\n# === cluster shape %s launch failed: %s ===\n", s.name,
                   cudaGetErrorString(err));
            cudaFree(d_smid);
            cudaFree(d_crank);
            continue;
        }
        gpuErrchk(cudaDeviceSynchronize());

        vector<unsigned> h_smid(total_blocks), h_crank(total_blocks);
        gpuErrchk(cudaMemcpy(h_smid.data(), d_smid,
                             total_blocks * sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_crank.data(), d_crank,
                             total_blocks * sizeof(unsigned),
                             cudaMemcpyDeviceToHost));
        cudaFree(d_smid);
        cudaFree(d_crank);

        // Verify single-GPC-per-cluster invariant and record per-cluster GPC.
        map<int, set<uint32_t>> cluster_gpcs;
        map<int, set<uint32_t>> cluster_tpcs;
        map<int, vector<unsigned>> cluster_smids;
        for (unsigned i = 0; i < total_blocks; ++i) {
            unsigned smid = h_smid[i];
            if (smid >= mapping.size())
                continue;
            int bx = (int)(i % grid.x);
            int by = (int)((i / grid.x) % grid.y);
            int bz = (int)(i / (grid.x * grid.y));
            int cid =
                cluster_id_from_block(bx, by, bz, grid.x, grid.y, s);
            cluster_gpcs[cid].insert(mapping[smid].gpcId);
            cluster_tpcs[cid].insert(mapping[smid].tpcId);
            cluster_smids[cid].push_back(smid);
        }

        bool single_gpc = true;
        for (auto &kv : cluster_gpcs)
            if (kv.second.size() != 1)
                single_gpc = false;

        printf(
            "\n# === cluster shape %s (size=%d, nclusters=%d, grid=%ux%ux%u, "
            "blocks=%u) ===\n",
            s.name, csize, nclusters, grid.x, grid.y, grid.z, total_blocks);
        printf("# CHECK every cluster -> single GPC: %s\n",
               single_gpc ? "OK" : "MISMATCH");

        // cluster_id -> gpc summary, ordered by cluster_id
        printf("# cluster_id -> gpcId  (sm_count, smids):\n");
        for (auto &kv : cluster_gpcs) {
            int cid = kv.first;
            uint32_t g = *kv.second.begin();
            auto &sms = cluster_smids[cid];
            printf("#   cluster %3d -> GPC %u  (%zu SMs: ", cid, g, sms.size());
            for (size_t i = 0; i < sms.size(); ++i)
                printf("%s%u", i ? "," : "", sms[i]);
            printf(")\n");
        }

        // Per-block CSV.
        printf("# CSV: linear,blockIdx_x,blockIdx_y,blockIdx_z,cluster_id,"
               "rank_in_cluster,smid,gpcId,tpcId\n");
        for (unsigned i = 0; i < total_blocks; ++i) {
            int bx = (int)(i % grid.x);
            int by = (int)((i / grid.x) % grid.y);
            int bz = (int)(i / (grid.x * grid.y));
            int cid = cluster_id_from_block(bx, by, bz, grid.x, grid.y, s);
            unsigned smid = h_smid[i];
            uint32_t gpc =
                (smid < mapping.size()) ? mapping[smid].gpcId : 0xffffffffu;
            uint32_t tpc =
                (smid < mapping.size()) ? mapping[smid].tpcId : 0xffffffffu;
            printf("%u,%d,%d,%d,%d,%u,%u,%u,%u\n", i, bx, by, bz, cid,
                   h_crank[i], smid, gpc, tpc);
        }
    }

    return 0;
}
