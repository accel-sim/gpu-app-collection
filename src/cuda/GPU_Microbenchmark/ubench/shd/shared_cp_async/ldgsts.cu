#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>

// ==============================
// Compile-time data type
// ==============================
#ifndef DATA_SIZE
#define DATA_SIZE 8   // default 16B (safe for .cg)
#endif

struct alignas(DATA_SIZE) Data {
    char bytes[DATA_SIZE];
};
static_assert(sizeof(Data) == DATA_SIZE, "Data struct size mismatch");


// sm_80+ required
__global__ void pipeline_kernel_async(const Data* __restrict__ global,
                                      uint64_t* __restrict__ clk_out,
                                      size_t total_elems,     // dataset size (global)
                                      size_t tile_copy_count, // per-thread slots in shared
                                      size_t loop)
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Data* shared = reinterpret_cast<Data*>(smem_raw);

    int tid = threadIdx.x;
    int threads = blockDim.x;
    size_t block_tile = threads * tile_copy_count; // elems per block per tile

    if (clk_out && tid == 0)
        clk_out[blockIdx.x * 2 + 0] = clock64();

    for (size_t j = 0; j < loop; j++) {
        // Load one tile (all threads cooperate)
        #pragma unroll
        for (size_t i = 0; i < tile_copy_count; ++i) {
            size_t g_idx = (j * block_tile + i * threads + tid) % total_elems;
            size_t s_idx = i * threads + tid;

            const void* gmem_ptr = &global[g_idx];
            void* smem_ptr       = &shared[s_idx];
            unsigned smem_addr   = __cvta_generic_to_shared(smem_ptr);

            if constexpr (DATA_SIZE == 16) {
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;\n"
                             :: "r"(smem_addr), "l"(gmem_ptr));
            } else if constexpr (DATA_SIZE == 8) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;\n"
                             :: "r"(smem_addr), "l"(gmem_ptr));
            } else if constexpr (DATA_SIZE == 4) {
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;\n"
                             :: "r"(smem_addr), "l"(gmem_ptr));
            }
        }

   
    }
     asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    if (clk_out && tid == 0)
        clk_out[blockIdx.x * 2 + 1] = clock64();
}



// ==============================
// Host Benchmark Driver
// ==============================
int main(int argc, char** argv) {
    size_t loop              = 1024;
    size_t num_blocks        = 4;
    size_t threads_per_block = 256;
    double l2_multiplier     = 1.0;  // dataset = multiplier × L2 size

    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <loop> <num_blocks> <threads_per_block> [L2_multiplier]\n";
        return 1;
    }

    loop              = std::atoi(argv[1]);
    num_blocks        = std::atoi(argv[2]);
    threads_per_block = std::atoi(argv[3]);
    if (argc == 5) l2_multiplier = std::atof(argv[4]);

    // ==============================
    // Device properties
    // ==============================
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t l2_size_bytes = prop.l2CacheSize;
    if (l2_size_bytes == 0) {
        std::cerr << "Warning: L2 cache size not reported, assuming 4MB\n";
        l2_size_bytes = 4 * 1024 * 1024;
    }

    // Dataset size ~ multiple of L2
    size_t total_bytes = static_cast<size_t>(l2_size_bytes * l2_multiplier);
    size_t total_elems = total_bytes / sizeof(Data);

    // Shared memory per block (opt-in max if available)
    size_t max_shared_mem = prop.sharedMemPerBlockOptin ?
                            prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;
    cudaFuncSetAttribute(pipeline_kernel_async,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem);

    // Round down to multiple of blockDim.x
    size_t smem_elems = (max_shared_mem / sizeof(Data) / threads_per_block) * threads_per_block;
    size_t smem_bytes = smem_elems * sizeof(Data);

    // Per-thread tile size
    size_t tile_copy_count = smem_elems / threads_per_block;
    if (tile_copy_count == 0) {
        std::cerr << "Error: not enough shared memory for even 1 element per thread.\n";
        return 1;
    }

    // ==============================
    // Print config
    // ==============================
    std::cout << "GPU Model              = " << prop.name << "\n";
    std::cout << "DATA_SIZE               = " << DATA_SIZE << " B\n";
    std::cout << "Device L2 size          = " << l2_size_bytes / 1024 << " KB\n";
    std::cout << "Working set multiplier  = " << l2_multiplier << "\n";
    std::cout << "Total elems             = " << total_elems << "\n";
    std::cout << "Total bytes             = " << total_bytes / (1024.0*1024) << " MB\n";
    std::cout << "Shared mem per SM (max) = "
          << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
    std::cout << "Shared mem per block    = " << smem_bytes / 1024.0 << " KB\n";
    std::cout << "Total shared mem (all blocks) = "
          << (smem_bytes * num_blocks) / (1024.0*1024.0) << " MB\n";
    std::cout << "Tile copy count/thread  = " << tile_copy_count << "\n";

    // ==============================
    // Allocate dataset
    // ==============================
    Data* h_data = new Data[total_elems];
    for (size_t i = 0; i < total_elems; i++) {
        h_data[i].bytes[0] = static_cast<char>(i & 0xFF);
    }

    Data* d_data;
    cudaMalloc(&d_data, total_bytes);
    cudaMemcpy(d_data, h_data, total_bytes, cudaMemcpyHostToDevice);

    uint64_t* d_clock;
    cudaMalloc(&d_clock, 2 * num_blocks * sizeof(uint64_t));
    cudaMemset(d_clock, 0, 2 * num_blocks * sizeof(uint64_t));

    // ==============================
    // Launch kernel
    // ==============================
    pipeline_kernel_async<<<num_blocks, threads_per_block, smem_bytes>>>(
        d_data, d_clock, total_elems, tile_copy_count, loop);

    // ==============================
    // Collect results
    // ==============================
// Collect results
uint64_t* h_clock = new uint64_t[2 * num_blocks];
cudaMemcpy(h_clock, d_clock, 2 * num_blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost);

// Average cycles across blocks
uint64_t total_cycles = 0;
for (size_t b = 0; b < num_blocks; b++) {
    total_cycles += (h_clock[2*b+1] - h_clock[2*b]);
}
double avg_cycles = static_cast<double>(total_cycles) / num_blocks;

// GPU frequency (kHz → Hz)
double gpu_clock_hz = static_cast<double>(prop.clockRate) * 1000.0;

// Time in seconds
double time_sec = avg_cycles / gpu_clock_hz;

// Total bytes moved
double bytes_moved = static_cast<double>(num_blocks) *
                     static_cast<double>(threads_per_block) *
                     static_cast<double>(tile_copy_count) *
                     static_cast<double>(DATA_SIZE) *
                     static_cast<double>(loop);

// Derived metrics
double bw_gbs   = bytes_moved / time_sec / 1e9;
double bytesclk = bytes_moved / avg_cycles;   // Bytes per GPU cycle

std::cout << "---------------------------------\n";

std::cout << "Avg cycles (per block) = " << avg_cycles << "\n";
std::cout << "Time (s)               = " << time_sec << "\n";
std::cout << "Bytes moved            = " << bytes_moved / (1024.0*1024*1024) << " GB\n";
std::cout << "Effective BW           = " << bw_gbs << " GB/s\n";
std::cout << "Bytes per cycle        = " << bytesclk << " B/clk\n";


    // ==============================
    // Cleanup
    // ==============================
    cudaFree(d_data);
    cudaFree(d_clock);
    delete[] h_data;
    delete[] h_clock;
    return 0;
}
