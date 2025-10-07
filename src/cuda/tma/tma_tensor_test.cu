// Adapt from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=tma#using-tma-to-transfer-multi-dimensional-arrays
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda.h>         // CUtensormap
#include <cuda/barrier>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unordered_map>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

/*
 * Test application for TMA tensor operations.
 * 
 * Usage: ./tma_tensor_test -w <width> -h <height> -o <opcode>
 * 
 */

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err)                                             \
        {                                                                   \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#define GMEM_WIDTH 1024
#define GMEM_HEIGHT 1024
#define SMEM_WIDTH 32
#define SMEM_HEIGHT 32

enum class TestType {
    UTMAPF,
    UTMALDG,
    UTMALDG_L2Hint,
    UTMASTG,
    UTMAREDG,
    REGULAR_LOAD
};

static const std::unordered_map<std::string, TestType> opcode_map = {
    {"UTMAPF", TestType::UTMAPF},
    {"UTMALDG", TestType::UTMALDG},
    {"UTMASTG", TestType::UTMASTG},
    {"UTMAREDG", TestType::UTMAREDG},
    {"REGULAR_LOAD", TestType::REGULAR_LOAD}
};

__global__ void test_kernel(const __grid_constant__ CUtensorMap tensor_map, TestType test_type, int width_stride);
__device__ void test_UTMAPF_kernel(CUtensorMap const& tensor_map, int x, int y);
__device__ void test_UTMALDG_kernel_L2Hint(CUtensorMap const& tensor_map, int x, int y);
__device__ void test_UTMALDG_kernel(CUtensorMap const& tensor_map, int x, int y);
__device__ void test_UTMASTG_kernel(CUtensorMap const& tensor_map, int x, int y);
__device__ void test_UTMAREDG_kernel(CUtensorMap const& tensor_map, int x, int y);
__device__ void test_REGULAR_LOAD_kernel(int *mat, int x, int y, int width_stride);

__global__ void test_kernel(const __grid_constant__ CUtensorMap tensor_map, int *mat, TestType test_type, int width_stride) {
    int x = blockDim.x * blockIdx.x;
    int y = blockDim.y * blockIdx.y;
    if (blockIdx.x == 0 && blockIdx.y == 0 &&
        threadIdx.x == 0 && threadIdx.y == 0) {
        printf("TensorMap address: %p\n", &tensor_map);
    }
    switch (test_type) {
        case TestType::UTMAPF:
            test_UTMAPF_kernel(tensor_map, x, y);
            break;
        case TestType::UTMALDG:
            test_UTMALDG_kernel(tensor_map, x, y);
            break;
        case TestType::UTMASTG:
            test_UTMASTG_kernel(tensor_map, x, y);
            break;
        case TestType::UTMAREDG:
            test_UTMAREDG_kernel(tensor_map, x, y);
            break;
        case TestType::REGULAR_LOAD:
            test_REGULAR_LOAD_kernel(mat, x, y, width_stride);
            break;
        default:
            test_UTMAPF_kernel(tensor_map, x, y);
            break;
    }
}

__device__ void test_UTMAPF_kernel(CUtensorMap const& tensor_map, int x, int y) {
    // TensorMap prefetch at tensor_map with tensor coord {x, y}
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        asm volatile (
            "cp.async.bulk.prefetch.tensor.2d.L2.global.tile"
            " [%0, {%1, %2}];"
            :
            : "l"(&tensor_map),
              "r"(x),
              "r"(y)
            : "memory");
    }
}

__device__ void test_UTMALDG_kernel(CUtensorMap const& tensor_map, int x, int y) {
    // The destination shared memory buffer of a bulk tensor operation should be
    // 128 byte aligned.
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        // Initialize barrier. All threads in block participate.
        init(&bar, blockDim.x * blockDim.y);
        // Make initialized barrier visible in async proxy.
        ptx::fence_proxy_async(ptx::space_shared);
    }
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    barrier::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Initiate bulk tensor copy.
        ptx::cp_async_bulk_tensor(
            ptx::space_cluster,
            ptx::space_global,
            &smem_buffer, 
            &tensor_map,
            {x, y},
            cuda::device::barrier_native_handle(bar)
        );
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    }
    else
    {
        // Other threads just arrive.
        token = bar.arrive();
    }
    // Wait for the data to have arrived.
    bar.wait(std::move(token));
}

__device__ void test_UTMASTG_kernel(CUtensorMap const& tensor_map, int x, int y) {
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // Compute a unique value for the thread
    int thread_x = threadIdx.x + x;
    int thread_y = threadIdx.y + y;
    smem_buffer[threadIdx.y][threadIdx.x] = 1 + thread_x + thread_y * blockDim.x * gridDim.x;

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ptx::cp_async_bulk_tensor(
            ptx::space_global,
            ptx::space_shared,
            &tensor_map,
            {x, y},
            &smem_buffer
        );
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}

__device__ void test_UTMAREDG_kernel(CUtensorMap const& tensor_map, int x, int y) {
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // Compute a unique value for the thread
    int thread_x = threadIdx.x + x;
    int thread_y = threadIdx.y + y;
    // Add 1 so the max op will not be effective
    smem_buffer[threadIdx.y][threadIdx.x] = 1 + thread_x + thread_y * blockDim.x * gridDim.x;

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ptx::cp_reduce_async_bulk_tensor(
            ptx::space_global,
            ptx::space_shared,
            ptx::op_max,
            &tensor_map,
            {x, y},
            &smem_buffer
        );
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
}

__device__ void test_REGULAR_LOAD_kernel(int *mat, int x, int y, int width_stride) {
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // Compute a unique value for the thread
    int thread_x = threadIdx.x + x;
    int thread_y = threadIdx.y + y;
    // Mimic a TMA load pattern here
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int row = 0; row < SMEM_HEIGHT; row++) {
            for (int col = 0; col < SMEM_WIDTH; col++) {
                smem_buffer[row][col] = mat[(y + row) * width_stride + (x + col)];
            }
        }
    }
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled()
{
    // Get pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void *cuTensorMapEncodeTiled_ptr = nullptr;
    CUDA_SAFECALL(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
    assert(driver_status == cudaDriverEntryPointSuccess);

    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}


int main(int argc, char *argv[]) {
    uint64_t width = GMEM_WIDTH;
    uint64_t height = GMEM_HEIGHT;
    std::string opcode = "UTMAPF";
    TestType test_type = TestType::UTMAPF;
    int opt;
    while ((opt = getopt(argc, argv, "w:h:o:")) != -1) {
        switch (opt) {
            case 'w':
                width = uint64_t(atoi(optarg));
                break;
            case 'h':
                height = uint64_t(atoi(optarg));
                break;
            case 'o':
                opcode = std::string(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s -w <width> -h <height> -o <opcode>\n", argv[0]);
                fprintf(stderr, "  Block size: %d x %d\n", SMEM_WIDTH, SMEM_HEIGHT);
                fprintf(stderr, "  -w <width>: width of the matrix\n");
                fprintf(stderr, "  -h <height>: height of the matrix\n");
                fprintf(stderr, "  -o <opcode>: opcode\n");
                fprintf(stderr, "  -o UTMAPF: tensor prefetch\n");
                fprintf(stderr, "  -o UTMALDG: tensor load async\n");
                fprintf(stderr, "  -o UTMASTG: tensor store async\n");
                fprintf(stderr, "  -o UTMAREDG: tensor reduce async\n");
                return 1;
        }
    }

    // Check if opcode is valid
    if (opcode_map.find(opcode) == opcode_map.end()) {
        fprintf(stderr, "Invalid opcode\n");
        return 1;
    }

    test_type = opcode_map.at(opcode);

    // Initialize data matrix
    int *mat, *out_mat, *d_mat;
    // height|width_stride must be a multiple of 16 and must be greater than height|width
    // Here we make it multiple of SMEM_HEIGHT and SMEM_WIDTH to fit our shmem size
    uint64_t height_stride = ((height + (SMEM_HEIGHT - 1)) / SMEM_HEIGHT) * SMEM_HEIGHT;
    uint64_t width_stride = ((width + (SMEM_WIDTH - 1)) / SMEM_WIDTH) * SMEM_WIDTH;
    printf("height: %lu, width: %lu\n", height, width);
    printf("height_stride: %lu, width_stride: %lu\n", height_stride, width_stride);
    size_t byte_count = height_stride * width_stride * sizeof(int);
    mat = (int *)malloc(byte_count);
    out_mat = (int *)malloc(byte_count);
    cudaMalloc(&d_mat, byte_count);

    // Initialize the matrix, set the values to 0 for out of bounds
    uint64_t i = 1;
    for (uint64_t r = 0; r < height_stride; r++) {
        for (uint64_t c = 0; c < width_stride; c++) {
            // If within the tensor, set some unique values
            if (r < height && c < width) {
                mat[r * width_stride + c] = i;
                i++;
            }
            // Otherwise, set the value to 0
            else {
                mat[r * width_stride + c] = 0;
            }
        }
    }
    cudaMemcpy(d_mat, mat, byte_count, cudaMemcpyHostToDevice);

    // TMA tensor map object
    CUtensorMap tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;
    // The tensor size
    uint64_t size[rank] = {width, height};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {width_stride * sizeof(int)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride[rank] = {1, 1};

    // Get a function pointer to the cuTensorMapEncodeTiled driver API.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map, // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,        // cuuint32_t tensorRank,
        d_mat,       // void *globalAddress,
        size,        // const cuuint64_t *globalDim,
        stride,      // const cuuint64_t *globalStrides,
        box_size,    // const cuuint32_t *boxDim,
        elem_stride, // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Kernel launch
    dim3 grid_dim(width_stride / SMEM_WIDTH, height_stride / SMEM_HEIGHT);
    dim3 block_dim(SMEM_WIDTH, SMEM_HEIGHT);
    printf("grid_dim: x: %d, y: %d\n", grid_dim.x, grid_dim.y);
    printf("block_dim: x: %d, y: %d\n", block_dim.x, block_dim.y);
    CUDA_SAFECALL((test_kernel<<<grid_dim, block_dim>>>(tensor_map, d_mat, test_type, width_stride)));
    CUDA_SAFECALL(cudaMemcpy(out_mat, d_mat, byte_count, cudaMemcpyDeviceToHost));

    // Print the matrix to output file
    // Dump the values to a file in hex format
    // For load operations, use host source array
    int *ptr = mat;

    // For store operations, use destination array
    if (test_type == TestType::UTMASTG || test_type == TestType::UTMAREDG) {
        ptr = out_mat;
    }

    char filename[100];
    sprintf(filename, "tma_tensor_test_%s_%lu_%lu.txt", opcode.c_str(), height, width);
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < height_stride * width_stride; i++)
    {
        fprintf(f, "0x%x ", ptr[i]);
        // Add line break after every 512 values
        if ((i + 1) % 512 == 0)
            fprintf(f, "\n");
    }
    fclose(f);
    printf("Values dumped to %s\n", filename);

    // Release device memory
    cudaFree(d_mat);

    // Release host memory
    free(mat);
    return 0;
}
