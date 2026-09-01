#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kWarpSize = 32;
constexpr int kUnset = -1;

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                               \
    if (error != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error));                                \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                        \
  } while (0)

// An atomic read makes the polling operation visible to both real hardware
// and the timing simulator.  Volta+ may schedule a runnable sibling lane while
// the current divergent path repeatedly executes this loop.
__device__ __forceinline__ int poll(const int *address) {
  return *reinterpret_cast<const volatile int *>(address);
}

// Lanes 1..31 can reach the wait before lane 0 reaches the publishing path.
// A non-preemptive PDOM stack can pin those waiters above lane 0 forever.
__global__ void forward_publish_kernel(int *output) {
  __shared__ int ready;
  __shared__ int payload;
  const int lane = threadIdx.x;

  if (lane == 0) {
    ready = 0;
    payload = 0;
  }
  __syncthreads();

  if (lane != 0) {
    while (poll(&ready) == 0) {
    }
    output[lane] = payload + lane;
  } else {
    payload = 1000;
    *reinterpret_cast<volatile int *>(&ready) = 1;
    output[0] = 1000;
  }
}

// Mirror the predicate and producer lane.  This catches implementations that
// accidentally make progress only for one branch direction or lane ordering.
__global__ void reverse_publish_kernel(int *output) {
  __shared__ int ready;
  __shared__ int payload;
  const int lane = threadIdx.x;

  if (lane == 0) {
    ready = 0;
    payload = 0;
  }
  __syncthreads();

  if (lane != kWarpSize - 1) {
    while (poll(&ready) == 0) {
    }
    output[lane] = payload - lane;
  } else {
    payload = 2000;
    *reinterpret_cast<volatile int *>(&ready) = 1;
    output[lane] = 2000 - lane;
  }
}

// The active owner changes 32 times. Each lane must leave the polling path,
// publish its trace slot, and wake the next lane. This exercises many dynamic
// split masks rather than a single producer/consumer split.
__global__ void token_ring_kernel(int *output) {
  __shared__ int token;
  const int lane = threadIdx.x;

  if (lane == 0) *reinterpret_cast<volatile int *>(&token) = 0;
  __syncthreads();

  while (poll(&token) != lane) {
  }
  output[lane] = 3000 + lane;
  *reinterpret_cast<volatile int *>(&token) = lane + 1;
}

enum TestId { kRing, kForward, kReverse, kTestCount };

const char *const kTestNames[kTestCount] = {
    "token-ring", "forward-publish", "reverse-publish"};

int expected_value(TestId test, int lane) {
  switch (test) {
    case kForward:
      return 1000 + lane;
    case kReverse:
      return 2000 - lane;
    case kRing:
      return 3000 + lane;
    default:
      return kUnset;
  }
}

void launch(TestId test, int *device_output) {
  switch (test) {
    case kForward:
      forward_publish_kernel<<<1, kWarpSize>>>(device_output);
      break;
    case kReverse:
      reverse_publish_kernel<<<1, kWarpSize>>>(device_output);
      break;
    case kRing:
      token_ring_kernel<<<1, kWarpSize>>>(device_output);
      break;
    default:
      std::abort();
  }
}

bool run_test(TestId test) {
  int *device_output = nullptr;
  int host_output[kWarpSize];
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(host_output)));
  CUDA_CHECK(cudaMemset(device_output, 0xff, sizeof(host_output)));

  launch(test, device_output);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_output, device_output, sizeof(host_output),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));

  int failures = 0;
  for (int lane = 0; lane < kWarpSize; ++lane) {
    const int expected = expected_value(test, lane);
    if (host_output[lane] != expected) {
      std::printf("  lane %2d: got %d, expected %d\n", lane,
                  host_output[lane], expected);
      ++failures;
    }
  }
  std::printf("[%s] %s\n", failures == 0 ? "PASS" : "FAIL",
              kTestNames[test]);
  return failures == 0;
}

int find_test(const char *name) {
  for (int i = 0; i < kTestCount; ++i) {
    if (std::strcmp(name, kTestNames[i]) == 0) return i;
  }
  return -1;
}

}  // namespace

int main(int argc, char **argv) {
  int first = 0;
  int last = kTestCount;
  if (argc == 2) {
    first = find_test(argv[1]);
    if (first < 0) {
      std::fprintf(stderr, "Unknown test '%s'. Available tests:\n", argv[1]);
      for (const char *name : kTestNames) std::fprintf(stderr, "  %s\n", name);
      return EXIT_FAILURE;
    }
    last = first + 1;
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [test-name]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int failures = 0;
  for (int test = first; test < last; ++test) {
    if (!run_test(static_cast<TestId>(test))) ++failures;
  }

  std::printf("RESULT: %s (%d/%d tests passed)\n",
              failures == 0 ? "PASSED" : "FAILED",
              last - first - failures, last - first);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
