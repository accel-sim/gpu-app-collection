#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

#define CUDACHK(call)                                                         \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " @ "       \
                << #call << "\n";                                            \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

float bits_to_f32(uint32_t bits) {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

enum class rs_minifloat_family_t {
  kE4M3,
  kE5M2,
  kE2M1,
  kE2M3,
  kE3M2,
};

struct rs_minifloat_entry_t {
  uint16_t code;
  float value;
};

unsigned rs_minifloat_code_limit(rs_minifloat_family_t fmt) {
  switch (fmt) {
    case rs_minifloat_family_t::kE4M3:
    case rs_minifloat_family_t::kE5M2:
      return 256u;
    case rs_minifloat_family_t::kE2M1:
      return 16u;
    case rs_minifloat_family_t::kE2M3:
    case rs_minifloat_family_t::kE3M2:
      return 64u;
  }
  return 0u;
}

float decode_rs_minifloat_code(rs_minifloat_family_t fmt, uint16_t code) {
  int exp_bits = 0;
  int mant_bits = 0;
  int bias = 0;
  bool has_inf = false;
  switch (fmt) {
    case rs_minifloat_family_t::kE4M3:
      exp_bits = 4;
      mant_bits = 3;
      bias = 7;
      has_inf = false;
      break;
    case rs_minifloat_family_t::kE5M2:
      exp_bits = 5;
      mant_bits = 2;
      bias = 15;
      has_inf = true;
      break;
    case rs_minifloat_family_t::kE2M1:
      exp_bits = 2;
      mant_bits = 1;
      bias = 1;
      has_inf = false;
      break;
    case rs_minifloat_family_t::kE2M3:
      exp_bits = 2;
      mant_bits = 3;
      bias = 1;
      has_inf = false;
      break;
    case rs_minifloat_family_t::kE3M2:
      exp_bits = 3;
      mant_bits = 2;
      bias = 3;
      has_inf = false;
      break;
  }

  const int sign_shift = exp_bits + mant_bits;
  const bool neg = ((code >> sign_shift) & 0x1u) != 0u;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;
  const int exp = (code >> mant_bits) & exp_mask;
  const int mant = code & mant_mask;

  if (fmt == rs_minifloat_family_t::kE4M3 &&
      (code == 0x7fu || code == 0xffu)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (has_inf && exp == exp_mask) {
    if (mant == 0) {
      return neg ? -std::numeric_limits<float>::infinity()
                 : std::numeric_limits<float>::infinity();
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  float value = 0.0f;
  if (exp == 0) {
    if (mant != 0) {
      value = std::ldexp(static_cast<float>(mant), 1 - bias - mant_bits);
    }
  } else {
    const int sig = (1 << mant_bits) | mant;
    value = std::ldexp(static_cast<float>(sig), exp - bias - mant_bits);
  }
  return neg ? -value : value;
}

std::vector<float> collect_rs_midpoint_ties(rs_minifloat_family_t fmt,
                                            size_t max_count) {
  std::vector<rs_minifloat_entry_t> entries;
  entries.reserve(rs_minifloat_code_limit(fmt));
  for (unsigned code = 0; code < rs_minifloat_code_limit(fmt); ++code) {
    const float value = decode_rs_minifloat_code(fmt, static_cast<uint16_t>(code));
    if (!std::isfinite(value)) continue;
    entries.push_back({static_cast<uint16_t>(code), value});
  }

  std::sort(entries.begin(), entries.end(),
            [](const rs_minifloat_entry_t &a, const rs_minifloat_entry_t &b) {
              if (a.value < b.value) return true;
              if (a.value > b.value) return false;
              return a.code < b.code;
            });

  std::vector<float> out;
  out.reserve(max_count);
  for (size_t i = 0; i + 1 < entries.size() && out.size() < max_count; ++i) {
    const float lo = entries[i].value;
    const float hi = entries[i + 1].value;
    if (!(lo < hi)) continue;
    const long double midpoint =
        (static_cast<long double>(lo) + static_cast<long double>(hi)) * 0.5L;
    if (!std::isfinite(static_cast<double>(midpoint))) continue;
    const float mid_f32 = static_cast<float>(midpoint);
    if (!std::isfinite(mid_f32)) continue;
    if (static_cast<long double>(mid_f32) != midpoint) continue;
    out.push_back(mid_f32);
  }
  return out;
}

std::vector<uint32_t> edge_f32_patterns() {
  return {
      0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu,
      0x807fffffu, 0x00800000u, 0x80800000u, 0x00800001u, 0x80800001u,
      0x3f7fffffu, 0x3f800000u, 0x3f800001u, 0xbf7fffffu, 0xbf800000u,
      0xbf800001u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u,
      0x7fc00000u, 0xffc00000u, 0x7fa00001u, 0xffa00001u,
  };
}

void build_pair_rs_inputs(std::vector<float> &a, std::vector<float> &b,
                          std::vector<uint32_t> &rbits) {
  a.clear();
  b.clear();
  rbits.clear();

  std::mt19937 rng(0x84f0a313u);

  const std::vector<uint32_t> edges = edge_f32_patterns();
  for (size_t i = 0; i + 1 < edges.size(); i += 2) {
    a.push_back(bits_to_f32(edges[i]));
    b.push_back(bits_to_f32(edges[i + 1]));

    const uint32_t ra = rng() & 0x1fffu;
    const uint32_t rb = rng() & 0x1fffu;
    rbits.push_back((ra << 16) | rb);
  }

  const uint32_t manual_rbits[] = {
      0x00000000u, 0x00010001u, 0x1fff1fffu, 0x1fff0000u,
      0x00001fffu, 0x12340789u, 0x15550aaau,
  };
  for (uint32_t r : manual_rbits) {
    a.push_back(bits_to_f32(rng()));
    b.push_back(bits_to_f32(rng()));
    rbits.push_back(static_cast<uint32_t>(((r >> 16) & 0x1fffu) << 16) |
                    (r & 0x1fffu));
  }

  // Deterministic tie-focused probes for .rs pair conversions.
  const uint32_t midpoint_bits[] = {
      0x3f801000u, 0xbf801000u, 0x3f001000u, 0xbf001000u,
  };
  const uint16_t tie_rbits[] = {0x0fffu, 0x1000u};
  for (uint32_t bits : midpoint_bits) {
    const float x = bits_to_f32(bits);
    for (uint16_t r : tie_rbits) {
      a.push_back(x);
      b.push_back(x);
      rbits.push_back((static_cast<uint32_t>(r) << 16) | r);
    }
  }

  while (a.size() < 1024) {
    a.push_back(bits_to_f32(rng()));
    b.push_back(bits_to_f32(rng()));

    const uint32_t ra = rng() & 0x1fffu;
    const uint32_t rb = rng() & 0x1fffu;
    rbits.push_back((ra << 16) | rb);
  }
}

void build_quad_rs_inputs(std::vector<float> &a, std::vector<float> &b,
                          std::vector<float> &e, std::vector<float> &f,
                          std::vector<uint32_t> &rbits) {
  a.clear();
  b.clear();
  e.clear();
  f.clear();
  rbits.clear();

  std::mt19937 rng(0x2f08bc41u);

  const std::vector<uint32_t> edges = edge_f32_patterns();
  for (size_t i = 0; i + 3 < edges.size(); i += 4) {
    a.push_back(bits_to_f32(edges[i]));
    b.push_back(bits_to_f32(edges[i + 1]));
    e.push_back(bits_to_f32(edges[i + 2]));
    f.push_back(bits_to_f32(edges[i + 3]));
    rbits.push_back(rng());
  }

  const uint32_t manual_rbits[] = {
      0x00000000u, 0xffffffffu, 0x01020304u, 0x11223344u,
      0x89abcdefu, 0xfedcba98u,
  };
  for (uint32_t r : manual_rbits) {
    a.push_back(bits_to_f32(rng()));
    b.push_back(bits_to_f32(rng()));
    e.push_back(bits_to_f32(rng()));
    f.push_back(bits_to_f32(rng()));
    rbits.push_back(r);
  }

  // Deterministic midpoint/tie probes for each minifloat x4 .rs family.
  const rs_minifloat_family_t families[] = {
      rs_minifloat_family_t::kE4M3, rs_minifloat_family_t::kE5M2,
      rs_minifloat_family_t::kE2M1, rs_minifloat_family_t::kE2M3,
      rs_minifloat_family_t::kE3M2,
  };
  for (rs_minifloat_family_t fmt : families) {
    const std::vector<float> ties = collect_rs_midpoint_ties(fmt, 8);
    for (float tie : ties) {
      a.push_back(tie);
      b.push_back(tie);
      e.push_back(tie);
      f.push_back(tie);
      rbits.push_back(0x7f7f7f7fu);

      a.push_back(tie);
      b.push_back(tie);
      e.push_back(tie);
      f.push_back(tie);
      rbits.push_back(0x80808080u);

      const float down =
          std::nextafter(tie, -std::numeric_limits<float>::infinity());
      const float up =
          std::nextafter(tie, std::numeric_limits<float>::infinity());
      if (std::isfinite(down) && std::isfinite(up)) {
        a.push_back(down);
        b.push_back(tie);
        e.push_back(up);
        f.push_back(tie);
        rbits.push_back(0x7f80807fu);
      }
    }
  }

  while (a.size() < 768) {
    a.push_back(bits_to_f32(rng()));
    b.push_back(bits_to_f32(rng()));
    e.push_back(bits_to_f32(rng()));
    f.push_back(bits_to_f32(rng()));
    rbits.push_back(rng());
  }
}

__global__ void k_f16x2_rs(uint32_t *out, const float *a, const float *b,
                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.f16x2.f32 %0, %1, %2, %3;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_f16x2_satfinite_relu_rs(uint32_t *out, const float *a,
                                          const float *b,
                                          const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.f16x2.f32 %0, %1, %2, %3;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_bf16x2_rs(uint32_t *out, const float *a, const float *b,
                            const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.bf16x2.f32 %0, %1, %2, %3;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_bf16x2_satfinite_relu_rs(uint32_t *out, const float *a,
                                           const float *b,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.bf16x2.f32 %0, %1, %2, %3;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e4m3x4_satfinite_rs(uint32_t *out, const float *a,
                                      const float *b, const float *e,
                                      const float *f, const uint32_t *rbits,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e4m3x4_satfinite_relu_rs(uint32_t *out, const float *a,
                                           const float *b, const float *e,
                                           const float *f,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e5m2x4_satfinite_rs(uint32_t *out, const float *a,
                                      const float *b, const float *e,
                                      const float *f, const uint32_t *rbits,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e5m2x4_satfinite_relu_rs(uint32_t *out, const float *a,
                                           const float *b, const float *e,
                                           const float *f,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e2m1x4_satfinite_rs(uint16_t *out, const float *a,
                                      const float *b, const float *e,
                                      const float *f, const uint32_t *rbits,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e2m1x4_satfinite_relu_rs(uint16_t *out, const float *a,
                                           const float *b, const float *e,
                                           const float *f,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rs.satfinite.relu.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e2m3x4_satfinite_rs(uint32_t *out, const float *a,
                                      const float *b, const float *e,
                                      const float *f, const uint32_t *rbits,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.e2m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e2m3x4_satfinite_relu_rs(uint32_t *out, const float *a,
                                           const float *b, const float *e,
                                           const float *f,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.e2m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e3m2x4_satfinite_rs(uint32_t *out, const float *a,
                                      const float *b, const float *e,
                                      const float *f, const uint32_t *rbits,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.e3m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

__global__ void k_e3m2x4_satfinite_relu_rs(uint32_t *out, const float *a,
                                           const float *b, const float *e,
                                           const float *f,
                                           const uint32_t *rbits, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rs.satfinite.relu.e3m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
               : "=r"(d)
               : "f"(a[i]), "f"(b[i]), "f"(e[i]), "f"(f[i]), "r"(rbits[i]));
  out[i] = d;
}

template <typename TOut, typename LaunchFn>
void run_pair_kernel_with_rbits(const std::vector<float> &a,
                                const std::vector<float> &b,
                                const std::vector<uint32_t> &rbits,
                                std::vector<TOut> &out,
                                LaunchFn launch_kernel) {
  float *d_a = nullptr;
  float *d_b = nullptr;
  uint32_t *d_rbits = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(a.size());

  CUDACHK(cudaMalloc(&d_a, sizeof(float) * a.size()));
  CUDACHK(cudaMalloc(&d_b, sizeof(float) * b.size()));
  CUDACHK(cudaMalloc(&d_rbits, sizeof(uint32_t) * rbits.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * a.size()));

  CUDACHK(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_rbits, rbits.data(), sizeof(uint32_t) * rbits.size(),
                     cudaMemcpyHostToDevice));

  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_a, d_b, d_rbits, n);
  CUDACHK(cudaDeviceSynchronize());

  out.resize(a.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));

  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_rbits));
  CUDACHK(cudaFree(d_b));
  CUDACHK(cudaFree(d_a));
}

template <typename TOut, typename LaunchFn>
void run_quad_kernel_with_rbits(const std::vector<float> &a,
                                const std::vector<float> &b,
                                const std::vector<float> &e,
                                const std::vector<float> &f,
                                const std::vector<uint32_t> &rbits,
                                std::vector<TOut> &out,
                                LaunchFn launch_kernel) {
  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_e = nullptr;
  float *d_f = nullptr;
  uint32_t *d_rbits = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(a.size());

  CUDACHK(cudaMalloc(&d_a, sizeof(float) * a.size()));
  CUDACHK(cudaMalloc(&d_b, sizeof(float) * b.size()));
  CUDACHK(cudaMalloc(&d_e, sizeof(float) * e.size()));
  CUDACHK(cudaMalloc(&d_f, sizeof(float) * f.size()));
  CUDACHK(cudaMalloc(&d_rbits, sizeof(uint32_t) * rbits.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * a.size()));

  CUDACHK(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_e, e.data(), sizeof(float) * e.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_f, f.data(), sizeof(float) * f.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_rbits, rbits.data(), sizeof(uint32_t) * rbits.size(),
                     cudaMemcpyHostToDevice));

  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_a, d_b, d_e, d_f, d_rbits, n);
  CUDACHK(cudaDeviceSynchronize());

  out.resize(a.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));

  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_rbits));
  CUDACHK(cudaFree(d_f));
  CUDACHK(cudaFree(d_e));
  CUDACHK(cudaFree(d_b));
  CUDACHK(cudaFree(d_a));
}

template <typename T>
void dump_hex(std::ofstream &ofs, const std::string &label,
              const std::vector<T> &vals, unsigned width) {
  for (size_t i = 0; i < vals.size(); ++i) {
    ofs << label << " " << i << " 0x" << std::hex << std::setw(width)
        << std::setfill('0')
        << static_cast<unsigned long long>(vals[i]) << std::dec << "\n";
  }
}

int run_and_dump(const char *out_path) {
  std::vector<float> pair_a, pair_b;
  std::vector<uint32_t> pair_rbits;
  build_pair_rs_inputs(pair_a, pair_b, pair_rbits);

  std::vector<float> quad_a, quad_b, quad_e, quad_f;
  std::vector<uint32_t> quad_rbits;
  build_quad_rs_inputs(quad_a, quad_b, quad_e, quad_f, quad_rbits);

  std::vector<uint16_t> out_u16;
  std::vector<uint32_t> out_u32;

  std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    std::cerr << "failed to open output file: " << out_path << "\n";
    return 1;
  }

  run_pair_kernel_with_rbits<uint32_t>(
      pair_a, pair_b, pair_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const uint32_t *d_rbits, int n) {
        k_f16x2_rs<<<grid, block>>>(d_out, d_a, d_b, d_rbits, n);
      });
  bool all_zero = true;
  for (size_t i = 0; i < out_u32.size(); ++i) {
    if (out_u32[i] != 0u) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    std::cerr << "RS_FEATURE_UNAVAILABLE: k_f16x2_rs produced all-zero output"
              << "\n";
    return 3;
  }
  dump_hex(ofs, "k_f16x2_rs", out_u32, 8);

  run_pair_kernel_with_rbits<uint32_t>(
      pair_a, pair_b, pair_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const uint32_t *d_rbits, int n) {
        k_f16x2_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_rbits,
                                                   n);
      });
  dump_hex(ofs, "k_f16x2_satfinite_relu_rs", out_u32, 8);

  run_pair_kernel_with_rbits<uint32_t>(
      pair_a, pair_b, pair_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const uint32_t *d_rbits, int n) {
        k_bf16x2_rs<<<grid, block>>>(d_out, d_a, d_b, d_rbits, n);
      });
  dump_hex(ofs, "k_bf16x2_rs", out_u32, 8);

  run_pair_kernel_with_rbits<uint32_t>(
      pair_a, pair_b, pair_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const uint32_t *d_rbits, int n) {
        k_bf16x2_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_rbits,
                                                    n);
      });
  dump_hex(ofs, "k_bf16x2_satfinite_relu_rs", out_u32, 8);

  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e4m3x4_satfinite_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                               d_rbits, n);
      });
  dump_hex(ofs, "k_e4m3x4_satfinite_rs", out_u32, 8);
  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e4m3x4_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                                    d_rbits, n);
      });
  dump_hex(ofs, "k_e4m3x4_satfinite_relu_rs", out_u32, 8);

  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e5m2x4_satfinite_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                               d_rbits, n);
      });
  dump_hex(ofs, "k_e5m2x4_satfinite_rs", out_u32, 8);
  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e5m2x4_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                                    d_rbits, n);
      });
  dump_hex(ofs, "k_e5m2x4_satfinite_relu_rs", out_u32, 8);

  run_quad_kernel_with_rbits<uint16_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e2m1x4_satfinite_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                               d_rbits, n);
      });
  dump_hex(ofs, "k_e2m1x4_satfinite_rs", out_u16, 4);
  run_quad_kernel_with_rbits<uint16_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e2m1x4_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                                    d_rbits, n);
      });
  dump_hex(ofs, "k_e2m1x4_satfinite_relu_rs", out_u16, 4);

  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e2m3x4_satfinite_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                               d_rbits, n);
      });
  dump_hex(ofs, "k_e2m3x4_satfinite_rs", out_u32, 8);
  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e2m3x4_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                                    d_rbits, n);
      });
  dump_hex(ofs, "k_e2m3x4_satfinite_relu_rs", out_u32, 8);

  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e3m2x4_satfinite_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                               d_rbits, n);
      });
  dump_hex(ofs, "k_e3m2x4_satfinite_rs", out_u32, 8);
  run_quad_kernel_with_rbits<uint32_t>(
      quad_a, quad_b, quad_e, quad_f, quad_rbits, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a,
         const float *d_b, const float *d_e, const float *d_f,
         const uint32_t *d_rbits, int n) {
        k_e3m2x4_satfinite_relu_rs<<<grid, block>>>(d_out, d_a, d_b, d_e, d_f,
                                                    d_rbits, n);
      });
  dump_hex(ofs, "k_e3m2x4_satfinite_relu_rs", out_u32, 8);

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--output") {
    std::cerr << "usage: " << argv[0] << " --output <path>\n";
    return 1;
  }
  return run_and_dump(argv[2]);
}
