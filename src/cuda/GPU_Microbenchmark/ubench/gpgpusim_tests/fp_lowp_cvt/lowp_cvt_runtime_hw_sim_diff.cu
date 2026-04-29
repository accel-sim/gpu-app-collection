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

#ifndef LOWP_CVT_HAS_UE8M0X2_BF16X2_RELU
#define LOWP_CVT_HAS_UE8M0X2_BF16X2_RELU 0
#endif

namespace {

#define CUDACHK(call)                                                         \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " @ "       \
                << #call << "\n";                                            \
      std::exit(2);                                                           \
    }                                                                         \
  } while (0)

float bits_to_f32(uint32_t bits) {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint32_t f32_to_bits(float v) {
  uint32_t bits = 0u;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

uint16_t bf16_bits_from_f32(float v) {
  return static_cast<uint16_t>(f32_to_bits(v) >> 16);
}

double bits_to_f64(uint64_t bits) {
  double out = 0.0;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

enum class minifloat_family_t {
  kE4M3,
  kE5M2,
  kE2M1,
  kE2M3,
  kE3M2,
  kUE8M0,
};

struct minifloat_entry_t {
  uint16_t code;
  float value;
};

unsigned minifloat_code_limit(minifloat_family_t fmt) {
  switch (fmt) {
    case minifloat_family_t::kE4M3:
    case minifloat_family_t::kE5M2:
    case minifloat_family_t::kUE8M0:
      return 256u;
    case minifloat_family_t::kE2M1:
      return 16u;
    case minifloat_family_t::kE2M3:
    case minifloat_family_t::kE3M2:
      return 64u;
  }
  return 0u;
}

float decode_minifloat_code(minifloat_family_t fmt, uint16_t code) {
  if (fmt == minifloat_family_t::kUE8M0) {
    if (code == 0xffu) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, static_cast<int>(code) - 127);
  }

  int exp_bits = 0;
  int mant_bits = 0;
  int bias = 0;
  bool has_inf = false;
  switch (fmt) {
    case minifloat_family_t::kE4M3:
      exp_bits = 4;
      mant_bits = 3;
      bias = 7;
      has_inf = false;
      break;
    case minifloat_family_t::kE5M2:
      exp_bits = 5;
      mant_bits = 2;
      bias = 15;
      has_inf = true;
      break;
    case minifloat_family_t::kE2M1:
      exp_bits = 2;
      mant_bits = 1;
      bias = 1;
      has_inf = false;
      break;
    case minifloat_family_t::kE2M3:
      exp_bits = 2;
      mant_bits = 3;
      bias = 1;
      has_inf = false;
      break;
    case minifloat_family_t::kE3M2:
      exp_bits = 3;
      mant_bits = 2;
      bias = 3;
      has_inf = false;
      break;
    case minifloat_family_t::kUE8M0:
      break;
  }

  const int sign_shift = exp_bits + mant_bits;
  const bool neg = ((code >> sign_shift) & 0x1u) != 0u;
  const int exp_mask = (1 << exp_bits) - 1;
  const int mant_mask = (1 << mant_bits) - 1;
  const int exp = (code >> mant_bits) & exp_mask;
  const int mant = code & mant_mask;

  if (fmt == minifloat_family_t::kE4M3 && (code == 0x7fu || code == 0xffu)) {
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

std::vector<float> collect_minifloat_midpoint_ties(minifloat_family_t fmt,
                                                   size_t max_count) {
  std::vector<minifloat_entry_t> entries;
  entries.reserve(minifloat_code_limit(fmt));
  for (unsigned code = 0; code < minifloat_code_limit(fmt); ++code) {
    const float value = decode_minifloat_code(fmt, static_cast<uint16_t>(code));
    if (!std::isfinite(value)) continue;
    entries.push_back({static_cast<uint16_t>(code), value});
  }

  std::sort(entries.begin(), entries.end(),
            [](const minifloat_entry_t &a, const minifloat_entry_t &b) {
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

std::vector<float> build_minifloat_midpoint_probe_values() {
  const minifloat_family_t families[] = {
      minifloat_family_t::kE4M3, minifloat_family_t::kE5M2,
      minifloat_family_t::kE2M1, minifloat_family_t::kE2M3,
      minifloat_family_t::kE3M2, minifloat_family_t::kUE8M0,
  };

  std::vector<float> out;
  for (minifloat_family_t fmt : families) {
    const std::vector<float> ties = collect_minifloat_midpoint_ties(fmt, 12);
    for (float tie : ties) {
      out.push_back(tie);
      const float down =
          std::nextafter(tie, -std::numeric_limits<float>::infinity());
      const float up =
          std::nextafter(tie, std::numeric_limits<float>::infinity());
      if (std::isfinite(down)) out.push_back(down);
      if (std::isfinite(up)) out.push_back(up);
    }
  }

  std::sort(out.begin(), out.end(),
            [](float a, float b) { return f32_to_bits(a) < f32_to_bits(b); });
  out.erase(std::unique(out.begin(), out.end(),
                        [](float a, float b) {
                          return f32_to_bits(a) == f32_to_bits(b);
                        }),
            out.end());
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

std::vector<float> build_scalar_inputs() {
  std::vector<float> out;
  out.reserve(384);
  const std::vector<uint32_t> edges = edge_f32_patterns();
  for (size_t i = 0; i < edges.size(); ++i) {
    out.push_back(bits_to_f32(edges[i]));
  }
  std::mt19937 rng(0x40f8b37du);
  while (out.size() < 384) out.push_back(bits_to_f32(rng()));
  return out;
}

void build_pair_inputs(std::vector<float> &a, std::vector<float> &b) {
  a.clear();
  b.clear();
  a.reserve(448);
  b.reserve(448);

  const std::vector<uint32_t> edges = edge_f32_patterns();
  for (size_t i = 0; i + 1 < edges.size(); i += 2) {
    a.push_back(bits_to_f32(edges[i]));
    b.push_back(bits_to_f32(edges[i + 1]));
  }

  const std::vector<float> midpoint_probes = build_minifloat_midpoint_probe_values();
  for (size_t i = 0; i < midpoint_probes.size(); i += 2) {
    const float lhs = midpoint_probes[i];
    const float rhs =
        (i + 1 < midpoint_probes.size()) ? midpoint_probes[i + 1] : lhs;
    a.push_back(lhs);
    b.push_back(rhs);
  }

  std::mt19937 rng(0x7b91d00du);
  while (a.size() < 384) {
    a.push_back(bits_to_f32(rng()));
    b.push_back(bits_to_f32(rng()));
  }
}

std::vector<uint16_t> build_bf16_inputs() {
  std::vector<uint16_t> out = {
      0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x007fu, 0x807fu, 0x0080u, 0x8080u,
      0x3f80u, 0xbf80u, 0x7f80u, 0xff80u, 0x7fffu,
  };

  const float int_edges[] = {
      -65536.0f, -32768.0f, -32767.0f, -256.0f, -255.0f, -128.0f, -127.5f,
      -127.0f,   -1.5f,     -1.0f,     -0.5f,   0.0f,    0.5f,    1.0f,
      1.5f,      127.0f,    127.5f,    128.0f,  255.0f,  256.0f,  32767.0f,
      32768.0f,  65535.0f,  65536.0f,
  };
  for (float v : int_edges) out.push_back(bf16_bits_from_f32(v));

  out.reserve(384);
  std::mt19937 rng(0x13572468u);
  while (out.size() < 384) out.push_back(static_cast<uint16_t>(rng()));
  return out;
}

std::vector<uint16_t> build_f16_inputs() {
  std::vector<uint16_t> out = {
      0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x03ffu, 0x83ffu, 0x0400u,
      0x8400u, 0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu, 0x7c00u, 0xfc00u,
      0x7e00u, 0xfe00u,
  };
  out.reserve(384);
  std::mt19937 rng(0x97531bdcu);
  while (out.size() < 384) out.push_back(static_cast<uint16_t>(rng()));
  return out;
}

std::vector<int32_t> build_s32_inputs() {
  std::vector<int32_t> out = {
      0,  1,  -1, 2, -2, 127, -127, 255, -255, 1024, -1024,
      std::numeric_limits<int32_t>::max(),
      std::numeric_limits<int32_t>::min(),
      0x00ffffff, static_cast<int32_t>(0xff000001u),
  };
  out.reserve(384);
  std::mt19937 rng(0x2468ace0u);
  while (out.size() < 384) out.push_back(static_cast<int32_t>(rng()));
  return out;
}

std::vector<uint32_t> build_u32_inputs() {
  std::vector<uint32_t> out = {
      0u, 1u, 2u, 255u, 65535u, 65536u, 0x00ffffffu, 0x7fffffffu, 0xffffffffu,
  };
  out.reserve(384);
  std::mt19937 rng(0x10293847u);
  while (out.size() < 384) out.push_back(rng());
  return out;
}

std::vector<double> build_f64_inputs() {
  std::vector<double> out;
  const std::vector<uint64_t> edges = {
      0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull,
      0x8000000000000001ull, 0x000fffffffffffffull, 0x800fffffffffffffull,
      0x0010000000000000ull, 0x8010000000000000ull, 0x3ff0000000000000ull,
      0xbff0000000000000ull, 0x3fefffffffffffffull, 0xbfefffffffffffffull,
      0x7fefffffffffffffull, 0xffefffffffffffffull, 0x7ff0000000000000ull,
      0xfff0000000000000ull, 0x7ff8000000000000ull, 0xfff8000000000000ull,
      0x7ff4000000000001ull, 0xfff4000000000001ull,
  };
  out.reserve(384);
  for (size_t i = 0; i < edges.size(); ++i) out.push_back(bits_to_f64(edges[i]));
  std::mt19937_64 rng(0x1122334455667788ull);
  while (out.size() < 384) out.push_back(bits_to_f64(rng()));
  return out;
}

std::vector<uint16_t> build_e2m1x2_inputs_exhaustive() {
  std::vector<uint16_t> out;
  out.reserve(256);
  for (unsigned packed = 0; packed <= 0xff; ++packed) {
    out.push_back(static_cast<uint16_t>(packed));
  }
  return out;
}

std::vector<uint16_t> build_e2m3x2_inputs_exhaustive() {
  std::vector<uint16_t> out;
  out.reserve(64u * 64u);
  for (unsigned hi = 0; hi < 64; ++hi) {
    for (unsigned lo = 0; lo < 64; ++lo) {
      out.push_back(static_cast<uint16_t>((hi << 8) | lo));
    }
  }
  return out;
}

std::vector<uint16_t> build_e3m2x2_inputs_exhaustive() {
  std::vector<uint16_t> out;
  out.reserve(64u * 64u);
  for (unsigned hi = 0; hi < 64; ++hi) {
    for (unsigned lo = 0; lo < 64; ++lo) {
      out.push_back(static_cast<uint16_t>((hi << 8) | lo));
    }
  }
  return out;
}

std::vector<uint16_t> build_u16_inputs_exhaustive() {
  std::vector<uint16_t> out;
  out.reserve(1u << 16);
  for (unsigned packed = 0; packed <= 0xffff; ++packed) {
    out.push_back(static_cast<uint16_t>(packed));
  }
  return out;
}

std::vector<uint16_t> build_scale_inputs(size_t count, uint32_t seed) {
  std::vector<uint16_t> out;
  out.reserve(count);

  // Packed ue8m0x2 scale-factors; include notable edge patterns and random.
  const uint16_t edge_scales[] = {
      0x0000u, 0x0101u, 0x7f7fu, 0xfefeu, 0x0080u,
      0x8000u, 0x7f00u, 0x007fu, 0xff00u, 0x00ffu,
  };
  for (size_t i = 0; i < sizeof(edge_scales) / sizeof(edge_scales[0]) &&
                     out.size() < count;
       ++i) {
    out.push_back(edge_scales[i]);
  }

  std::mt19937 rng(seed);
  while (out.size() < count) out.push_back(static_cast<uint16_t>(rng()));
  return out;
}

void build_s2f6_relu_focus_inputs(std::vector<float> &a, std::vector<float> &b,
                                  std::vector<uint16_t> &scale) {
  a.clear();
  b.clear();
  scale.clear();

  const uint32_t a_bits[] = {
      0xff800000u, 0x7fc00000u, 0x80000000u, 0xbf800000u, 0x00000000u,
      0x7f800000u, 0x7f7fffffu, 0xff7fffffu,
  };
  const uint32_t b_bits[] = {
      0x7f800000u, 0xffc00000u, 0x00000000u, 0x3f800000u, 0x80000000u,
      0xbf800000u, 0xff800000u, 0x7f800000u,
  };
  const uint16_t scale_words[] = {
      0x0000u, 0x0000u, 0x7f7fu, 0x0101u, 0x8000u, 0x0080u, 0x00ffu, 0xff00u,
  };

  const size_t n = sizeof(a_bits) / sizeof(a_bits[0]);
  for (size_t i = 0; i < n; ++i) {
    a.push_back(bits_to_f32(a_bits[i]));
    b.push_back(bits_to_f32(b_bits[i]));
    scale.push_back(scale_words[i]);
  }
}

__global__ void k_f16_rz(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.f16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_f16_rm(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rm.f16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_f16_rp(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.f16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_f16_sat_rn(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.sat.f16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_f16_satfinite_relu_rn(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.relu.f16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_rz(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.bf16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_rm(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rm.bf16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_rp(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.bf16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_satfinite_relu_rn(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.relu.bf16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_tf32_rz(uint32_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rz.tf32.f32 %0, %1;" : "=r"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_tf32_rna(uint32_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_tf32_satfinite_relu_rn(uint32_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.satfinite.relu.tf32.f32 %0, %1;" : "=r"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_f32_bf16_ftz_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float d = 0.0f;
  asm volatile("cvt.rn.ftz.f32.bf16 %0, %1;" : "=f"(d) : "h"(in[i]));
  out[i] = __float_as_uint(d);
}

__global__ void k_f16_bf16_rn(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.f16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f64_bf16(uint64_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double d = 0.0;
  asm volatile("cvt.f64.bf16 %0, %1;" : "=d"(d) : "h"(in[i]));
  out[i] = __double_as_longlong(d);
}

__global__ void k_s32_bf16_rzi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.s32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u32_bf16_rzi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.u32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s32_bf16_rni(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rni.s32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s32_bf16_rmi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rmi.s32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s32_bf16_rpi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rpi.s32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u32_bf16_rni(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rni.u32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u32_bf16_rmi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rmi.u32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u32_bf16_rpi(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rpi.u32.bf16 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s16_bf16_rzi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rzi.s16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s16_bf16_rni(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rni.s16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s16_bf16_rmi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rmi.s16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s16_bf16_rpi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rpi.s16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_u16_bf16_rzi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rzi.u16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u16_bf16_rni(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rni.u16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u16_bf16_rmi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rmi.u16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u16_bf16_rpi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rpi.u16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s8_bf16_rzi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("{ .reg .s8 t; cvt.rzi.s8.bf16 t, %1; cvt.s16.s8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s8_bf16_rni(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("{ .reg .s8 t; cvt.rni.s8.bf16 t, %1; cvt.s16.s8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s8_bf16_rmi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("{ .reg .s8 t; cvt.rmi.s8.bf16 t, %1; cvt.s16.s8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_s8_bf16_rpi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("{ .reg .s8 t; cvt.rpi.s8.bf16 t, %1; cvt.s16.s8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_u8_bf16_rzi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .u8 t; cvt.rzi.u8.bf16 t, %1; cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u8_bf16_rni(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .u8 t; cvt.rni.u8.bf16 t, %1; cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u8_bf16_rmi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .u8 t; cvt.rmi.u8.bf16 t, %1; cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_u8_bf16_rpi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .u8 t; cvt.rpi.u8.bf16 t, %1; cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s64_bf16_rzi(uint64_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long d = 0;
  asm volatile("cvt.rzi.s64.bf16 %0, %1;" : "=l"(d) : "h"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_u64_bf16_rzi(uint64_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long d = 0;
  asm volatile("cvt.rzi.u64.bf16 %0, %1;" : "=l"(d) : "h"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_s64_f32_rzi(uint64_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long d = 0;
  asm volatile("cvt.rzi.s64.f32 %0, %1;" : "=l"(d) : "f"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_u64_f32_rzi(uint64_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long d = 0;
  asm volatile("cvt.rzi.u64.f32 %0, %1;" : "=l"(d) : "f"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_s32_f32_rzi_sat(uint32_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.sat.s32.f32 %0, %1;" : "=r"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_u32_f32_rzi_sat(uint32_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.sat.u32.f32 %0, %1;" : "=r"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_s32_f64_rzi_sat(uint32_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.sat.s32.f64 %0, %1;" : "=r"(d) : "d"(in[i]));
  out[i] = d;
}

__global__ void k_u32_f64_rzi_sat(uint32_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rzi.sat.u32.f64 %0, %1;" : "=r"(d) : "d"(in[i]));
  out[i] = d;
}

__global__ void k_s16_f32_rzi_sat(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rzi.sat.s16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_u16_f32_rzi_sat(uint16_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rzi.sat.u16.f32 %0, %1;" : "=h"(d) : "f"(in[i]));
  out[i] = d;
}

__global__ void k_s64_f32_rzi_sat(uint64_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long d = 0;
  asm volatile("cvt.rzi.sat.s64.f32 %0, %1;" : "=l"(d) : "f"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_u64_f32_rzi_sat(uint64_t *out, const float *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long d = 0;
  asm volatile("cvt.rzi.sat.u64.f32 %0, %1;" : "=l"(d) : "f"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_s16_f64_rzi_sat(uint16_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  short d = 0;
  asm volatile("cvt.rzi.sat.s16.f64 %0, %1;" : "=h"(d) : "d"(in[i]));
  out[i] = static_cast<uint16_t>(d);
}

__global__ void k_u16_f64_rzi_sat(uint16_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rzi.sat.u16.f64 %0, %1;" : "=h"(d) : "d"(in[i]));
  out[i] = d;
}

__global__ void k_s64_f64_rzi_sat(uint64_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long d = 0;
  asm volatile("cvt.rzi.sat.s64.f64 %0, %1;" : "=l"(d) : "d"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_u64_f64_rzi_sat(uint64_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long d = 0;
  asm volatile("cvt.rzi.sat.u64.f64 %0, %1;" : "=l"(d) : "d"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_s64_f64_rzi(uint64_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long d = 0;
  asm volatile("cvt.rzi.s64.f64 %0, %1;" : "=l"(d) : "d"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_u64_f64_rzi(uint64_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long d = 0;
  asm volatile("cvt.rzi.u64.f64 %0, %1;" : "=l"(d) : "d"(in[i]));
  out[i] = static_cast<uint64_t>(d);
}

__global__ void k_bf16_f16_rn(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.bf16.f16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_s32_rn(uint16_t *out, const int32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.bf16.s32 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_u32_rz(uint16_t *out, const uint32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.bf16.u32 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_f64_rn(uint16_t *out, const double *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.bf16.f64 %0, %1;" : "=h"(d) : "d"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_bf16(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.bf16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_bf16_bf16_rzi(uint16_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rzi.bf16.bf16 %0, %1;" : "=h"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f16x2_rn(uint32_t *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_bf16x2_rz(uint32_t *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rz.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e2m1x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .b8 t; "
               "cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; "
               "cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e4m3x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e5m2x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e2m3x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e3m2x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_f16x2_e4m3x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f16x2_e5m2x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f16x2_e2m1x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("{ .reg .b8 t; "
               "cvt.u8.u16 t, %1; "
               "cvt.rn.f16x2.e2m1x2 %0, t; }"
               : "=r"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f16x2_e2m3x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_f16x2_e3m2x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_satfinite_rz(uint16_t *out, const float *a,
                                       const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_rz(uint16_t *out, const float *a, const float *b,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.ue8m0x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_satfinite_rp(uint16_t *out, const float *a,
                                       const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_rp(uint16_t *out, const float *a, const float *b,
                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.ue8m0x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_satfinite_rn(uint16_t *out, const float *a,
                                      const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.s2f6x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_satfinite_relu_rn(uint16_t *out, const float *a,
                                           const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.relu.s2f6x2.f32 %0, %1, %2;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]));
  out[i] = d;
}

__global__ void k_e4m3x2_f16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e5m2x2_f16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e2m1x2_f16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .b8 t; "
               "cvt.rn.satfinite.e2m1x2.f16x2 t, %1; "
               "cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e2m3x2_f16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e2m3x2.f16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e3m2x2_f16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e3m2x2.f16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e4m3x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e5m2x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e5m2x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e2m1x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("{ .reg .b8 t; "
               "cvt.rn.satfinite.e2m1x2.bf16x2 t, %1; "
               "cvt.u16.u8 %0, t; }"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e2m3x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e2m3x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_e3m2x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.e3m2x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_bf16x2_satfinite_rz(uint16_t *out, const uint32_t *in,
                                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1;"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_bf16x2_rz(uint16_t *out, const uint32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.ue8m0x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_bf16x2_satfinite_rp(uint16_t *out, const uint32_t *in,
                                              int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.satfinite.ue8m0x2.bf16x2 %0, %1;"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_ue8m0x2_bf16x2_rp(uint16_t *out, const uint32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rp.ue8m0x2.bf16x2 %0, %1;" : "=h"(d) : "r"(in[i]));
  out[i] = d;
}

#if LOWP_CVT_HAS_UE8M0X2_BF16X2_RELU
__global__ void k_ue8m0x2_bf16x2_satfinite_relu_rz(uint16_t *out,
                                                   const uint32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rz.satfinite.relu.ue8m0x2.bf16x2 %0, %1;"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}
#endif

__global__ void k_bf16x2_ue8m0x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_bf16x2_satfinite_rn(uint16_t *out, const uint32_t *in,
                                             int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.s2f6x2.bf16x2 %0, %1;"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_bf16x2_satfinite_relu_rn(uint16_t *out,
                                                  const uint32_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.relu.s2f6x2.bf16x2 %0, %1;"
               : "=h"(d)
               : "r"(in[i]));
  out[i] = d;
}

__global__ void k_bf16x2_s2f6x2_rn(uint32_t *out, const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.bf16x2.s2f6x2 %0, %1;" : "=r"(d) : "h"(in[i]));
  out[i] = d;
}

__global__ void k_bf16x2_s2f6x2_satfinite_relu_rn(uint32_t *out,
                                                  const uint16_t *in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile("cvt.rn.satfinite.relu.bf16x2.s2f6x2 %0, %1;"
               : "=r"(d)
               : "h"(in[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_f32_sc(uint16_t *out, const float *a, const float *b,
                                const uint16_t *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile("cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32 %0, %1, %2, %3;"
               : "=h"(d)
               : "f"(a[i]), "f"(b[i]), "h"(scale[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_f32_sc_relu(uint16_t *out, const float *a,
                                     const float *b, const uint16_t *scale,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile(
      "cvt.rn.satfinite.relu.scaled::n2::ue8m0.s2f6x2.f32 %0, %1, %2, %3;"
      : "=h"(d)
      : "f"(a[i]), "f"(b[i]), "h"(scale[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_bf16x2_sc(uint16_t *out, const uint32_t *in,
                                   const uint16_t *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile(
      "cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2 %0, %1, %2;"
      : "=h"(d)
      : "r"(in[i]), "h"(scale[i]));
  out[i] = d;
}

__global__ void k_s2f6x2_bf16x2_sc_relu(uint16_t *out, const uint32_t *in,
                                        const uint16_t *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned short d = 0;
  asm volatile(
      "cvt.rn.satfinite.relu.scaled::n2::ue8m0.s2f6x2.bf16x2 %0, %1, %2;"
      : "=h"(d)
      : "r"(in[i]), "h"(scale[i]));
  out[i] = d;
}

__global__ void k_bf16x2_s2f6x2_sc(uint32_t *out, const uint16_t *in,
                                   const uint16_t *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile(
      "cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;"
      : "=r"(d)
      : "h"(in[i]), "h"(scale[i]));
  out[i] = d;
}

__global__ void k_bf16x2_s2f6x2_sc_relu(uint32_t *out, const uint16_t *in,
                                        const uint16_t *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned d = 0;
  asm volatile(
      "cvt.rn.satfinite.relu.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;"
      : "=r"(d)
      : "h"(in[i]), "h"(scale[i]));
  out[i] = d;
}

template <typename TIn, typename TOut, typename LaunchFn>
void run_unary_kernel(const std::vector<TIn> &in, std::vector<TOut> &out,
                      LaunchFn launch_kernel) {
  TIn *d_in = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(in.size());
  CUDACHK(cudaMalloc(&d_in, sizeof(TIn) * in.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * in.size()));
  CUDACHK(cudaMemcpy(d_in, in.data(), sizeof(TIn) * in.size(),
                     cudaMemcpyHostToDevice));
  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_in, n);
  CUDACHK(cudaDeviceSynchronize());
  out.resize(in.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));
  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_in));
}

template <typename TOut, typename LaunchFn>
void run_pair_kernel(const std::vector<float> &a, const std::vector<float> &b,
                     std::vector<TOut> &out, LaunchFn launch_kernel) {
  float *d_a = nullptr;
  float *d_b = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(a.size());
  CUDACHK(cudaMalloc(&d_a, sizeof(float) * a.size()));
  CUDACHK(cudaMalloc(&d_b, sizeof(float) * b.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * a.size()));
  CUDACHK(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(),
                     cudaMemcpyHostToDevice));
  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_a, d_b, n);
  CUDACHK(cudaDeviceSynchronize());
  out.resize(a.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));
  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_b));
  CUDACHK(cudaFree(d_a));
}

template <typename TIn, typename TScale, typename TOut, typename LaunchFn>
void run_unary_kernel_with_scale(const std::vector<TIn> &in,
                                 const std::vector<TScale> &scale,
                                 std::vector<TOut> &out,
                                 LaunchFn launch_kernel) {
  if (in.size() != scale.size()) {
    std::cerr << "input/scale size mismatch\n";
    std::exit(2);
  }
  TIn *d_in = nullptr;
  TScale *d_scale = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(in.size());
  CUDACHK(cudaMalloc(&d_in, sizeof(TIn) * in.size()));
  CUDACHK(cudaMalloc(&d_scale, sizeof(TScale) * scale.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * in.size()));
  CUDACHK(
      cudaMemcpy(d_in, in.data(), sizeof(TIn) * in.size(), cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_scale, scale.data(), sizeof(TScale) * scale.size(),
                     cudaMemcpyHostToDevice));
  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_in, d_scale, n);
  CUDACHK(cudaDeviceSynchronize());
  out.resize(in.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));
  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_scale));
  CUDACHK(cudaFree(d_in));
}

template <typename TOut, typename LaunchFn>
void run_pair_kernel_with_scale(const std::vector<float> &a,
                                const std::vector<float> &b,
                                const std::vector<uint16_t> &scale,
                                std::vector<TOut> &out,
                                LaunchFn launch_kernel) {
  if (a.size() != b.size() || a.size() != scale.size()) {
    std::cerr << "pair/scale size mismatch\n";
    std::exit(2);
  }
  float *d_a = nullptr;
  float *d_b = nullptr;
  uint16_t *d_scale = nullptr;
  TOut *d_out = nullptr;
  const int n = static_cast<int>(a.size());
  CUDACHK(cudaMalloc(&d_a, sizeof(float) * a.size()));
  CUDACHK(cudaMalloc(&d_b, sizeof(float) * b.size()));
  CUDACHK(cudaMalloc(&d_scale, sizeof(uint16_t) * scale.size()));
  CUDACHK(cudaMalloc(&d_out, sizeof(TOut) * a.size()));
  CUDACHK(cudaMemcpy(d_a, a.data(), sizeof(float) * a.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_b, b.data(), sizeof(float) * b.size(),
                     cudaMemcpyHostToDevice));
  CUDACHK(cudaMemcpy(d_scale, scale.data(), sizeof(uint16_t) * scale.size(),
                     cudaMemcpyHostToDevice));
  const int block = 128;
  const int grid = (n + block - 1) / block;
  launch_kernel(grid, block, d_out, d_a, d_b, d_scale, n);
  CUDACHK(cudaDeviceSynchronize());
  out.resize(a.size());
  CUDACHK(cudaMemcpy(out.data(), d_out, sizeof(TOut) * out.size(),
                     cudaMemcpyDeviceToHost));
  CUDACHK(cudaFree(d_out));
  CUDACHK(cudaFree(d_scale));
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
  const std::vector<float> scalars = build_scalar_inputs();
  std::vector<float> pair_a, pair_b;
  build_pair_inputs(pair_a, pair_b);
  const std::vector<uint16_t> bf16_inputs = build_bf16_inputs();
  const std::vector<uint16_t> f16_inputs = build_f16_inputs();
  const std::vector<int32_t> s32_inputs = build_s32_inputs();
  const std::vector<uint32_t> u32_inputs = build_u32_inputs();
  const std::vector<double> f64_inputs = build_f64_inputs();
  const std::vector<uint16_t> e2m1x2_inputs_exhaustive =
      build_e2m1x2_inputs_exhaustive();
  const std::vector<uint16_t> e2m3x2_inputs_exhaustive =
      build_e2m3x2_inputs_exhaustive();
  const std::vector<uint16_t> e3m2x2_inputs_exhaustive =
      build_e3m2x2_inputs_exhaustive();
  const std::vector<uint16_t> u16_inputs_exhaustive =
      build_u16_inputs_exhaustive();
  const std::vector<uint16_t> pair_scales =
      build_scale_inputs(pair_a.size(), 0x4f2a9c13u);
  std::vector<float> s2f6_relu_focus_a, s2f6_relu_focus_b;
  std::vector<uint16_t> s2f6_relu_focus_scales;
  build_s2f6_relu_focus_inputs(s2f6_relu_focus_a, s2f6_relu_focus_b,
                               s2f6_relu_focus_scales);
  const std::vector<uint16_t> exhaustive_scales =
      build_scale_inputs(u16_inputs_exhaustive.size(), 0x6be03491u);

  std::vector<uint16_t> out_u16;
  std::vector<uint32_t> out_u32;
  std::vector<uint64_t> out_u64;
  std::vector<uint16_t> tmp_u16;
  std::vector<uint32_t> tmp_u32;

  std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    std::cerr << "failed to open output file: " << out_path << "\n";
    return 2;
  }

  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_f16_rz<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_f16_rz", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_f16_rm<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_f16_rm", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_f16_rp<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_f16_rp", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_in, int n) {
        k_f16_sat_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16_sat_rn", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_in, int n) {
        k_f16_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16_satfinite_relu_rn", out_u16, 4);

  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_bf16_rz<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_bf16_rz", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_bf16_rm<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_bf16_rm", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) { k_bf16_rp<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_bf16_rp", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_in, int n) {
        k_bf16_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_satfinite_relu_rn", out_u16, 4);

  run_unary_kernel<float, uint32_t>(
      scalars, out_u32, [](int grid, int block, uint32_t *d_out, const float *d_in,
                           int n) { k_tf32_rz<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_tf32_rz", out_u32, 8);
  run_unary_kernel<float, uint32_t>(
      scalars, out_u32, [](int grid, int block, uint32_t *d_out, const float *d_in,
                           int n) { k_tf32_rna<<<grid, block>>>(d_out, d_in, n); });
  dump_hex(ofs, "k_tf32_rna", out_u32, 8);
  run_unary_kernel<float, uint32_t>(
      scalars, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_in, int n) {
        k_tf32_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_tf32_satfinite_relu_rn", out_u32, 8);

  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f32_bf16_ftz_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f32_bf16_ftz_rn", out_u32, 8);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_f16_bf16_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16_bf16_rn", out_u16, 4);
  run_unary_kernel<uint16_t, uint64_t>(
      bf16_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const uint16_t *d_in, int n) {
        k_f64_bf16<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f64_bf16", out_u64, 16);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_s32_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_bf16_rzi", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_u32_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_bf16_rzi", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_s32_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_bf16_rni", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_s32_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_bf16_rmi", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_s32_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_bf16_rpi", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_u32_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_bf16_rni", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_u32_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_bf16_rmi", out_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      bf16_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_u32_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_bf16_rpi", out_u32, 8);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s16_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_bf16_rzi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s16_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_bf16_rni", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s16_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_bf16_rmi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s16_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_bf16_rpi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u16_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_bf16_rzi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u16_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_bf16_rni", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u16_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_bf16_rmi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u16_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_bf16_rpi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s8_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s8_bf16_rzi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s8_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s8_bf16_rni", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s8_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s8_bf16_rmi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_s8_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s8_bf16_rpi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u8_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u8_bf16_rzi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u8_bf16_rni<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u8_bf16_rni", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u8_bf16_rmi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u8_bf16_rmi", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_u8_bf16_rpi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u8_bf16_rpi", out_u16, 4);
  run_unary_kernel<uint16_t, uint64_t>(
      bf16_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const uint16_t *d_in, int n) {
        k_s64_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s64_bf16_rzi", out_u64, 16);
  run_unary_kernel<uint16_t, uint64_t>(
      bf16_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const uint16_t *d_in, int n) {
        k_u64_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u64_bf16_rzi", out_u64, 16);
  run_unary_kernel<float, uint64_t>(
      scalars, out_u64, [](int grid, int block, uint64_t *d_out, const float *d_in,
                           int n) {
        k_s64_f32_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s64_f32_rzi", out_u64, 16);
  run_unary_kernel<float, uint64_t>(
      scalars, out_u64, [](int grid, int block, uint64_t *d_out, const float *d_in,
                           int n) {
        k_u64_f32_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u64_f32_rzi", out_u64, 16);
  run_unary_kernel<float, uint32_t>(
      scalars, out_u32, [](int grid, int block, uint32_t *d_out, const float *d_in,
                           int n) {
        k_s32_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_f32_rzi_sat", out_u32, 8);
  run_unary_kernel<float, uint32_t>(
      scalars, out_u32, [](int grid, int block, uint32_t *d_out, const float *d_in,
                           int n) {
        k_u32_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_f32_rzi_sat", out_u32, 8);
  run_unary_kernel<double, uint32_t>(
      f64_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const double *d_in, int n) {
        k_s32_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s32_f64_rzi_sat", out_u32, 8);
  run_unary_kernel<double, uint32_t>(
      f64_inputs, out_u32,
      [](int grid, int block, uint32_t *d_out, const double *d_in, int n) {
        k_u32_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u32_f64_rzi_sat", out_u32, 8);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) {
        k_s16_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_f32_rzi_sat", out_u16, 4);
  run_unary_kernel<float, uint16_t>(
      scalars, out_u16, [](int grid, int block, uint16_t *d_out, const float *d_in,
                           int n) {
        k_u16_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_f32_rzi_sat", out_u16, 4);
  run_unary_kernel<float, uint64_t>(
      scalars, out_u64, [](int grid, int block, uint64_t *d_out, const float *d_in,
                           int n) {
        k_s64_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s64_f32_rzi_sat", out_u64, 16);
  run_unary_kernel<float, uint64_t>(
      scalars, out_u64, [](int grid, int block, uint64_t *d_out, const float *d_in,
                           int n) {
        k_u64_f32_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u64_f32_rzi_sat", out_u64, 16);
  run_unary_kernel<double, uint16_t>(
      f64_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const double *d_in, int n) {
        k_s16_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s16_f64_rzi_sat", out_u16, 4);
  run_unary_kernel<double, uint16_t>(
      f64_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const double *d_in, int n) {
        k_u16_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u16_f64_rzi_sat", out_u16, 4);
  run_unary_kernel<double, uint64_t>(
      f64_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const double *d_in, int n) {
        k_s64_f64_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s64_f64_rzi", out_u64, 16);
  run_unary_kernel<double, uint64_t>(
      f64_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const double *d_in, int n) {
        k_u64_f64_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u64_f64_rzi", out_u64, 16);
  run_unary_kernel<double, uint64_t>(
      f64_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const double *d_in, int n) {
        k_s64_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s64_f64_rzi_sat", out_u64, 16);
  run_unary_kernel<double, uint64_t>(
      f64_inputs, out_u64,
      [](int grid, int block, uint64_t *d_out, const double *d_in, int n) {
        k_u64_f64_rzi_sat<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_u64_f64_rzi_sat", out_u64, 16);

  run_unary_kernel<uint16_t, uint16_t>(
      f16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_bf16_f16_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_f16_rn", out_u16, 4);
  run_unary_kernel<int32_t, uint16_t>(
      s32_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const int32_t *d_in, int n) {
        k_bf16_s32_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_s32_rn", out_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      u32_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_bf16_u32_rz<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_u32_rz", out_u16, 4);
  run_unary_kernel<double, uint16_t>(
      f64_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const double *d_in, int n) {
        k_bf16_f64_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_f64_rn", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_bf16_bf16<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_bf16", out_u16, 4);
  run_unary_kernel<uint16_t, uint16_t>(
      bf16_inputs, out_u16,
      [](int grid, int block, uint16_t *d_out, const uint16_t *d_in, int n) {
        k_bf16_bf16_rzi<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16_bf16_rzi", out_u16, 4);

  run_pair_kernel<uint32_t>(
      pair_a, pair_b, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a, const float *d_b,
         int n) { k_f16x2_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_f16x2_rn", out_u32, 8);
  const std::vector<uint32_t> f16x2_pairs = out_u32;
  run_pair_kernel<uint32_t>(
      pair_a, pair_b, out_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a, const float *d_b,
         int n) { k_bf16x2_rz<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_bf16x2_rz", out_u32, 8);
  const std::vector<uint32_t> bf16x2_pairs = out_u32;

  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_e4m3x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_e4m3x2_satfinite_rn", out_u16, 4);
  const std::vector<uint16_t> e4m3x2_pairs = out_u16;
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_e5m2x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_e5m2x2_satfinite_rn", out_u16, 4);
  const std::vector<uint16_t> e5m2x2_pairs = out_u16;
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_e2m1x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_e2m1x2_satfinite_rn", out_u16, 4);
  const std::vector<uint16_t> e2m1x2_pairs = out_u16;
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_e2m3x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_e2m3x2_satfinite_rn", out_u16, 4);
  const std::vector<uint16_t> e2m3x2_pairs = out_u16;
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_e3m2x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_e3m2x2_satfinite_rn", out_u16, 4);
  const std::vector<uint16_t> e3m2x2_pairs = out_u16;
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_ue8m0x2_satfinite_rz<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_ue8m0x2_satfinite_rz", out_u16, 4);
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_ue8m0x2_rz<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_ue8m0x2_rz", out_u16, 4);
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_ue8m0x2_satfinite_rp<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_ue8m0x2_satfinite_rp", out_u16, 4);
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_ue8m0x2_rp<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_ue8m0x2_rp", out_u16, 4);
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_s2f6x2_satfinite_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_s2f6x2_satfinite_rn", out_u16, 4);
  run_pair_kernel<uint16_t>(
      pair_a, pair_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_s2f6x2_satfinite_relu_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_s2f6x2_satfinite_relu_rn", out_u16, 4);
  run_pair_kernel_with_scale<uint16_t>(
      pair_a, pair_b, pair_scales, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a,
         const float *d_b, const uint16_t *d_scale, int n) {
        k_s2f6x2_f32_sc<<<grid, block>>>(d_out, d_a, d_b, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_f32_satfinite_scaled_rn", out_u16, 4);
  run_pair_kernel_with_scale<uint16_t>(
      pair_a, pair_b, pair_scales, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a,
         const float *d_b, const uint16_t *d_scale, int n) {
        k_s2f6x2_f32_sc_relu<<<grid, block>>>(d_out, d_a, d_b, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_f32_satfinite_relu_scaled_rn", out_u16, 4);
  run_pair_kernel<uint16_t>(
      s2f6_relu_focus_a, s2f6_relu_focus_b, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a, const float *d_b,
         int n) { k_s2f6x2_satfinite_relu_rn<<<grid, block>>>(d_out, d_a, d_b, n); });
  dump_hex(ofs, "k_s2f6x2_satfinite_relu_rn_focus", out_u16, 4);
  run_pair_kernel_with_scale<uint16_t>(
      s2f6_relu_focus_a, s2f6_relu_focus_b, s2f6_relu_focus_scales, out_u16,
      [](int grid, int block, uint16_t *d_out, const float *d_a,
         const float *d_b, const uint16_t *d_scale, int n) {
        k_s2f6x2_f32_sc_relu<<<grid, block>>>(d_out, d_a, d_b, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_f32_satfinite_relu_scaled_rn_focus", out_u16, 4);
  run_pair_kernel<uint32_t>(
      s2f6_relu_focus_a, s2f6_relu_focus_b, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const float *d_a, const float *d_b,
         int n) { k_bf16x2_rz<<<grid, block>>>(d_out, d_a, d_b, n); });
  const std::vector<uint32_t> s2f6_relu_focus_bf16x2 = tmp_u32;
  run_unary_kernel<uint32_t, uint16_t>(
      s2f6_relu_focus_bf16x2, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_s2f6x2_bf16x2_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_relu_rn_focus", tmp_u16, 4);
  run_unary_kernel_with_scale<uint32_t, uint16_t, uint16_t>(
      s2f6_relu_focus_bf16x2, s2f6_relu_focus_scales, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in,
         const uint16_t *d_scale, int n) {
        k_s2f6x2_bf16x2_sc_relu<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_relu_scaled_rn_focus", tmp_u16, 4);

  run_unary_kernel<uint16_t, uint32_t>(
      e4m3x2_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e4m3x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e4m3x2_rn", tmp_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      e5m2x2_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e5m2x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e5m2x2_rn", tmp_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      e2m1x2_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e2m1x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e2m1x2_rn", tmp_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      e2m3x2_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e2m3x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e2m3x2_rn", tmp_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      e3m2x2_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e3m2x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e3m2x2_rn", tmp_u32, 8);

  run_unary_kernel<uint32_t, uint16_t>(
      f16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e4m3x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e4m3x2_f16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      f16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e5m2x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e5m2x2_f16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      f16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m1x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m1x2_f16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      f16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m3x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m3x2_f16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      f16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e3m2x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e3m2x2_f16x2_satfinite_rn", tmp_u16, 4);

  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e4m3x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e4m3x2_bf16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e5m2x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e5m2x2_bf16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m1x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m1x2_bf16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m3x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m3x2_bf16x2_satfinite_rn", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e3m2x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e3m2x2_bf16x2_satfinite_rn", tmp_u16, 4);

  run_unary_kernel<uint16_t, uint32_t>(
      u16_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e4m3x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e4m3x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e4m3x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e4m3x2_f16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint16_t, uint32_t>(
      u16_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e5m2x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e5m2x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e5m2x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e5m2x2_f16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint16_t, uint32_t>(
      e2m1x2_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e2m1x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e2m1x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m1x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m1x2_f16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint16_t, uint32_t>(
      e2m3x2_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e2m3x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e2m3x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e2m3x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e2m3x2_f16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint16_t, uint32_t>(
      e3m2x2_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_f16x2_e3m2x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_f16x2_e3m2x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_e3m2x2_f16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_e3m2x2_f16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_satfinite_rz<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_satfinite_rz", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_rz<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_rz", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_satfinite_rp<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_satfinite_rp", tmp_u16, 4);
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_rp<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_rp", tmp_u16, 4);
#if LOWP_CVT_HAS_UE8M0X2_BF16X2_RELU
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_satfinite_relu_rz<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_satfinite_relu_rz", tmp_u16, 4);
#endif
  run_unary_kernel<uint16_t, uint32_t>(
      tmp_u16, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_bf16x2_ue8m0x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16x2_ue8m0x2_rn", tmp_u32, 8);

  run_unary_kernel<uint16_t, uint32_t>(
      u16_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_bf16x2_ue8m0x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16x2_ue8m0x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_ue8m0x2_bf16x2_satfinite_rz<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_ue8m0x2_bf16x2_satfinite_rz_roundtrip_exhaustive", tmp_u16,
           4);

  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_s2f6x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_rn", tmp_u16, 4);
  const std::vector<uint16_t> s2f6_pairs = tmp_u16;
  run_unary_kernel<uint32_t, uint16_t>(
      bf16x2_pairs, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_s2f6x2_bf16x2_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_relu_rn", tmp_u16, 4);
  run_unary_kernel<uint16_t, uint32_t>(
      s2f6_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_bf16x2_s2f6x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_rn", tmp_u32, 8);
  run_unary_kernel<uint16_t, uint32_t>(
      s2f6_pairs, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_bf16x2_s2f6x2_satfinite_relu_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_satfinite_relu_rn", tmp_u32, 8);
  run_unary_kernel_with_scale<uint32_t, uint16_t, uint16_t>(
      bf16x2_pairs, pair_scales, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in,
         const uint16_t *d_scale, int n) {
        k_s2f6x2_bf16x2_sc<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_scaled_rn", tmp_u16, 4);
  run_unary_kernel_with_scale<uint32_t, uint16_t, uint16_t>(
      bf16x2_pairs, pair_scales, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in,
         const uint16_t *d_scale, int n) {
        k_s2f6x2_bf16x2_sc_relu<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_relu_scaled_rn", tmp_u16, 4);
  run_unary_kernel_with_scale<uint16_t, uint16_t, uint32_t>(
      s2f6_pairs, pair_scales, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in,
         const uint16_t *d_scale, int n) {
        k_bf16x2_s2f6x2_sc<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_satfinite_scaled_rn", tmp_u32, 8);
  run_unary_kernel_with_scale<uint16_t, uint16_t, uint32_t>(
      s2f6_pairs, pair_scales, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in,
         const uint16_t *d_scale, int n) {
        k_bf16x2_s2f6x2_sc_relu<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_satfinite_relu_scaled_rn", tmp_u32, 8);

  run_unary_kernel<uint16_t, uint32_t>(
      u16_inputs_exhaustive, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in, int n) {
        k_bf16x2_s2f6x2_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel<uint32_t, uint16_t>(
      tmp_u32, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in, int n) {
        k_s2f6x2_bf16x2_satfinite_rn<<<grid, block>>>(d_out, d_in, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_rn_roundtrip_exhaustive", tmp_u16,
           4);
  run_unary_kernel_with_scale<uint16_t, uint16_t, uint32_t>(
      u16_inputs_exhaustive, exhaustive_scales, tmp_u32,
      [](int grid, int block, uint32_t *d_out, const uint16_t *d_in,
         const uint16_t *d_scale, int n) {
        k_bf16x2_s2f6x2_sc<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_bf16x2_s2f6x2_satfinite_scaled_rn_exhaustive", tmp_u32, 8);
  run_unary_kernel_with_scale<uint32_t, uint16_t, uint16_t>(
      tmp_u32, exhaustive_scales, tmp_u16,
      [](int grid, int block, uint16_t *d_out, const uint32_t *d_in,
         const uint16_t *d_scale, int n) {
        k_s2f6x2_bf16x2_sc<<<grid, block>>>(d_out, d_in, d_scale, n);
      });
  dump_hex(ofs, "k_s2f6x2_bf16x2_satfinite_scaled_rn_roundtrip_exhaustive",
           tmp_u16, 4);

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3 || std::string(argv[1]) != "--output") {
    std::cerr << "usage: " << argv[0] << " --output <path>\n";
    return 2;
  }
  return run_and_dump(argv[2]);
}
