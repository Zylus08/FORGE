// FORGE M1 — Fused ReLU + Bias kernels (C++20, ATen-based)
// Implementation: single-pass forward stores ReLU mask alongside output,
// eliminating a second read/write cycle vs. separate relu_ + add_ calls.
// Uses at::parallel_for for multi-core dispatch.
// AVX2 path enabled when __AVX2__ is defined (MSVC /arch:AVX2).

#include "forge_ops.h"
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Helper: ensure tensor is contiguous float32 on CPU
// ---------------------------------------------------------------------------
static inline torch::Tensor ensure_contiguous_f32(const torch::Tensor& t,
                                                   const char* name) {
    TORCH_CHECK(t.is_cpu(),
                "forge_ops: ", name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32,
                "forge_ops: ", name, " must be float32");
    return t.contiguous();
}

// ---------------------------------------------------------------------------
// fused_relu_forward
// Single memory pass: output[i] = max(0, input[i]), mask[i] = input[i] > 0
// Returns {output, mask}
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_forward(torch::Tensor input) {
    auto x     = ensure_contiguous_f32(input, "input");
    auto out   = torch::empty_like(x);
    auto mask  = torch::empty(x.sizes(), torch::TensorOptions()
                               .dtype(torch::kBool).device(x.device()));

    const float* __restrict x_ptr   = x.data_ptr<float>();
    float*       __restrict o_ptr   = out.data_ptr<float>();
    bool*        __restrict m_ptr   = mask.data_ptr<bool>();
    const int64_t N = x.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            // AVX2 vectorised path: process 8 floats per iteration
            const __m256 zero8 = _mm256_setzero_ps();
            const __m128i one8 = _mm_set1_epi8(1);
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 v   = _mm256_loadu_ps(x_ptr + i);
                __m256 r   = _mm256_max_ps(v, zero8);
                _mm256_storeu_ps(o_ptr + i, r);
                // Mask: v > 0 => 0xFFFFFFFF, else 0x00000000
                __m256 cmp = _mm256_cmp_ps(v, zero8, _CMP_GT_OQ);
                __m256i cmpi = _mm256_castps_si256(cmp);
                __m128i lo = _mm256_castsi256_si128(cmpi);
                __m128i hi = _mm256_extractf128_si256(cmpi, 1);
                // Pack 8x32 to 8x16, then to 8x8
                __m128i p16 = _mm_packs_epi32(lo, hi);
                __m128i p8  = _mm_packs_epi16(p16, p16);
                // Convert 0xFF to 0x01 (true boolean)
                __m128i b8  = _mm_and_si128(p8, one8);
                // Store 8 bytes
                _mm_storel_epi64(reinterpret_cast<__m128i*>(m_ptr + i), b8);
            }
            // scalar tail
            for (; i < end; ++i) {
                float v    = x_ptr[i];
                bool  pos  = v > 0.0f;
                m_ptr[i]   = pos;
                o_ptr[i]   = pos ? v : 0.0f;
            }
#else
            for (int64_t i = begin; i < end; ++i) {
                float v    = x_ptr[i];
                bool  pos  = v > 0.0f;
                m_ptr[i]   = pos;
                o_ptr[i]   = pos ? v : 0.0f;
            }
#endif
        });

    return {out, mask};
}

// ---------------------------------------------------------------------------
// fused_relu_backward
// Single pass: grad_in[i] = grad_out[i] * mask[i]
// Returns grad_input
// ---------------------------------------------------------------------------
torch::Tensor fused_relu_backward(torch::Tensor grad_output,
                                   torch::Tensor mask) {
    TORCH_CHECK(grad_output.is_cpu(), "forge_ops: grad_output must be CPU");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "forge_ops: grad_output must be float32");
    TORCH_CHECK(mask.is_cpu(), "forge_ops: mask must be CPU");
    TORCH_CHECK(mask.scalar_type() == torch::kBool,
                "forge_ops: mask must be bool");
    TORCH_CHECK(grad_output.sizes() == mask.sizes(),
                "forge_ops: grad_output and mask must have same shape");

    auto g_in   = grad_output.contiguous();
    auto m      = mask.contiguous();
    auto g_out  = torch::empty_like(g_in);

    const float* __restrict gi_ptr = g_in.data_ptr<float>();
    const bool*  __restrict m_ptr  = m.data_ptr<bool>();
    float*       __restrict go_ptr = g_out.data_ptr<float>();
    const int64_t N = g_in.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 g = _mm256_loadu_ps(gi_ptr + i);
                // Build mask register from 8 bool bytes
                // We load 8 booleans individually (bool is 1 byte on x86)
                // and build a blendv mask (high bit = sign bit of float)
                // If mask byte != 0 => keep gradient, else zero it
                __m128i mb = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(m_ptr + i));
                // Convert 8 bytes to 8 int32 (0 or 1), scale to 0x00000000/0xFFFFFFFF
                // Using a cast: bool!=0 => need to build float sign bits
                // Simpler: broadcast each bool to float via _mm256_blendv_ps
                // Build float_mask: 0.0 if false, -0.0 (sign bit set) if true
                // Actually use _mm256_cmp to avoid byte-by-byte loop:
                // Convert 8 uint8 to 8 float32, compare != 0
                __m256i mi = _mm256_cvtepu8_epi32(mb);  // 8 x int32
                __m256  mf = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(mi, _mm256_set1_epi32(0)));
                // mf has 0x00000000 where mask==0 (false), 0xFFFFFFFF where mask!=0
                // We want: keep g where mask is true (!=0), zero where false
                // _mm256_andnot_ps(mf, g) zeros where mf is all-ones (mask==false)
                __m256 result = _mm256_andnot_ps(mf, g);
                _mm256_storeu_ps(go_ptr + i, result);
            }
            for (; i < end; ++i)
                go_ptr[i] = m_ptr[i] ? gi_ptr[i] : 0.0f;
#else
            for (int64_t i = begin; i < end; ++i)
                go_ptr[i] = m_ptr[i] ? gi_ptr[i] : 0.0f;
#endif
        });

    return g_out;
}

// ---------------------------------------------------------------------------
// fused_relu_bias_forward
// Single pass: out[i] = max(0, input[i] + bias[channel(i)])
// input shape: (N, C, *) — bias is 1D of size C
// Returns {output, mask}
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_bias_forward(torch::Tensor input,
                                                    torch::Tensor bias) {
    auto x = ensure_contiguous_f32(input, "input");
    auto b = ensure_contiguous_f32(bias,  "bias");

    TORCH_CHECK(b.dim() == 1, "forge_ops: bias must be 1-D");
    TORCH_CHECK(x.dim() >= 2,
                "forge_ops: input must have at least 2 dims (N, C, ...)");
    TORCH_CHECK(x.size(1) == b.size(0),
                "forge_ops: input.size(1)=", x.size(1),
                " != bias.size(0)=", b.size(0));

    auto out  = torch::empty_like(x);
    auto mask = torch::empty(x.sizes(),
                             torch::TensorOptions()
                             .dtype(torch::kBool).device(x.device()));

    const float* __restrict x_ptr = x.data_ptr<float>();
    const float* __restrict b_ptr = b.data_ptr<float>();
    float*       __restrict o_ptr = out.data_ptr<float>();
    bool*        __restrict m_ptr = mask.data_ptr<bool>();

    // Compute strides for channel indexing
    // channel of element i = (i / spatial_size) % C
    const int64_t N      = x.size(0);
    const int64_t C      = x.size(1);
    const int64_t numel  = x.numel();
    // Elements per (n, c) slice — product of dims 2..end
    int64_t spatial = 1;
    for (int64_t d = 2; d < x.dim(); ++d) spatial *= x.size(d);
    const int64_t nc_stride = C * spatial;  // stride of batch dim

    at::parallel_for(0, N * C, /*grain_size=*/16,
        [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                int64_t n = nc / C;
                int64_t c = nc % C;
                float bval = b_ptr[c];
                const float* xi = x_ptr + nc * spatial;
                float*        oi = o_ptr + nc * spatial;
                bool*         mi = m_ptr + nc * spatial;
#ifdef __AVX2__
                const __m256 bv8  = _mm256_set1_ps(bval);
                const __m256 zero8 = _mm256_setzero_ps();
                int64_t s = 0;
                for (; s + 8 <= spatial; s += 8) {
                    __m256 v   = _mm256_add_ps(_mm256_loadu_ps(xi + s), bv8);
                    __m256 r   = _mm256_max_ps(v, zero8);
                    _mm256_storeu_ps(oi + s, r);
                    int bitmask = _mm256_movemask_ps(
                        _mm256_cmp_ps(v, zero8, _CMP_GT_OQ));
                    for (int k = 0; k < 8; ++k)
                        mi[s + k] = (bool)((bitmask >> k) & 1);
                }
                for (; s < spatial; ++s) {
                    float v  = xi[s] + bval;
                    bool pos = v > 0.0f;
                    mi[s]    = pos;
                    oi[s]    = pos ? v : 0.0f;
                }
#else
                for (int64_t s = 0; s < spatial; ++s) {
                    float v  = xi[s] + bval;
                    bool pos = v > 0.0f;
                    mi[s]    = pos;
                    oi[s]    = pos ? v : 0.0f;
                }
#endif
            }
        });

    return {out, mask};
}

// ---------------------------------------------------------------------------
// fused_relu_bias_backward
// grad_input[i]  = grad_output[i] * mask[i]
// grad_bias[c]   = sum over {i : channel(i)==c} of grad_output[i] * mask[i]
// Returns {grad_input, grad_bias}
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_bias_backward(torch::Tensor grad_output,
                                                     torch::Tensor mask) {
    TORCH_CHECK(grad_output.is_cpu(), "forge_ops: grad_output must be CPU");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "forge_ops: grad_output must be float32");
    TORCH_CHECK(mask.is_cpu(), "forge_ops: mask must be CPU");
    TORCH_CHECK(mask.scalar_type() == torch::kBool,
                "forge_ops: mask must be bool");
    TORCH_CHECK(grad_output.sizes() == mask.sizes(),
                "forge_ops: grad_output and mask must have same shape");
    TORCH_CHECK(grad_output.dim() >= 2,
                "forge_ops: grad_output must have at least 2 dims");

    auto g_in  = grad_output.contiguous();
    auto m     = mask.contiguous();
    auto g_out = torch::empty_like(g_in);

    const int64_t N      = g_in.size(0);
    const int64_t C      = g_in.size(1);
    int64_t spatial = 1;
    for (int64_t d = 2; d < g_in.dim(); ++d) spatial *= g_in.size(d);

    const float* __restrict gi_ptr = g_in.data_ptr<float>();
    const bool*  __restrict m_ptr  = m.data_ptr<bool>();
    float*       __restrict go_ptr = g_out.data_ptr<float>();

    // Compute grad_input using parallel_for
    at::parallel_for(0, N * C, /*grain_size=*/16,
        [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const float* gi_nc = gi_ptr + nc * spatial;
                const bool*  m_nc  = m_ptr  + nc * spatial;
                float*       go_nc = go_ptr + nc * spatial;
#ifdef __AVX2__
                const __m256 zero8 = _mm256_setzero_ps();
                int64_t s = 0;
                for (; s + 8 <= spatial; s += 8) {
                    __m256 g = _mm256_loadu_ps(gi_nc + s);
                    __m128i mb = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(m_nc + s));
                    __m256i mi = _mm256_cvtepu8_epi32(mb);
                    __m256  mf = _mm256_castsi256_ps(
                        _mm256_cmpeq_epi32(mi, _mm256_set1_epi32(0)));
                    __m256 result = _mm256_andnot_ps(mf, g);
                    _mm256_storeu_ps(go_nc + s, result);
                }
                for (; s < spatial; ++s) {
                    go_nc[s] = m_nc[s] ? gi_nc[s] : 0.0f;
                }
#else
                for (int64_t s = 0; s < spatial; ++s) {
                    go_nc[s] = m_nc[s] ? gi_nc[s] : 0.0f;
                }
#endif
            }
        });

    // Compute grad_bias: sum of grad_input over batch dim (0) and spatial dims (2..dim-1)
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t d = 2; d < g_out.dim(); ++d) {
        reduce_dims.push_back(d);
    }
    auto grad_bias = g_out.sum(reduce_dims);

    return {g_out, grad_bias};
}

// ---------------------------------------------------------------------------
// pybind11 module bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FORGE M1 native CPU acceleration kernels";

    m.def("fused_relu_forward", &fused_relu_forward,
          "Fused ReLU forward: returns (output, mask)",
          py::arg("input"));

    m.def("fused_relu_backward", &fused_relu_backward,
          "Fused ReLU backward: returns grad_input",
          py::arg("grad_output"), py::arg("mask"));

    m.def("fused_relu_bias_forward", &fused_relu_bias_forward,
          "Fused ReLU+bias forward: returns (output, mask)",
          py::arg("input"), py::arg("bias"));

    m.def("fused_relu_bias_backward", &fused_relu_bias_backward,
          "Fused ReLU+bias backward: returns (grad_input, grad_bias)",
          py::arg("grad_output"), py::arg("mask"));
}
