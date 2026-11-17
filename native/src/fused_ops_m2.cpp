// FORGE M2 — Native CPU Acceleration Kernels (C++20, AVX2, C++ Autograd Dispatch)
// Zero-layout-conversion contiguous & channels_last tensor processing

#include "forge_ops.h"
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/Parallel.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// ---------------------------------------------------------------------------
// Helper: ensure tensor is float32 on CPU and flat-contiguous (NCHW or NHWC)
// Zero layout conversion if already contiguous or channels_last!
// ---------------------------------------------------------------------------
static inline torch::Tensor ensure_flat_f32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cpu(), "forge_ops: ", name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, "forge_ops: ", name, " must be float32");
    if (t.is_contiguous() || t.is_contiguous(at::MemoryFormat::ChannelsLast)) {
        return t;
    }
    return t.contiguous();
}

// ---------------------------------------------------------------------------
// 1. fused_relu_forward
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_forward(torch::Tensor input) {
    auto x     = ensure_flat_f32(input, "input");
    auto out   = torch::empty_like(x, x.suggest_memory_format());
    auto mask  = torch::empty_like(x, x.options().dtype(torch::kBool), x.suggest_memory_format());

    const float* __restrict x_ptr   = x.data_ptr<float>();
    float*       __restrict o_ptr   = out.data_ptr<float>();
    bool*        __restrict m_ptr   = mask.data_ptr<bool>();
    const int64_t N = x.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            const __m128i one8 = _mm_set1_epi8(1);
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 v   = _mm256_loadu_ps(x_ptr + i);
                __m256 r   = _mm256_max_ps(v, zero8);
                _mm256_storeu_ps(o_ptr + i, r);

                __m256 cmp = _mm256_cmp_ps(v, zero8, _CMP_GT_OQ);
                __m256i cmpi = _mm256_castps_si256(cmp);
                __m128i lo = _mm256_castsi256_si128(cmpi);
                __m128i hi = _mm256_extractf128_si256(cmpi, 1);
                __m128i p16 = _mm_packs_epi32(lo, hi);
                __m128i p8  = _mm_packs_epi16(p16, p16);
                __m128i b8  = _mm_and_si128(p8, one8);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(m_ptr + i), b8);
            }
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
// 1b. fused_relu_backward
// ---------------------------------------------------------------------------
torch::Tensor fused_relu_backward(torch::Tensor grad_output, torch::Tensor mask) {
    TORCH_CHECK(grad_output.is_cpu(), "forge_ops: grad_output must be CPU");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "forge_ops: grad_output must be float32");
    TORCH_CHECK(mask.is_cpu(), "forge_ops: mask must be CPU");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "forge_ops: mask must be bool");

    auto g_in   = ensure_flat_f32(grad_output, "grad_output");
    auto m      = (mask.is_contiguous() || mask.is_contiguous(at::MemoryFormat::ChannelsLast)) ? mask : mask.contiguous();
    auto g_out  = torch::empty_like(g_in, g_in.suggest_memory_format());

    const float* __restrict gi_ptr = g_in.data_ptr<float>();
    const bool*  __restrict m_ptr  = m.data_ptr<bool>();
    float*       __restrict go_ptr = g_out.data_ptr<float>();
    const int64_t N = g_in.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 g = _mm256_loadu_ps(gi_ptr + i);
                __m128i mb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(m_ptr + i));
                __m256i mi = _mm256_cvtepu8_epi32(mb);
                __m256  mf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, _mm256_set1_epi32(0)));
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
// 2. fused_relu_bias_forward & backward
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_bias_forward(torch::Tensor input, torch::Tensor bias) {
    auto x = input.contiguous();
    auto b = bias.contiguous();

    TORCH_CHECK(b.dim() == 1, "forge_ops: bias must be 1-D");
    TORCH_CHECK(x.dim() >= 2, "forge_ops: input must have >= 2 dims");
    TORCH_CHECK(x.size(1) == b.size(0), "forge_ops: channel mismatch");

    auto out  = torch::empty_like(x);
    auto mask = torch::empty(x.sizes(), torch::TensorOptions().dtype(torch::kBool).device(x.device()));

    const float* __restrict x_ptr = x.data_ptr<float>();
    const float* __restrict b_ptr = b.data_ptr<float>();
    float*       __restrict o_ptr = out.data_ptr<float>();
    bool*        __restrict m_ptr = mask.data_ptr<bool>();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    int64_t spatial = 1;
    for (int64_t d = 2; d < x.dim(); ++d) spatial *= x.size(d);

    at::parallel_for(0, N * C, /*grain_size=*/16,
        [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
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
                    int bitmask = _mm256_movemask_ps(_mm256_cmp_ps(v, zero8, _CMP_GT_OQ));
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

std::vector<torch::Tensor> fused_relu_bias_backward(torch::Tensor grad_output, torch::Tensor mask) {
    auto g_in  = grad_output.contiguous();
    auto m     = mask.contiguous();
    auto g_out = torch::empty_like(g_in);

    const int64_t N = g_in.size(0);
    const int64_t C = g_in.size(1);
    int64_t spatial = 1;
    for (int64_t d = 2; d < g_in.dim(); ++d) spatial *= g_in.size(d);

    const float* __restrict gi_ptr = g_in.data_ptr<float>();
    const bool*  __restrict m_ptr  = m.data_ptr<bool>();
    float*       __restrict go_ptr = g_out.data_ptr<float>();

    at::parallel_for(0, N * C, /*grain_size=*/16,
        [&](int64_t begin, int64_t end) {
            for (int64_t nc = begin; nc < end; ++nc) {
                const float* gi_nc = gi_ptr + nc * spatial;
                const bool*  m_nc  = m_ptr  + nc * spatial;
                float*       go_nc = go_ptr + nc * spatial;
#ifdef __AVX2__
                int64_t s = 0;
                for (; s + 8 <= spatial; s += 8) {
                    __m256 g = _mm256_loadu_ps(gi_nc + s);
                    __m128i mb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(m_nc + s));
                    __m256i mi = _mm256_cvtepu8_epi32(mb);
                    __m256  mf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, _mm256_set1_epi32(0)));
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

    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    for (int64_t d = 2; d < g_out.dim(); ++d) reduce_dims.push_back(d);
    auto grad_bias = g_out.sum(reduce_dims);

    return {g_out, grad_bias};
}

// ---------------------------------------------------------------------------
// 3. fused_add_relu_forward & backward (Zero layout conversion!)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_add_relu_forward(torch::Tensor x, torch::Tensor residual) {
    auto x_c = ensure_flat_f32(x, "x");
    auto r_c = ensure_flat_f32(residual, "residual");
    TORCH_CHECK(x_c.sizes() == r_c.sizes(), "forge_ops: x and residual must have same shape");

    auto out  = torch::empty_like(x_c, x_c.suggest_memory_format());
    auto mask = torch::empty_like(x_c, x_c.options().dtype(torch::kBool), x_c.suggest_memory_format());

    const float* __restrict xp = x_c.data_ptr<float>();
    const float* __restrict rp = r_c.data_ptr<float>();
    float*       __restrict op = out.data_ptr<float>();
    bool*        __restrict mp = mask.data_ptr<bool>();
    const int64_t N = x_c.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            const __m128i one8 = _mm_set1_epi8(1);
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 xv = _mm256_loadu_ps(xp + i);
                __m256 rv = _mm256_loadu_ps(rp + i);
                __m256 sum = _mm256_add_ps(xv, rv);
                __m256 r   = _mm256_max_ps(sum, zero8);
                _mm256_storeu_ps(op + i, r);

                __m256 cmp = _mm256_cmp_ps(sum, zero8, _CMP_GT_OQ);
                __m256i cmpi = _mm256_castps_si256(cmp);
                __m128i lo = _mm256_castsi256_si128(cmpi);
                __m128i hi = _mm256_extractf128_si256(cmpi, 1);
                __m128i p16 = _mm_packs_epi32(lo, hi);
                __m128i p8  = _mm_packs_epi16(p16, p16);
                __m128i b8  = _mm_and_si128(p8, one8);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(mp + i), b8);
            }
            for (; i < end; ++i) {
                float sum = xp[i] + rp[i];
                bool pos = sum > 0.0f;
                mp[i] = pos;
                op[i] = pos ? sum : 0.0f;
            }
#else
            for (int64_t i = begin; i < end; ++i) {
                float sum = xp[i] + rp[i];
                bool pos = sum > 0.0f;
                mp[i] = pos;
                op[i] = pos ? sum : 0.0f;
            }
#endif
        });

    return {out, mask};
}

std::vector<torch::Tensor> fused_add_relu_backward(torch::Tensor grad_output, torch::Tensor mask) {
    auto grad_in = fused_relu_backward(grad_output, mask);
    return {grad_in, grad_in};
}

// ---------------------------------------------------------------------------
// 4. C++ Autograd Custom Functions
// ---------------------------------------------------------------------------
class FusedReLUFunction : public torch::autograd::Function<FusedReLUFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input) {
        auto res = fused_relu_forward(input);
        ctx->save_for_backward({res[1]});
        return res[0];
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto grad_input = fused_relu_backward(grad_outputs[0], saved[0]);
        return {grad_input};
    }
};

class FusedAddReLUFunction : public torch::autograd::Function<FusedAddReLUFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor x,
                                 torch::Tensor residual) {
        auto res = fused_add_relu_forward(x, residual);
        ctx->save_for_backward({res[1]});
        return res[0];
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto grads = fused_add_relu_backward(grad_outputs[0], saved[0]);
        return {grads[0], grads[1]};
    }
};

torch::Tensor native_autograd_fused_relu(torch::Tensor input) {
    if (!at::GradMode::is_enabled()) {
        return at::relu(input);
    }
    return FusedReLUFunction::apply(input);
}

torch::Tensor native_autograd_fused_add_relu(torch::Tensor x, torch::Tensor residual) {
    if (!at::GradMode::is_enabled()) {
        return at::relu(x + residual);
    }
    return FusedAddReLUFunction::apply(x, residual);
}

// ---------------------------------------------------------------------------
// 5. M3 Zero-Mask Fused ReLU (No Boolean Mask Allocation!)
// ---------------------------------------------------------------------------
torch::Tensor fused_relu_zero_mask_forward(torch::Tensor input) {
    auto x   = ensure_flat_f32(input, "input");
    auto out = torch::empty_like(x, x.suggest_memory_format());

    const float* __restrict xp = x.data_ptr<float>();
    float*       __restrict op = out.data_ptr<float>();
    const int64_t N = x.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 v = _mm256_loadu_ps(xp + i);
                __m256 r = _mm256_max_ps(v, zero8);
                _mm256_storeu_ps(op + i, r);
            }
            for (; i < end; ++i) {
                float v = xp[i];
                op[i] = v > 0.0f ? v : 0.0f;
            }
#else
            for (int64_t i = begin; i < end; ++i) {
                float v = xp[i];
                op[i] = v > 0.0f ? v : 0.0f;
            }
#endif
        });
    return out;
}

torch::Tensor fused_relu_zero_mask_backward(torch::Tensor grad_output, torch::Tensor output) {
    auto g_in  = ensure_flat_f32(grad_output, "grad_output");
    auto out   = ensure_flat_f32(output, "output");
    auto g_out = torch::empty_like(g_in, g_in.suggest_memory_format());

    const float* __restrict gip = g_in.data_ptr<float>();
    const float* __restrict op  = out.data_ptr<float>();
    float*       __restrict gop = g_out.data_ptr<float>();
    const int64_t N = g_in.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 g    = _mm256_loadu_ps(gip + i);
                __m256 o    = _mm256_loadu_ps(op + i);
                __m256 mask = _mm256_cmp_ps(o, zero8, _CMP_GT_OQ);
                __m256 r    = _mm256_and_ps(g, mask);
                _mm256_storeu_ps(gop + i, r);
            }
            for (; i < end; ++i) {
                gop[i] = (op[i] > 0.0f) ? gip[i] : 0.0f;
            }
#else
            for (int64_t i = begin; i < end; ++i) {
                gop[i] = (op[i] > 0.0f) ? gip[i] : 0.0f;
            }
#endif
        });
    return g_out;
}

// ---------------------------------------------------------------------------
// 6. M3 Zero-Mask Fused Add + ReLU (No Boolean Mask Allocation!)
// ---------------------------------------------------------------------------
torch::Tensor fused_add_relu_zero_mask_forward(torch::Tensor x, torch::Tensor residual) {
    auto xc = ensure_flat_f32(x, "x");
    auto rc = ensure_flat_f32(residual, "residual");
    TORCH_CHECK(xc.sizes() == rc.sizes(), "forge_ops: x and residual must match shape");
    auto out = torch::empty_like(xc, xc.suggest_memory_format());

    const float* __restrict xp = xc.data_ptr<float>();
    const float* __restrict rp = rc.data_ptr<float>();
    float*       __restrict op = out.data_ptr<float>();
    const int64_t N = xc.numel();

    at::parallel_for(0, N, /*grain_size=*/32768,
        [&](int64_t begin, int64_t end) {
#ifdef __AVX2__
            const __m256 zero8 = _mm256_setzero_ps();
            int64_t i = begin;
            for (; i + 8 <= end; i += 8) {
                __m256 xv  = _mm256_loadu_ps(xp + i);
                __m256 rv  = _mm256_loadu_ps(rp + i);
                __m256 sum = _mm256_add_ps(xv, rv);
                __m256 r   = _mm256_max_ps(sum, zero8);
                _mm256_storeu_ps(op + i, r);
            }
            for (; i < end; ++i) {
                float sum = xp[i] + rp[i];
                op[i] = sum > 0.0f ? sum : 0.0f;
            }
#else
            for (int64_t i = begin; i < end; ++i) {
                float sum = xp[i] + rp[i];
                op[i] = sum > 0.0f ? sum : 0.0f;
            }
#endif
        });
    return out;
}

std::vector<torch::Tensor> fused_add_relu_zero_mask_backward(torch::Tensor grad_output, torch::Tensor output) {
    auto grad_in = fused_relu_zero_mask_backward(grad_output, output);
    return {grad_in, grad_in};
}

class FusedReLUZeroMaskFunction : public torch::autograd::Function<FusedReLUZeroMaskFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input) {
        auto out = fused_relu_zero_mask_forward(input);
        ctx->save_for_backward({out});
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto grad_in = fused_relu_zero_mask_backward(grad_outputs[0], saved[0]);
        return {grad_in};
    }
};

class FusedAddReLUZeroMaskFunction : public torch::autograd::Function<FusedAddReLUZeroMaskFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor x,
                                 torch::Tensor residual) {
        auto out = fused_add_relu_zero_mask_forward(x, residual);
        ctx->save_for_backward({out});
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                   torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto grads = fused_add_relu_zero_mask_backward(grad_outputs[0], saved[0]);
        return {grads[0], grads[1]};
    }
};

torch::Tensor native_autograd_fused_relu_v3(torch::Tensor input) {
    if (!at::GradMode::is_enabled()) {
        return at::relu(input);
    }
    return FusedReLUZeroMaskFunction::apply(input);
}

torch::Tensor native_autograd_fused_add_relu_v3(torch::Tensor x, torch::Tensor residual) {
    if (!at::GradMode::is_enabled()) {
        return at::relu(x + residual);
    }
    return FusedAddReLUZeroMaskFunction::apply(x, residual);
}

// ---------------------------------------------------------------------------
// pybind11 module bindings
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FORGE M3 native CPU acceleration kernels with C++ Autograd Dispatch";

    m.def("fused_relu_forward", &fused_relu_forward, "Fused ReLU forward: (out, mask)");
    m.def("fused_relu_backward", &fused_relu_backward, "Fused ReLU backward: grad_input");
    m.def("fused_relu_bias_forward", &fused_relu_bias_forward, "Fused ReLU+bias forward: (out, mask)");
    m.def("fused_relu_bias_backward", &fused_relu_bias_backward, "Fused ReLU+bias backward: (grad_in, grad_bias)");

    m.def("fused_add_relu_forward", &fused_add_relu_forward, "Fused Add+ReLU forward: (out, mask)");
    m.def("fused_add_relu_backward", &fused_add_relu_backward, "Fused Add+ReLU backward: (grad_x, grad_residual)");

    m.def("native_autograd_fused_relu", &native_autograd_fused_relu, "Native C++ Autograd Fused ReLU (M2)");
    m.def("native_autograd_fused_add_relu", &native_autograd_fused_add_relu, "Native C++ Autograd Fused Add+ReLU (M2)");

    // M3 Zero-Mask bindings
    m.def("fused_relu_zero_mask_forward", &fused_relu_zero_mask_forward, "M3 Fused ReLU forward (zero mask)");
    m.def("fused_relu_zero_mask_backward", &fused_relu_zero_mask_backward, "M3 Fused ReLU backward (zero mask)");
    m.def("fused_add_relu_zero_mask_forward", &fused_add_relu_zero_mask_forward, "M3 Fused Add+ReLU forward (zero mask)");
    m.def("fused_add_relu_zero_mask_backward", &fused_add_relu_zero_mask_backward, "M3 Fused Add+ReLU backward (zero mask)");
    m.def("native_autograd_fused_relu_v3", &native_autograd_fused_relu_v3, "Native C++ Autograd Fused ReLU (M3 Zero Mask)");
    m.def("native_autograd_fused_add_relu_v3", &native_autograd_fused_add_relu_v3, "Native C++ Autograd Fused Add+ReLU (M3 Zero Mask)");
}
