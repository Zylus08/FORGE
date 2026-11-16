# FORGE Milestone M2: Native Kernel Architecture & Engineering

**Date:** 2026-11-17  
**Auditor:** Principal C++ Optimization & CPU Kernel Architect  
**Artifact Location:** [`audit/m2/NATIVE_KERNEL_ARCHITECTURE.md`](file:///E:/FORGE/audit/m2/NATIVE_KERNEL_ARCHITECTURE.md)  
**Source Code:** [`native/src/fused_ops_m2.cpp`](file:///E:/FORGE/native/src/fused_ops_m2.cpp), [`native/include/forge_ops.h`](file:///E:/FORGE/native/include/forge_ops.h)

---

## 1. Architectural Scope

In Milestone M2, native CPU kernel development expanded beyond standalone elementwise activation to target **Candidate C: Fused Residual Path Optimization**.

In standard ResNet architectures, each residual block executes:
```text
out = conv2(x)
out = bn2(out)
out = out + residual    <-- Tensor allocation + memory read/write pass
out = relu(out)         <-- Activation pass + mask allocation
```
During backward propagation:
```text
grad_out = relu_backward(grad, mask)
grad_x = grad_out       <-- Distributed to branch 1
grad_res = grad_out     <-- Distributed to branch 2
```

In FORGE M2, we engineered **`fused_add_relu`**, combining residual summation and activation into a single vectorized pass:
$$\text{out}[i] = \max(0, x[i] + \text{residual}[i])$$
$$\text{mask}[i] = (x[i] + \text{residual}[i] > 0)$$

---

## 2. AVX2 Vectorization Mechanics

The kernel leverages x86-64 AVX2 intrinsics processing 8 single-precision float32 elements per vector register:

### Forward Vector Loop (`fused_add_relu_forward`):
```cpp
const __m256 zero8 = _mm256_setzero_ps();
const __m128i one8 = _mm_set1_epi8(1);
int64_t i = begin;
for (; i + 8 <= end; i += 8) {
    __m256 xv  = _mm256_loadu_ps(xp + i);
    __m256 rv  = _mm256_loadu_ps(rp + i);
    __m256 sum = _mm256_add_ps(xv, rv);
    __m256 r   = _mm256_max_ps(sum, zero8);
    _mm256_storeu_ps(op + i, r);

    // Compute binary comparison mask (0xFFFFFFFF or 0x00000000)
    __m256 cmp   = _mm256_cmp_ps(sum, zero8, _CMP_GT_OQ);
    __m256i cmpi = _mm256_castps_si256(cmp);
    __m128i lo   = _mm256_castsi256_si128(cmpi);
    __m128i hi   = _mm256_extractf128_si256(cmpi, 1);
    
    // Pack 8 x 32-bit integers to 8 x 16-bit, then to 8 x 8-bit bytes
    __m128i p16  = _mm_packs_epi32(lo, hi);
    __m128i p8   = _mm_packs_epi16(p16, p16);
    __m128i b8   = _mm_and_si128(p8, one8);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(mp + i), b8);
}
```

### Backward Vector Loop (`fused_add_relu_backward`):
```cpp
for (; i + 8 <= end; i += 8) {
    __m256 g    = _mm256_loadu_ps(gi_ptr + i);
    __m128i mb  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(m_ptr + i));
    __m256i mi  = _mm256_cvtepu8_epi32(mb);
    __m256  mf  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(mi, _mm256_set1_epi32(0)));
    // _mm256_andnot_ps zeros gradient where mask is false (0)
    __m256 res  = _mm256_andnot_ps(mf, g);
    _mm256_storeu_ps(go_ptr + i, res);
}
```

---

## 3. Parallelism & Thread Scheduling

- **Grain Size:** Configured to `grain_size = 32,768` elements per task slice in `at::parallel_for`.
- **Cache Locality:** Slices are sized to fit comfortably inside the per-core L1 Data Cache (48 KB on Golden Cove P-cores), preventing L2/L3 cache misses during the joint read/write pass.
- **Scalar Tail Guard:** A scalar tail loop handles any remaining elements when `numel % 8 != 0`, guaranteeing numerical correctness for arbitrary non-vector-aligned tensor dimensions.

---

## 4. C++20 Language Compliance

- **Standard:** Modern C++20 (`/std:c++20` on MSVC v19.43).
- **Memory Safety:** Pointer references use `__restrict` qualifiers to inform the compiler of zero-aliasing between input, residual, and output pointers.
- **Strict Exception Safety:** PyTorch `TORCH_CHECK` macros validate tensor shape conformity, device residency (CPU), and dtype (float32) prior to pointer extraction.
