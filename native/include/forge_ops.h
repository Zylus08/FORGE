#pragma once
// FORGE M2 — Native CPU Acceleration Kernels
// C++20 header declaring fused operations with C++ autograd integration
#include <torch/extension.h>
#include <vector>

// ---------------------------------------------------------------------------
// 1. Fused ReLU forward / backward
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_forward(torch::Tensor input);
torch::Tensor fused_relu_backward(torch::Tensor grad_output, torch::Tensor mask);

// ---------------------------------------------------------------------------
// 2. Fused ReLU + Bias forward / backward
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_bias_forward(torch::Tensor input, torch::Tensor bias);
std::vector<torch::Tensor> fused_relu_bias_backward(torch::Tensor grad_output, torch::Tensor mask);

// ---------------------------------------------------------------------------
// 3. Fused Residual Add + ReLU forward / backward (Candidate C: Fused Residual Path)
// Computes: out = max(0, x + residual) in a single pass over memory.
// Saves 1-byte mask or bit-mask to eliminate intermediate tensor buffer.
// In backward: grad_x = mask * grad_out, grad_residual = mask * grad_out.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_add_relu_forward(torch::Tensor x, torch::Tensor residual);
std::vector<torch::Tensor> fused_add_relu_backward(torch::Tensor grad_output, torch::Tensor mask);

// ---------------------------------------------------------------------------
// 4. Native C++ Autograd entry points (Zero Python Autograd overhead)
// ---------------------------------------------------------------------------
torch::Tensor native_autograd_fused_relu(torch::Tensor input);
torch::Tensor native_autograd_fused_add_relu(torch::Tensor x, torch::Tensor residual);

// ---------------------------------------------------------------------------
// 5. M3 Zero-Mask Fused Operations (Eliminating boolean mask DRAM traffic)
// ---------------------------------------------------------------------------
torch::Tensor fused_relu_zero_mask_forward(torch::Tensor input);
torch::Tensor fused_relu_zero_mask_backward(torch::Tensor grad_output, torch::Tensor output);
torch::Tensor fused_add_relu_zero_mask_forward(torch::Tensor x, torch::Tensor residual);
std::vector<torch::Tensor> fused_add_relu_zero_mask_backward(torch::Tensor grad_output, torch::Tensor output);

torch::Tensor native_autograd_fused_relu_v3(torch::Tensor input);
torch::Tensor native_autograd_fused_add_relu_v3(torch::Tensor x, torch::Tensor residual);
