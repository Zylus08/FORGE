#pragma once
// FORGE M1 — Native CPU Acceleration Kernels
// C++20 header declaring fused ReLU operations
#include <torch/extension.h>
#include <vector>

// ---------------------------------------------------------------------------
// Fused ReLU forward: applies ReLU and saves binary mask in one memory pass.
// Returns {output, mask} where mask is bool tensor (true where input > 0).
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_forward(torch::Tensor input);

// Fused ReLU backward: applies saved mask to upstream gradient.
// Returns {grad_input} (mask zeros out gradient where input was ≤ 0).
torch::Tensor fused_relu_backward(torch::Tensor grad_output,
                                   torch::Tensor mask);

// ---------------------------------------------------------------------------
// Fused ReLU + bias forward: adds per-channel bias then applies ReLU.
// bias must be 1-D with size == input.size(1) (channel dim).
// Returns {output, mask}.
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> fused_relu_bias_forward(torch::Tensor input,
                                                    torch::Tensor bias);

// Fused ReLU + bias backward: returns {grad_input, grad_bias}.
// grad_bias is the sum of masked grad_output across all non-channel dims.
std::vector<torch::Tensor> fused_relu_bias_backward(torch::Tensor grad_output,
                                                     torch::Tensor mask);
