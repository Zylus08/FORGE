# FORGE M1 Profiler Findings

**Profile:** 20 steps of eager ResNet-20 training (batch=128, CPU)

## Top-20 Operators by Self CPU Time

| Operator | Count | Self CPU (ms) | CPU Total (ms) |
|----------|-------|--------------|----------------|
| aten::convolution_backward | 380 | 8123.418 | 8234.900 |
| aten::mkldnn_convolution | 380 | 3698.740 | 3713.882 |
| enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__ | 20 | 1783.918 | 2935.481 |
| aten::native_batch_norm | 380 | 1291.647 | 1306.121 |
| aten::native_batch_norm_backward | 380 | 960.642 | 989.184 |
| aten::clamp_min | 380 | 534.160 | 534.160 |
| aten::threshold_backward | 380 | 523.112 | 523.112 |
| aten::add_ | 3100 | 425.766 | 425.766 |
| aten::copy_ | 16760 | 313.108 | 313.108 |
| autograd::engine::evaluate_function: ReluBackward0 | 380 | 284.669 | 815.588 |
| autograd::engine::evaluate_function: NativeBatchNormBackward0 | 380 | 156.118 | 1155.772 |
| forward | 20 | 148.562 | 6007.589 |
| aten::_to_copy | 11420 | 132.530 | 233.528 |
| aten::empty | 21960 | 121.336 | 121.336 |
| autograd::engine::evaluate_function: ConvolutionBackward0 | 380 | 110.701 | 8533.205 |
| aten::fill_ | 2740 | 94.859 | 94.859 |
| Optimizer.step#SGD.step | 20 | 83.022 | 325.374 |
| backward | 20 | 77.901 | 10795.419 |
| aten::div | 2560 | 76.269 | 129.071 |
| aten::clone | 5200 | 71.866 | 287.606 |

## Time by Layer Type

| Category | Time (ms) | % of Total |
|----------|-----------|------------|
| conv | 11965.020 | 59.6% |
| other | 4688.646 | 23.4% |
| batch_norm | 2436.489 | 12.1% |
| backward | 654.201 | 3.3% |
| relu | 306.578 | 1.5% |
| loss | 10.142 | 0.1% |
| linear | 5.433 | 0.0% |

## Bottleneck Analysis

**#1 Bottleneck: CONV** — 11965.0ms (59.6% of total)

**#2 Bottleneck: OTHER** — 4688.6ms (23.4% of total)

**#3 Bottleneck: BATCH_NORM** — 2436.5ms (12.1% of total)


## Amdahl Feasibility Note

- If conv+backward is 62.9% of total and kernel speedup = 2x → overall speedup = 1.46x
- If conv+backward is 62.9% of total and kernel speedup = 5x → overall speedup = 2.01x
- If conv+backward is 62.9% of total and kernel speedup = 10x → overall speedup = 2.30x

*Chrome trace exported to: benchmarks/results/profile_eager.json*
