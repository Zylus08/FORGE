# FORGE M1 Workload Selection Rationale

## 1. Primary Workload: ResNet-20 on CIFAR-10

### Overview & Parameters
- **Architecture**: ResNet-20 for CIFAR-10 (He et al., 2015, *Deep Residual Learning for Image Recognition*)
- **Total Parameters**: 272,474 (compact CNN)
- **Dataset**: CIFAR-10 (50,000 training images, 10,000 test images across 10 classes, 32x32x3)
- **Batch Size**: 128
- **Loss**: `nn.CrossEntropyLoss`
- **Optimizer**: SGD with momentum=0.9, weight_decay=1e-4, initial learning rate=0.01
- **Learning Rate Schedule**: MultiStepLR at milestones [80, 120] with gamma=0.1 over a full 160-epoch schedule
- **Accuracy Target**: >85.0% top-1 validation accuracy at 160 epochs
- **M1 Timing Regime**: 5 epochs (reproducible, sufficient for steady-state throughput measurement and per-epoch time extrapolation)
- **Random Seeds**: PyTorch: 42, NumPy: 42, Python random: 42

### Why ResNet-20 on CIFAR-10?
1. **Representative Training Pipeline**: ResNet-20 includes all standard building blocks of modern deep neural networks:
   - 2D convolutions (3x3 kernels with stride 1 and stride 2)
   - Batch Normalization (`BatchNorm2d`)
   - Rectified Linear Units (`ReLU`)
   - Residual shortcut connections (identity additions and 1x1 downsampling convolutions / zero-padding)
   - Global Average Pooling (`AdaptiveAvgPool2d` / `avg_pool2d`)
   - Linear projection (`nn.Linear` classifier)
2. **Realistic Computational Complexity**: Unlike toy MLPs, ResNet-20 exercises memory layouts (NCHW vs NHWC / channels-last), cache locality across intermediate feature maps, and multi-core thread scaling.
3. **CPU Feasibility & Iteration Speed**: Full 160 epochs take ~15-20 minutes on modern multi-core CPU; 5 epochs run in ~30-60 seconds, allowing tight feedback loops for profiling, kernel benchmarking, and ablation studies.
4. **Established Reference Accuracy**: Published baselines consistently achieve ~91.5-92.0% top-1 accuracy on CIFAR-10. Meeting >85% ensures numerical sanity and contract adherence.
5. **Amdahl's Law Profiling**: The mixture of conv2d (compute-bound, MKL/oneDNN), batch_norm / relu (memory-bandwidth bound), and backprop allows granular identification of acceleration opportunities and theoretical speedup bounds.

### Why Not Simpler Workloads?
- **MNIST / Simple MLP**: Dominated by memory bandwidth and tiny matrix-vector products. It fails to trigger vectorization across spatial convolutions, hides cache thrashing issues, and does not represent computer vision / tensor workloads.
- **ResNet-50 on ImageNet**: Prohibitively slow on CPU for rapid iterative compiler/native kernel prototyping without large multi-socket servers. ResNet-20 on CIFAR-10 retains the identical structural challenges at accessible scale.

---

## 2. Secondary Diagnostic Workloads

To isolate specific hardware bottlenecks and kernel scaling limits, two synthetic diagnostic microbenchmarks are maintained:

### 1. Compute-Bound Benchmark: 4-Layer MLP
- **Structure**: Linear(1024, 2048) -> ReLU -> Linear(2048, 2048) -> ReLU -> Linear(2048, 1024) -> ReLU -> Linear(1024, 10)
- **Batch Size**: 1024 (synthetic tensors, float32)
- **Characteristic**: Dominated by large dense matrix multiplications (GEMM). Tests raw BLAS/MKL/OpenMP thread scaling and AVX2 FMA throughput without spatial convolution or channel layout overhead.

### 2. Memory-Bound Operator Benchmark: Embedding + Elementwise Matmul Chain
- **Structure**: Embedding bag / lookup table followed by serialized elementwise tensor additions, scale multiplications, and activations.
- **Characteristic**: High memory traffic, low arithmetic intensity (<1 FLOP/byte). Stresses memory bus bandwidth (DDR4-3200 dual-channel), cache misses, and operator fusion opportunities.
