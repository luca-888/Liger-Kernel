# Triton 与 Liger Kernel 实战学习计划

这份计划面向已经了解一些 Triton、但还不能独立开发和优化生产级 kernel 的学习者。目标不是通读整个 Liger Kernel 仓库，也不是机械复现官方教程，而是通过几个纵向算子建立完整能力：

```text
数学分析
  → Triton 实现
  → forward/backward
  → 正确性与边界测试
  → benchmark 与性能诊断
  → PyTorch/Liger 集成
  → 真实 Issue/PR
  → CuTe DSL 架构特化
```

建议周期为 8 周，每周投入 8～12 小时。时间不足时可以延长周期，但不要跳过阶段验收。

## 最终目标

完成计划后，应当能够：

- 从 PyTorch reference 和数学公式独立设计 Triton kernel。
- 独立实现并验证 forward/backward。
- 正确处理非 2 次幂、极值、低精度、超大维度和 pointer overflow。
- 判断 kernel 是 launch-bound、memory-bound、reduction-bound 还是 compute-bound。
- 使用 benchmark 和 profiler 解释性能变化，而不是盲目调整参数。
- 理解 `BLOCK_SIZE`、`num_warps`、`num_stages` 和 grid 的取舍。
- 为 kernel 编写 reference、数学性质、边界、编译兼容和性能测试。
- 独立复现、定位并修复一个范围明确的 Liger Issue。
- 解释 Triton 实现的性能边界，以及 CuTe DSL 能解决的具体调度问题。

## 学习原则

时间建议按以下比例分配：

```text
30% 阅读源码、测试和 Git 历史
50% 独立实现、实验、benchmark 和 profile
20% Issue、代码审查和 PR 实战
```

每次实验都遵循同一个循环：

1. 写下对正确性或性能的预测。
2. 设计能够证伪预测的最小实验。
3. 记录输入形状、dtype、GPU、软件版本和结果。
4. 用数据解释结果。
5. 无法解释时查看 IR、PTX 或 profiler，而不是继续随机调参。

不要从完整仓库开始横向阅读。每个阶段只纵向研究一个算子，把数学、kernel、封装、测试、benchmark 和历史问题串起来。

## 开始前：环境与学习记录

### 环境准备

```bash
pip install -e ".[dev]"
python -m liger_kernel.env_report
make checkstyle
```

运行单个测试文件：

```bash
python -m pytest test/transformers/test_softmax.py -v
```

运行单个 benchmark：

```bash
python benchmark/scripts/benchmark_softmax.py
```

### 建立实验记录

每个实验至少记录：

```markdown
## 实验名称

- 日期：
- Git commit：
- GPU：
- PyTorch/Triton/CUDA：
- 输入 shape/dtype：
- 修改内容：
- 修改前预测：
- 正确性结果：
- 性能结果：
- profiler 证据：
- 当前解释：
- 尚未解释的问题：
```

### 基线检查

- [ ] 能运行 Softmax 单元测试。
- [ ] 能运行 Softmax benchmark。
- [ ] 知道当前 GPU 的 compute capability、SM 数量和显存带宽。
- [ ] 能定位 Triton cache 和生成的编译产物。
- [ ] 能使用 `git log`、`git blame` 和 GitHub PR 查找代码历史。

## 第 1～2 周：Softmax 与 reduction 基础

### 学习目标

- 建立 Triton blocked-program 执行模型。
- 掌握 grid、program、tile、mask 和 stride。
- 掌握稳定 Softmax 和 online Softmax。
- 独立推导和实现 backward。
- 理解 single-block、循环式 multi-block 和多 program 协作的区别。

### 阅读顺序

1. `src/liger_kernel/ops/softmax.py`
2. `test/transformers/test_softmax.py`
3. `src/liger_kernel/ops/utils.py` 中的 `calculate_settings` 和 `ensure_contiguous`
4. `src/liger_kernel/transformers/softmax.py`
5. `benchmark/scripts/benchmark_softmax.py`
6. GitHub PR #689 和 PR #1252
7. `src/liger_kernel/ops/backends/_ascend/ops/softmax.py`，只用于比较分块策略

先独立判断代码行为，再阅读 PR 结论。

### 数学任务

- [ ] 推导稳定 Softmax forward。
- [ ] 推导 Softmax backward：

  \[
  dx_i = y_i\left(dy_i - \sum_j dy_jy_j\right)
  \]

- [ ] 推导 online max/sum 的状态合并公式。
- [ ] 解释旧分母在最大值更新后为什么需要 rescale。
- [ ] 计算 single-block 和循环式 multi-block forward/backward 的理论内存读写量。

### 独立实现任务

不要先复制 Liger kernel。单独实现一个最小练习版本：

- [ ] single-block forward。
- [ ] single-block backward。
- [ ] 支持非 2 次幂列数。
- [ ] online multi-block forward。
- [ ] multi-block backward。
- [ ] 给 dispatch 增加可验证的路径标识或测试钩子。

### 测试矩阵

```text
n_rows: 1, 8, 128, 4096
n_cols: 1, 127, 1024, 4096, 32000, 65536, 70000, 128256
dtype: fp32, bf16
```

正确性测试：

- [ ] 输出与 `torch.softmax` 对齐。
- [ ] backward 与 PyTorch autograd 对齐。
- [ ] 每行输出非负。
- [ ] 每行和约等于 1。
- [ ] `softmax(x + c) ≈ softmax(x)`。
- [ ] 常量输入得到均匀分布。
- [ ] 极端 logits 不产生意外 NaN/Inf。
- [ ] 每行输入梯度和约等于 0。
- [ ] 全 1 upstream gradient 对应输入梯度约等于 0。
- [ ] 尾块 mask 覆盖非 2 次幂和超宽行。

### 性能实验

对每个代表形状记录：

- Triton 延迟。
- PyTorch 延迟。
- 估算读写字节数。
- 有效显存带宽。
- `BLOCK_SIZE` 和 `num_warps`。
- 寄存器、occupancy 和 spill 情况。

至少完成以下对比：

- [ ] 不同 `BLOCK_SIZE`。
- [ ] 不同 `num_warps`。
- [ ] `n_rows=1` 与大量行。
- [ ] 2 次幂与非 2 次幂列数。
- [ ] single-block 与循环式 multi-block。
- [ ] Triton 与 `torch.softmax`。

### 阶段产出

- 一份 Softmax 数学与 kernel 设计说明。
- 一个独立可运行的 forward/backward 练习实现。
- 一张 shape × dtype 的正确性表。
- 一张 shape × provider 的性能表。
- 对 Liger 当前 Softmax dispatch 的代码审查笔记。

### 阶段验收

不看代码，能够回答：

- [ ] 为什么当前默认实现的 multi-block 分支不可达？
- [ ] PR #1252 解决的是 correctness、performance，还是两者兼有？
- [ ] one-program-per-row 与 multi-program-per-row 有什么区别？
- [ ] 为什么 forward 和 backward 都需要跨 block 的全行状态？
- [ ] 哪些 shape 可能由 launch overhead 主导？
- [ ] 哪些 shape 可能需要 fallback 或完全不同的并行方案？

## 第 3～4 周：RMSNorm、dtype 与参数梯度

### 学习目标

- 掌握 norm reduction。
- 处理 fp32 accumulation 和模型特定 casting 语义。
- 实现输入梯度与参数梯度。
- 理解 int32 pointer overflow、inplace 和 `torch.compile` 标量类型问题。

### 阅读顺序

1. `src/liger_kernel/ops/rms_norm.py`
2. `test/transformers/test_rms_norm.py`
3. `src/liger_kernel/transformers/rms_norm.py`
4. `benchmark/scripts/benchmark_rms_norm.py`
5. GitHub Issue #803 / PR #804
6. fp64 scalar 修复 PR #1350、#1358

### 独立实现任务

- [ ] 简化版 RMSNorm forward。
- [ ] `dX`。
- [ ] `dW`。
- [ ] `elementwise_affine=True/False`。
- [ ] fp32 与 bf16 输入。
- [ ] fp32 accumulation。
- [ ] 大行数下使用 int64 pointer arithmetic。

第一版只实现一种清晰的 casting 语义；验证正确后再对照 Liger 的 Llama、Gemma 和 none casting mode。

### 必做分析

- [ ] 推导 RMSNorm backward。
- [ ] 标出 kernel 中所有计算 dtype 和存储 dtype。
- [ ] 标出所有可能超过 int32 的 pointer offset 乘法。
- [ ] 比较 `dX` 和 `dW` 的并行与 reduction 需求。
- [ ] 解释 Python float 参数为什么可能在 `torch.compile` 下提升为 fp64。
- [ ] 解释 universal int64 cast 为什么可能增加性能成本。

### 测试与性能

- [ ] forward、`dX`、`dW` 对齐 PyTorch reference。
- [ ] irregular hidden size。
- [ ] 超长行数。
- [ ] inplace 与非 inplace。
- [ ] affine 与非 affine。
- [ ] fp32/bf16。
- [ ] eager 与 `torch.compile`，环境支持时。
- [ ] benchmark 不同 hidden size、batch 和 sequence length。

### 阶段产出

- 一个独立 RMSNorm forward/backward 实现。
- 一份 dtype 流转图。
- 一份 pointer overflow 审计清单。
- 一份 bandwidth/occupancy 分析。

### 阶段验收

- [ ] 能解释何时应该以 fp32 累积、何时转换回输入 dtype。
- [ ] 能解释为什么 `tl.program_id` 可能导致超大 tensor OOB。
- [ ] 能说明 `dW` 为什么比 `dX` 更难并行。
- [ ] 能定位寄存器、occupancy 或内存带宽瓶颈。

## 第 5 周：第一次真实贡献

### 目标

完成一个范围明确、能够独立验证的 Liger 贡献。第一次贡献不追求新算子或最大性能提升，重点是完整工程闭环。

### 推荐方向

按优先级选择一个：

1. 为 Softmax 增加确定性数学性质测试。
2. 为 PR #1252 补 BF16、非 2 次幂和不同架构 benchmark。
3. 增加能够证明 single/multi-block dispatch 的 regression test。
4. 为 Issue #1272 建立 Softmax 的 bandwidth-efficiency benchmark 样例。
5. 针对已掌握算子做 dtype/index 静态审计，并先提交问题报告或小范围修复。

### Issue 选择规则

认领前必须确认：

- [ ] 当前仍可复现。
- [ ] 没有活跃 PR 或其他贡献者正在处理。
- [ ] 自己拥有所需硬件。
- [ ] 范围集中在一个算子和少量文件。
- [ ] 有明确 PyTorch/reference oracle。
- [ ] 能先写出失败测试。
- [ ] 性能修复能够提供可靠 benchmark。

第一次贡献应避开：

- 没有对应硬件的 B300/SM103 性能问题。
- 新后端或全新调度框架。
- 大型 MoE/FLCE 重构。
- 已有开放修复 PR 的 Issue。
- 根因主要位于 PyTorch/Inductor 上游的问题。

### PR 完成定义

- [ ] 修复前新增测试稳定失败。
- [ ] 修复后新增测试稳定通过。
- [ ] 相关算子完整测试通过。
- [ ] `make checkstyle` 通过。
- [ ] 性能没有意外退化。
- [ ] PR 描述包含问题、根因、修复、验证和硬件环境。
- [ ] 对没有运行的测试明确说明原因。

## 第 6～7 周：Cross Entropy 与复杂融合

### 学习目标

- 综合 online reduction、dtype、mask 和复杂 backward。
- 理解 loss kernel 的生产语义。
- 为学习 FLCE 和 CuTe DSL GEMM pipeline 建立基础。

### 阅读顺序

1. `src/liger_kernel/ops/cross_entropy.py`
2. `test/transformers/test_cross_entropy.py`
3. `src/liger_kernel/transformers/cross_entropy.py`
4. `benchmark/scripts/benchmark_cross_entropy.py`
5. 最后阅读 `fused_linear_cross_entropy.py`

不要从 FLCE 开始。先把不带 GEMM 的 Cross Entropy 读透。

### 数学与实现任务

- [ ] 推导 logsumexp 和 Cross Entropy gradient。
- [ ] 实现 online logsumexp。
- [ ] 支持 `ignore_index`。
- [ ] 支持 `none/sum/mean` reduction。
- [ ] 理解 label smoothing。
- [ ] 理解 z-loss。
- [ ] 理解 class weight 对 loss 和 gradient 的影响。
- [ ] 解释为什么某些实现会原地保存 gradient。

### 必测边界

- [ ] 所有 token 都有效。
- [ ] 部分 `ignore_index`。
- [ ] 所有 token 都被 ignore。
- [ ] 非 2 次幂 vocab。
- [ ] 超大 vocab。
- [ ] 极端 logits。
- [ ] target 位于首列和末列。
- [ ] eager 与 compile 结果一致，环境支持时。

### FLCE 阅读问题

完成普通 CE 后，阅读 FLCE 并回答：

- [ ] 为什么要 chunk logits？
- [ ] 哪些中间 tensor 被避免 materialize？
- [ ] forward、`dX`、`dW` 分别包含什么 GEMM？
- [ ] 为什么 `dW` shape 可能需要不同调度？
- [ ] chunk size 如何影响显存、launch overhead 和 GEMM 效率？
- [ ] 哪些部分是 Triton reduction，哪些部分是 GEMM 性能问题？

### 阶段产出

- Cross Entropy 数学与 kernel 数据流说明。
- 一组 ignore/mask/极值的边界测试。
- FLCE 显存与计算数据流图。
- 对一个生产 shape 的 profile 报告。

## 第 8 周：CuTe DSL 对照学习

### 进入条件

只有满足以下条件才进入本阶段：

- [ ] 能独立实现和分析 Softmax 或 RMSNorm。
- [ ] 能解释 Cross Entropy 的 online reduction。
- [ ] 会用 profiler 识别 Triton 版本瓶颈。
- [ ] 能明确说出某个瓶颈为什么需要更显式的 TMA、MMA layout 或 warp specialization。
- [ ] 拥有能够运行目标 CuTe kernel 的 NVIDIA 硬件和软件环境。

如果不满足，继续深化前面阶段，不要为了追近期 PR 强行进入 CuTe DSL。

### 对照对象

只选择已经熟悉的算子：

- Triton RMSNorm vs CuTe DSL RMSNorm，或
- Triton Cross Entropy vs CuTe DSL Cross Entropy。

相关路径：

- `src/liger_kernel/ops/cutedsl/ops/rms_norm.py`
- `src/liger_kernel/ops/cutedsl/ops/rms_norm_fastpath.py`
- `src/liger_kernel/ops/cutedsl/ops/cross_entropy.py`
- `test/cutedsl/`

### 对照问题

- [ ] Triton program tile 对应哪个 CuTe CTA tile？
- [ ] `tl.arange`/pointer arithmetic 对应什么 CuTe layout？
- [ ] `tl.load` 对应普通 tiled copy 还是 TMA？
- [ ] `tl.dot` 对应什么 MMA atom/tiled MMA？
- [ ] shared-memory layout 如何避免 bank conflict？
- [ ] producer 和 consumer warp 如何分工？
- [ ] pipeline 有多少 stage？
- [ ] barrier phase 为什么这样更新？
- [ ] 性能收益来自算法、减少访存、降低 launch overhead，还是硬件 schedule？
- [ ] Hopper 和 Blackwell 为什么可能需要不同实现？

### 阶段产出

- 一份 Triton/CuTe 同算子概念映射表。
- 相同 shape/dtype 下的正确性与性能对比。
- 一份 CuTe 实现的 pipeline 图。
- 一份结论：哪些收益 Triton也能实现，哪些需要更显式的硬件控制。

第一轮 CuTe 学习不要求提交性能 PR。目标是能够解释现有实现，而不是修改几个 tile 参数后碰巧获得提升。

## 硬件分支建议

### RTX 3090/4090 或 A100

- 优先完成 Triton、正确性、dtype 和通用性能分析。
- 不认领只在 Hopper/Blackwell 生效的 CuTe fast path。
- 可以阅读 CuTe，但不要提交无法在目标硬件验证的性能结论。

### H100

- 完成基础路线后学习 TMA、WGMMA、persistent kernel 和 warp specialization。
- 优先比较 Hopper CuTe 与 Triton实现。

### B200/B300

- 在基础路线之后重点学习 TCGen05、TMEM、multi-CTA 和 Blackwell schedule。
- 任何优化都同时报告 Triton、CuTe、PyTorch/官方库基线。

### 没有稳定 NVIDIA GPU

- 优先做数学推导、代码审查、测试、API 和 compile 语义。
- 不认领需要真实性能结论的优化 Issue。

## 每周复盘模板

```markdown
# Week N Review

## 本周完成

- [ ]

## 我现在能独立解释的内容

-

## 有数据支持的结论

-

## 尚未解释的现象

-

## 读过的源码/PR

-

## 下周最小目标

-
```

## 最终毕业检查

### 正确性

- [ ] 能从公式独立写出 Softmax forward/backward。
- [ ] 能实现 RMSNorm forward、`dX` 和 `dW`。
- [ ] 能正确处理 mask、非 2 次幂、极值和低精度。
- [ ] 能发现并解释 int32 pointer overflow。
- [ ] 能区分计算 dtype、累积 dtype 和存储 dtype。

### 性能

- [ ] 能判断 kernel 的主要瓶颈类别。
- [ ] 能估算理论内存流量和有效带宽。
- [ ] 能解释 grid、tile、warp 和 stage 的取舍。
- [ ] 能使用 profiler 验证优化假设。
- [ ] 能发现 occupancy、寄存器 spill、launch overhead 或低效访存。
- [ ] 能说明优化对不同 shape 和 GPU 架构的影响。

### 工程

- [ ] 能完成 autograd Function 和公开 API 封装。
- [ ] 能编写 reference、性质、边界和性能测试。
- [ ] 能验证 eager、autocast 和 compile 的关键路径。
- [ ] 能复现并修复一个真实 Issue。
- [ ] 能提交包含充分证据、测试和 benchmark 的 PR。

### CuTe DSL

- [ ] 能把 Triton tile 与 CuTe layout/partition 对应起来。
- [ ] 能解释 copy、MMA、pipeline 和 warp specialization。
- [ ] 能明确指出使用 CuTe 的具体性能理由，而不是因为它是近期热点。
- [ ] 能判断某个算子应保留 Triton、增加 CuTe backend，还是直接使用框架原生实现。

## 计划完成的判断标准

计划完成不以“看完多少文件”或“写过多少 kernel”为标准，而以是否能够回答下面四个问题为标准：

1. 它为什么正确？
2. 它为什么快或为什么慢？
3. 哪些输入、dtype、编译模式或硬件会使它失效？
4. 如果换一个 shape 或 GPU，应该如何重新设计和验证？

能够用公式、代码、测试和 profiler 数据回答这四个问题，才算真正建立了独立开发生产级 Triton kernel 的能力。
