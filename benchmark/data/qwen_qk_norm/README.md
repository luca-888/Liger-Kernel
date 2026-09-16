# Qwen instance Q/K norm benchmark

This report measures the additional Q/K RMSNorm patch on already-created HF models. The completed results cover **96 one-GPU norm/attention cases** across four official configurations and **12 eight-GPU Qwen3-8B training cases**. The other three full-model evaluations are blocked by an external workspace shutdown. The PR remains Draft because those results are unavailable and one baseline-reproduced convergence failure remains unresolved. No earlier measurements are reused.

## Comparison and model scope

**B** creates a native HF instance, applies Liger RMSNorm/SwiGLU/fused linear cross entropy, and restores only the saved native `self_attn.q_norm` and `k_norm` forward bindings. **C** uses the same instance patch with the additional Q/K norm bindings. Both groups assert their actual forward methods. B's unused Liger attributes do not affect its native forward.

Each group runs in a fresh process. Configurations, initial parameter bytes, inputs, dtype and other settings must match before a B/C ratio is emitted. Only official configuration JSON is downloaded; parameters are randomly initialized. [Pinned configurations](configs/) include immutable revisions and source URLs.

| Official model | Layers (full / linear) | Hidden | Head dim | Q / KV heads | Experts / active | Text parameters |
|---|---:|---:|---:|---:|---:|---:|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) | 36 / 0 | 4096 | 128 | 32 / 8 | — | 8,190,735,360 |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/config.json) | 8 / 24 | 4096 | 256 | 16 / 4 | — | 8,953,803,264 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/config.json) | 48 / 0 | 2048 | 128 | 32 / 4 | 128 / 8 | 30,532,122,624 |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json) | 10 / 30 | 2048 | 256 | 16 / 2 | 256 / 8 | 34,660,610,688 |

The original HF text causal LM is constructed on meta before patching. Norm/attention cases materialize one actual Full Attention module; the other model parameters remain meta. These measurements do not execute MoE routing or establish full-model throughput. The training configuration retains all text layers, experts and vocabulary; Qwen3.5's vision encoder is excluded.

Qwen3 Next is covered by instance and convergence validation; its full-model performance is outside this four-model matrix.

Norm inputs come from real Q/K projections with shape `[batch, sequence, heads, head_dim]`. Qwen3 Q/K and Qwen3.5 K are contiguous. Qwen3.5 Q retains its native query/gate chunk layout, including a `2 * head_dim` head stride. Projection cost is excluded only from isolated norm timing. Liger's contiguous-copy cost stays inside forward and forward+backward timing. Qwen3 uses offset 0 / llama casting; Qwen3.5 uses offset 1 / gemma casting / `in_place=False`.

## Environments and measurement

The completed norm/attention runs used one NVIDIA H100 80GB HBM3 (85,017,624,576 bytes), Python 3.12.10, PyTorch 2.9.1, CUDA 12.8, Triton 3.5.1, Transformers 5.15.1, flash-linear-attention/fla-core 0.5.2, causal-conv1d 1.7.0 and driver 580.95.05. They did not install TileLang or a CUDA compiler and do not invoke Linear Attention.

The **training and final convergence environment** preserves those Torch/CUDA/Triton/Transformers pins and adds TileLang 0.1.14 with supplemental CUDA 13.0 compiler packages: nvcc/nvvm/crt 13.0.88, CCCL 13.0.85 and runtime 13.0.96. Torch still reports CUDA 12.8. This addresses FLA 0.5.2's Hopper/Triton 3.5.1 gated backward incompatibility; [FLA's TileLang backend](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/backends/tilelang/__init__.py) requires usable nvcc. `FLA_TILELANG=1` selects that supported path. Hybrid full-model results require actual FLA/causal-conv1d bindings and successful TileLang backward dispatch; HF's Torch Linear Attention fallback is not an accepted training result. The dependency addition does not invalidate the completed Full Attention/norm cases.

All levels use BF16 compute, SDPA, training mode, zero dropout, no `torch.compile`, no TF32 and no KV cache. Liger RMSNorm, SwiGLU and fused linear cross entropy are enabled; Liger RoPE and standalone cross entropy are disabled. B/C differ only in Q/K norm forward bindings.

- **Norm/attention:** all six batch/sequence combinations, `batch={1,4}` × `sequence={128,2048,8192}`; Q, K and Q+K each have forward, backward and forward+backward measurements. Complete attention has the same three modes. CUDA events exclude preparation and three warmup iterations. Five rounds contain 20 norm or 10 attention iterations each.
- **Full-model training configuration:** eight H100s, FSDP2 decoder-layer/root sharding, non-reentrant activation checkpointing, FP32 master parameters, gradients/reduction and both AdamW moments; BF16 unsharded compute parameters. AdamW uses `lr=1e-5, foreach=False, fused=False`. The primary workloads, 4×2048 and 1×8192 per GPU, both contain 65,536 global tokens per step. Qwen3-8B already completed all six shapes; its four additional shapes are retained as supplementary results. The remaining three models are limited to the two primary shapes, four B/C cases each. Five rounds contain three steps each after three warmup steps. Synchronized wall time includes zero-grad, forward, backward, optimizer and communication; each step reports the slowest rank. Peak allocated memory is reported per rank and as its maximum, not total cluster memory.

Every backward iteration uses a fresh graph and fresh upstream gradients; Qwen3's in-place RMSNorm backward makes retained-graph/gradient reuse unsuitable. Parameters use HF initialization distributions and special constants with deterministic per-name/per-rank RNG, not HF's serial RNG order. Training checks actual FP32 gradient and Adam allocation after warmup. Initial local-shard hashes are combined in rank order; RoPE buffers and inputs are also checked.

## One-GPU results

The two primary workloads hold tokens per GPU at 8192 while varying batch versus context length. The remaining four shapes are retained in raw results. Below, each B/C cell is **median ms [minimum–maximum of five round medians]**; ratios are B÷C. These repeated rounds are within one fresh process per case, not five independent GPU allocations.

| Model | Batch × sequence | Q+K forward+backward B / C (ms) | B/C | Full Attention forward+backward B / C (ms) | B/C |
|---|---:|---|---:|---|---:|
| Qwen3-8B | 4 × 2048 | 2.4393 [2.4365–2.4396] / 0.5616 [0.5454–0.5748] | 4.343× | 8.8828 [8.8806–8.8900] / 6.7049 [6.6936–6.7379] | 1.325× |
| Qwen3-8B | 1 × 8192 | 2.4353 [2.4335–2.4356] / 0.5417 [0.5405–0.5427] | 4.496× | 13.5520 [13.5438–13.7456] / 11.2690 [11.2570–11.3044] | 1.203× |
| Qwen3.5-9B | 4 × 2048 | 2.5602 [2.5595–2.5627] / 0.7390 [0.6974–0.8668] | 3.465× | 10.7630 [10.7188–10.7777] / 8.8019 [8.6617–9.0901] | 1.223× |
| Qwen3.5-9B | 1 × 8192 | 2.5642 [2.5621–2.5661] / 0.7294 [0.7109–0.7836] | 3.515× | 16.0631 [15.9923–16.2252] / 14.1081 [14.0874–14.1913] | 1.139× |
| Qwen3-30B-A3B | 4 × 2048 | 2.2192 [2.2135–2.2277] / 0.6434 [0.6331–0.6815] | 3.449× | 7.0712 [7.0584–7.0830] / 5.2796 [5.2684–5.3102] | 1.339× |
| Qwen3-30B-A3B | 1 × 8192 | 2.2115 [2.2041–2.2177] / 0.6591 [0.6135–0.7068] | 3.355× | 11.5970 [11.5859–11.6052] / 9.8578 [9.8262–9.8811] | 1.176× |
| Qwen3.5-35B-A3B | 4 × 2048 | 2.2921 [2.2907–2.2981] / 0.7404 [0.6791–0.7576] | 3.096× | 8.2482 [8.2348–8.2742] / 6.3236 [6.3204–6.3561] | 1.304× |
| Qwen3.5-35B-A3B | 1 × 8192 | 2.2883 [2.2876–2.2901] / 0.6915 [0.6907–0.7110] | 3.309× | 13.4753 [13.4552–13.4863] / 11.5388 [11.5164–11.5790] | 1.168× |

There are regressions outside this primary table. Qwen3-30B-A3B attention **forward** at 4×128 takes B 0.662464 ms [0.645888–0.698320] versus C 0.831888 ms [0.770448–0.836416]: **0.796×**, or 25.6% higher C latency. Its K-norm forward at the same shape is 0.076816 → 0.089280 ms (**0.860×**). Qwen3.5-35B-A3B attention **forward+backward** at 1×128 is 3.056144 ms [2.985024–3.132624] → 3.214304 ms [3.187392–3.222672], **0.951×**. The complete CSVs retain these regressions. Overall, 39 of 288 mode/component comparisons have B/C below 1; some round ranges overlap. Range overlap is descriptive, not a significance test.

Norm/attention improvements must not be presented as full-model speedups. Random weights do not reproduce pretrained behavior; full MoE training remains unmeasured, and random routing would not reproduce pretrained expert utilization. Results apply to the stated eager BF16/SDPA workloads and preserve native copy costs.

## Full text-model training results

Qwen3-8B completed all six shapes: **12 successful cases / six B/C pairs**, independently audited for eight H100 80GB ranks, FP32 master/gradient/Adam storage, matching initial shards and inputs, and recomputed timing/throughput statistics. Across those shapes, B/C is 1.034–1.080×. The primary rows below use 65,536 global tokens/step; time is maximum-rank synchronized wall milliseconds, brackets are the five-round minimum–maximum, and peak memory is maximum-rank allocated GiB.

| Model | Per-GPU batch × sequence | B ms [min–max] | C ms [min–max] | B/C | Tokens/s B / C | Peak GiB B / C | Status |
|---|---:|---|---|---:|---:|---:|---|
| Qwen3-8B | 4 × 2048 | 1217.574 [1212.056–1217.999] | 1127.726 [1123.544–1128.963] | 1.080× | 53825.1 / 58113.4 | 21.698 / 21.698 | Complete |
| Qwen3-8B | 1 × 8192 | 1425.893 [1422.769–1427.826] | 1332.106 [1328.582–1334.046] | 1.070× | 45961.4 / 49197.3 | 21.698 / 21.698 | Complete |
| Qwen3.5-9B | 4 × 2048 | Unavailable | Unavailable | — | — | — | Blocked: interrupted |
| Qwen3.5-9B | 1 × 8192 | Unavailable | Unavailable | — | — | — | Blocked: interrupted |
| Qwen3-30B-A3B | 4 × 2048 | Unavailable | Unavailable | — | — | — | Blocked: not run |
| Qwen3-30B-A3B | 1 × 8192 | Unavailable | Unavailable | — | — | — | Blocked: not run |
| Qwen3.5-35B-A3B | 4 × 2048 | Unavailable | Unavailable | — | — | — | Blocked: not run |
| Qwen3.5-35B-A3B | 1 × 8192 | Unavailable | Unavailable | — | — | — | Blocked: not run |

The Qwen3 primary rows have identical B/C peak allocated memory. Norm speedups therefore do not imply a memory saving in this checkpointed, sharded training setup.

The remaining allocation was terminated when Modal disabled the execution workspace; the archived `train-remaining/run_error.json` records the error. Qwen3.5-9B was interrupted without a retained valid B/C raw pair, and the two MoE full-model runs were not reached. No partial console observation is used as a measurement. These are **external execution blocks, not OOM results**; their step time, throughput, memory and ratios are unavailable. Compiler/backend probes establish functionality only and are excluded from performance totals.

The remaining validation scope is **12 training cases**: B/C at 4×2048 and 1×8192 for Qwen3.5-9B and the two official MoE models. Completed correctness, norm/attention and Qwen3 training results do not need rerunning. Historical commands retain the original six-shape request; future training commands use `--primary`.

## Correctness validation

Fresh H100 80GB runs passed **21 instance tests and 64 RMSNorm numerical tests**; **`make checkstyle` passed**. Original main (`95b01e94027cc70ffec7233e89d3e9fa31fb72f7`) fails all **13 new Q/K forward-binding assertions**; the eight existing VL checks pass. The instance tests create native HF models before patching and check weights, epsilon, flags, disabled RMSNorm, nonempty Full Attention coverage and unchanged Linear Attention norms.

| Model / entry points | Instance tests | BF16 FLCE / logits | FP32 FLCE / logits | Existing multimodal BF16 / FP32 |
|---|---:|---|---|---|
| Qwen3 LM / base | 2 passed | Pass / Pass | Pass / Pass | — |
| Qwen3 MoE LM / base | 2 passed | Pass / **Fail** | Pass / Pass | — |
| Qwen3 Next LM / base | 2 passed | Pass / Pass | Skip / Skip | — |
| Qwen3.5 LM / text base / conditional generation | 3 passed | Pass / Pass | Skip / Skip | Pass / Pass |
| Qwen3.5 MoE LM / text base / conditional generation / multimodal base | 4 passed | Pass / No existing case | Skip / No existing case | Pass / Pass |
| Existing Qwen3 VL / VL MoE instance and RoPE checks | 8 passed | Not selected | Not selected | Not selected |

Text convergence totals **12 passed, 1 failed, 5 existing skips**; multimodal convergence totals **4 passed**. The BF16 Qwen3 MoE logits top-k log-probability assertion also fails on original main. Its FLCE case passes on both versions. These convergence tests patch classes before constructing models, so they do not exercise the modified instance branch; they cannot replace the instance regression tests. The numerical failure remains unresolved and the PR stays Draft. No tolerance was changed.

Eight multi-GPU DTensor RMSNorm cases were deselected; the complete repository suites were not run. An initial convergence attempt hit FLA's known Hopper/Triton guard; its logs are retained as a superseded environment diagnostic. The final convergence runs use the supported TileLang backend, whose actual forward/backward probe passed with finite gradients. All final correctness runs used H100 80GB; the initial incompatible-environment attempt used H100 NVL.

## Reproduce and inspect raw data

From the repository root with an authenticated Modal 1.5.5 installation, the following commands reproduce the measurements. Norm/attention supports all four model names (`qwen3`, `qwen3_5`, `qwen3_moe`, `qwen3_5_moe`); completing the blocked training scope needs only `qwen3_5`, `qwen3_moe` and `qwen3_5_moe`. The runner reuses the existing Debian/Python image and H100 allocation conventions; [its source](../../../dev/modal/qwen_qk_norm.py) pins dependencies and records the resolved environment.

```bash
# 24 cases: B/C × norm/attention × six shapes, separate process per case.
python -m modal run dev/modal/qwen_qk_norm.py --gpu 'H100!' \
  --command 'python benchmark/scripts/benchmark_qwen_qk_norm.py --model qwen3 --matrix --levels norm attention --rounds 5 --output /tmp/qwen-results' \
  --output /tmp/qwen-reproduction/kernels-qwen3

# 4 cases per remaining model: B/C × two primary shapes, fresh eight-rank groups.
# Repeat for qwen3_moe and qwen3_5_moe: 12 remaining cases in total.
python -m modal run dev/modal/qwen_qk_norm.py --gpu 'H100!:8' \
  --command 'python benchmark/scripts/benchmark_qwen_qk_norm.py --model qwen3_5 --matrix --primary --levels model --rounds 5 --output /tmp/qwen-results' \
  --output /tmp/qwen-reproduction/training-qwen3_5
```

Use a new local output directory for every invocation. The benchmark starts `torchrun --nproc_per_node=8` for model cases; no weights are downloaded. For one norm/attention case, replace `--matrix --levels ...` with `--group B --level norm --batch-size 4 --seq-len 2048`. `--validate-only` checks meta construction/bindings/layouts and emits no timings. A process/group has an 1800-second timeout; the remote command has a 6900-second cap. Failures remain JSON records and cannot become successful timing samples.

[Raw validation archive](validation-logs.tar.gz) extracts to `qwen-qk-norm-results/`. It contains the four `kernels-*` runs, the completed `train-qwen3/` run, the `train-remaining/` execution error, actual validation/probe diagnostics, JUnit, `validation-summary.json`, source/environment manifests, resolved requirements, commands, checkstyle output and the independent auditor. Completed case JSON retains every round, binding, hash and native stride. The archive contains no synthetic auditor fixtures.

The four readable comparison files are [Qwen3](qwen3_comparison.csv), [Qwen3.5](qwen3_5_comparison.csv), [Qwen3 MoE](qwen3_moe_comparison.csv) and [Qwen3.5 MoE](qwen3_5_moe_comparison.csv). Qwen3 includes all six complete-model training pairs; the other three contain norm/attention results only. In total, **108 successful cases / 54 B/C pairs** provide 294 component/mode comparisons. Missing training results have no CSV timing row. The correctness commands are in `command-plan.json`; `validation-summary.json` distinguishes final results from earlier environment diagnostics.

```bash
tar -xzf benchmark/data/qwen_qk_norm/validation-logs.tar.gz -C /tmp
python /tmp/qwen-qk-norm-results/audit-fresh-benchmarks.py \
  --root /tmp/qwen-qk-norm-results --repo "$PWD"
```
