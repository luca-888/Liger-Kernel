# Qwen instance Q/K norm benchmark

Measured on Modal NVIDIA H100 80GB HBM3 with BF16, PyTorch 2.9.1 / CUDA 12.8, Triton 3.5.1 and Transformers 5.15.1. The change only adds instance Q/K norm patching; all other B/C settings are held fixed. Results include small attention cases within noise or slightly slower, as well as larger improvements. The PR remains Draft because three existing BF16 MoE convergence tests fail on both the original main implementation and this patch; no tolerances were relaxed.

## Reproduce

The runner reuses `dev/modal/tests.py`'s Debian/Python 3.12 base image and H100 selection, with PyTorch 2.9.1, CUDA supplied by that PyTorch wheel, Triton 3.5.1 and Transformers 5.15.1 pinned in `dev/modal/qwen_qk_norm.py`. It returns the actual GPU, driver, CUDA and package versions, source hashes, logs and raw results. Run from the repository root with an authenticated Modal account:

```bash
python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python benchmark/scripts/benchmark_qwen_qk_norm.py --model qwen3 --matrix --output /tmp/qwen-results' \
  --output benchmark/data/qwen_qk_norm/h100-qwen3

python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python benchmark/scripts/benchmark_qwen_qk_norm.py --model qwen3_5 --matrix --output /tmp/qwen-results' \
  --output benchmark/data/qwen_qk_norm/h100-qwen3_5
```

Each primary model matrix runs 36 independent subprocesses: B/C × norm/attention/model × batch 1/4 × sequence 128/2048/8192. B/C order alternates between shapes on the same GPU. Each process has a 600-second timeout; the Modal runner caps the whole command at 6900 seconds and returns any completed case files if that cap is reached. OOM, timeout and other errors are saved per case and do not become timing samples. A single case can be reproduced with:

```bash
python benchmark/scripts/benchmark_qwen_qk_norm.py \
  --model qwen3_5 --group C --level norm --batch-size 4 --seq-len 8192 \
  --rounds 5 --output /tmp/qwen-results/qwen3_5_norm_b4_s8192_C.json
```

Add `--validate-only` to a single case (or matrix) and use Modal `--gpu none` to check the full model on the meta device without GPU allocation. Those files have `status=validated`, contain no measurements, and are excluded from B/C summaries. Every new Modal run needs a new local `--output` directory.

## Comparison

- **B:** Construct native HF model first, apply the existing instance patch with RMSNorm, SwiGLU and fused linear cross entropy enabled, then restore only the saved native `self_attn.q_norm` / `k_norm` forward bindings.
- **C:** Same construction and patch settings, including this change's Q/K norm instance patch.

Both groups assert actual Q/K forward bindings. Parameters and inputs are seeded identically; each successful result contains an exact SHA-256 of initial parameter bytes, a configuration SHA-256, input checksum and all settings. The CSV comparison asserts equality of these before computing B/C ratios. B's unused Liger norm attributes remain attached after native forward restoration; the native methods do not read them.

BF16, SDPA, training mode, no gradient checkpointing, no `torch.compile`, no TF32, no KV cache, dropout zero, RoPE patch off and cross-entropy patch off are fixed. Model training uses fused AdamW (`lr=1e-5`) with BF16 parameters and PyTorch's corresponding optimizer state, without an FP32 master-weight copy. The timed interval includes forward loss, backward and optimizer update; `zero_grad(set_to_none=True)` and preloaded input preparation are outside the CUDA events in both groups. Warmup and subsequent optimizer steps use the same seeded initial state and fixed token inputs in both groups. The full Qwen3.5 text model's linear attention kernel bindings are recorded; the default runner does not install FLA or causal-conv1d and therefore uses HF's Torch fallback.

## Configurations and layouts

Only official configuration JSON is saved; no pretrained weights are downloaded. Both configurations are pinned to immutable Hugging Face revisions and source URLs inside `configs/*.json`.

| Configuration | Depth | Hidden | Head dimension | Query / KV heads | Vocabulary | Attention |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) | 28 | 1024 | 128 | 16 / 8 | 151936 | 28 full |
| [Qwen3.5-0.8B text](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json) | 24 | 1024 | 256 | 8 / 2 | 248320 | 6 full + 18 linear |

The model benchmark constructs the complete specified text causal LM, with its original depth, dimensions, vocabulary and tied embeddings. Qwen3.5's vision encoder is excluded, so results must be described as full **text-model** training. No reduced-depth model is used.

The norm benchmark obtains Q/K inputs from the actual attention projections. Qwen3's inputs are contiguous `[B,S,H,D]`; Qwen3.5 Q comes from the query/gate interleaved projection's chunk and retains a stride of `2*D` between heads. K remains contiguous. Each result records shapes and strides. Any contiguous copies inside Liger remain in the measured forward/full path. Projection cost is excluded from the isolated norm measurement and included in the complete attention measurement.

Qwen3 uses offset 0 / llama casting. Qwen3.5 uses offset 1 / gemma casting / `in_place=False`. These two configurations cover the two norm semantics and the gated-query layout. Supplementary norm and attention measurements use the official Qwen3-30B-A3B configuration (hidden 2048, head dimension 128, Q/KV heads 32/4) and Qwen3-Next-80B-A3B configuration (hidden 2048, head dimension 256, Q/KV heads 16/2). These run at batch/sequence 1/2048 and 4/8192. Qwen3.5-35B-A3B has the same Q/K geometry and gated-query strides as Next; its configuration and B/C binding checks are included without duplicating the same norm timing matrix. Its complete attention is not separately timed. All three additional configurations are pinned in `configs/`.

For these large variants, the original full model is constructed on meta before patching, then only the actual Full Attention module is materialized with identical random weights in B/C. This preserves real attention dimensions while making no claim about full MoE training. Full-model BF16 parameters plus gradients alone require 113.741 GiB (Qwen3 MoE), 296.810 GiB (Next), and 129.121 GiB (3.5 MoE), exceeding the allocated H100's 79.179 GiB before optimizer state or activations. These are capacity exclusions, not measured OOMs. No reduced model is substituted. MoE routing is not measured; random routing would not represent pretrained expert utilization.

## Measurements and artifacts

For Q, K and Q+K, the script measures forward, backward and forward+backward separately; complete attention has the same three modes. The complete model records a training step, tokens/s and peak allocated bytes. CUDA events exclude preparation, three warmup iterations and compilation. Five rounds are collected, each containing 20 norm iterations, 10 attention iterations or 3 model steps. Each round reports its median; the final median, minimum, maximum and standard deviation describe variation across rounds. These are repeated measurement rounds within a fresh process per B/C case, not five separate GPU allocations.

Backward gets a fresh forward graph and upstream gradient each iteration: retaining graphs/reusing gradient buffers is unsafe for Qwen3's in-place RMSNorm backward. Setup and gradient cloning occur before the CUDA events. Forward measures training forward with autograd enabled. Peak allocated memory is absolute process allocation. The complete primary text models remain resident at all levels; supplementary large variants materialize only one attention module. Use the primary model rows for whole-training memory comparisons.

The Modal output directories contain one JSON per case with all round timings and settings, `*_comparison.csv` with B/C medians, ranges, speedup ratios, peak allocation and training throughput, plus environment and source manifests. A ratio above 1 means C was faster. Inspect the variation before attributing a small difference to this change. Missing, failed or unexecuted cases have no ratio; an OOM must be reported as OOM. Readable `*_comparison.csv` files and the tables below contain all successful B/C pairs. `validation-logs.tar.gz` contains the raw JSON rounds, logs, source manifests and resolved environments for every run; errors or capacity exclusions remain explicit JSON records.

## Validation recorded for this change

Actual H100 environment: Python 3.12.10, PyTorch 2.9.1, CUDA runtime 12.8, Triton 3.5.1, Transformers 5.15.1, driver 580.95.05; total GPU memory 85,017,624,576 bytes. `validation.json` records exact statuses. Raw pytest logs/JUnit, source hashes, packages, benchmark JSON and checkstyle output are in `validation-logs.tar.gz`.

| Model / scope | H100 instance checks | BF16 FLCE / logits | FP32 FLCE / logits | Existing multimodal BF16 / FP32 |
| --- | ---: | --- | --- | --- |
| Qwen3 LM / base | 2 passed | Pass / Pass | Pass / Pass | N/A |
| Qwen3 MoE LM / base | 2 passed | Fail / Fail | Pass / Pass | N/A |
| Qwen3 Next LM / base | 2 passed | Pass / Pass | Skip / Skip | N/A |
| Qwen3.5 LM / text base / conditional generation | 3 passed | Pass / Pass | Skip / Skip | Pass / Pass |
| Qwen3.5 MoE LM / text base / conditional generation / multimodal base | 4 passed | Fail / No existing case | Skip / No existing case | Pass / Pass |
| Existing Qwen3 VL / VL MoE instance and RoPE hooks | 8 passed | Not selected | Not selected | Not selected |

- H100 RMSNorm numerical tests: **64 passed**, 8 multi-GPU DTensor cases deselected.
- H100 instance suite: **21 passed, 0 skipped**; preserves weights, epsilon, flags and untouched Linear Attention norms.
- Text convergence: **10 passed / 3 failed / 5 existing skips**. Multimodal convergence: **4 passed**.
- Baseline `95b01e9` replay: all three failing BF16 MoE rows also fail. Specific values and failing assertions vary between runs; this is not a claim of identical numerical failures. The cause remains unresolved and the PR stays Draft.
- Before-fix regression proof: the original `95b01e94027cc70ffec7233e89d3e9fa31fb72f7` implementation fails all 13 new Q/K forward assertions on Modal CPU. Its class-level convergence tests cannot replace this pre-created-instance coverage.
- Ten CPU/meta benchmark construction/binding/layout checks passed (B/C for all five variants); these are not GPU execution tests.
- `make checkstyle` passed. No new full-attention gradient, multimodal training or checkpointing test matrix was added.

The five FP32 hybrid skips are existing test marks. The two with-logits files contain no Qwen3.5 MoE case. The complete repository correctness/convergence suites and multi-GPU DTensor tests were not run. No tolerance was changed.

The following commands reproduce CPU red/green binding checks and the GPU validation; use a fresh local output directory each time:

```bash
python -m modal run dev/modal/qwen_qk_norm.py --gpu none \
  --baseline-ref 95b01e94027cc70ffec7233e89d3e9fa31fb72f7 \
  --command 'python -m pytest test/transformers/test_monkey_patch.py -q -k "qwen3 and not qwen3_vl" --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-red
# Expected: exit 1 with 13 assertion failures on the original implementation.

python -m modal run dev/modal/qwen_qk_norm.py --gpu none \
  --command 'python -m pytest test/transformers/test_monkey_patch.py -q -k qwen3 --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-instances-cpu

python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python -m pytest test/transformers/test_rms_norm.py -q -k "not dtensor" --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-rms-h100

python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python -m pytest test/transformers/test_monkey_patch.py -q -k qwen3 --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-instances-h100

python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python -m pytest test/convergence/bf16/test_mini_models.py test/convergence/bf16/test_mini_models_with_logits.py test/convergence/fp32/test_mini_models.py test/convergence/fp32/test_mini_models_with_logits.py -q -k "qwen3 and not qwen3_vl" --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-convergence-h100

python -m modal run dev/modal/qwen_qk_norm.py \
  --command 'python -m pytest test/convergence/bf16/test_mini_models_multimodal.py test/convergence/fp32/test_mini_models_multimodal.py -q -k qwen3_5 --junitxml=/tmp/qwen-results/junit.xml' \
  --output /tmp/qwen-existing-multimodal-h100

make checkstyle
```

## H100 results

All timings are CUDA-event milliseconds; brackets show the minimum and maximum of the five round medians. B/C > 1 favors C. The CSVs include individual Q and K, combined Q+K, separate forward/backward and forward+backward; the tables here focus on combined work and complete training.

The primary matrix completed **70 measured cases and 2 OOMs**. All 36 Qwen3 cases succeeded; Qwen3.5 completed 34 cases, with **both B and C out of memory at full-model batch 4 / sequence 8192**. No time, throughput or ratio is assigned to that pair. All 16 supplementary norm/attention cases succeeded. Three full MoE configurations are separately recorded as capacity exclusions.

**Forward-only regressions are real in some small cases.** Qwen3 K forward at 4×128 is 0.058592 ms [0.058448, 0.062992] → 0.068048 ms [0.067392, 0.069152], B/C 0.861x (16.1% higher C latency). Qwen3 complete attention forward at that shape is 0.493696 ms [0.490832, 0.513232] → 0.561360 ms [0.553728, 0.590640], B/C 0.879x (13.7% higher C latency). These ranges do not overlap and should not be dismissed as noise. Qwen3.5 also has forward-only regressions (for example complete attention at 1×128, B/C 0.901x). The supplementary CSVs retain their forward-only regressions as well.

Qwen3 full training improves in this setup, while Qwen3.5 training gains are much smaller: short cases and batch 4 / sequence 2048 overlap across rounds, whereas batch 1 / sequence 2048 and 8192 show about 4% improvement. Qwen3.5's HF Torch Linear Attention fallback dominates much of the step. Do not generalize norm speedups to model training or to a production FLA setup.

### Qwen3-0.6B

#### Q + K norm, forward + backward

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.4604 [0.4552, 0.4692] | 0.3876 [0.3805, 0.4100] | 1.188x |
| 1 | 2048 | 0.4896 [0.4869, 0.4948] | 0.3895 [0.3856, 0.4091] | 1.257x |
| 1 | 8192 | 1.6061 [1.6027, 1.6126] | 0.4354 [0.4202, 0.4512] | 3.689x |
| 4 | 128 | 0.4329 [0.4229, 0.4960] | 0.3859 [0.3744, 0.4025] | 1.122x |
| 4 | 2048 | 1.6042 [1.6020, 1.6058] | 0.4406 [0.4281, 0.4478] | 3.641x |
| 4 | 8192 | 5.5350 [5.5345, 5.5377] | 0.8365 [0.7637, 0.8530] | 6.617x |

#### Complete attention, forward + backward

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1.8575 [1.8306, 1.8692] | 1.7846 [1.7049, 1.8881] | 1.041x |
| 1 | 2048 | 1.8628 [1.8293, 1.8893] | 1.8632 [1.8016, 1.8869] | 1.000x |
| 1 | 8192 | 6.7493 [6.7261, 6.7539] | 5.5323 [5.5229, 5.5425] | 1.220x |
| 4 | 128 | 1.7979 [1.6951, 1.8558] | 1.8196 [1.8113, 1.8242] | 0.988x |
| 4 | 2048 | 4.3319 [4.3258, 4.3356] | 3.1083 [3.0920, 3.1221] | 1.394x |
| 4 | 8192 | 24.1383 [24.0657, 24.2008] | 18.9685 [18.9586, 18.9947] | 1.273x |

#### Full text-model training step

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 89.8307 [86.8161, 90.6086] | 85.4614 [81.6939, 85.9939] | 1.051x |
| 1 | 2048 | 87.6655 [86.4113, 90.8722] | 82.2174 [81.2611, 85.0644] | 1.066x |
| 1 | 8192 | 234.6182 [233.7221, 235.9698] | 194.3267 [193.9314, 194.9091] | 1.207x |
| 4 | 128 | 87.6365 [85.9571, 89.0954] | 80.6684 [79.1690, 82.7989] | 1.086x |
| 4 | 2048 | 167.0588 [166.1978, 167.6446] | 126.1442 [125.6589, 126.8863] | 1.324x |
| 4 | 8192 | 838.9922 [837.4209, 840.4976] | 700.6868 [698.3565, 701.5800] | 1.197x |

| Batch | Sequence | B tokens/s | C tokens/s | B peak GiB | C peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1425 | 1498 | 4.794 | 4.794 |
| 1 | 2048 | 23362 | 24910 | 6.913 | 6.257 |
| 1 | 8192 | 34916 | 42156 | 16.602 | 13.977 |
| 4 | 128 | 5842 | 6347 | 4.832 | 4.828 |
| 4 | 2048 | 49037 | 64942 | 16.599 | 13.974 |
| 4 | 8192 | 39056 | 46766 | 55.347 | 44.847 |

### Qwen3.5-0.8B text

#### Q + K norm, forward + backward

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 0.4752 [0.4663, 0.5118] | 0.4494 [0.4382, 0.4725] | 1.058x |
| 1 | 2048 | 0.5457 [0.5433, 0.5527] | 0.4394 [0.4314, 0.4720] | 1.242x |
| 1 | 8192 | 1.3554 [1.3544, 1.3567] | 0.4951 [0.4856, 0.5211] | 2.738x |
| 4 | 128 | 0.4628 [0.4561, 0.5158] | 0.3991 [0.3944, 0.4033] | 1.160x |
| 4 | 2048 | 1.3523 [1.3517, 1.3534] | 0.4941 [0.4880, 0.5012] | 2.737x |
| 4 | 8192 | 4.8327 [4.8319, 4.8338] | 0.8963 [0.8957, 0.9307] | 5.392x |

#### Complete attention, forward + backward

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 2.0805 [2.0090, 2.1583] | 2.0340 [2.0155, 2.2874] | 1.023x |
| 1 | 2048 | 2.0285 [2.0151, 2.0342] | 2.0441 [1.9558, 2.0700] | 0.992x |
| 1 | 8192 | 6.9064 [6.8891, 6.9375] | 6.0171 [6.0067, 6.0356] | 1.148x |
| 4 | 128 | 2.0514 [1.9882, 2.2261] | 2.1940 [1.9734, 2.2535] | 0.935x |
| 4 | 2048 | 4.2529 [4.2504, 4.2573] | 3.3490 [3.3423, 3.3677] | 1.270x |
| 4 | 8192 | 25.3058 [25.2976, 25.3209] | 21.1935 [21.1724, 21.2160] | 1.194x |

#### Full text-model training step

| Batch | Sequence | B median ms [min, max] | C median ms [min, max] | B/C |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 360.0613 [355.0375, 385.6257] | 361.5039 [359.4940, 388.7575] | 0.996x |
| 1 | 2048 | 1128.4083 [1114.1154, 1135.5679] | 1081.3909 [1069.5734, 1091.4086] | 1.043x |
| 1 | 8192 | 3521.7375 [3471.6785, 3604.6450] | 3393.0374 [3338.7551, 3422.1724] | 1.038x |
| 4 | 128 | 360.6518 [359.0733, 376.8817] | 358.1805 [354.9912, 381.4587] | 1.007x |
| 4 | 2048 | 1206.1482 [1187.0504, 1216.1846] | 1201.7935 [1198.9758, 1203.3347] | 1.004x |

| Batch | Sequence | B tokens/s | C tokens/s | B peak GiB | C peak GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 355 | 354 | 6.198 | 6.198 |
| 1 | 2048 | 1815 | 1894 | 16.625 | 16.443 |
| 1 | 8192 | 2326 | 2414 | 51.941 | 51.234 |
| 4 | 128 | 1420 | 1429 | 7.723 | 7.677 |
| 4 | 2048 | 6792 | 6816 | 51.976 | 51.237 |


Qwen3.5 full-model 4×8192: **B OOM / C OOM**, omitted from successful timing and throughput tables. The original model configuration was not reduced.

### Supplementary attention geometries

These are full-size attention modules, not full MoE training. All rows below are forward + backward; separate forward/backward and individual Q/K values are in the CSVs.

| Model geometry | Batch × sequence | Component | B median ms [min, max] | C median ms [min, max] | B/C |
| --- | --- | --- | ---: | ---: | ---: |
| qwen3_moe | 1 × 2048 | attention | 2.4376 [2.3289, 2.5004] | 2.1949 [2.0900, 2.3050] | 1.111x |
| qwen3_moe | 4 × 8192 | attention | 43.6091 [43.5532, 43.8374] | 36.5248 [36.4741, 36.6477] | 1.194x |
| qwen3_moe | 1 × 2048 | qk | 0.7597 [0.7509, 0.7763] | 0.4985 [0.4831, 0.5421] | 1.524x |
| qwen3_moe | 4 × 8192 | qk | 7.9603 [7.9585, 7.9620] | 1.0633 [1.0617, 1.0696] | 7.486x |
| qwen3_next | 1 × 2048 | attention | 2.7508 [2.7482, 2.7912] | 2.7426 [2.7148, 2.7464] | 1.003x |
| qwen3_next | 4 × 8192 | attention | 51.4647 [51.2570, 51.4952] | 44.3646 [44.2976, 44.4190] | 1.160x |
| qwen3_next | 1 × 2048 | qk | 0.8220 [0.8132, 0.8300] | 0.5123 [0.5007, 0.5425] | 1.604x |
| qwen3_next | 4 × 8192 | qk | 8.3528 [8.3523, 8.3548] | 1.6031 [1.6030, 1.6034] | 5.211x |

Supplementary single cases can be reproduced with the same Modal runner, replacing `--model` with `qwen3_moe` or `qwen3_next`, and choosing `--level norm` or `attention`, B/C, and 1×2048 or 4×8192. `--matrix --levels norm attention` runs all six shapes if a broader follow-up is desired. `--model qwen3_5_moe --group C --level model --batch-size 1 --seq-len 128` records the full-model capacity exclusion. The exact commands for every recorded run are in the archived `environment.json` files.

Extract raw data with `tar -xzf benchmark/data/qwen_qk_norm/validation-logs.tar.gz -C /tmp`. Files under `benchmark-qwen3-h100/`, `benchmark-qwen3_5-h100/` and `benchmark-supplement-h100/` contain the per-case rounds, bindings, parameter hashes, layouts, errors and memory results. Model configurations are in `configs/`; no pretrained weight file is included.
