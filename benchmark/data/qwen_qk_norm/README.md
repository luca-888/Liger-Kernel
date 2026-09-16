# Qwen instance Q/K norm benchmark

H100 measurements are **not available** for this change: Modal rejected GPU allocation because the workspace requires a payment method. No speedup, attention timing, training throughput, or H100 memory result is claimed. The scripts and pinned configurations below are ready for execution after that external block is resolved. CPU/meta validation only checks construction, layouts and method bindings; it does not validate Triton execution or performance.

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

Each model matrix runs 36 independent subprocesses: B/C × norm/attention/model × batch 1/4 × sequence 128/2048/8192. B/C order alternates between shapes on the same GPU. Each process has a 600-second timeout; the Modal runner caps the whole command at 6900 seconds and returns any completed case files if that cap is reached. OOM, timeout and other errors are saved per case and do not become timing samples. A single case can be reproduced with:

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

BF16, SDPA, training mode, no gradient checkpointing, no `torch.compile`, no TF32, no KV cache, dropout zero, RoPE patch off and cross-entropy patch off are fixed. Model training uses fused AdamW (`lr=1e-5`) and includes forward loss, backward and optimizer update. Warmup and subsequent optimizer steps use the same seeded initial state and fixed token inputs in both groups. The full Qwen3.5 text model's linear attention kernel bindings are recorded; the default runner does not install FLA or causal-conv1d and therefore uses HF's Torch fallback.

## Configurations and layouts

Only official configuration JSON is saved; no pretrained weights are downloaded. Both configurations are pinned to immutable Hugging Face revisions and source URLs inside `configs/*.json`.

| Configuration | Depth | Hidden | Head dimension | Query / KV heads | Vocabulary | Attention |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) | 28 | 1024 | 128 | 16 / 8 | 151936 | 28 full |
| [Qwen3.5-0.8B text](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json) | 24 | 1024 | 256 | 8 / 2 | 248320 | 6 full + 18 linear |

The model benchmark constructs the complete specified text causal LM, with its original depth, dimensions, vocabulary and tied embeddings. Qwen3.5's vision encoder is excluded, so results must be described as full **text-model** training. No reduced-depth model is used.

The norm benchmark obtains Q/K inputs from the actual attention projections. Qwen3's inputs are contiguous `[B,S,H,D]`; Qwen3.5 Q comes from the query/gate interleaved projection's chunk and retains a stride of `2*D` between heads. K remains contiguous. Each result records shapes and strides. Any contiguous copies inside Liger remain in the measured forward/full path. Projection cost is excluded from the isolated norm measurement and included in the complete attention measurement.

Qwen3 uses offset 0 / llama casting. Qwen3.5 uses offset 1 / gemma casting / `in_place=False`. These two configurations cover the two norm semantics and the gated-query layout. Other model variants and MoE configurations have not been benchmarked; conclusions cannot be generalized to their head counts, depth or routing. Future randomly initialized MoE measurements must disclose that random routing is not representative of pretrained expert utilization.

## Measurements and artifacts

For Q, K and Q+K, the script measures forward, backward and forward+backward separately; complete attention has the same three modes. The complete model records a training step, tokens/s and peak allocated bytes. CUDA events exclude preparation, three warmup iterations and compilation. Five rounds are collected, each containing 20 norm iterations, 10 attention iterations or 3 model steps. Each round reports its median; the final median, minimum, maximum and standard deviation describe variation across rounds. These are repeated measurement rounds within a fresh process per B/C case, not five separate GPU allocations.

Backward gets a fresh forward graph and upstream gradient each iteration: retaining graphs/reusing gradient buffers is unsafe for Qwen3's in-place RMSNorm backward. Setup and gradient cloning occur before the CUDA events. Forward measures training forward with autograd enabled. Peak allocated memory is absolute process allocation (the complete text model remains resident at all levels); use the model rows for whole-training memory comparisons.

The Modal output directories contain one JSON per case with all round timings and settings, `*_comparison.csv` with B/C medians, ranges, speedup ratios, peak allocation and training throughput, plus environment and source manifests. A ratio above 1 means C was faster. Inspect the variation before attributing a small difference to this change. Missing, failed or unexecuted cases have no ratio; an OOM must be reported as OOM. At present, no H100 timing JSON or comparison CSV exists because allocation was blocked.

## Validation recorded for this change

Actual available validation environment: Modal **CPU**, Python 3.12.10, PyTorch 2.9.1 (CUDA runtime 12.8 wheel), Triton 3.5.1, Transformers 5.15.1. H100 was requested but never allocated; there is no measured GPU name, driver or memory value. `validation.json` lists exact test names and statuses. `validation-logs.tar.gz` contains raw pytest logs/JUnit, source hashes, resolved packages, four meta-device benchmark results, checkstyle output and the Modal allocation failures.

| Model / scope | Instance checks on Modal CPU | Before fix |
| --- | ---: | ---: |
| Qwen3 LM / base | 2 passed | 2 failed at new Q/K forward assertion |
| Qwen3 MoE LM / base | 2 passed | 2 failed |
| Qwen3 Next LM / base | 2 passed | 2 failed |
| Qwen3.5 LM / text base / conditional generation | 3 passed | 3 failed |
| Qwen3.5 MoE LM / text base / conditional generation / multimodal base | 4 passed | 4 failed |
| Existing Qwen3 VL / VL MoE instance and RoPE hooks | 8 passed | Unchanged; not part of red run |
| Qwen3 / Qwen3.5 B/C meta-device benchmark construction | 4 passed | Not applicable |
| `make checkstyle` | Passed | Not applicable |

The baseline regression used commit `95b01e94027cc70ffec7233e89d3e9fa31fb72f7`'s original `monkey_patch.py` and the new assertions; all 13 failures reached the missing Q/K forward binding assertion. The fixed run had 21 passed, zero failed and zero skipped. This validates instance binding and preservation of weights/epsilon/settings, not numerical Triton execution.

**Not executed:** H100 RMSNorm numerical tests, GPU instance checks, all affected convergence tests and all three performance levels. The four existing text convergence files have 18 relevant cases; five FP32 hybrid cases are already marked skipped upstream. Those are static observations, not results from an executed convergence run. The two existing with-logits files have no Qwen3.5 MoE case. No tolerances were changed and no new convergence/checkpointing/multimodal matrix was added.

The following commands reproduce CPU red/green binding checks and the required blocked GPU validation; use a fresh local output directory each time:

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
