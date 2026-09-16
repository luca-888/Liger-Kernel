"""Isolated B/C benchmark for Qwen instance Q/K RMSNorm patching.

Run each (model, group, level, shape) in a new process; see the accompanying
benchmark/data/qwen_qk_norm/README.md for Modal commands and interpretation.
Only configuration JSON is loaded. All parameters and inputs are seeded random.
"""

import argparse
import csv
import hashlib
import inspect
import json
import statistics
import subprocess
import sys
import traceback

from importlib.metadata import version
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3", "qwen3_5"), required=True)
    parser.add_argument("--group", choices=("B", "C"))
    parser.add_argument("--level", choices=("norm", "attention", "model"))
    parser.add_argument("--batch-size", type=int, choices=(1, 4))
    parser.add_argument("--seq-len", type=int, choices=(128, 2048, 8192))
    parser.add_argument("--overwrite", action="store_true", help="Accepted for make run-benchmarks compatibility.")
    parser.add_argument("--matrix", action="store_true", help="Run every group/level/shape in separate subprocesses.")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate construction, bindings and layouts on meta device."
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-process matrix timeout in seconds.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    if len(sys.argv) == 1 or sys.argv[1:] == ["--overwrite"]:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("At least five independent measurement rounds are required.")
    if not args.matrix and any(getattr(args, key) is None for key in ("group", "level", "batch_size", "seq_len")):
        parser.error("Single cases require --group, --level, --batch-size and --seq-len.")
    return args


def binding(module):
    fn = module.forward.__func__
    return f"{fn.__module__}.{fn.__qualname__}"


def implementation_name(function):
    # Transformers' hybrid functions capture their selected implementation in
    # a wrapper closure (optional FLA/causal-conv1d or the native torch fallback).
    while True:
        selected = inspect.getclosurevars(function).nonlocals.get("implementation")
        if selected is not None:
            return f"{selected.__module__}.{selected.__qualname__}"
        if not hasattr(function, "__wrapped__"):
            return f"{function.__module__}.{function.__qualname__}"
        function = function.__wrapped__


def summarize(samples):
    return {
        "round_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.stdev(samples),
    }


def measure(prepare, execute, rounds, repetitions, warmup=3):
    """Fresh graphs/gradients per iteration; compile and setup excluded.

    Unlike retain_graph timing, this is safe for RMSNorm's in-place backward.
    Preparation resets gradients and makes a fresh upstream gradient *before*
    recording events. Forward/full modes include all norm layout conversions.
    """
    import torch

    for _ in range(warmup):
        execute(prepare())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(rounds):
        times = []
        for _ in range(repetitions):
            payload = prepare()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = execute(payload)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            del result, payload
        samples.append(statistics.median(times))
    return {
        **summarize(samples),
        "repetitions_per_round": repetitions,
        "warmup_iterations": warmup,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def measure_operation(forward, inputs, parameters, rounds, repetitions):
    import torch

    # Upstream gradients are fixed across groups/modes, cloned outside timing
    # because Qwen3 Liger backward may overwrite them in place.
    prototype = forward()
    gradients = tuple(torch.randn_like(output) for output in prototype)
    del prototype

    def prepare(mode):
        for tensor in (*inputs, *parameters):
            tensor.grad = None
        grads = tuple(gradient.clone() for gradient in gradients)
        outputs = forward() if mode == "backward" else None
        return outputs, grads

    def execute(payload, mode):
        outputs, grads = payload
        if outputs is None:
            outputs = forward()
        if mode != "forward":
            torch.autograd.backward(outputs, grads)
        return outputs

    return {
        mode: measure(
            lambda mode=mode: prepare(mode),
            lambda payload, mode=mode: execute(payload, mode),
            rounds,
            repetitions,
        )
        for mode in ("forward", "backward", "full")
    }


def run(args, report):
    import torch
    import transformers

    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "meta" if args.validate_only else "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config_path = Path(__file__).resolve().parents[1] / "data/qwen_qk_norm/configs" / f"{args.model}.json"
    source = json.loads(config_path.read_text())
    raw_config = source["config"].get("text_config", source["config"])
    if args.model == "qwen3":
        config_cls = transformers.Qwen3Config
        model_cls = transformers.Qwen3ForCausalLM
        patch = apply_liger_kernel_to_qwen3
    else:
        config_cls = transformers.Qwen3_5TextConfig
        model_cls = transformers.Qwen3_5ForCausalLM
        patch = apply_liger_kernel_to_qwen3_5
    config = config_cls.from_dict(raw_config)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    report.update(
        environment={
            "gpu": None if args.validate_only else torch.cuda.get_device_name(),
            "gpu_memory_bytes": None if args.validate_only else torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": version("triton"),
            "transformers": transformers.__version__,
        },
        model_id=source["model_id"],
        config_source=source["source"],
        config_revision=source["revision"],
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        model_config=config.to_dict(),
        settings={
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "gradient_checkpointing": False,
            "torch_compile": False,
            "tf32": False,
            "use_cache": False,
            "seed": 42,
            "rms_norm": True,
            "swiglu": True,
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "optimizer": "AdamW(lr=1e-5, fused=True)" if args.level == "model" else None,
        },
    )
    # The full, unmodified text configuration is constructed even for isolated
    # norm/attention measurements. There is no model-depth reduction.
    with torch.device(device):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = model_cls(config).train()
        finally:
            torch.set_default_dtype(previous_dtype)
    if args.model == "qwen3_5":
        from transformers.models.qwen3_5 import modeling_qwen3_5

        report["linear_attention_kernels"] = {
            name: implementation_name(getattr(modeling_qwen3_5, name))
            for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")
        }
    qk_norms = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(("self_attn.q_norm", "self_attn.k_norm"))
    ]
    assert qk_norms, "The benchmark must encounter Full Attention Q/K norms."
    native_forwards = {name: module.forward for name, module in qk_norms}
    assert all("transformers.models." in binding(module) for _, module in qk_norms)
    report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    report["qk_norm_count"] = len(qk_norms)
    patch(
        model=model,
        rms_norm=True,
        swiglu=True,
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )
    if args.group == "B":
        for name, module in qk_norms:
            module.forward = native_forwards[name]
    for _, module in qk_norms:
        assert (module.forward.__func__ is LigerRMSNorm.forward) == (args.group == "C")
    report["bindings"] = {name: binding(module) for name, module in qk_norms}
    report["norm_settings"] = {
        "offset": qk_norms[0][1].offset,
        "casting_mode": qk_norms[0][1].casting_mode,
        "in_place": qk_norms[0][1].in_place,
        "note": "B restores native forward; Liger-only attributes are unused in B.",
    }
    if args.validate_only:
        query_width = config.head_dim * (2 if args.model == "qwen3_5" else 1)
        query = torch.empty(args.batch_size, args.seq_len, config.num_attention_heads, query_width, device="meta")
        if args.model == "qwen3_5":
            query = query.chunk(2, dim=-1)[0]
            assert not query.is_contiguous()
        report["validated_query_layout"] = {"shape": list(query.shape), "stride": list(query.stride())}
        report["validation_scope"] = (
            "Meta-device configuration, model construction, forward bindings and query layout only; no CUDA execution."
        )
        return
    # Exact initial-parameter identity, computed before warmup/timing. No model
    # weights are downloaded; these bytes came from seeded random initialization.
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().view(torch.uint8).cpu().numpy().tobytes())
    report["initial_weights_sha256"] = digest.hexdigest()
    if args.level == "model":
        ids = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
        report["input_checksum"] = ids.sum().item()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

        def prepare():
            optimizer.zero_grad(set_to_none=True)

        def step(_):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            loss.backward()
            optimizer.step()
            return loss

        result = measure(prepare, step, args.rounds, repetitions=3)
        result["tokens_per_second"] = args.batch_size * args.seq_len * 1000 / result["median_ms"]
        report["measurements"] = {"training_step": result}
        report["scope"] = "Full text causal LM; Qwen3.5 vision encoder is not instantiated."
        return

    base = model.model
    attention = next(layer.self_attn for layer in base.layers if hasattr(layer, "self_attn"))
    hidden = torch.randn(
        args.batch_size, args.seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    report["input_checksum"] = hidden.detach().double().sum().item()
    if args.level == "attention":
        positions = torch.arange(args.seq_len, device="cuda").unsqueeze(0).expand(args.batch_size, -1)
        rotary = base.rotary_emb(hidden, positions)
        report["measurements"] = {
            "attention": measure_operation(
                lambda: (attention(hidden, position_embeddings=rotary, attention_mask=None)[0],),
                [hidden],
                list(attention.parameters()),
                args.rounds,
                repetitions=10,
            )
        }
        return

    # Materialize projections once, preserving exactly HF's pre-norm view/chunk
    # strides. Norm timing includes any contiguous copies performed by Liger.
    with torch.no_grad():
        if args.model == "qwen3_5":
            query, _ = (
                attention.q_proj(hidden).view(args.batch_size, args.seq_len, -1, 2 * config.head_dim).chunk(2, -1)
            )
        else:
            query = attention.q_proj(hidden).view(args.batch_size, args.seq_len, -1, config.head_dim)
        key = attention.k_proj(hidden).view(args.batch_size, args.seq_len, -1, config.head_dim)
    query = query.detach().requires_grad_()
    key = key.detach().requires_grad_()
    report["layouts"] = {
        name: {"shape": list(tensor.shape), "stride": list(tensor.stride()), "contiguous": tensor.is_contiguous()}
        for name, tensor in (("q", query), ("k", key))
    }
    if args.model == "qwen3_5":
        assert not query.is_contiguous(), "Query/gate layout must retain its native gaps."
    report["measurements"] = {}
    for name, forward, inputs, norms in (
        ("q", lambda: (attention.q_norm(query),), [query], [attention.q_norm]),
        ("k", lambda: (attention.k_norm(key),), [key], [attention.k_norm]),
        (
            "qk",
            lambda: (attention.q_norm(query), attention.k_norm(key)),
            [query, key],
            [attention.q_norm, attention.k_norm],
        ),
    ):
        report["measurements"][name] = measure_operation(
            forward,
            inputs,
            [parameter for norm in norms for parameter in norm.parameters()],
            args.rounds,
            repetitions=20,
        )


def write_comparison(directory, model):
    rows = []
    for path in sorted(directory.glob(f"{model}_*_B.json")):
        baseline = json.loads(path.read_text())
        candidate = json.loads(path.with_name(path.name.replace("_B.json", "_C.json")).read_text())
        if baseline["status"] != "ok" or candidate["status"] != "ok":
            continue
        for key in (
            "initial_weights_sha256",
            "input_checksum",
            "config_sha256",
            "model_config",
            "environment",
            "settings",
        ):
            assert baseline[key] == candidate[key], f"B/C mismatch for {key}: {path.name}"
        assert baseline.get("linear_attention_kernels") == candidate.get("linear_attention_kernels")
        for component, measurements in baseline["measurements"].items():
            modes = {"full": measurements} if component == "training_step" else measurements
            for mode, before in modes.items():
                after = candidate["measurements"][component]
                if component != "training_step":
                    after = after[mode]
                rows.append(
                    {
                        "model": model,
                        "level": baseline["level"],
                        "batch_size": baseline["batch_size"],
                        "seq_len": baseline["seq_len"],
                        "component": component,
                        "mode": mode,
                        "B_median_ms": before["median_ms"],
                        "C_median_ms": after["median_ms"],
                        "B_over_C": before["median_ms"] / after["median_ms"],
                        "B_min_ms": before["min_ms"],
                        "B_max_ms": before["max_ms"],
                        "C_min_ms": after["min_ms"],
                        "C_max_ms": after["max_ms"],
                        "B_peak_allocated_bytes": before["peak_allocated_bytes"],
                        "C_peak_allocated_bytes": after["peak_allocated_bytes"],
                        "B_tokens_per_second": before.get("tokens_per_second", ""),
                        "C_tokens_per_second": after.get("tokens_per_second", ""),
                    }
                )
    if rows:
        with (directory / f"{model}_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def run_matrix(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for level in ("norm", "attention", "model"):
        for shape_index, (batch, seq) in enumerate((b, s) for b in (1, 4) for s in (128, 2048, 8192)):
            for group in ("B", "C") if shape_index % 2 == 0 else ("C", "B"):
                name = f"{args.model}_{level}_b{batch}_s{seq}_{group}"
                path = args.output / f"{name}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--model",
                    args.model,
                    "--group",
                    group,
                    "--level",
                    level,
                    "--batch-size",
                    str(batch),
                    "--seq-len",
                    str(seq),
                    "--rounds",
                    str(args.rounds),
                    "--output",
                    str(path),
                ]
                if args.validate_only:
                    command.append("--validate-only")
                print(f"Running {name}", flush=True)
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    if path.exists():
                        data = json.loads(path.read_text())
                    else:
                        data = {"status": "process_error", "returncode": result.returncode}
                    data["stdout"] = result.stdout
                    data["stderr"] = result.stderr
                except subprocess.TimeoutExpired:
                    data = {"status": "timeout", "timeout_seconds": args.timeout}
                data.update(model=args.model, group=group, level=level, batch_size=batch, seq_len=seq)
                path.write_text(json.dumps(data, indent=2, default=str) + "\n")
                print(f"Finished {name}: {data['status']}", flush=True)
    write_comparison(args.output, args.model)


def main():
    args = parse_args()
    if args.matrix:
        run_matrix(args)
        return
    report = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        run(args, report)
        report["status"] = "validated" if args.validate_only else "ok"
    except Exception as error:
        import torch

        report["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    if report["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
