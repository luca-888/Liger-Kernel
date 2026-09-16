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
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback

from importlib.metadata import version
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"), required=True)
    parser.add_argument("--group", choices=("B", "C"))
    parser.add_argument("--level", choices=("norm", "attention", "model"))
    parser.add_argument("--levels", nargs="+", choices=("norm", "attention", "model"), help="Limit matrix levels.")
    parser.add_argument("--batch-size", type=int, choices=(1, 4))
    parser.add_argument("--seq-len", type=int, choices=(128, 2048, 8192))
    parser.add_argument("--overwrite", action="store_true", help="Accepted for make run-benchmarks compatibility.")
    parser.add_argument("--primary", action="store_true", help="Only per-rank B4/S2048 and B1/S8192.")
    parser.add_argument(
        "--probe-only", action="store_true", help="One untimed forward/backward/optimizer validation step."
    )
    parser.add_argument("--matrix", action="store_true", help="Run every group/level/shape in separate subprocesses.")
    parser.add_argument(
        "--validate-only", action="store_true", help="Validate construction, bindings and layouts on meta device."
    )
    parser.add_argument("--timeout", type=int, default=1800, help="Per-process/group timeout in seconds.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    if len(sys.argv) == 1 or sys.argv[1:] == ["--overwrite"]:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error("At least five independent measurement rounds are required.")
    if args.probe_only and ((args.matrix and args.levels != ["model"]) or (not args.matrix and args.level != "model")):
        parser.error("--probe-only applies to full-model steps; use --level model or --matrix --levels model.")
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


def parameter_initializers(model, gemma):
    """Capture native HF initialization rules before monkey-patching classes."""
    import torch

    rules = {}
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{name}" if module_name else name
            padding_idx = module.padding_idx if isinstance(module, torch.nn.Embedding) else None
            if name == "A_log":
                kind = "log_uniform_0_16"
            elif name == "dt_bias":
                kind = "ones"
            elif "RMSNorm" in type(module).__name__ and name == "weight":
                kind = "zeros" if gemma and type(module).__name__.endswith("RMSNorm") else "ones"
            elif name == "bias":
                kind = "zeros"
            elif parameter.ndim >= 2:
                kind = "normal"
            else:
                raise ValueError(f"Unrecognized HF parameter initialization: {full_name}")
            rules[full_name] = (kind, padding_idx)
    return rules


def initialize_parameters(model, rules, std, rank=0, world_size=1, prefix=""):
    """Initialize only local storage, with HF distributions and deterministic seeds.

    This preserves HF's initialization semantics, not its serial RNG sequence.
    FSDP shards use independent per-name/per-rank generators; B/C hashes verify
    the actual resulting bytes rather than assuming equal seeds imply equality.
    """
    import torch

    from torch.distributed.tensor import DTensor

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            kind, padding_idx = rules[full_name]
            seed = int.from_bytes(hashlib.sha256(f"42:{full_name}:{rank}".encode()).digest()[:8], "little")
            generator = torch.Generator(device=local.device).manual_seed(seed)
            if kind == "normal":
                local.normal_(mean=0.0, std=std, generator=generator)
            elif kind == "log_uniform_0_16":
                local.uniform_(0.0, 16.0, generator=generator).log_()
            elif kind == "ones":
                local.fill_(1)
            else:
                local.zero_()
            if padding_idx is not None:
                shard_rows = (parameter.shape[0] + world_size - 1) // world_size
                index = padding_idx - rank * shard_rows
                if 0 <= index < local.shape[0]:
                    local[index].zero_()


def tensor_hash(named_tensors, rank=0):
    import torch

    from torch.distributed.tensor import DTensor

    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        metadata = (name, tuple(tensor.shape), str(tensor.dtype), str(getattr(tensor, "placements", None)), rank)
        digest.update(repr(metadata).encode())
        digest.update(local.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def distributed_training(args, report, model, rules, config):
    import torch
    import torch.distributed as dist

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8, "Full-model training requires exactly eight H100 workers."
    mesh = init_device_mesh("cuda", (world_size,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    # Keep embedding and LM head in the root group: FLCE accesses head.weight
    # directly and would bypass a separately sharded head's forward hook.
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    model.to_empty(device=torch.device("cuda", rank))
    initialize_parameters(model, rules, config.initializer_range, rank, world_size)
    # to_empty also empties nonpersistent buffers. Recreate both RoPE buffers.
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config, device=torch.device("cuda", rank))
    assert all(not buffer.is_meta for buffer in model.buffers())
    shard_hashes = [None] * world_size
    dist.all_gather_object(shard_hashes, tensor_hash(model.named_parameters(), rank))
    report["rank_initial_weights_sha256"] = shard_hashes
    report["initial_weights_sha256"] = hashlib.sha256("".join(shard_hashes).encode()).hexdigest()
    report["buffer_sha256"] = tensor_hash(model.named_buffers())
    buffer_hashes = [None] * world_size
    dist.all_gather_object(buffer_hashes, report["buffer_sha256"])
    assert len(set(buffer_hashes)) == 1, "RoPE buffers must match on every rank."
    report["weight_hash_scope"] = "All actual FP32 local shards, combined in rank order."
    report["initialization"] = (
        "HF distributions/special values; deterministic parameter-name/rank RNG, not HF serial RNG."
    )
    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(1234 + rank)
    ids = torch.randint(config.vocab_size, (args.batch_size, args.seq_len), device=f"cuda:{rank}", generator=generator)
    input_hashes = [None] * world_size
    dist.all_gather_object(input_hashes, tensor_hash([("input_ids", ids)], rank))
    report["rank_input_sha256"] = input_hashes
    report["input_checksum"] = hashlib.sha256("".join(input_hashes).encode()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False, fused=False)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        optimizer.step()
        return loss.detach()

    def check_loss(loss):
        valid = torch.isfinite(loss).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        assert valid.item(), "A rank produced a non-finite loss."

    delta = next(
        (module for module in model.modules() if hasattr(module, "A_log") and hasattr(module, "dt_bias")), None
    )
    observed_delta_dtypes = {}
    hook = None
    if delta is not None:

        def observe_delta(module, _):
            observed_delta_dtypes.update(A_log=str(module.A_log.dtype), dt_bias=str(module.dt_bias.dtype))

        hook = delta.register_forward_pre_hook(observe_delta)
    for _ in range(1 if args.probe_only else 3):
        loss = step()
        check_loss(loss)
    if hook is not None:
        hook.remove()
    report["observed_delta_compute_parameter_dtypes"] = observed_delta_dtypes
    if args.model in {"qwen3_5", "qwen3_5_moe"}:
        from fla.ops.backends import BackendRegistry

        required_dispatch = "common:chunk_bwd_dqkwg:tilelang"
        executed_dispatches = sorted(BackendRegistry._registries["common"]._logged)
        assert required_dispatch in executed_dispatches, "The supported TileLang GDN backward must actually execute."
        report["hybrid_backend_dispatch"] = {
            "required": required_dispatch,
            "executed": executed_dispatches,
            "FLA_TILELANG": os.environ.get("FLA_TILELANG"),
            "tilelang_version": version("tilelang"),
        }
    torch.cuda.synchronize()
    for parameter in model.parameters():
        assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        for key in ("exp_avg", "exp_avg_sq"):
            state = optimizer.state.get(parameter, {}).get(key)
            assert state is not None and state.dtype == torch.float32
    report["verified_precision"] = {
        "master": "float32",
        "gradient": "float32",
        "exp_avg": "float32",
        "exp_avg_sq": "float32",
    }
    local_state_bytes = sum(
        state.to_local().numel() * state.element_size()
        for states in optimizer.state.values()
        for key, state in states.items()
        if key in {"exp_avg", "exp_avg_sq"}
    )
    state_bytes = [None] * world_size
    dist.all_gather_object(state_bytes, local_state_bytes)
    report["optimizer_state_bytes_per_rank"] = state_bytes
    report["scope"] = "Complete text causal LM on eight GPUs; original depth/experts/vocabulary, no vision encoder."
    if args.probe_only:
        report["status"] = "probed"
        report["probe_loss_per_rank"] = [None] * world_size
        dist.all_gather_object(report["probe_loss_per_rank"], float(loss))
        return
    torch.cuda.reset_peak_memory_stats()
    wall_rounds, cuda_rounds = [], []
    for _ in range(args.rounds):
        walls, events = [], []
        for _ in range(3):
            dist.barrier()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start.record()
            loss = step()
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            # Reduction is outside the timed interval; include the slowest rank.
            times = torch.tensor([elapsed, start.elapsed_time(end)], device=f"cuda:{rank}")
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
            walls.append(times[0].item())
            events.append(times[1].item())
            check_loss(loss)
        wall_rounds.append(statistics.median(walls))
        cuda_rounds.append(statistics.median(events))
    peaks = [None] * world_size
    dist.all_gather_object(peaks, torch.cuda.max_memory_allocated())
    result = {
        **summarize(wall_rounds),
        "cuda_event_round_ms": cuda_rounds,
        "timing": "Synchronized wall time, maximum rank; includes zero_grad, forward, backward, optimizer and FSDP communication.",
        "warmup_iterations": 3,
        "repetitions_per_round": 3,
        "peak_allocated_bytes": max(peaks),
        "peak_allocated_bytes_per_rank": peaks,
    }
    result["tokens_per_second"] = world_size * args.batch_size * args.seq_len * 1000 / result["median_ms"]
    report["measurements"] = {"training_step": result}


def run(args, report):
    import torch

    distributed = args.level == "model" and not args.validate_only
    if distributed:
        from datetime import timedelta

        import torch.distributed as dist

        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        assert dist.get_world_size() == 8
        hardware = {
            "rank": rank,
            "name": torch.cuda.get_device_name(rank),
            "total_memory_bytes": torch.cuda.get_device_properties(rank).total_memory,
        }
        report["rank_hardware"] = [None] * 8
        dist.all_gather_object(report["rank_hardware"], hardware)
        assert all(
            gpu["name"] == "NVIDIA H100 80GB HBM3" and 78 * 2**30 <= gpu["total_memory_bytes"] <= 81 * 2**30
            for gpu in report["rank_hardware"]
        ), f"Hardware mismatch: training requires 8 H100 80GB HBM3 GPUs; got {report['rank_hardware']}"
        assert len({(gpu["name"], gpu["total_memory_bytes"]) for gpu in report["rank_hardware"]}) == 1
    # Select the rank's device before imports that may inspect CUDA (e.g. FLA).
    import transformers

    from liger_kernel.transformers import monkey_patch
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    gemma = args.model in {"qwen3_5", "qwen3_5_moe"}
    device = "meta"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config_path = Path(__file__).resolve().parents[1] / "data/qwen_qk_norm/configs" / f"{args.model}.json"
    source = json.loads(config_path.read_text())
    raw_config = source["config"].get("text_config", source["config"])
    classes = {
        "qwen3": ("Qwen3Config", "Qwen3ForCausalLM"),
        "qwen3_moe": ("Qwen3MoeConfig", "Qwen3MoeForCausalLM"),
        "qwen3_5": ("Qwen3_5TextConfig", "Qwen3_5ForCausalLM"),
        "qwen3_5_moe": ("Qwen3_5MoeTextConfig", "Qwen3_5MoeForCausalLM"),
    }
    config_cls, model_cls = (getattr(transformers, name) for name in classes[args.model])
    patch = getattr(monkey_patch, f"apply_liger_kernel_to_{args.model}")
    config = config_cls.from_dict(raw_config)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    report.update(
        environment={
            "gpu": None if args.validate_only else torch.cuda.get_device_name(),
            "gpu_memory_bytes": None
            if args.validate_only
            else torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
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
            "gradient_checkpointing": distributed,
            "checkpoint_use_reentrant": False if distributed else None,
            "world_size": 8 if distributed else 1,
            "parameter_master_dtype": "float32" if distributed else "bfloat16",
            "gradient_reduce_dtype": "float32" if distributed else None,
            "optimizer_state_dtype": "float32" if distributed else None,
            "A_log_dt_bias": "FP32 master, BF16 unsharded parameter, explicit HF float32 math"
            if gemma and distributed
            else None,
            "fsdp": "FSDP2 decoder+root, reshard_after_forward=True" if distributed else None,
            "torch_compile": False,
            "tf32": False,
            "use_cache": False,
            "seed": 42,
            "rms_norm": True,
            "swiglu": True,
            "rope": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
            "optimizer": "AdamW(lr=1e-5, foreach=False, fused=False)" if distributed else None,
        },
    )
    # Always create the native, full-config HF instance on meta before patching.
    with torch.device(device):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            model = model_cls(config).train()
        finally:
            torch.set_default_dtype(previous_dtype)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters()), (
        "Native meta parameters must be FP32."
    )
    rules = parameter_initializers(model, gemma)
    if gemma:
        import importlib

        modeling = importlib.import_module(f"transformers.models.{args.model}.modeling_{args.model}")
        report["linear_attention_kernels"] = {
            name: implementation_name(getattr(modeling, name))
            for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")
        }
        if not args.validate_only:
            assert report["linear_attention_kernels"]["torch_chunk_gated_delta_rule"].startswith("fla."), (
                "FLA must actually be bound."
            )
            assert report["linear_attention_kernels"]["causal_conv1d_fn"].startswith("causal_conv1d."), (
                "causal-conv1d must actually be bound."
            )
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
        query_width = config.head_dim * (2 if gemma else 1)
        query = torch.empty(args.batch_size, args.seq_len, config.num_attention_heads, query_width, device="meta")
        if gemma:
            query = query.chunk(2, dim=-1)[0]
            assert not query.is_contiguous()
        report["validated_query_layout"] = {"shape": list(query.shape), "stride": list(query.stride())}
        report["validation_scope"] = (
            "Meta-device configuration, model construction, forward bindings and query layout only; no CUDA execution."
        )
        return
    if distributed:
        distributed_training(args, report, model, rules, config)
        return
    base = model.model
    layer_index, layer = next((i, layer) for i, layer in enumerate(base.layers) if hasattr(layer, "self_attn"))
    attention = layer.self_attn.to(dtype=torch.bfloat16)
    attention.to_empty(device="cuda")
    initialize_parameters(attention, rules, config.initializer_range, prefix=f"model.layers.{layer_index}.self_attn")
    rotary_embedding = type(base.rotary_emb)(config, device="cuda")
    report["scope"] = "Full-size attention from original full-model instance; remaining parameters stay meta."
    report["materialized_parameter_count"] = sum(parameter.numel() for parameter in attention.parameters())
    report["weight_hash_scope"] = "isolated_attention"
    report["initial_weights_sha256"] = tensor_hash(attention.named_parameters())
    report["buffer_sha256"] = tensor_hash(rotary_embedding.named_buffers())
    hidden = torch.randn(
        args.batch_size, args.seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    report["input_checksum"] = tensor_hash([("hidden", hidden)])
    if args.level == "attention":
        positions = torch.arange(args.seq_len, device="cuda").unsqueeze(0).expand(args.batch_size, -1)
        rotary = rotary_embedding(hidden, positions)
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
        if gemma:
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
    if gemma:
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
        if baseline.get("model") != model:
            continue
        candidate = json.loads(path.with_name(path.name.replace("_B.json", "_C.json")).read_text())
        if baseline["status"] != "ok" or candidate["status"] != "ok":
            continue
        for key in (
            "initial_weights_sha256",
            "input_checksum",
            "config_sha256",
            "buffer_sha256",
            "model_config",
            "environment",
            "settings",
        ):
            assert baseline[key] == candidate[key], f"B/C mismatch for {key}: {path.name}"
        assert baseline.get("linear_attention_kernels") == candidate.get("linear_attention_kernels")
        assert baseline.get("hybrid_backend_dispatch") == candidate.get("hybrid_backend_dispatch")
        assert baseline.get("rank_hardware") == candidate.get("rank_hardware")
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
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def run_process(command, output, timeout):
    """Keep a failed worker/group visible and terminate all ranks on timeout."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite a prior benchmark case: {output}")
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            data = json.loads(output.read_text()) if output.exists() else {"status": "process_error"}
            data.update(stdout=stdout, stderr=stderr, returncode=process.returncode)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            data = {"status": "timeout", "timeout_seconds": timeout, "stdout": stdout, "stderr": stderr}
    failures = [json.loads(path.read_text()) for path in sorted(output.parent.glob(f"{output.stem}.rank*.json"))]
    if failures:
        data["rank_failures"] = failures
        data["status"] = "oom" if any(failure["status"] == "oom" for failure in failures) else "error"
    if data.get("returncode", 0) != 0 and data["status"] in {"ok", "probed", "validated"}:
        data["status"] = "process_error"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return data


def distributed_command(arguments):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=8",
        str(Path(__file__).resolve()),
        *arguments,
    ]


def run_matrix(args):
    args.output.mkdir(parents=True, exist_ok=True)
    levels = args.levels or ("norm", "attention", "model")
    shapes = [(4, 2048), (1, 8192)]
    if not args.primary:
        shapes += [(1, 128), (1, 2048), (4, 128), (4, 8192)]
    for level in levels:
        for shape_index, (batch, seq) in enumerate(shapes):
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
                elif level == "model":
                    command = distributed_command(command[2:])
                if args.probe_only:
                    command.append("--probe-only")
                print(f"Running {name}", flush=True)
                data = run_process(command, path, args.timeout)
                data.update(model=args.model, group=group, level=level, batch_size=batch, seq_len=seq)
                path.write_text(json.dumps(data, indent=2, default=str) + "\n")
                print(f"Finished {name}: {data['status']}", flush=True)
    write_comparison(args.output, args.model)


def main():
    args = parse_args()
    if args.matrix:
        run_matrix(args)
        return
    if args.level == "model" and not args.validate_only and "RANK" not in os.environ:
        data = run_process(distributed_command(sys.argv[1:]), args.output, args.timeout)
        print(json.dumps({"output": str(args.output), "status": data["status"]}))
        if data["status"] not in {"ok", "probed", "oom"}:
            raise SystemExit(1)
        return
    report = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    try:
        run(args, report)
        report.setdefault("status", "validated" if args.validate_only else "ok")
    except Exception as error:
        import torch

        report["status"] = "oom" if isinstance(error, torch.OutOfMemoryError) else "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    rank = int(os.environ.get("RANK", "0"))
    failed = report["status"] in {"error", "oom"}
    path = (
        args.output.with_name(f"{args.output.stem}.rank{rank}.json") if "RANK" in os.environ and failed else args.output
    )
    if rank == 0 or failed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({"output": str(path), "status": report["status"]}), flush=True)
    if "RANK" in os.environ:
        if failed:
            # Let torchrun stop peers immediately, rather than hang in collectives.
            raise SystemExit(1)
        import torch.distributed as dist

        dist.destroy_process_group()
    elif failed and report["status"] != "oom":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
