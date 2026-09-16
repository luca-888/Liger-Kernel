"""Focused Qwen instance-patch validation and B/C benchmarks on Modal H100.

Uses the same Python image and GPU selection as dev/modal/tests.py. See
benchmark/data/qwen_qk_norm/README.md for exact reproduction commands.
"""

import hashlib
import json
import shlex
import subprocess
import sys

from pathlib import Path

import modal

if modal.is_local():
    from dev.modal.tests import REMOTE_ROOT_PATH
    from dev.modal.tests import ROOT_PATH
    from dev.modal.tests import image as base_image

    REMOTE = Path(REMOTE_ROOT_PATH)
    PINS = ["torch==2.9.1", "torchvision==0.24.1", "triton==3.5.1", "transformers==5.15.1"]
    CAUSAL_CONV_WHEEL = (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
        "causal_conv1d-1.7.0%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )
    image = (
        base_image.apt_install("build-essential", "git")
        .uv_pip_install(
            *PINS,
            "pytest==8.4.2",
            "pytest-asyncio==1.2.0",
            "pytest-xdist==3.8.0",
            "pytest-cov==7.0.0",
            "pytest-rerunfailures==16.1",
            "datasets==4.4.1",
            "matplotlib==3.10.7",
            "seaborn==0.13.2",
            "ruff==0.15.22",
            "flash-linear-attention==0.5.2",
            "tilelang==0.1.14",
            "nvidia-cuda-nvcc==13.0.88",
            "nvidia-cuda-cccl==13.0.85",
            "nvidia-nvvm==13.0.88",
            "nvidia-cuda-crt==13.0.88",
            "nvidia-cuda-runtime==13.0.96",
            CAUSAL_CONV_WHEEL,
        )
        .env(
            {
                "PYTHONPATH": f"{REMOTE}/src:{REMOTE}",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_DATASETS_OFFLINE": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "LIGER_KERNEL_IMPL": "",
                "FLA_TILELANG": "1",
            }
        )
        .workdir(str(REMOTE))
    )
    for name in ("pyproject.toml", "setup.py", "README.md", "LICENSE", "Makefile"):
        image = image.add_local_file(ROOT_PATH / name, str(REMOTE / name), copy=True)
    image = image.run_commands(
        f"mkdir -p {REMOTE}/src/liger_kernel && touch {REMOTE}/src/liger_kernel/__init__.py",
        "uv pip install --system --no-deps -e .",
    )
    for name in ("src", "test", "benchmark"):
        image = image.add_local_dir(
            ROOT_PATH / name, str(REMOTE / name), ignore=["**/__pycache__/**", "**/*.pyc", "**/.pytest_cache/**"]
        )
else:
    REMOTE = Path("/root/liger-kernel")
    image = None

app = modal.App("liger-qwen-qk-norm")


@app.function(image=image, cpu=8, memory=65536, timeout=7200, retries=0, single_use_containers=True, scaledown_window=2)
def run_job(command: list[str], baseline_source: str = ""):
    import importlib.metadata
    import platform
    import traceback

    import torch

    out = Path("/tmp/qwen-results")
    out.mkdir(exist_ok=True)
    (out / "requirements.txt").write_bytes(subprocess.check_output(["uv", "pip", "freeze", "--system"]))
    report = {
        "command": command,
        "python": platform.python_version(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "triton",
                "transformers",
                "flash-linear-attention",
                "fla-core",
                "causal-conv1d",
                "tilelang",
                "nvidia-cuda-nvcc",
            )
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip()
        if torch.cuda.is_available()
        else None,
    }
    patch_file = REMOTE / "src/liger_kernel/transformers/monkey_patch.py"
    original = patch_file.read_text()
    try:
        if baseline_source:
            patch_file.write_text(baseline_source)
        result = subprocess.run(command, cwd=REMOTE, capture_output=True, timeout=6900)
        report["exit_code"] = result.returncode
        files = {"stdout.log": result.stdout, "stderr.log": result.stderr}
    except Exception:
        report["exit_code"] = 1
        files = {"stderr.log": traceback.format_exc().encode()}
    finally:
        if baseline_source:
            patch_file.write_text(original)
    files.update({p.name: p.read_bytes() for p in out.iterdir() if p.is_file()})
    return {"environment": report, "files": files}


@app.local_entrypoint()
def main(command: str, output: str, baseline_ref: str = "", gpu: str = "H100!"):
    """Each invocation executes the command in a fresh Python subprocess."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline_ref": baseline_ref,
        "source_sha256": {
            str(p.relative_to(ROOT_PATH)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("src", "test", "benchmark/scripts")
            for p in sorted((ROOT_PATH / directory).rglob("*.py"))
        },
    }
    (out / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    baseline = ""
    if baseline_ref:
        baseline = subprocess.check_output(
            ["git", "show", f"{baseline_ref}:src/liger_kernel/transformers/monkey_patch.py"], text=True
        )
    if gpu not in {"H100!", "H100!:8", "none"}:
        raise ValueError("gpu must be H100!, H100!:8, or none (CPU binding checks only)")
    worker = run_job if gpu == "none" else run_job.with_options(gpu=gpu)
    if gpu == "H100!:8":
        worker = worker.with_options(cpu=32, memory=196608)
    try:
        result = worker.remote(shlex.split(command), baseline)
    except Exception as error:
        (out / "run_error.json").write_text(
            json.dumps({"gpu_requested": gpu, "command": command, "error": str(error)}, indent=2) + "\n"
        )
        raise
    (out / "environment.json").write_text(json.dumps(result["environment"], indent=2) + "\n")
    for name, data in result["files"].items():
        (out / name).write_bytes(data)
        if name.endswith(".log"):
            print(data.decode(errors="replace")[-16000:])
    print(json.dumps(result["environment"], indent=2))
    print(f"Results: {out}")
    if result["environment"]["exit_code"]:
        sys.exit(result["environment"]["exit_code"])
