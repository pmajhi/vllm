# CacheGen Reproducibility Snapshot

## Snapshot

- Timestamp (UTC): 20260907T224316Z
- Repository branch: hetero-quant-paged-kv
- Commit before snapshot: d8c50a21120debfb197ff72b5aa500798dad4c7a
- Python executable: /lambda/nfs/pm-asplos/src/vllm/.venv/bin/python
- Virtual environment: /lambda/nfs/pm-asplos/src/vllm/.venv

## Contents

- `requirements-freeze.txt`: exact installed Python package versions.
- `python_version.txt`, `python_runtime.txt`: interpreter version and paths.
- `torch_cuda.txt`: PyTorch, CUDA runtime, GPU properties, and FP8 availability.
- `nvidia_smi_full.txt`, `nvidia_smi_summary.csv`: GPU and driver state.
- `nvcc_version.txt`: CUDA compiler information, if installed.
- `os_release.txt`, `uname.txt`, `lscpu.txt`: operating-system and CPU details.
- `environment_sanitized.txt`: relevant environment configuration with credential values redacted.
- `git_status_before_commit.txt`: repository state prior to committing.
- `working_tree_before_commit.patch`: complete uncommitted patch before committing.
- `experimental_python_files.txt`: CacheGen experimental module inventory.
- `cachegen_test_files.txt`: CacheGen test inventory.
- `cachegen_benchmark_files.txt`: benchmark script inventory.


## Credential handling

Full process environments are deliberately excluded from version control because they
may contain access tokens, API keys, cookies, or passwords. The sanitized environment
snapshot records only performance-relevant variables and redacts credential values.

## Restore

On a new compatible Linux host:

```bash
git clone <your-repository-url>
cd vllm
git checkout <the-commit-recorded-after-this-snapshot>
./scripts/restore_cachegen_environment.sh \
  artifacts/reproducibility/20260907T224316Z/requirements-freeze.txt
```

Then verify GPU visibility:

```bash
source .venv/bin/activate
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

## Important compatibility note

The package lockfile captures Python dependencies exactly, but GPU environments also
require a compatible NVIDIA driver, CUDA runtime, Python version, PyTorch build,
FlashAttention build, Triton build, and GPU compute capability. Rebuild local CUDA
extensions such as FlashAttention on the new machine if the driver, CUDA, Python,
PyTorch, or GPU architecture changes.
