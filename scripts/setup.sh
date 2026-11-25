#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
HOST_OS="$(uname -s)"

case "${HOST_OS}" in
    Darwin)
        DEFAULT_TORCH_INDEX_URL=""
        ;;
    Linux)
        DEFAULT_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
        ;;
    *)
        printf 'Error: this setup script supports macOS and Linux, not %s.\n' "${HOST_OS}" >&2
        exit 1
        ;;
esac
TORCH_INDEX_URL="${TORCH_INDEX_URL-${DEFAULT_TORCH_INDEX_URL}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    printf 'Error: %s was not found. Install Python 3.10+ or set PYTHON=/path/to/python.\n' "${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    printf 'Error: Python 3.10 or newer is required (selected: %s).\n' "$("${PYTHON_BIN}" --version 2>&1)" >&2
    exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
    if [[ -e "${VENV_DIR}" ]]; then
        printf 'Error: %s exists but is not a usable virtual environment. Move it aside and rerun setup.\n' "${VENV_DIR}" >&2
        exit 1
    fi
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || {
        printf 'Error: could not create the virtual environment. On Ubuntu, install python3-venv and retry.\n' >&2
        exit 1
    }
fi

if ! "${VENV_PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    printf 'Error: %s uses an unsupported Python version. Remove .venv and rerun with PYTHON=python3.10+.\n' "${VENV_PYTHON}" >&2
    exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${VENV_PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" 'torch>=2.2'
else
    "${VENV_PYTHON}" -m pip install 'torch>=2.2'
fi
"${VENV_PYTHON}" -m pip install --editable "${PROJECT_ROOT}"
"${VENV_PYTHON}" -m unittest discover -s "${PROJECT_ROOT}/tests" -v

"${VENV_PYTHON}" -c '
import platform
import torch

print("Platform:", platform.platform())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda or "not included")
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Apple MPS device: available")
else:
    print("No GPU backend detected; CPU execution is available.")
'

if [[ "${HOST_OS}" == "Linux" ]] && ! "${VENV_PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    printf 'WARNING: CUDA is unavailable on Linux. Check the NVIDIA driver (nvidia-smi) and, if needed, set TORCH_INDEX_URL to a compatible PyTorch CUDA wheel index.\n' >&2
fi

printf '\nSetup complete. Activate the project environment with:\n  source "%s/bin/activate"\n' "${VENV_DIR}"
