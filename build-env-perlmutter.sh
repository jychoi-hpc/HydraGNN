#!/usr/bin/env bash

set -Eeuo pipefail

PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch/2.13.0}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
MAX_JOBS="${MAX_JOBS:-16}"
INSTALL_DEV="${INSTALL_DEV:-0}"
INSTALL_OPTIONAL="${INSTALL_OPTIONAL:-0}"
INSTALL_DEEPSPEED="${INSTALL_DEEPSPEED:-0}"
DDSTORE_ONLY="${DDSTORE_ONLY:-0}"
DDSTORE_SOURCE_DIR="${DDSTORE_SOURCE_DIR:-${PWD}/DDStore-Perlmutter/DDStore}"
ADIOS2_ONLY="${ADIOS2_ONLY:-0}"
ADIOS2_VERSION="${ADIOS2_VERSION:-v2.12.1}"
ADIOS2_JOBS="${ADIOS2_JOBS:-32}"
ADIOS2_SOURCE_DIR="${ADIOS2_SOURCE_DIR:-${PWD}/ADIOS2-Perlmutter/ADIOS2-${ADIOS2_VERSION}}"
ADIOS2_BUILD_DIR="${ADIOS2_BUILD_DIR:-${PWD}/ADIOS2-Perlmutter/adios2-build-${ADIOS2_VERSION}}"
MODULE_LOAD_SCRIPT="${MODULE_LOAD_SCRIPT:-module-to-load-perlmutter.sh}"

init_modules() {
  if command -v module >/dev/null 2>&1; then
    return
  fi

  local init_candidates=(
    "/etc/profile.d/modules.sh"
    "/usr/share/lmod/lmod/init/bash"
    "/usr/share/Modules/init/bash"
  )
  local init_file
  for init_file in "${init_candidates[@]}"; do
    if [[ -f "${init_file}" ]]; then
      # shellcheck disable=SC1090
      source "${init_file}"
      break
    fi
  done

  if ! command -v module >/dev/null 2>&1; then
    printf 'ERROR: module command not found. Run this on Perlmutter with modules enabled.\n' >&2
    exit 1
  fi
}

pip_retry() {
  local tries=3
  local delay=3
  local attempt
  for ((attempt = 1; attempt <= tries; attempt++)); do
    if python -m pip install "$@"; then
      return 0
    fi
    if (( attempt == tries )); then
      break
    fi
    printf 'pip install failed (attempt %d/%d). Retrying in %ds...\n' "${attempt}" "${tries}" "${delay}" >&2
    sleep "${delay}"
    delay=$((delay * 2))
  done
  return 1
}

parse_requirements() {
  local file="$1"
  shift || true
  python - "$file" "$@" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
exclude = {arg.lower().replace("_", "-") for arg in sys.argv[2:]}

for raw in path.read_text().splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line or line.startswith("-r"):
        continue
    candidate = line.split(";", 1)[0].strip()
    candidate = re.split(r"[<>=!~\[]", candidate, maxsplit=1)[0]
    normalized = candidate.lower().replace("_", "-")
    if normalized in exclude:
        continue
    print(line)
PY
}

install_from_requirements() {
  local file="$1"
  shift || true
  local -a packages=()
  mapfile -t packages < <(parse_requirements "$file" "$@")
  if (( ${#packages[@]} == 0 )); then
    return 0
  fi
  pip_retry --no-build-isolation -v "${packages[@]}"
}

install_pyg_extensions() {
  local -a packages=()
  mapfile -t packages < <(parse_requirements "requirements-pyg.txt" torch_geometric)
  local package
  for package in "${packages[@]}"; do
    pip_retry --no-build-isolation --no-binary :all: -v "$package"
  done
}

install_ddstore() {
  if [[ ! -f "${DDSTORE_SOURCE_DIR}/setup.py" ]]; then
    if [[ -e "${DDSTORE_SOURCE_DIR}" ]]; then
      printf 'ERROR: %s exists but is not a DDStore source checkout.\n' "${DDSTORE_SOURCE_DIR}" >&2
      return 1
    fi
    mkdir -p "$(dirname "${DDSTORE_SOURCE_DIR}")"
    git clone https://github.com/ORNL/DDStore.git "${DDSTORE_SOURCE_DIR}"
  fi

  if ! pkg-config --exists libfabric; then
    printf 'ERROR: libfabric development files are unavailable (pkg-config libfabric).\n' >&2
    return 1
  fi
  python -c 'import numpy, Cython, mpi4py'
  printf 'Building PyDDStore from %s with Cray MPI wrappers...\n' "${DDSTORE_SOURCE_DIR}"
  # Keep the environment's Cray mpi4py/NumPy and avoid an isolated MPI build.
  CC=cc CXX=CC DDSTORE_MPI=1 pip_retry --no-build-isolation --no-deps -v "${DDSTORE_SOURCE_DIR}"
  python - <<'PY'
import pyddstore
from pyddstore import PyDDStore

assert pyddstore.HAVE_MPI, "PyDDStore was built without MPI support"
print(f"PyDDStore import: OK ({pyddstore.__file__}); MPI: {pyddstore.HAVE_MPI}")
PY
}

check_adios2() {
  python - "${ADIOS2_VERSION#v}" <<'PY'
import adios2
import adios2.bindings as bindings
import sys

assert adios2.__version__ == sys.argv[1], (adios2.__version__, sys.argv[1])
assert hasattr(adios2, "Stream"), "ADIOS2 Stream API is unavailable"
assert hasattr(bindings, "ADIOS"), "ADIOS2 C++ Python bindings are unavailable"
print(f"ADIOS2 import: OK ({adios2.__file__}); version: {adios2.__version__}")
PY
}

install_adios2() {
  if [[ ! -f "${ADIOS2_SOURCE_DIR}/CMakeLists.txt" ]]; then
    if [[ -e "${ADIOS2_SOURCE_DIR}" ]]; then
      printf 'ERROR: %s exists but is not an ADIOS2 source checkout.\n' "${ADIOS2_SOURCE_DIR}" >&2
      return 1
    fi
    mkdir -p "$(dirname "${ADIOS2_SOURCE_DIR}")"
    git clone --branch "${ADIOS2_VERSION}" --depth 1 https://github.com/ornladios/ADIOS2.git "${ADIOS2_SOURCE_DIR}"
  fi
  if [[ "$(git -C "${ADIOS2_SOURCE_DIR}" describe --tags --exact-match 2>/dev/null)" != "${ADIOS2_VERSION}" ]]; then
    printf 'ERROR: %s is not checked out at %s. Set ADIOS2_SOURCE_DIR to the matching checkout.\n' "${ADIOS2_SOURCE_DIR}" "${ADIOS2_VERSION}" >&2
    return 1
  fi

  local venv_prefix
  venv_prefix="$(python -c 'import sys; print(sys.prefix)')"
  printf 'Building ADIOS2 %s against Cray MPI into %s...\n' "${ADIOS2_VERSION}" "${venv_prefix}"
  CC=cc CXX=CC FC=ftn \
  cmake -DCMAKE_INSTALL_PREFIX="${venv_prefix}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Fortran=OFF \
    -DADIOS2_BUILD_EXAMPLES_EXPERIMENTAL=OFF \
    -DADIOS2_BUILD_TESTING=OFF \
    -DADIOS2_USE_HDF5=OFF \
    -DADIOS2_USE_SST=OFF \
    -DADIOS2_USE_BZip2=OFF \
    -DADIOS2_USE_PNG=OFF \
    -DADIOS2_USE_DataSpaces=OFF \
    -DADIOS2_USE_Python=ON \
    -DPython_EXECUTABLE="$(command -v python)" \
    -B "${ADIOS2_BUILD_DIR}" -S "${ADIOS2_SOURCE_DIR}"
  cmake --build "${ADIOS2_BUILD_DIR}" -j"${ADIOS2_JOBS}"
  cmake --install "${ADIOS2_BUILD_DIR}"
  check_adios2
}

write_module_load_script() {
  local venv_path="$1"
  cat > "${MODULE_LOAD_SCRIPT}" <<EOF
#!/bin/bash

module load ${PYTORCH_MODULE}

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
export MAX_JOBS=${MAX_JOBS}
export CC=${CC}
export CXX=${CXX}

source ${venv_path}/bin/activate
EOF
  chmod +x "${MODULE_LOAD_SCRIPT}"
}

printf 'Loading Perlmutter PyTorch module: %s\n' "${PYTORCH_MODULE}"
init_modules
module load "${PYTORCH_MODULE}"
module unload darshan >/dev/null 2>&1 || true
module unload darshan-runtime >/dev/null 2>&1 || true

export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export CC="${CC:-cc}"
export CXX="${CXX:-CC}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

printf 'Using Python: %s\n' "$(which python)"
python --version

if [[ "${DDSTORE_ONLY}" == "1" ]]; then
  install_ddstore
  printf 'DDStore repair complete.\n'
  exit 0
fi

if [[ "${ADIOS2_ONLY}" == "1" ]]; then
  install_adios2
  printf 'ADIOS2 repair complete.\n'
  exit 0
fi

pip_retry --upgrade pip setuptools wheel
pip_retry wheel Cython ninja cmake

python - <<'PY'
import torch

print(f"module torch version: {torch.__version__}")
print(f"torch cuda version: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
PY

printf 'Installing HydraGNN base requirements (excluding mpi4py)...\n'
install_from_requirements "requirements-base.txt" mpi4py

printf 'Installing mpi4py against Cray MPI...\n'
MPICC="${MPICC:-cc -shared}" pip_retry --no-cache-dir --no-binary=mpi4py -v mpi4py==4.1.1

printf 'Installing DDStore with Cray MPI support...\n'
install_ddstore

printf 'Installing ADIOS2 with Cray MPI and Python bindings...\n'
install_adios2

printf 'Installing HydraGNN torch-side Python packages (excluding torch/vision/audio)...\n'
install_from_requirements "requirements-torch.txt" torch torchvision torchaudio

printf 'Building PyG compiled extensions against the module PyTorch...\n'
install_pyg_extensions

printf 'Installing torch_geometric Python package...\n'
install_from_requirements "requirements-pyg.txt" torch_scatter torch_sparse torch_cluster torch_spline_conv

if [[ "${INSTALL_DEV}" == "1" ]]; then
  printf 'Installing development requirements...\n'
  install_from_requirements "requirements-dev.txt"
fi

if [[ "${INSTALL_OPTIONAL}" == "1" ]]; then
  printf 'Installing optional requirements...\n'
  install_from_requirements "requirements-optional.txt"
fi

if [[ "${INSTALL_DEEPSPEED}" == "1" ]]; then
  printf 'Installing DeepSpeed requirements...\n'
  install_from_requirements "requirements-deepspeed.txt"
fi

printf 'Installing HydraGNN in editable mode without pulling a second PyTorch...\n'
CC="${CC}" CXX="${CXX}" pip_retry --no-build-isolation -e . --no-deps

printf 'Running import sanity checks...\n'
python - <<'PY'
import hydragnn
import adios2
import adios2.bindings
import mpi4py
import pyddstore
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
import torch_spline_conv

print(f"hydragnn import: OK ({hydragnn.__file__})")
print(f"adios2 version: {adios2.__version__}")
print(f"mpi4py version: {mpi4py.__version__}")
print(f"pyddstore MPI support: {pyddstore.HAVE_MPI}")
print(f"torch version: {torch.__version__}")
print(f"torch_geometric version: {torch_geometric.__version__}")
PY

printf 'Writing module reload helper: %s\n' "${MODULE_LOAD_SCRIPT}"
write_module_load_script "${VENV_DIR}"

printf '\nEnvironment is ready at %s\n' "${VENV_DIR}"
printf 'Activate with: source %s/bin/activate\n' "${VENV_DIR}"
printf 'Or reload with: source %s\n' "${MODULE_LOAD_SCRIPT}"
printf 'Optional installs: INSTALL_DEV=1 INSTALL_OPTIONAL=1 INSTALL_DEEPSPEED=1 ./build-env-perlmutter.sh\n'
printf 'Repair DDStore only: DDSTORE_ONLY=1 ./build-env-perlmutter.sh\n'
printf 'Repair ADIOS2 only: ADIOS2_ONLY=1 ./build-env-perlmutter.sh\n'
