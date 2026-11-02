#!/bin/bash
# Build perseus._pc (Rust + CUDA, sm_120) for THIS checkout without installing anything: `maturin build` makes a wheel
# (its own cargo target dir per checkout), and only the compiled module is copied into python/perseus/ here. Run the
# code with this checkout's python/ first on the path (selective/ scripts do it), with any venv that has the Python deps.
#   bash selective/build_perseus.sh
set -eu
REPO="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
export RUSTUP_HOME=/home/lzhou/data/tools/rust/rustup
export CARGO_HOME=/home/lzhou/data/tools/rust/cargo
export CUDA_HOME=/usr/local/cuda-13.2
VENV=${VENV:-/home/lzhou/data/venvs/perseus}
export PATH=$VENV/bin:$CARGO_HOME/bin:$CUDA_HOME/bin:$PATH
export CARGO_TARGET_DIR=/home/lzhou/data/selective-fhe/build/target_$(basename "$REPO")
cd "$REPO"
rm -rf "$CARGO_TARGET_DIR/wheels"
maturin build --release --skip-auditwheel --interpreter "$VENV/bin/python"
WHL=$(ls "$CARGO_TARGET_DIR"/wheels/*.whl | head -1)
"$VENV/bin/python" - "$WHL" "$REPO/python" <<'PY'
import sys, zipfile
whl, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z:
    so = [n for n in z.namelist() if n.startswith("perseus/_pc") and n.endswith(".so")]
    for n in so:
        z.extract(n, dst)
print("installed", so, "into", dst)
PY
