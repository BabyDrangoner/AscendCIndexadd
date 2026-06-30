#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/releases}"
ZIP_NAME="${2:-IndexAdd_candidate.zip}"

IMAGE="${IMAGE:-swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8}"
CANN_PATH="${CANN_PATH:-/home/ma-user/Ascend/cann-8.5.0}"
LIMA_INSTANCE="${LIMA_INSTANCE:-cann-docker}"
LIMACTL="${LIMACTL:-$HOME/.local/lima-2.1.3/bin/limactl}"
MODE="${MODE:-auto}"
KEEP_WORK="${KEEP_WORK:-0}"

if [[ "$ZIP_NAME" != *.zip ]]; then
  echo "zip name must end with .zip" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(python3 - "$OUT_DIR" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)"

RUN_ID="indexadd_build_$(date +%Y%m%d_%H%M%S)_$$"
LOCAL_WORK="$(mktemp -d "${TMPDIR:-/tmp}/${RUN_ID}.XXXXXX")"

cleanup() {
  if [[ "$KEEP_WORK" != "1" ]]; then
    rm -rf "$LOCAL_WORK"
  else
    echo "kept local work dir: $LOCAL_WORK"
  fi
}
trap cleanup EXIT

SRC_COPY="$LOCAL_WORK/IndexAdd"
mkdir -p "$SRC_COPY"
rsync -a \
  --exclude .git \
  --exclude build_out \
  --exclude releases \
  --exclude .cache \
  "$ROOT_DIR/" "$SRC_COPY/"
chmod -R a+rX "$SRC_COPY"

CONTAINER_SCRIPT="$LOCAL_WORK/container_build.sh"
python3 - "$CONTAINER_SCRIPT" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(r'''#!/usr/bin/env bash
set -Eeuo pipefail

: "${ZIP_NAME:?}"
: "${CANN_PATH:?}"

set +u
source "$CANN_PATH/set_env.sh"
set -u

cd /src
rm -rf build_out
bash build.sh package

run_file="$(find build_out -type f -name 'custom_opp_*_aarch64.run' | head -n 1 || true)"
if [[ -z "$run_file" ]]; then
  run_file="$(find build_out -type f -name '*.run' | head -n 1 || true)"
fi
[[ -n "$run_file" ]] || { echo "no run package produced" >&2; exit 1; }

pkg="/tmp/indexadd_pkg"
rm -rf "$pkg"
mkdir -p "$pkg"
cp -a op_host "$pkg/op_host"
cp -a op_kernel "$pkg/op_kernel"
cp -a "$run_file" "$pkg/custom_opp_euleros_aarch64.run"

[[ -f "$pkg/op_host/index_add.cpp" ]] || { echo "missing op_host/index_add.cpp" >&2; exit 1; }
[[ -f "$pkg/op_host/index_add_tiling.h" ]] || { echo "missing op_host/index_add_tiling.h" >&2; exit 1; }
[[ -f "$pkg/op_kernel/index_add.cpp" ]] || { echo "missing op_kernel/index_add.cpp" >&2; exit 1; }
[[ -f "$pkg/custom_opp_euleros_aarch64.run" ]] || { echo "missing run package" >&2; exit 1; }

rm -f "/out/$ZIP_NAME" "/out/$ZIP_NAME.sha256"
python3 - "$pkg" "/out/$ZIP_NAME" <<'ZIP'
import os
import sys
import zipfile

root, out_zip = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for base in ("op_host", "op_kernel"):
        full_base = os.path.join(root, base)
        for dirpath, _, filenames in os.walk(full_base):
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                arc = os.path.relpath(path, root)
                zf.write(path, arc)
    zf.write(os.path.join(root, "custom_opp_euleros_aarch64.run"), "custom_opp_euleros_aarch64.run")
ZIP

sha256sum "/out/$ZIP_NAME" | tee "/out/$ZIP_NAME.sha256"
''', encoding="utf-8")
PY
chmod +x "$CONTAINER_SCRIPT"

LIMA_RUNNING=0
if [[ -x "$LIMACTL" ]]; then
  if "$LIMACTL" list --format '{{.Name}} {{.Status}}' 2>/dev/null |
      awk -v n="$LIMA_INSTANCE" '$1 == n && $2 == "Running" { found = 1 } END { exit(found ? 0 : 1) }'; then
    LIMA_RUNNING=1
  fi
fi

USE_MODE="$MODE"
if [[ "$USE_MODE" == "auto" ]]; then
  if [[ "$LIMA_RUNNING" == "1" ]]; then
    USE_MODE="lima"
  else
    USE_MODE="docker"
  fi
fi

if [[ "$USE_MODE" == "lima" ]]; then
  [[ "$LIMA_RUNNING" == "1" ]] || { echo "Lima instance '$LIMA_INSTANCE' is not running" >&2; exit 1; }
  VM_WORK="/tmp/$RUN_ID"
  VM_SRC="$VM_WORK/IndexAdd"
  VM_OUT="$VM_WORK/out"
  VM_SCRIPT="$VM_WORK/container_build.sh"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- rm -rf "$VM_WORK"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- mkdir -p "$VM_WORK" "$VM_OUT"
  "$LIMACTL" copy "$SRC_COPY" "$LIMA_INSTANCE:$VM_WORK/"
  "$LIMACTL" copy "$CONTAINER_SCRIPT" "$LIMA_INSTANCE:$VM_SCRIPT"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- chmod +x "$VM_SCRIPT"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- chmod -R a+rwX "$VM_SRC"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- chmod a+rwx "$VM_OUT"
  "$LIMACTL" shell "$LIMA_INSTANCE" -- docker run \
    --rm \
    --platform linux/arm64 \
    --entrypoint /bin/bash \
    -e "ZIP_NAME=$ZIP_NAME" \
    -e "CANN_PATH=$CANN_PATH" \
    -v "$VM_SRC:/src" \
    -v "$VM_OUT:/out" \
    -v "$VM_SCRIPT:/work/container_build.sh:ro" \
    "$IMAGE" /work/container_build.sh
  "$LIMACTL" shell "$LIMA_INSTANCE" -- sudo chmod -R a+rX "$VM_OUT"
  "$LIMACTL" copy "$LIMA_INSTANCE:$VM_OUT/$ZIP_NAME" "$OUT_DIR/$ZIP_NAME"
  "$LIMACTL" copy "$LIMA_INSTANCE:$VM_OUT/$ZIP_NAME.sha256" "$OUT_DIR/$ZIP_NAME.sha256"
  if [[ "$KEEP_WORK" != "1" ]]; then
    "$LIMACTL" shell "$LIMA_INSTANCE" -- sudo rm -rf "$VM_WORK"
  else
    echo "kept VM work dir: $VM_WORK"
  fi
else
  docker run \
    --rm \
    --platform linux/arm64 \
    --entrypoint /bin/bash \
    -e "ZIP_NAME=$ZIP_NAME" \
    -e "CANN_PATH=$CANN_PATH" \
    -v "$SRC_COPY:/src" \
    -v "$OUT_DIR:/out" \
    -v "$CONTAINER_SCRIPT:/work/container_build.sh:ro" \
    "$IMAGE" /work/container_build.sh
fi

ZIP_PATH="$OUT_DIR/$ZIP_NAME"
[[ -f "$ZIP_PATH" ]] || { echo "missing output zip: $ZIP_PATH" >&2; exit 1; }

for entry in \
  op_host/index_add.cpp \
  op_host/index_add_tiling.h \
  op_kernel/index_add.cpp \
  custom_opp_euleros_aarch64.run; do
  unzip -l "$ZIP_PATH" "$entry" >/dev/null || {
    echo "zip missing $entry" >&2
    exit 1
  }
done

echo "$ZIP_PATH"
