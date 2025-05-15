#!/usr/bin/env bash

set -euo pipefail

# ───────────── Configuration ─────────────
WORKDIR="${WORKDIR:-$HOME/modelPerf}"
ACCELSIM_DIR="$WORKDIR/accel-sim-framework"
MLP_DIR="$WORKDIR/inference"
GCOM_DIR="$WORKDIR/gcom"

TRACES_ROOT="$WORKDIR/mlperf_traces"     # {traces,processed}
CUDNN_DIR="$HOME/cudnn/lib64"
MAKE_JOBS="${MAKE_JOBS:-$(nproc)}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

[[ -x "$GCOM_DIR/bin/GCoM" ]] || { echo "ERROR: GCoM not built"; exit 1; }

# ───────── clone repos if missing ─────────
clone() { [[ -d "$2/.git" ]] || git clone --depth 1 "$1" "$2"; }
clone https://github.com/accel-sim/accel-sim-framework.git "$ACCELSIM_DIR"
clone https://github.com/mlcommons/inference.git            "$MLP_DIR"

# ───────── build tracer (SM-75 patch) ────
pushd "$ACCELSIM_DIR/util/tracer_nvbit" >/dev/null
  ./install_nvbit.sh
popd >/dev/null

TRACER_CU=$(find "$ACCELSIM_DIR/util/tracer_nvbit" -path '*/tracer_tool/tracer_tool.cu' -print -quit)
TRACER_DIR=$(dirname "$TRACER_CU")

# remove stale lines, add clean hard-code once
sed -i '/binary_version[[:space:]]*=[[:space:]]*75;/d' "$TRACER_CU"
if ! grep -q '^  binary_version = 75;$' "$TRACER_CU"; then
  sed -i '/CU_FUNC_ATTRIBUTE_BINARY_VERSION.*p->f.*;/a\
  // Force all traces to report SM-75\n  binary_version = 75;' "$TRACER_CU"
fi

make -C "$TRACER_DIR"                     -j"$MAKE_JOBS"
make -C "$TRACER_DIR/traces-processing"   -j"$MAKE_JOBS"

TRACER_SO="$TRACER_DIR/tracer_tool.so"
POST_PROC="$ACCELSIM_DIR/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
echo "✔ tracer ready (SM-75)"

# ───────── python deps ─────────
python3 -m pip -q install --upgrade pip
python3 -m pip -q install numpy transformers mlcommons-loadgen onnxruntime-gpu
export LD_LIBRARY_PATH="$CUDNN_DIR:$LD_LIBRARY_PATH"

# ───────── fresh trace dirs ─────
rm -rf "$TRACES_ROOT"; mkdir -p "$TRACES_ROOT"/{traces,processed}

# ───────── run BERT with tracing ─────────
export ACTIVE_FROM_START=1
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER="$TRACES_ROOT"
export MLC_MAX_NUM_THREADS=2
export LD_PRELOAD="$TRACER_SO"
export CUDA_VISIBLE_DEVICES

pushd "$MLP_DIR/language/bert" >/dev/null
  make -s setup
  python3 -u run.py --backend=onnxruntime --scenario=Offline
popd >/dev/null
unset LD_PRELOAD

# ───────── post-process & gzip ───────────
"$POST_PROC" "$TRACES_ROOT/traces"
mv "$TRACES_ROOT"/traces/*.traceg.xz "$TRACES_ROOT/processed"
pushd "$TRACES_ROOT/processed" >/dev/null
  for f in *.traceg.xz; do xzcat "$f" | gzip > "${f%.xz}.gz" && rm "$f"; done
popd >/dev/null

# ───────── generate rep_warp_out.bin ─────
pushd "$TRACES_ROOT/processed" >/dev/null
find . -maxdepth 1 -name '*.traceg.gz' | sed 's|^\./||' | sort -V > kernelslist.g
[[ -s kernelslist.g ]] || { echo "ERROR: no .traceg.gz files"; exit 1; }

cat > make_rep_warp.cpp <<'CPP'
#include <fstream>
#include <vector>
#include <iostream>
#include <string>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
struct KernelRepWarp{int kernelNumber,repWarpIdx;
 template<class A>void serialize(A&ar,const unsigned){ar&kernelNumber&repWarpIdx;}};
int main(){std::ifstream in("kernelslist.g");std::string l;std::vector<KernelRepWarp> v;
 for(int k=1;std::getline(in,l);) if(!l.empty()) v.push_back({k++,0});
 std::ofstream ofs("rep_warp_out.bin",std::ios::binary);
 boost::archive::binary_oarchive(ofs)<<v;
 std::cerr<<"rep_warp_out.bin with "<<v.size()<<" entries\n";
}
CPP

if pkg-config --exists boost_serialization; then
  CXXFLAGS=$(pkg-config --cflags boost_serialization)
  LDFLAGS=$(pkg-config --libs   boost_serialization)
else
  CXXFLAGS="-I$HOME/boost/include"
  LDFLAGS="-L$HOME/boost/lib -Wl,-rpath,$HOME/boost/lib -lboost_serialization"
fi

g++ -std=c++17 -O2 make_rep_warp.cpp -o make_rep_warp $CXXFLAGS $LDFLAGS
./make_rep_warp
rm make_rep_warp make_rep_warp.cpp
popd >/dev/null

# ───────── run GCoM ───────────────────────
"$GCOM_DIR/bin/GCoM" -w 1 \
  -r "$TRACES_ROOT/processed/rep_warp_out.bin" \
  -t "$TRACES_ROOT/processed/" \
  -C "$GCOM_DIR/configs/RTX2060.config"

echo -e "\n✅  Pipeline finished, rep_warp_out.bin kept in processed/, SM 75."
