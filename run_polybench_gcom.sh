#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined variable, or pipe failure
IFS=$'\n\t'        # Set Internal Field Separator to newline and tab

# ───────── Config ─────────
WORKDIR="${WORKDIR:-$HOME/modelPerf}"
ACCELSIM_DIR="$WORKDIR/accel-sim-framework"
POLY_DIR="$WORKDIR/PolyBench-ACC"
GCOM_DIR="${GCOM_DIR:-$WORKDIR/gcom}"
TRACES_ROOT="$WORKDIR/polybench_traces"
TRACES_DIR="$TRACES_ROOT/traces"    # Benchmark .out/.err logs & parent for NVBit's own 'traces'
PROC_DIR="$TRACES_ROOT/processed"
NVCC_ARCH="${NVCC_ARCH:-sm_75}"
JOBS="${JOBS:-$(nproc)}"

# Function to print error and exit
error_exit() {
    echo "ERROR: ${1:-"Unknown error"} (line $LINENO, script $0, command: $BASH_COMMAND)" >&2
    exit 1
}

# Trap ERR to call error_exit, providing more context
trap 'error_exit "Command failed with exit code $?"' ERR

echo "INFO: Starting PolyBench GCoM Pipeline..."
echo "INFO: Workdir: $WORKDIR"
echo "INFO: Accel-Sim Dir: $ACCELSIM_DIR"
echo "INFO: PolyBench Dir: $POLY_DIR"
echo "INFO: GCoM Dir: $GCOM_DIR"
echo "INFO: Traces Root: $TRACES_ROOT"
echo "INFO: Script's main trace log dir (TRACES_DIR): $TRACES_DIR"
echo "INFO: Processed traces dir (PROC_DIR): $PROC_DIR"

# ───────── Sanity Checks ─────────
[[ -x "$GCOM_DIR/bin/GCoM" ]] || error_exit "GCoM executable not found at $GCOM_DIR/bin/GCoM"
command -v git >/dev/null || error_exit "git command not found. Please install git."
command -v make >/dev/null || error_exit "make command not found. Please install make."
command -v g++ >/dev/null || error_exit "g++ command not found. Please install g++."
command -v xzcat >/dev/null || error_exit "xzcat command not found. Please install xz-utils."
command -v gzip >/dev/null || error_exit "gzip command not found. Please install gzip."
command -v pkg-config >/dev/null || error_exit "pkg-config command not found. Please install pkg-config."
command -v sed >/dev/null || error_exit "sed command not found. Please install sed."

# ───────── Clone Repositories (if missing) ─────────
clone_repo() {
    local repo_url="$1"
    local target_dir="$2"
    if [[ -d "$target_dir/.git" ]]; then
        echo "INFO: Repository $target_dir already cloned."
    else
        echo "INFO: Cloning $repo_url into $target_dir..."
        git clone --depth=1 "$repo_url" "$target_dir" || error_exit "Failed to clone $repo_url"
    fi
}
mkdir -p "$WORKDIR" || error_exit "Failed to create workdir: $WORKDIR"
clone_repo https://github.com/accel-sim/accel-sim-framework.git "$ACCELSIM_DIR"
clone_repo https://github.com/cavazos-lab/PolyBench-ACC.git "$POLY_DIR"

# ───────── Build & Patch NVBit Tracer ─────────
echo "INFO: Building and patching NVBit tracer..."
NVBIT_TRACER_SRC_DIR="$ACCELSIM_DIR/util/tracer_nvbit"
[[ -d "$NVBIT_TRACER_SRC_DIR" ]] || error_exit "NVBit tracer source directory not found: $NVBIT_TRACER_SRC_DIR"

pushd "$NVBIT_TRACER_SRC_DIR" >/dev/null
[[ -f "./install_nvbit.sh" ]] || error_exit "install_nvbit.sh not found in $NVBIT_TRACER_SRC_DIR"
if ./install_nvbit.sh > install_nvbit.log 2>&1; then
    echo "INFO: install_nvbit.sh completed."
else
    error_exit "install_nvbit.sh failed. Check log: $NVBIT_TRACER_SRC_DIR/install_nvbit.log"
fi
popd >/dev/null

TR_CU=$(find "$NVBIT_TRACER_SRC_DIR" -path '*/tracer_tool/tracer_tool.cu' -print -quit)
[[ -n "$TR_CU" && -f "$TR_CU" ]] || error_exit "tracer_tool.cu not found under $NVBIT_TRACER_SRC_DIR"
TR_DIR=$(dirname "$TR_CU")

# --- Apply SM-75 patch right after the binary_version query line ---
# Remove any existing forced SM-75 lines
sed -i.sm75_patch_bak '/binary_version *= *75; *$/d' "$TR_CU"

# Insert force line immediately after the cuFuncGetAttribute for binary_version
if grep -q 'CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));' "$TR_CU"; then
  sed -i '/CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));/a \
  // Force SM-75 (override detected version)\
  binary_version = 75;' "$TR_CU"
  echo "INFO: SM-75 patch applied after the full attribute query."
else
  error_exit "Could not find the complete binary_version query to patch in $TR_CU"
fi
# Patch should_trace_kernel
sed -i.should_trace_kernel_patch_bak \
'/^bool should_trace_kernel(uint64_t kernel_id, const std::string& kernel_name) {$/,/^}$/c\
bool should_trace_kernel(uint64_t kernel_id, const std::string& kernel_name) {\
  for (const auto& range : g_kernel_ranges) {\
    bool id_match = (range.end == 0 ? kernel_id >= range.start : (kernel_id >= range.start && kernel_id <= range.end));\
    if (!id_match) continue;\
    if (range.kernel_name_regexes.empty()) return true;\
    for (auto &rx : range.kernel_name_regexes) try { if (std::regex_match(kernel_name, rx)) return true; } catch(...){};\
  }\
  return false;\
}' "$TR_CU"
echo "INFO: should_trace_kernel patch applied."

# Build tracer_tool.so
make -C "$TR_DIR" -j"$JOBS" || error_exit "Failed to build tracer_tool.so"
TR_SO="$TR_DIR/tracer_tool.so"
[[ -f "$TR_SO" ]] || error_exit "tracer_tool.so not found at $TR_SO"

# Build post-traces-processing tool
POST_PROC_DIR_BUILD="$TR_DIR/traces-processing"
make -C "$POST_PROC_DIR_BUILD" -j"$JOBS" || error_exit "Failed to build post-traces-processing tool"
POST="$POST_PROC_DIR_BUILD/post-traces-processing"
[[ -x "$POST" ]] || error_exit "post-traces-processing tool not found or not executable at $POST"

# ───────── Build PolyBench-ACC Kernels ─────────
echo "INFO: Building PolyBench-ACC kernels for $NVCC_ARCH..."
find "$POLY_DIR/CUDA" -name Makefile -print0 | while IFS= read -r -d '' mf; do
    dir=$(dirname "$mf")
    echo "INFO: Building in $dir"
    make -C "$dir" NVCC_FLAGS="-w -arch=$NVCC_ARCH -O3 -use_fast_math" -j"$JOBS" || error_exit "Build failed in $dir"
done
echo "INFO: Kernel builds complete."

# ───────── Collect Executables ─────────
echo "INFO: Collecting PolyBench executables..."
mapfile -t EXES < <(
    find "$POLY_DIR/CUDA" -type f -executable \
        ! -name '*.cu' ! -name '*.cuh' ! -name '*.c' ! -name '*.cpp' \
        ! -name 'Makefile*' ! -name '*.sh' ! -name '*.dat' ! -name '*.ptx' \
        -print
)
(( ${#EXES[@]} )) || error_exit "No executables found under $POLY_DIR/CUDA."
echo "INFO: Found ${#EXES[@]} executables."

# ───────── Prepare Trace Directories ─────────
echo "INFO: Preparing trace directories..."
rm -rf "$TRACES_DIR" "$PROC_DIR"
mkdir -p "$TRACES_DIR" "$PROC_DIR" || error_exit "Failed to create trace directories"

NVBIT_ACTUAL_OUTPUT_PATH="$TRACES_DIR/traces"
echo "INFO: NVBit will output traces to: $NVBIT_ACTUAL_OUTPUT_PATH"

# ───────── Set NVBit Environment Variables ─────────
export ACTIVE_FROM_START=1
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER="$TRACES_DIR"
export CUDA_VISIBLE_DEVICES=0
export TRACE_FILE_COMPRESS=0
export NOBANNER=1
export TOOL_VERBOSE=1

# ───────── Trace All Benchmarks ─────────
echo "INFO: Tracing all benchmarks..."
SUCCESS=0
FAIL=0
for exe in "${EXES[@]}"; do
    rel="${exe#"$POLY_DIR/CUDA/"}"
    name=$(basename "$exe")
    logbase="${rel//\//_}_$name"
    printf 'INFO: Tracing %-60s ... ' "$rel"
    if LD_PRELOAD="$TR_SO" "$exe" >"$TRACES_DIR/${logbase}.out" 2>"$TRACES_DIR/${logbase}.err"; then
        echo "OK"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
    fi
done
echo "INFO: Tracing done. Success: $SUCCESS, Fail: $FAIL"
(( FAIL )) && echo "WARNING: Some benchmarks failed. Check .err logs."

# ───────── Verify Raw Trace File Generation (.trace) ─────────
echo "INFO: Verifying raw '.trace' files..."
[[ -d "$NVBIT_ACTUAL_OUTPUT_PATH" ]] || {
    ls -A "$TRACES_DIR"
    error_exit "NVBit did not create its output directory."
}
shopt -s nullglob
trace_files=("$NVBIT_ACTUAL_OUTPUT_PATH"/kernel-*.trace)
shopt -u nullglob
(( ${#trace_files[@]} )) || {
    ls -A "$NVBIT_ACTUAL_OUTPUT_PATH"
    error_exit "No raw '.trace' files found."
}
echo "INFO: Found ${#trace_files[@]} raw '.trace' files. Sample:"
printf '%s\n' "${trace_files[@]:0:5}"

# ───────── Clean Up NVBit Environment Variables ─────────
unset ACTIVE_FROM_START USER_DEFINED_FOLDERS TRACES_FOLDER CUDA_VISIBLE_DEVICES NOBANNER TRACE_FILE_COMPRESS TOOL_VERBOSE

# ───────── Post-process Traces (.trace → .traceg) ─────────
echo "INFO: Post-processing raw traces..."
"$POST" "$NVBIT_ACTUAL_OUTPUT_PATH" || error_exit "Post-traces-processing failed."
echo "INFO: Moving .traceg files to $PROC_DIR..."
if ls "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg >/dev/null 2>&1; then
    mv "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg "$PROC_DIR"/
elif ls "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg.xz >/dev/null 2>&1; then
    mv "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg.xz "$PROC_DIR"/
else
    error_exit "No .traceg or .traceg.xz files found after post-processing."
fi

# ───────── Convert to .traceg.gz ─────────
echo "INFO: Converting to .traceg.gz..."
pushd "$PROC_DIR" >/dev/null
shopt -s nullglob
for f in *.traceg; do gzip "$f"; done
for f in *.traceg.xz; do xzcat "$f" | gzip >"${f%.xz}.gz"; rm -f "$f"; done
shopt -u nullglob
popd >/dev/null

# ───────── Generate rep_warp_out.bin ─────────
echo "INFO: Generating rep_warp_out.bin..."
pushd "$PROC_DIR" >/dev/null
find . -maxdepth 1 -name '*.traceg.gz' -printf "%f\n" | sort -V > kernelslist.g
[[ -s kernelslist.g ]] || error_exit "kernelslist.g is empty or missing."
cat > make_rep_warp.cpp <<'EOF'
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
struct KernelInfo {
    int kernelNumber;
    int repWarpIdx;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & kernelNumber;
        ar & repWarpIdx;
    }
};
int main() {
    std::ifstream kernel_list_file("kernelslist.g");
    if (!kernel_list_file.is_open()) {
        std::cerr << "Error: Cannot open kernelslist.g\n";
        return 1;
    }
    std::string line;
    std::vector<KernelInfo> kernels;
    int current = 1;
    while (std::getline(kernel_list_file, line)) {
        if (!line.empty()) {
            kernels.push_back({current++, 0});
        }
    }
    std::ofstream ofs("rep_warp_out.bin", std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error: Cannot open rep_warp_out.bin\n";
        return 1;
    }
    try {
        boost::archive::binary_oarchive oa(ofs);
        oa << kernels;
    } catch (...) {
        std::cerr << "Error during Boost serialization\n";
        return 1;
    }
    return 0;
}
EOF

if pkg-config --exists boost_serialization; then
    CXXFLAGS="$(pkg-config --cflags boost_serialization)"
    LDFLAGS="$(pkg-config --libs boost_serialization)"
elif [[ -d "$HOME/boost/include" ]]; then
    CXXFLAGS="-I$HOME/boost/include"
    LDFLAGS="-L$HOME/boost/lib -lboost_serialization"
else
    echo "WARNING: Boost not found; make_rep_warp may fail."
fi
g++ -std=c++17 -O2 make_rep_warp.cpp -o make_rep_warp $CXXFLAGS $LDFLAGS \
    || error_exit "Failed to compile make_rep_warp.cpp"
./make_rep_warp || error_exit "make_rep_warp execution failed"
rm -f make_rep_warp make_rep_warp.cpp kernelslist.g
popd >/dev/null
[[ -f "$PROC_DIR/rep_warp_out.bin" ]] || error_exit "rep_warp_out.bin missing."
echo "INFO: rep_warp_out.bin generated."

# ───────── Run GCoM with Dedicated Log ─────────
GCOM_EXEC="$GCOM_DIR/bin/GCoM"
REP_WARP="$PROC_DIR/rep_warp_out.bin"
PROC_DIR_ABS="$(cd "$PROC_DIR" && pwd)"
GCOM_CONFIG_FILE="$GCOM_DIR/configs/RTX2060.config"
if [[ ! -f "$GCOM_CONFIG_FILE" ]]; then
    ALT=$(find "$GCOM_DIR/configs" -name '*.config' -print -quit)
    [[ -n "$ALT" ]] && GCOM_CONFIG_FILE="$ALT" || error_exit "No GCoM .config found."
fi
GCOM_CONFIG="$GCOM_CONFIG_FILE"

echo "Starting GCoM..."
echo "INFO: GCoM command: $GCOM_EXEC -w 1 -r \"$REP_WARP\" -t \"$PROC_DIR_ABS\" -C \"$GCOM_CONFIG\""

GCOM_LOG_DIR="$WORKDIR/gcom_simulation_logs"
mkdir -p "$GCOM_LOG_DIR"
GCOM_OUTPUT_LOG="$GCOM_LOG_DIR/gcom_simulation_$(date +%Y%m%d_%H%M%S).log"

pushd "$GCOM_LOG_DIR" >/dev/null
"$GCOM_EXEC" -w 1 \
    -r "$REP_WARP" \
    -t "$PROC_DIR_ABS" \
    -C "$GCOM_CONFIG" >"$GCOM_OUTPUT_LOG" 2>&1 || {
    cat "$GCOM_OUTPUT_LOG"
    echo "ERROR: GCoM simulation failed. Log: $GCOM_OUTPUT_LOG" >&2
    popd >/dev/null
    exit 1
}
popd >/dev/null

echo "INFO: GCoM simulation finished. Output log: $GCOM_OUTPUT_LOG"
echo "-----------------------------------"

echo "✅ All done!"
exit 0
