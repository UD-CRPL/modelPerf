#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKDIR="${WORKDIR:-$HOME/modelPerf}"
GCOM_DIR="$WORKDIR/gcom"
ACCELSIM_DIR="$WORKDIR/accel-sim-framework"
GPT2_RUN_DIR="$WORKDIR/gpt2_inference"
USER_CUDNN_DIR="$HOME/cudnn/lib64" # Path where my cuDNN files are found

# Trace Directory Structure
TRACES_ROOT="$WORKDIR/gpt2_traces_root"
TRACES_DIR="$TRACES_ROOT/traces" # This is where NVBit's 'traces' subdir will be
PROC_DIR="$TRACES_ROOT/processed" # Processed traces and rep_warp_out.bin go here

# Final location for the rep_warp file
REP_WARP="$PROC_DIR/rep_warp_out.bin"

MAKE_JOBS="${MAKE_JOBS:-$(nproc --all || echo 4)}" # Added --all and fallback
# -----------------------------------------------------------------------------

echo "=== Environment & Version Info ==="
echo -n "GCoM Directory: "; echo "$GCOM_DIR" $( [ -d "$GCOM_DIR/.git" ] && echo "(commit:" $(git -C "$GCOM_DIR" rev-parse --short HEAD)")" || echo "(Not a git repo or not found)" )
echo -n "Accel-Sim commit: "
if [ -d "$ACCELSIM_DIR/.git" ]; then git -C "$ACCELSIM_DIR" rev-parse --short HEAD; else echo "Not a git repo or not found"; fi
echo -n "CUDA / nvcc:    "; nvcc --version | head -n1 || echo "nvcc not found"
echo -n "g++:            "; g++ --version | head -n1 || echo "g++ not found"
echo -n "Python:         "; python3 --version || echo "python3 not found"
echo -n "ONNX Info:      ";
python3 - <<'PYONNX'
import importlib.metadata
try:
    import onnxruntime as ort
    import transformers
    print(f"ONNX Runtime: {ort.__version__} (Device: {ort.get_device()})")
    try:
        optimum_version = importlib.metadata.version("optimum")
        print(f"Optimum:      {optimum_version}")
    except importlib.metadata.PackageNotFoundError:
        print("Optimum:      (version not detected)")
    print(f"Transformers: {transformers.__version__}")
except Exception as e:
    print(f"Python package check error (some packages might not be installed yet): {e}")
PYONNX
echo

# -----------------------------------------------------------------------------
# Locate libcudnn.so.X (Set CUDNN_MAJOR_VERSION based on verification)
# -----------------------------------------------------------------------------
CUDNN_MAJOR_VERSION=9 # This is correct based on your files
# -----------------------------------------------------------------------------
echo "Locating libcudnn.so.${CUDNN_MAJOR_VERSION}..."
CANDIDATES=()
if c1=$(ldconfig -p 2>/dev/null | grep -oP "libcudnn\.so\.${CUDNN_MAJOR_VERSION}\s+\(.*\)\s+=>\s+\K\S+" | head -n1); then
  CANDIDATES+=("$c1")
fi
[[ -f "$USER_CUDNN_DIR/libcudnn.so.${CUDNN_MAJOR_VERSION}" ]] && CANDIDATES+=("$USER_CUDNN_DIR/libcudnn.so.${CUDNN_MAJOR_VERSION}")

ACTUAL_CUDA_LIB_PATH="/opt/cuda/12.8/lib64"
COMMON_CUDA_PATHS=( "${ACTUAL_CUDA_LIB_PATH}" "${CUDA_INSTALL_PATH:-/opt}/lib64" "/usr/lib/x86_64-linux-gnu" "/lib/x86_64-linux-gnu" )
for d in "${COMMON_CUDA_PATHS[@]}"; do
  [[ -f "$d/libcudnn.so.${CUDNN_MAJOR_VERSION}" ]] && CANDIDATES+=("$d/libcudnn.so.${CUDNN_MAJOR_VERSION}")
done
UNIQUE_CANDIDATES=($(printf "%s\n" "${CANDIDATES[@]}" | sort -u))

FOUND_PATH=""
if [[ ${#UNIQUE_CANDIDATES[@]} -gt 0 ]]; then
  for candidate in "${UNIQUE_CANDIDATES[@]}"; do
    if [[ "$candidate" == "$USER_CUDNN_DIR/libcudnn.so.${CUDNN_MAJOR_VERSION}" && -f "$candidate" ]]; then
      FOUND_PATH="$candidate"
      break
    fi
    if [[ -f "$candidate" && -z "$FOUND_PATH" ]]; then
      FOUND_PATH="$candidate"
    fi
  done
fi

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [[ -n "$FOUND_PATH" ]]; then
    export LD_LIBRARY_PATH="$(dirname "$FOUND_PATH"):${LD_LIBRARY_PATH}"
    echo "  Found libcudnn.so.${CUDNN_MAJOR_VERSION} at: $FOUND_PATH"
    echo "  LD_LIBRARY_PATH prepended with $(dirname "$FOUND_PATH")"
else
 echo "  Warning: libcudnn.so.${CUDNN_MAJOR_VERSION} not found via explicit checks."
 echo "  Ensure cuDNN (version ${CUDNN_MAJOR_VERSION}.x) is installed correctly."
 echo "  Searched paths:" "$USER_CUDNN_DIR" "${COMMON_CUDA_PATHS[@]}" "(and ldconfig)"
fi
echo "-----------------------------------"


# -----------------------------------------------------------------------------
# 1) Build Accel‑Sim NVBit tracer & post‑processing
# -----------------------------------------------------------------------------
echo "[Building Accel-Sim NVBit tracer and tools...]"
if [ ! -d "$ACCELSIM_DIR" ]; then
    echo "Cloning Accel-Sim Framework..."
    git clone https://github.com/Wmoook/accel-sim-framework.git "$ACCELSIM_DIR" || { echo "ERROR: Failed to clone Accel-Sim." >&2; exit 1; }
else
    echo "Accel-Sim Framework directory already exists."
fi

NVBIT_TRACER_SRC_DIR="$ACCELSIM_DIR/util/tracer_nvbit"
if [ ! -d "$NVBIT_TRACER_SRC_DIR" ]; then
    echo "ERROR: NVBit tracer source directory not found: $NVBIT_TRACER_SRC_DIR" >&2; exit 1;
fi

pushd "$NVBIT_TRACER_SRC_DIR" >/dev/null
  ./install_nvbit.sh > install_nvbit.log 2>&1 || { cat install_nvbit.log; echo "ERROR: NVBit installation failed. Log above." >&2; popd >/dev/null; exit 1; }
popd >/dev/null
# --- Apply SM-75 patch (force Compute Capability 7.5) ---
TR_CU=$(find "$NVBIT_TRACER_SRC_DIR" -path '*/tracer_tool/tracer_tool.cu' -print -quit)
if [[ -z "$TR_CU" ]]; then
  echo "ERROR: tracer_tool.cu not found under $NVBIT_TRACER_SRC_DIR" >&2
  exit 1
fi

# Remove any existing forced SM-75 lines
sed -i.sm75_bak '/binary_version *= *75; *$/d' "$TR_CU"

# Insert override immediately after the binary-version query
if grep -q 'CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));' "$TR_CU"; then
  sed -i '/CU_FUNC_ATTRIBUTE_BINARY_VERSION, func));/a \
  // Force SM-75 (override detected version)\
  binary_version = 75;' "$TR_CU"
  echo "INFO: SM-75 patch applied to $TR_CU"
else
  echo "ERROR: Could not locate the binary-version query in $TR_CU" >&2
  exit 1
fi
make -C "$NVBIT_TRACER_SRC_DIR/tracer_tool" -j"$MAKE_JOBS" || { echo "ERROR: Failed to build tracer_tool." >&2; exit 1; }
make -C "$NVBIT_TRACER_SRC_DIR/tracer_tool/traces-processing" -j"$MAKE_JOBS" || { echo "ERROR: Failed to build traces-processing." >&2; exit 1; }
echo "INFO: Accel-Sim build complete."
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 2) GCoM check
# -----------------------------------------------------------------------------
echo "[Checking GCoM]"
if [ ! -d "$GCOM_DIR" ]; then echo "Error: GCoM directory not found at '$GCOM_DIR'." >&2; exit 1; fi
if [ ! -f "$GCOM_DIR/bin/GCoM" ]; then echo "Warning: GCoM executable not found at $GCOM_DIR/bin/GCoM. Will fail later if needed."; fi
echo "GCoM directory exists. Assuming it is built."
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 3) Python dependencies
# -----------------------------------------------------------------------------
echo "[Installing Python dependencies]"
pip install --upgrade pip || { echo "ERROR: Failed to upgrade pip." >&2; exit 1; }

echo "INFO: Uninstalling existing onnxruntime and onnxruntime-gpu (if any)..."
pip uninstall -y onnxruntime onnxruntime-gpu || echo "INFO: onnxruntime/onnxruntime-gpu not found or uninstall failed, proceeding..."

echo "INFO: Installing onnxruntime-gpu==1.21.1..."
pip install onnxruntime-gpu==1.21.1 --force-reinstall --no-cache-dir || \
    { echo "ERROR: Failed to install onnxruntime-gpu==1.21.1." >&2; exit 1; }

echo "INFO: Installing other Python packages (numpy, transformers, torch, onnx, optimum)..."
pip install numpy transformers torch onnx optimum || \
    { echo "ERROR: Failed to install other Python packages." >&2; exit 1; }
echo "INFO: Python dependencies installation step complete."
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 4) Prepare GPT-2 ONNX Model and Inference Script
# -----------------------------------------------------------------------------
echo "[Preparing GPT-2 ONNX model and inference script]"
mkdir -p "$GPT2_RUN_DIR"
GPT2_MODEL_DIR="$GPT2_RUN_DIR/gpt2-onnx"
GPT2_SCRIPT="$GPT2_RUN_DIR/run_gpt2_onnx.py"

if [ ! -d "$GPT2_MODEL_DIR" ]; then
    echo "Downloading and exporting GPT-2 model to ONNX format..."
    optimum-cli export onnx --model gpt2 --task text-generation --framework pt "$GPT2_MODEL_DIR" || \
        { echo "ERROR: Failed to export GPT-2 model to ONNX." >&2; exit 1; }
    echo "Model exported to $GPT2_MODEL_DIR"
else
    echo "GPT-2 ONNX model directory already exists: $GPT2_MODEL_DIR"
fi

# Create the Python inference script - includes position_ids fix
cat > "$GPT2_SCRIPT" << 'EOPYTHON'
import os
import time
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

print("--- GPT-2 ONNX Inference Script ---")
print(f"ONNX Runtime version: {ort.__version__}")
print(f"ONNX Runtime Available Execution Providers: {ort.get_available_providers()}")

model_dir = os.environ.get("GPT2_MODEL_DIR", "./gpt2-onnx")
num_tokens_to_generate = int(os.environ.get("NUM_TOKENS", 50))
prompt = os.environ.get("PROMPT", "The purpose of life is")

print(f"Model directory: {model_dir}")
print(f"Prompt: '{prompt}'")
print(f"Tokens to generate: {num_tokens_to_generate}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    print("Loading ONNX model...")
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0 # Enable verbose logging from ORT
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), sess_options=sess_options, providers=providers)
    session_providers = session.get_providers()
    print(f"ONNX Runtime Using Execution Provider(s): {session_providers}")
    if 'CUDAExecutionProvider' not in session_providers:
        print("Warning: CUDAExecutionProvider not available or not used. Running on CPU.")

    input_names = {inp.name for inp in session.get_inputs()}
    print(f"Model Input Names: {input_names}")

    print("Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"] # Not always used directly if generating
    batch_size, sequence_length = input_ids.shape
    # position_ids = np.arange(0, sequence_length, dtype=np.int64).reshape(1, -1) # Initial position_ids
    generated_ids = input_ids.copy()

    print(f"Generating {num_tokens_to_generate} tokens...")
    start_time = time.time()
    for i in range(num_tokens_to_generate):
        print(f"Step {i+1}/{num_tokens_to_generate}", end='\r')

        current_input_ids = generated_ids
        current_attention_mask = np.ones(current_input_ids.shape, dtype=np.int64) # Mask for current generated sequence
        current_sequence_length = current_input_ids.shape[1]
        current_position_ids = np.arange(0, current_sequence_length, dtype=np.int64).reshape(1, -1) # Dynamic position_ids

        ort_inputs = {
            "input_ids": current_input_ids,
            "attention_mask": current_attention_mask,
            "position_ids": current_position_ids,
        }

        # Filter inputs to only those the model expects, helps with models not needing all (e.g. position_ids if baked in)
        model_feed = {name: ort_inputs[name] for name in input_names if name in ort_inputs}

        # Check for missing non-past_key_values inputs
        required_model_inputs = {inp.name for inp in session.get_inputs() if 'past_key_values' not in inp.name}
        missing_keys = required_model_inputs - set(model_feed.keys())
        if missing_keys:
            print(f"\nWarning: Required inputs missing from feed: {missing_keys}")
            # Attempt to provide zero past_key_values if they are expected and not in model_feed
            # This is a basic handling; more sophisticated KV caching is needed for true efficiency
            for inp in session.get_inputs():
                if 'past_key_values' in inp.name and inp.name not in model_feed:
                    # Assuming shape [batch_size, num_heads, sequence_length, head_size] or similar
                    # This is a placeholder, actual shapes are model-specific.
                    # For GPT-2, past_key_values are not mandatory for the first pass.
                    pass


        try:
            ort_outputs = session.run(None, model_feed)
        except Exception as e:
            print(f"\nError during session.run: {e}")
            print("Inputs provided to model:")
            for name, arr in model_feed.items():
                print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            raise e

        next_token_logits = ort_outputs[0][:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
        generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)

        if next_token_id[0,0] == tokenizer.eos_token_id:
            print("\nEOS token generated.")
            break
    print() # Newline after progress indicator

    end_time = time.time()
    print(f"Inference loop took {end_time - start_time:.2f} seconds.")

    print("Decoding output...")
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------\n")

except Exception as e:
    print(f"\nAn error occurred in Python script: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)

print("--- Python Script Finished Successfully ---")
import sys
sys.exit(0)
EOPYTHON
echo "Python script created at $GPT2_SCRIPT"
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 5) Prepare Trace Directories
# -----------------------------------------------------------------------------
echo "[Preparing trace directories...]"
# rm -rf "$TRACES_ROOT" # Comment out if you want to append traces
mkdir -p "$TRACES_ROOT" "$TRACES_DIR" "$PROC_DIR"
echo "INFO: Trace directories ready. NVBit traces in '$TRACES_DIR/traces', processed in '$PROC_DIR'."
# Note: NVBit will create a 'traces' subdirectory inside TRACES_FOLDER ($TRACES_ROOT)
# So raw .trace files will be in $TRACES_ROOT/traces.
NVBIT_ACTUAL_OUTPUT_PATH="$TRACES_DIR/traces" # This is where kernel-*.trace files are expected.
mkdir -p "$NVBIT_ACTUAL_OUTPUT_PATH" # Ensure this sub-directory for NVBit traces exists
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 6) Trace GPT-2 Inference with NVBit
# -----------------------------------------------------------------------------
echo "[Running GPT-2 Inference with NVBit tracing]"

export ACTIVE_FROM_START=1
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER="$TRACES_DIR" # NVBit will create its 'traces' subdir here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPT2_MODEL_DIR="$GPT2_MODEL_DIR"
export NUM_TOKENS="${NUM_TOKENS:-20}" # Short run for testing, allow override
export PROMPT="${PROMPT:-Once upon a time in a galaxy far, far away}"
export TRACE_FILE_COMPRESS=0 # Explicitly no compression for .trace files
export TOOL_VERBOSE=1

TRACER_TOOL_SO="$NVBIT_TRACER_SRC_DIR/tracer_tool/tracer_tool.so"
if [ ! -f "$TRACER_TOOL_SO" ]; then echo "Error: NVBit tracer tool '$TRACER_TOOL_SO' not found!" >&2; exit 1; fi

# --- BEGIN LD_LIBRARY_PATH FIX ---
CUDA_LIB_PATH="${ACTUAL_CUDA_LIB_PATH}" # Using var from top
if [ -d "$CUDA_LIB_PATH" ]; then
  # Prepend general CUDA path. Specific cuDNN path found earlier is already prepended.
  export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH}"
  echo "INFO: Added '$CUDA_LIB_PATH' to LD_LIBRARY_PATH for main CUDA libs."
else
  echo "WARNING: Specific CUDA lib path '$CUDA_LIB_PATH' not found. LD_LIBRARY_PATH relies on earlier cuDNN detection or system defaults."
fi
echo "DEBUG: LD_LIBRARY_PATH before running python is: $LD_LIBRARY_PATH"
# --- END LD_LIBRARY_PATH FIX ---

export LD_PRELOAD="$TRACER_TOOL_SO"

echo "Starting traced execution... Output will be in gpt2_inference.out / .err"
# Python script output will go here, including NVBit verbose messages if TOOL_VERBOSE=1
python3 -u "$GPT2_SCRIPT" > "$GPT2_RUN_DIR/gpt2_inference.out" 2> "$GPT2_RUN_DIR/gpt2_inference.err" || {
    echo "ERROR: Python inference script failed. Check '$GPT2_RUN_DIR/gpt2_inference.err' and '.out'." >&2
    unset LD_PRELOAD
    exit 1
}

unset LD_PRELOAD ACTIVE_FROM_START USER_DEFINED_FOLDERS TRACES_FOLDER CUDA_VISIBLE_DEVICES TRACE_FILE_COMPRESS TOOL_VERBOSE
echo "Traced execution finished."
echo "Raw traces should be in $NVBIT_ACTUAL_OUTPUT_PATH"
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 7) Post-process traces (raw .trace to .traceg or .traceg.xz)
# -----------------------------------------------------------------------------
echo "[Post-processing raw traces...]"
POST_PROCESS_EXEC="$NVBIT_TRACER_SRC_DIR/tracer_tool/traces-processing/post-traces-processing"

if [ ! -f "$POST_PROCESS_EXEC" ]; then echo "Error: Post-processing executable '$POST_PROCESS_EXEC' not found!" >&2; exit 1; fi

# Check for raw .trace files using nullglob
shopt -s nullglob
raw_trace_files=("$NVBIT_ACTUAL_OUTPUT_PATH"/kernel-*.trace)
shopt -u nullglob

if [[ ${#raw_trace_files[@]} -eq 0 ]]; then
    echo "ERROR: No raw 'kernel-*.trace' files found in $NVBIT_ACTUAL_OUTPUT_PATH." >&2
    echo "INFO: Contents of $NVBIT_ACTUAL_OUTPUT_PATH:" >&2
    ls -Al "$NVBIT_ACTUAL_OUTPUT_PATH" >&2 || echo "INFO: ls failed for $NVBIT_ACTUAL_OUTPUT_PATH" >&2
    echo "INFO: Check $GPT2_RUN_DIR/gpt2_inference.err for NVBit messages." >&2
    exit 1
fi
echo "INFO: Found ${#raw_trace_files[@]} raw trace files in $NVBIT_ACTUAL_OUTPUT_PATH to process."

echo "Running post-processor on traces in $NVBIT_ACTUAL_OUTPUT_PATH..."
# The post-processor reads from its input dir and writes .traceg or .traceg.xz there.
"$POST_PROCESS_EXEC" "$NVBIT_ACTUAL_OUTPUT_PATH"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then echo "ERROR: Post-processing failed with exit code $EXIT_CODE." >&2; exit 1; fi

echo "Moving processed traces to $PROC_DIR..."
shopt -s nullglob
processed_files_xz=("$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg.xz)
processed_files_traceg=("$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg)
shopt -u nullglob

MOVED_COUNT=0
if [[ ${#processed_files_xz[@]} -gt 0 ]]; then
    echo "INFO: Found *.traceg.xz files. Moving to $PROC_DIR..."
    mv -- "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg.xz "$PROC_DIR"/ || { echo "ERROR: Failed to move *.traceg.xz files." >&2; exit 1; }
    MOVED_COUNT=${#processed_files_xz[@]}
elif [[ ${#processed_files_traceg[@]} -gt 0 ]]; then
    echo "INFO: Found *.traceg files (uncompressed). Moving to $PROC_DIR..."
    mv -- "$NVBIT_ACTUAL_OUTPUT_PATH"/*.traceg "$PROC_DIR"/ || { echo "ERROR: Failed to move *.traceg files." >&2; exit 1; }
    MOVED_COUNT=${#processed_files_traceg[@]}
else
    echo "ERROR: No *.traceg.xz or *.traceg files generated by post-processing in $NVBIT_ACTUAL_OUTPUT_PATH." >&2
    echo "INFO: Contents of $NVBIT_ACTUAL_OUTPUT_PATH after post-processing attempt:" >&2
    ls -Al "$NVBIT_ACTUAL_OUTPUT_PATH" >&2
    exit 1
fi

echo "INFO: Post-processing complete. Moved $MOVED_COUNT processed trace file(s) to $PROC_DIR."
echo "-----------------------------------"

# -----------------------------------------------------------------------------
# 8) Convert trace format -> .gz (if needed)
# -----------------------------------------------------------------------------
echo "[Converting traces to .traceg.gz format in $PROC_DIR...]"
pushd "$PROC_DIR" >/dev/null
shopt -s nullglob
CONVERTED_TO_GZ=0

# Case 1: Process .traceg.xz files
files_to_convert_xz=(*.traceg.xz)
if [[ ${#files_to_convert_xz[@]} -gt 0 ]]; then
    echo "INFO: Converting ${#files_to_convert_xz[@]} .traceg.xz files to .traceg.gz..."
    for f in "${files_to_convert_xz[@]}"; do
        target_gz="${f%.xz}.gz"
        echo "INFO: Converting $f to $target_gz"
        if command -v pigz > /dev/null; then
            xzcat "$f" | pigz > "$target_gz" || { echo "ERROR: Failed to convert $f using pigz"; popd >/dev/null; exit 1; }
        else
            xzcat "$f" | gzip > "$target_gz" || { echo "ERROR: Failed to convert $f using gzip"; popd >/dev/null; exit 1; }
        fi
        rm "$f"
        CONVERTED_TO_GZ=$((CONVERTED_TO_GZ + 1))
    done
fi

# Case 2: Process .traceg files (if post-processor output them uncompressed)
files_to_convert_plain=(*.traceg)
if [[ ${#files_to_convert_plain[@]} -gt 0 ]]; then
    echo "INFO: Gzipping ${#files_to_convert_plain[@]} .traceg files to .traceg.gz..."
    for f in "${files_to_convert_plain[@]}"; do
        target_gz="${f}.gz"
        echo "INFO: Gzipping $f to $target_gz"
        if command -v pigz > /dev/null; then
            pigz "$f" || { echo "ERROR: Failed to gzip $f using pigz"; popd >/dev/null; exit 1; }
        else
            gzip "$f" || { echo "ERROR: Failed to gzip $f using gzip"; popd >/dev/null; exit 1; }
        fi
        # gzip by default renames f to f.gz. If pigz doesn't, ensure original is removed.
        CONVERTED_TO_GZ=$((CONVERTED_TO_GZ + 1))
    done
fi

if [[ "$CONVERTED_TO_GZ" -gt 0 ]]; then
    echo "INFO: $CONVERTED_TO_GZ file(s) processed into .traceg.gz format."
else
    # Check if .gz files already exist (e.g. from a previous run)
    existing_gz_files=(*.traceg.gz)
    if [[ ${#existing_gz_files[@]} -gt 0 ]]; then
        echo "INFO: No new files converted, but ${#existing_gz_files[@]} .traceg.gz files already exist in $PROC_DIR."
    else
        echo "WARNING: No .traceg.xz or .traceg files found in $PROC_DIR to convert to .gz. GCoM might fail if it expects .gz files."
    fi
fi
shopt -u nullglob
popd >/dev/null
echo "INFO: Trace format conversion step finished."
echo "-----------------------------------"


# -----------------------------------------------------------------------------
# 9) Build rep_warp_out.bin
# -----------------------------------------------------------------------------
echo "[Generating representative warp file ($REP_WARP)...]"
pushd "$PROC_DIR" >/dev/null # Operate within PROC_DIR

echo "INFO: Generating kernelslist.g from .traceg.gz files in $PWD..."
find . -maxdepth 1 -name '*.traceg.gz' -printf "%f\n" | sort -V > kernelslist.g
if [[ ! -s kernelslist.g ]]; then
    echo "ERROR: kernelslist.g is empty or not created in $PROC_DIR. No .traceg.gz files found." >&2
    ls -Al "$PROC_DIR" >&2 # Show contents for debugging
    popd >/dev/null
    exit 1
fi
echo "INFO: Found $(wc -l < kernelslist.g) trace files listed in kernelslist.g."

echo "INFO: Creating make_rep_warp.cpp..."
cat > make_rep_warp.cpp <<'EOF'
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept> // Required for std::runtime_error
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

struct KernelInfo { // Changed struct name to KernelInfo for consistency with PolyBench example
    int kernelNumber;
    int repWarpIdx;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version) {
        ar & kernelNumber;
        ar & repWarpIdx;
    }
};

int main() {
    try {
        std::ifstream kernel_list_file("kernelslist.g");
        if (!kernel_list_file.is_open()) {
            throw std::runtime_error("Error: Cannot open kernelslist.g");
        }
        std::string line;
        std::vector<KernelInfo> kernels;
        int current_kernel_number = 1; // GCoM typically expects 1-based kernel numbers
        while (std::getline(kernel_list_file, line)) {
            if (!line.empty()) { // Ensure line is not empty before adding
                kernels.push_back({current_kernel_number, 0}); // repWarpIdx is 0
                current_kernel_number++;
            }
        }
        kernel_list_file.close();

        if (kernels.empty()) {
            // GCoM might require this file even if empty.
            std::cout << "Info: No kernels found in kernelslist.g. rep_warp_out.bin will be empty." << std::endl;
        } else {
            std::cout << "Info: Processing " << kernels.size() << " kernels for rep_warp_out.bin." << std::endl;
        }

        std::ofstream ofs("rep_warp_out.bin", std::ios::binary); // Output directly to rep_warp_out.bin
        if (!ofs.is_open()) {
            throw std::runtime_error("Error: Cannot open rep_warp_out.bin for writing.");
        }

        boost::archive::binary_oarchive oa(ofs);
        oa << kernels; // Serialize the vector of KernelInfo
        ofs.close();
        std::cout << "Info: rep_warp_out.bin generated successfully in current directory." << std::endl;
        return 0;

    } catch (const boost::archive::archive_exception& e) {
        std::cerr << "Error during Boost serialization: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }
}
EOF

BOOST_CPPFLAGS=""
BOOST_LDFLAGS=""
if pkg-config --exists boost_serialization 2>/dev/null; then
    echo "INFO: Using pkg-config for Boost flags."
    BOOST_CPPFLAGS=$(pkg-config --cflags boost_serialization)
    BOOST_LDFLAGS=$(pkg-config --libs boost_serialization)
elif [[ -d "$HOME/boost/include" && -d "$HOME/boost/lib" ]]; then
    echo "INFO: Using manual Boost paths: $HOME/boost"
    BOOST_CPPFLAGS="-I$HOME/boost/include"
    # Add rpath to ensure the executable can find the custom Boost library at runtime
    BOOST_LDFLAGS="-L$HOME/boost/lib -Wl,-rpath=$HOME/boost/lib -lboost_serialization"
else
    echo "WARNING: Boost serialization library not found via pkg-config or in $HOME/boost. Compilation of make_rep_warp might fail."
    # Attempt to compile without explicit flags, hoping g++ finds it in default paths.
fi

echo "INFO: Compiling make_rep_warp.cpp..."
g++ -std=c++17 -O2 make_rep_warp.cpp -o make_rep_warp_exec $BOOST_CPPFLAGS $BOOST_LDFLAGS || {
    echo "ERROR: Failed to compile make_rep_warp.cpp." >&2
    popd >/dev/null
    exit 1
}

echo "INFO: Running make_rep_warp_exec..."
./make_rep_warp_exec || {
    echo "ERROR: make_rep_warp_exec execution failed." >&2
    popd >/dev/null
    exit 1
}

# rep_warp_out.bin is now in $PWD (which is $PROC_DIR)
# REP_WARP is defined as $PROC_DIR/rep_warp_out.bin, so no move is needed if compiled in PROC_DIR

echo "INFO: Cleaning up temporary compilation files..."
rm -f make_rep_warp.cpp make_rep_warp_exec
# kernelslist.g is intentionally kept in PROC_DIR as GCoM uses it.

popd >/dev/null # Back to original directory from PROC_DIR

if [[ ! -f "$REP_WARP" ]]; then # REP_WARP is $PROC_DIR/rep_warp_out.bin
    echo "ERROR: $REP_WARP was not generated in $PROC_DIR." >&2; exit 1;
fi
echo "INFO: Representative warp file is at $REP_WARP"
echo "-----------------------------------"


# -----------------------------------------------------------------------------
# 10) Run GCoM Simulation
# -----------------------------------------------------------------------------
echo "[Running GCoM Simulation]"
GCOM_EXEC="$GCOM_DIR/bin/GCoM"
GCOM_CONFIG="$GCOM_DIR/configs/RTX2060.config" # Adjust if using a different GCoM config
PROC_DIR_ABS="$(realpath "$PROC_DIR")" # GCoM might prefer absolute paths for -t

if [ ! -x "$GCOM_EXEC" ]; then echo "ERROR: GCoM executable not found or not executable: $GCOM_EXEC" >&2; exit 1; fi
if [ ! -f "$GCOM_CONFIG" ]; then echo "ERROR: GCoM config file not found: $GCOM_CONFIG" >&2; exit 1; fi
if [ ! -d "$PROC_DIR_ABS" ]; then echo "ERROR: Processed traces directory not found: $PROC_DIR_ABS" >&2; exit 1; fi

if [ ! -f "$REP_WARP" ]; then # REP_WARP is $PROC_DIR/rep_warp_out.bin
    echo "ERROR: Rep warp file '$REP_WARP' not found!" >&2
    # Check if kernelslist.g existed, which might indicate an issue in rep_warp generation
    if [ -f "$PROC_DIR/kernelslist.g" ] && [ ! -s "$PROC_DIR/kernelslist.g" ]; then
        echo "INFO: kernelslist.g was empty in $PROC_DIR, so rep_warp_out.bin might be empty or not created as expected." >&2
    fi
    exit 1;
fi

shopt -s nullglob
TRACE_FILES_IN_PROC=("$PROC_DIR_ABS"/*.traceg.gz)
shopt -u nullglob

if [ ${#TRACE_FILES_IN_PROC[@]} -eq 0 ]; then
    echo "WARNING: No trace files (*.traceg.gz) found in $PROC_DIR_ABS." >&2
    echo "INFO: GCoM will likely do nothing or error out. Check previous steps." >&2
    # Optionally, exit here if traces are mandatory for GCoM to run meaningfully
    # exit 1;
fi

echo "Starting GCoM..."
echo "INFO: GCoM command: $GCOM_EXEC -w 1 -r \"$REP_WARP\" -t \"$PROC_DIR_ABS\" -C \"$GCOM_CONFIG\""
# Run GCoM from WORKDIR or a dedicated log directory for GCoM outputs.
# GCoM often writes output files to its current working directory.
GCOM_LOG_DIR="$WORKDIR/gcom_simulation_logs"
mkdir -p "$GCOM_LOG_DIR"
GCOM_OUTPUT_LOG="$GCOM_LOG_DIR/gcom_gpt2_output_$(date +%Y%m%d_%H%M%S).log"

pushd "$GCOM_LOG_DIR" >/dev/null # Run GCoM from its log dir
"$GCOM_EXEC" -w 1 \
    -r "$REP_WARP" \
    -t "$PROC_DIR_ABS" \
    -C "$GCOM_CONFIG" > "$GCOM_OUTPUT_LOG" 2>&1 || {
        cat "$GCOM_OUTPUT_LOG" # Print GCoM log on failure
        echo "ERROR: GCoM simulation failed. Log: $GCOM_OUTPUT_LOG" >&2
        popd >/dev/null
        exit 1
    }
popd >/dev/null
echo "INFO: GCoM simulation finished. Output log: $GCOM_OUTPUT_LOG"
echo "-----------------------------------"

echo "✅ All done!"
exit 0
