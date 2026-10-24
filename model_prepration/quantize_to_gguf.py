# import and setups

import os, sys, json, gc, time, shutil, subprocess, glob
from pathlib import Path
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed

set_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
if DEVICE == "cuda":
    print("GPU :", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")


# Model Directory and Paths
'''
`MODEL_DIR` is the source pre-trained weights. 
`HF_F16_DIR` is a clean FP16 copy of the source (used as input to the GGUF converter). 
`GGUF_DIR` holds the GGUF F16 file plus every quantized variant.
'''
MODEL_DIR   = "/llama-3.2-1B-Instruct"
HF_F16_DIR  = "/llama-3.2-1B-Instruct-f16"
GGUF_DIR    = "/llama-3.2-1B-Instruct-gguf"

assert os.path.isdir(MODEL_DIR), f"Model dir not found: {MODEL_DIR}"

os.makedirs(GGUF_DIR, exist_ok=True)

print("MODEL_DIR  :", MODEL_DIR)
print("HF_F16_DIR :", HF_F16_DIR)
print("GGUF_DIR   :", GGUF_DIR)    


# Save the HF model in FP16

'''
We reload the original model in FP16 (so the conversion to GGUF F16 is exact), and save the result with the tokenizer. 
We don't need this loaded for the GGUF step, we just need the FP16 safetensors on disk. 
Free the GPU memory before invoking the converter.
'''

if os.path.isdir(HF_F16_DIR):
    print(f"Removing existing dir: {HF_F16_DIR}")
    shutil.rmtree(HF_F16_DIR)
os.makedirs(HF_F16_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.float16 if DEVICE == "cuda" else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model.to(DEVICE)
model.eval()

fp16_params = sum(p.numel() for p in model.parameters())
fp16_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Model class    : {type(model).__name__}")
print(f"FP16 params    : {fp16_params:,}")
print(f"FP16 memory    : {fp16_bytes / 1024**3:.2f} GB")


t0 = time.time()
model.save_pretrained(HF_F16_DIR, safe_serialization=True)
tokenizer.save_pretrained(HF_F16_DIR)
dt = time.time() - t0
print(f"Saved FP16 HF model in {dt:.2f} s")
print(f"\nFiles in {HF_F16_DIR}:")
for f in sorted(os.listdir(HF_F16_DIR)):
    p = os.path.join(HF_F16_DIR, f)
    if os.path.isfile(p):
        print(f"  {f:35s} {os.path.getsize(p) / 1024**2:8.2f} MB")



del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
if DEVICE == "cuda":
    print("VRAM after FP16 free:",
          round(torch.cuda.memory_allocated() / 1024**3, 2), "GB allocated,",
          round(torch.cuda.memory_reserved()  / 1024**3, 2), "GB reserved")


# Locate the llama.cpp converter and quantizer
'''We need two llama.cpp binaries:

- `convert_hf_to_gguf.py` - turns the FP16 HF checkpoint into a GGUF F16 file.
- `llama-quantize` (CLI) -  quantizes GGUF F16 into Q4_K_M and other k-quant types.
'''

CANDIDATE_CONVERTER_PATHS = [
    "/llama.cpp/convert_hf_to_gguf.py",
    "/llama.cpp/convert_hf_to_gguf.py",
    "/llama.cpp/gguf-py/scripts/convert_hf_to_gguf.py",
    "/convert-hf-to-gguf.py",
    "/opt/llama.cpp/convert_hf_to_gguf.py",
]
CANDIDATE_QUANTIZER_NAMES = ["llama-quantize", "quantize"]
CANDIDATE_QUANTIZER_DIRS = [
    "/llama.cpp/build/bin",
    "/llama.cpp",
    "llama.cpp/build",
    "/llama.cpp/bin",
    "/usr/local/bin",
    "/opt/llama.cpp/build/bin",
]

# Convert HF FP16 to  GGUF F16

CONVERTER = "/llama.cpp/convert_hf_to_gguf.py"

assert os.path.isfile(CONVERTER), f"Converter not found: {CONVERTER}"
print(f"Using converter: {CONVERTER}")

assert CONVERTER is not None, "convert_hf_to_gguf.py not found — see install hints above."

GGUF_F16 = os.path.join(GGUF_DIR, "Llama-3.2-1B-Instruct.F16.gguf")
if os.path.exists(GGUF_F16):
    print(f"Removing existing file: {GGUF_F16}")
    os.remove(GGUF_F16)

cmd = [
    sys.executable, CONVERTER,
    HF_F16_DIR,      # input: HF checkpoint directory
    "--outfile", GGUF_F16,
    "--outtype", "f16",
]
print("Running:", " ".join(cmd))
t0 = time.time()
res = subprocess.run(cmd, capture_output=True, text=True)
dt = time.time() - t0
print(f"\n--- converter stdout (tail) ---\n{res.stdout[-1500:]}")
if res.stderr.strip():
    print(f"\n--- converter stderr (tail) ---\n{res.stderr[-1500:]}")
print(f"\nexit code: {res.returncode}  ({dt:.1f} s)")
assert res.returncode == 0 and os.path.exists(GGUF_F16), "Conversion failed."
print(f"\nGGUF F16 file: {GGUF_F16}  ({os.path.getsize(GGUF_F16)/1024**3:.3f} GB)")


# Quantize GGUF F16  to all standard k-quant types

QUANTIZER = "/home/virendra/work/llama.cpp/build/bin/llama-quantize"

assert os.path.isfile(QUANTIZER), f"Quantizer not found: {QUANTIZER}"
assert os.access(QUANTIZER, os.X_OK), f"Quantizer is not executable: {QUANTIZER}"

print(f"Using quantizer: {QUANTIZER}")

# Every standard k-quant type llama.cpp supports. Order is from largest (best quality)
# to smallest (most aggressive compression). Edit this list to limit to a subset, e.g.
#   QUANT_TYPES = ["Q4_K_M"]
QUANT_TYPES = [
    "Q8_0",
    "Q6_K",
    "Q5_K_M", "Q5_K_S",
    "Q4_K_M", "Q4_K_S",
    "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "Q2_K",
]

results = []
for qt in QUANT_TYPES:
    out_path = os.path.join(GGUF_DIR, f"Llama-3.2-1B-Instruct.{qt}.gguf")
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [QUANTIZER, GGUF_F16, out_path, qt]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    ok = (res.returncode == 0) and os.path.exists(out_path)
    sz_mb = os.path.getsize(out_path) / 1024**2 if ok else None
    results.append((qt, ok, dt, sz_mb, (res.stdout + res.stderr).strip().splitlines()[-1] if res.stdout or res.stderr else ""))
    status = f"{sz_mb:7.2f} MB  ({dt:5.1f} s)" if ok else f"FAILED ({dt:.1f} s)"
    print(f"  {qt:6s}  {status}")

print(f"\n{sum(1 for _,ok,_,_,_ in results if ok)} / {len(results)} succeeded.")



# Verify all GGUF files

ORIGINAL_F16_BYTES = fp16_bytes   # 2.30 GB logical, ~2.30 GB on disk as a single safetensors file

print(f"{'file':50s} {'size':>12s} {'vs FP16':>10s} {'vs HF-src':>12s}")
print("-" * 90)
gguf_files = sorted(glob.glob(os.path.join(GGUF_DIR, "*.gguf")))
for p in gguf_files:
    sz = os.path.getsize(p)
    name = os.path.basename(p)
    vs_f16 = sz / ORIGINAL_F16_BYTES
    # source HF safetensors size, with tie_word_embeddings the embed is duplicated
    src_safetensors = os.path.getsize(os.path.join(MODEL_DIR, "model.safetensors"))
    vs_src = sz / src_safetensors
    print(f"{name:50s} {sz/1024**3:>10.3f} GB {vs_f16:>9.2f}x {vs_src:>11.2f}x")


# Smoke-load the Q4_K_M GGUF

Q4KM_PATH = os.path.join(GGUF_DIR, "Llama-3.2-1B-Instruct.Q4_K_M.gguf")
assert os.path.exists(Q4KM_PATH), f"Q4_K_M file not found: {Q4KM_PATH}"

if HAVE_GGUF_PY:
    reader = GGUFReader(Q4KM_PATH)
    n_tensors = len(reader.tensors)
    n_bytes   = sum(len(t.data) for t in reader.tensors)
    print(f"Q4_K_M GGUF tensors : {n_tensors}")
    print(f"Q4_K_M tensor bytes : {n_bytes/1024**2:.2f} MB (uncompressed view)")

    # First few tensor names — should look like llama layers
    print("\nFirst 10 tensors:")
    for t in reader.tensors[:10]:
        print(f"  {t.name:55s} shape={tuple(t.shape)} type={t.tensor_type.name}")
else:
    # Fall back: just print the file size and trust llama-quantize
    sz = os.path.getsize(Q4KM_PATH)
    print(f"Q4_K_M file exists: {Q4KM_PATH}")
    print(f"Q4_K_M file size : {sz/1024**3:.3f} GB ({sz/1024**2:.2f} MB)")