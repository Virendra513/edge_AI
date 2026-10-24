#!/usr/bin/env python3
"""
Batch runner for gguf_token_benchmark.py.

Runs the token-level benchmark on every .gguf model in a directory and stores
each model's results in its own sub-folder named after the model file:

    results/
        Llama-3.2-1B-Instruct.Q2_K/
            individual_results.csv
            token_level.csv
            summary_statistics.csv
            logs/
            console_output.txt
        Llama-3.2-1B-Instruct.Q4_K_M/
            ...
        combined_summary.csv        <- one row per model for easy comparison

Any argument not recognised here is forwarded unchanged to
gguf_token_benchmark.py (e.g. --prompt-file, --temperature, --batch-size).

Example:
    python3 gguf_batch_benchmark.py \
        --models-dir /home/virendra/work/LFT/modle321B/llama-3.2-1B-Instruct-gguf \
        --server-bin "$HOME/llama.cpp/build/bin/llama-server" \
        --output-dir "$HOME/gguf_batch_results" \
        --runs 5 --warmup-runs 1 --n-predict 128 --ignore-eos \
        --ctx-size 512 --threads 4 --gpu-layers 0 --timeout 180
"""

import argparse
import csv
import re
import subprocess
import sys
import time

from pathlib import Path


# Metrics pulled into combined_summary.csv: (summary metric name, short label)
COMBINED_METRICS = [
    ("ttft_ms", "ttft_ms"),
    ("tpot_ms", "tpot_ms"),
    ("e2e_latency_ms", "e2e_ms"),
    ("decode_tokens_per_second", "decode_tps"),
    ("overall_tokens_per_second", "overall_tps"),
    ("server_prompt_tokens_per_second", "prefill_tps"),
    ("peak_memory_mib", "peak_mem_mib"),
]

SPLIT_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


# ============================================================
# MODEL DISCOVERY
# ============================================================

def discover_models(models_dir, recursive, pattern):
    """Find .gguf files; for split models keep only the first shard."""
    glob = models_dir.rglob if recursive else models_dir.glob
    models = []

    for path in sorted(glob(pattern)):
        if not path.is_file():
            continue

        shard = SPLIT_SHARD.search(path.name)
        if shard and int(shard.group(1)) != 1:
            continue  # llama.cpp loads remaining shards automatically

        models.append(path)

    return models


def assign_folder_names(models):
    """Folder = file name without extension; disambiguate duplicates."""
    names = {}
    counts = {}

    for m in models:
        counts[m.stem] = counts.get(m.stem, 0) + 1

    for m in models:
        name = m.stem
        if counts[name] > 1:
            name = f"{m.parent.name}__{m.stem}"
        names[m] = name

    return names


# ============================================================
# RUN ONE MODEL
# ============================================================

def run_model(model, out_dir, args, passthrough):
    """Run the benchmark for one model; stream and save console output."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(args.benchmark_script),
        "--model", str(model),
        "--server-bin", str(args.server_bin),
        "--runs", str(args.runs),
        "--warmup-runs", str(args.warmup_runs),
        "--n-predict", str(args.n_predict),
        "--ctx-size", str(args.ctx_size),
        "--threads", str(args.threads),
        "--gpu-layers", str(args.gpu_layers),
        "--timeout", str(args.timeout),
        "--output-dir", str(out_dir),
    ]

    if args.ignore_eos:
        cmd.append("--ignore-eos")

    cmd.extend(passthrough)

    console_path = out_dir / "console_output.txt"
    start = time.perf_counter()

    with console_path.open("w", encoding="utf-8") as console:
        console.write("COMMAND: " + " ".join(cmd) + "\n\n")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in process.stdout:
                print("    " + line, end="")
                console.write(line)
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            raise

    return process.returncode, time.perf_counter() - start


# ============================================================
# COMBINED SUMMARY
# ============================================================

def read_summary(summary_csv):
    """Return {metric: row} from a model's summary_statistics.csv."""
    if not summary_csv.is_file():
        return {}

    with summary_csv.open(newline="", encoding="utf-8") as f:
        return {row["metric"]: row for row in csv.DictReader(f)}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_combined_row(model, folder_name, out_dir, return_code, wall_s):
    summary = read_summary(out_dir / "summary_statistics.csv")

    row = {
        "model_folder": folder_name,
        "model_file": model.name,
        "model_size_mib": round(model.stat().st_size / (1024 * 1024), 1),
        "exit_code": return_code,
        "wall_time_s": round(wall_s, 1),
        "successful_runs": (
            summary.get("_successful_runs", {}).get("n")
        ),
        "failed_runs": summary.get("_failed_runs", {}).get("n"),
    }

    for metric, label in COMBINED_METRICS:
        s = summary.get(metric, {})
        row[f"{label}_mean"] = to_float(s.get("mean"))
        row[f"{label}_median"] = to_float(s.get("median"))
        row[f"{label}_std"] = to_float(s.get("std"))

    for metric, label in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms")):
        s = summary.get(metric, {})
        row[f"{label}_p95"] = to_float(s.get("p95"))

    pooled = summary.get("itl_ms (pooled all tokens)", {})
    row["itl_pooled_p50_ms"] = to_float(pooled.get("p50"))
    row["itl_pooled_p95_ms"] = to_float(pooled.get("p95"))
    row["itl_pooled_p99_ms"] = to_float(pooled.get("p99"))
    row["itl_pooled_max_ms"] = to_float(pooled.get("max"))

    return row


def write_combined(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    tmp = path.with_suffix(".csv.tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in r.items()
            })

    tmp.replace(path)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Run gguf_token_benchmark.py on every model in a directory.",
        epilog="Unrecognised arguments are forwarded to gguf_token_benchmark.py.",
    )

    p.add_argument("--models-dir", required=True,
                   help="Directory containing .gguf models.")
    p.add_argument("--server-bin", required=True,
                   help="Path to llama-server executable.")
    p.add_argument("--output-dir", default="gguf_batch_results",
                   help="Root folder; one sub-folder per model is created.")
    p.add_argument("--benchmark-script",
                   default=str(Path(__file__).with_name(
                       "gguf_token_benchmark.py")),
                   help="Path to gguf_token_benchmark.py.")
    p.add_argument("--pattern", default="*.gguf",
                   help="Glob pattern for model files (default: *.gguf).")
    p.add_argument("--recursive", action="store_true",
                   help="Search sub-directories of --models-dir too.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip models whose summary_statistics.csv exists.")
    p.add_argument("--stop-on-failure", action="store_true",
                   help="Abort the batch when one model fails.")

    # Benchmark settings (defaults match your usual command)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-layers", type=int, default=0)
    p.add_argument("--timeout", type=int, default=180)

    return p.parse_known_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args, passthrough = parse_arguments()

    models_dir = Path(args.models_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    args.benchmark_script = Path(args.benchmark_script).expanduser().resolve()

    if not models_dir.is_dir():
        print(f"ERROR: models directory not found: {models_dir}",
              file=sys.stderr)
        return 2

    if not args.benchmark_script.is_file():
        print(f"ERROR: benchmark script not found: {args.benchmark_script}",
              file=sys.stderr)
        return 2

    models = discover_models(models_dir, args.recursive, args.pattern)

    if not models:
        print(f"ERROR: no models matching '{args.pattern}' in {models_dir}",
              file=sys.stderr)
        return 2

    folder_names = assign_folder_names(models)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GGUF BATCH BENCHMARK")
    print("=" * 70)
    print(f"Models dir : {models_dir}")
    print(f"Output     : {output_root}")
    print(f"Models     : {len(models)}")
    for m in models:
        print(f"  - {m.name}  ->  {folder_names[m]}/")
    print("=" * 70)

    combined = []
    failures = 0
    combined_path = output_root / "combined_summary.csv"

    try:
        for i, model in enumerate(models, 1):
            name = folder_names[model]
            out_dir = output_root / name

            print(f"\n[{i}/{len(models)}] {model.name}")

            if args.skip_existing and (
                    out_dir / "summary_statistics.csv").is_file():
                print("    skipped (results already exist)")
                combined.append(
                    build_combined_row(model, name, out_dir, None, 0.0))
                write_combined(combined_path, combined)
                continue

            code, wall = run_model(model, out_dir, args, passthrough)

            if code != 0:
                failures += 1
                print(f"    FAILED (exit code {code}); see "
                      f"{out_dir / 'console_output.txt'}")
            else:
                print(f"    done in {wall:.1f}s")

            combined.append(build_combined_row(model, name, out_dir, code, wall))
            write_combined(combined_path, combined)  # partial save

            if code != 0 and args.stop_on_failure:
                print("Stopping: --stop-on-failure set.", file=sys.stderr)
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Combined summary saved for finished models.",
              file=sys.stderr)

    write_combined(combined_path, combined)

    print("\n" + "=" * 70)
    print("BATCH COMPLETED")
    print("=" * 70)
    print(f"Models processed : {len(combined)}/{len(models)}")
    print(f"Failed           : {failures}")
    print(f"Combined summary : {combined_path}")
    print("=" * 70)

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())