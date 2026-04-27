"""
Run all 5 extraction models sequentially.
Each model is launched as a subprocess to fully release GPU memory between runs.

Usage:
    python run_all.py --input data.xlsx --output_dir results/ --tensor_parallel 1
    python run_all.py --input data.xlsx --output_dir results/ --tensor_parallel 2 --models llama qwen gemma3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

MODELS = {
    "llama":     ("extract_llama31_8b.py",  "results_llama31_8b.csv"),
    "qwen":      ("extract_qwen25_7b.py",   "results_qwen25_7b.csv"),
    "gemma3":    ("extract_gemma3_12b.py",  "results_gemma3_12b.csv"),
    "med42":     ("extract_med42_8b.py",    "results_med42_8b.csv"),
    "medgemma":  ("extract_medgemma_4b.py", "results_medgemma_4b.csv"),
}

BASE_DIR = Path(__file__).parent


def run_model(script: str, input_path: str, output_path, tensor_parallel: int, batch_size: int):
    cmd = [
        sys.executable,
        str(BASE_DIR / script),
        "--input", input_path,
        "--tensor_parallel", str(tensor_parallel),
        "--batch_size", str(batch_size),
    ]
    if output_path:
        cmd += ["--output", output_path]
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else f"FAILED (code={result.returncode})"
    print(f"{script}: {status} [{elapsed:.1f}s]")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all gastric extraction models")
    parser.add_argument("--input", required=True, help="Input xlsx path")
    parser.add_argument("--output_dir", default=None, help="Output directory. If omitted, writes back to input file")
    parser.add_argument("--tensor_parallel", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Which models to run (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    total_start = time.time()

    for key in args.models:
        script, out_file = MODELS[key]
        output_path = str(out_dir / out_file) if out_dir else None
        success = run_model(script, args.input, output_path, args.tensor_parallel, args.batch_size)
        summary[key] = "✓ OK" if success else "✗ FAILED"

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, status in summary.items():
        print(f"  {model:<12} {status}")
    print(f"\nTotal time: {total_elapsed/60:.1f} min")
    dest = out_dir.resolve() if out_dir else Path(args.input).resolve()
    print(f"Results in: {dest}")


if __name__ == "__main__":
    main()