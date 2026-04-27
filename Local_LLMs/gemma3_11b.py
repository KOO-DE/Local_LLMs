"""
Gastric Cancer Pathology Extraction — Gemma-3-12B-IT (vLLM)
Note: Gemma3-12B requires ~24GB VRAM per GPU. Use tensor_parallel=1 on A6000 (48GB).
Usage:
    python extract_gemma3_12b.py --input data.xlsx --output results_gemma3.xlsx
    python extract_gemma3_12b.py --input data.xlsx --output results_gemma3.xlsx --tensor_parallel 2
"""

import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from config import MODEL_CONFIGS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, EXTRACTION_FIELDS
from utils import load_reports, parse_json_response, build_output_df, save_results

MODEL_KEY = "gemma3-12b"


def build_prompt(tokenizer, report: str) -> str:
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(report=report)},
    ]
    # Gemma3 instruct does not use a separate system role
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_extraction(input_path: str, output_path: str, tensor_parallel: int = 1, batch_size: int = 4):
    from vllm import LLM, SamplingParams

    cfg = MODEL_CONFIGS[MODEL_KEY].copy()
    cfg["tensor_parallel_size"] = tensor_parallel

    print(f"[{MODEL_KEY}] Loading {cfg['model_id']} (tensor_parallel={tensor_parallel})")
    llm = LLM(
        model=cfg["model_id"],
        tensor_parallel_size=cfg["tensor_parallel_size"],
        gpu_memory_utilization=cfg["gpu_memory_utilization"],
        max_model_len=cfg["max_model_len"],
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["<end_of_turn>", "<eos>"],
    )

    df = load_reports(input_path)
    reports = df["PathologyReport"].fillna("").tolist()
    prompts = [build_prompt(tokenizer, r) for r in reports]

    print(f"Processing {len(prompts)} reports (batch_size={batch_size})...")
    results = []
    start = time.time()

    for i in tqdm(range(0, len(prompts), batch_size), desc=MODEL_KEY):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            parsed = parse_json_response(out.outputs[0].text)
            results.append({field: parsed.get(field) for field in EXTRACTION_FIELDS})

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s ({elapsed / len(reports):.2f}s/report)")

    out_df = build_output_df(df, results)
    save_results(out_df, input_path=args.input, model_name=MODEL_KEY, output_path=args.output if args.output else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input xlsx path")
    parser.add_argument("--output", default=None, help="Output path (optional). If omitted, writes back to input file)")
    parser.add_argument("--tensor_parallel", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    run_extraction(args.input, args.output, args.tensor_parallel, args.batch_size)