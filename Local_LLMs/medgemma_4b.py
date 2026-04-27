"""
Gastric Cancer Pathology Extraction — MedGemma-4B-IT (vLLM)
MedGemma-4B is Google's medically-tuned Gemma model (4B multimodal IT variant).
Text-only inference is used here.
Usage:
    python extract_medgemma_4b.py --input data.xlsx --output results_medgemma4b.xlsx
    python extract_medgemma_4b.py --input data.xlsx --output results_medgemma4b.xlsx --tensor_parallel 1
"""

import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from config import MODEL_CONFIGS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, EXTRACTION_FIELDS
from utils import load_reports, parse_json_response, build_output_df, save_results

MODEL_KEY = "medgemma-4b"


def build_prompt(tokenizer, report: str) -> str:
    """
    MedGemma-4B is based on Gemma architecture.
    System role is merged into the user turn as Gemma does not support a separate system role.
    """
    combined_user = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(report=report)
    messages = [
        {"role": "user", "content": combined_user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_extraction(input_path: str, output_path: str, tensor_parallel: int = 1, batch_size: int = 8):
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
        # MedGemma uses HF gated model; ensure HF_TOKEN env var is set
        trust_remote_code=True,
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
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_extraction(args.input, args.output, args.tensor_parallel, args.batch_size)