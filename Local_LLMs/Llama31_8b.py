"""
Gastric Cancer Pathology Extraction — Llama-3.1-8B-Instruct (vLLM)
Usage:
    python extract_llama31_8b.py --input data.xlsx --output results_llama_zeroshot.csv --prompt_type zero_shot
    python extract_llama31_8b.py --input data.xlsx --output results_llama_fewshot.csv --prompt_type few_shot
    python extract_llama31_8b.py --input data.xlsx --output results_llama_cot.csv     --prompt_type cot
"""

import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from config import MODEL_CONFIGS, EXTRACTION_FIELDS, PROMPT_TYPES, get_messages
from utils import load_reports, parse_json_response, parse_cot_response, build_output_df, save_results

MODEL_KEY = "llama3.1-8b"


def build_prompt(tokenizer, report: str, prompt_type: str) -> str:
    messages = get_messages(prompt_type, report)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_extraction(input_path: str, output_path: str, prompt_type: str,
                   tensor_parallel: int = 1, batch_size: int = 8):
    from vllm import LLM, SamplingParams

    cfg = MODEL_CONFIGS[MODEL_KEY].copy()
    cfg["tensor_parallel_size"] = tensor_parallel

    # CoT는 reasoning 텍스트가 길어서 max_tokens 늘림
    max_tokens = 1024 if prompt_type == "cot" else 512
    parser = parse_cot_response if prompt_type == "cot" else parse_json_response

    print(f"[{MODEL_KEY}] prompt_type={prompt_type} | max_tokens={max_tokens} | tensor_parallel={tensor_parallel}")
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
        max_tokens=max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    df = load_reports(input_path)
    reports = df["PathologyReport"].fillna("").tolist()
    prompts = [build_prompt(tokenizer, r, prompt_type) for r in reports]

    print(f"Processing {len(prompts)} reports (batch_size={batch_size})...")
    results = []
    start = time.time()

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{MODEL_KEY}/{prompt_type}"):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            parsed = parser(out.outputs[0].text)
            results.append({field: parsed.get(field) for field in EXTRACTION_FIELDS})

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s ({elapsed / len(reports):.2f}s/report)")

    out_df = build_output_df(df, results)
    save_results(out_df, input_path=input_path, model_name=MODEL_KEY, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input xlsx or csv path")
    parser.add_argument("--output", default=None, help="Output path (optional). If omitted, writes back to input file")
    parser.add_argument("--prompt_type", default="zero_shot", choices=PROMPT_TYPES,
                        help="Prompt strategy: zero_shot / few_shot / cot (default: zero_shot)")
    parser.add_argument("--tensor_parallel", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_extraction(args.input, args.output, args.prompt_type, args.tensor_parallel, args.batch_size)