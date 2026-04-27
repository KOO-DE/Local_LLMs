import json
import re
import shutil
from pathlib import Path
import pandas as pd


def load_reports(input_path: str) -> pd.DataFrame:
    """Load xlsx or csv. Expects: col0=Age, col1=Sex, col2=OpDate, col3=PathReport"""
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(input_path, header=0)
    df.columns = ["Age", "Sex", "OperationDate", "PathologyReport"] + list(df.columns[4:])
    return df


def parse_json_response(response_text: str) -> dict:
    """Extract JSON from model response, handling markdown fences."""
    text = response_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def parse_cot_response(response_text: str) -> dict:
    """CoT 전용 파서: <json> 태그 내부에서 JSON 추출, fallback은 일반 파서."""
    match = re.search(r"<json>\s*(\{.*?\})\s*</json>", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return parse_json_response(response_text)


def build_output_df(input_df: pd.DataFrame, results: list) -> pd.DataFrame:
    """기존 DataFrame에 추출 결과 열을 추가하여 반환."""
    from config import EXTRACTION_FIELDS
    result_df = pd.DataFrame(results, columns=EXTRACTION_FIELDS)
    # 기존 열 전체 유지 + 추출 결과 열 추가 (중복 열 덮어쓰기)
    for col in EXTRACTION_FIELDS:
        input_df[col] = result_df[col].values
    return input_df


def _backup(input_path: str) -> str:
    """원본 파일 백업 (.bak 생성), 백업 경로 반환."""
    backup_path = input_path + ".bak"
    shutil.copy2(input_path, backup_path)
    return backup_path


def save_results(df: pd.DataFrame, input_path: str, model_name: str, output_path: str = None):
    """
    추출 결과를 저장합니다.

    - output_path가 None이면 input 파일에 직접 덮어씁니다 (원본은 .bak으로 백업).
    - output_path가 지정되면 해당 경로에 별도 저장합니다.
    - xlsx / csv 형식 모두 지원하며, 확장자 기준으로 자동 판별합니다.
    """
    target = output_path if output_path else input_path
    ext = Path(target).suffix.lower()

    if not output_path:
        backup = _backup(input_path)
        print(f"  Backup: {backup}")

    if ext == ".csv":
        df.to_csv(target, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(target, index=False, engine="openpyxl")

    print(f"  Saved : {target}")