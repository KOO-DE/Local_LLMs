# Gastric Cancer Pathology Extraction

위암 병리기록지에서 17개 항목을 LLM으로 자동 추출하는 파이프라인.

## 파일 구조

```
gastric_extraction/
├── config.py              # 모델 설정 + 프롬프트 정의
├── utils.py               # 공통 유틸 (Excel I/O, JSON 파싱)
├── extract_llama31_8b.py  # Llama-3.1-8B-Instruct
├── extract_qwen25_7b.py   # Qwen2.5-7B-Instruct
├── extract_gemma3_12b.py  # Gemma-3-12B-IT
├── extract_med42_8b.py    # Med42-v2-8B
├── extract_medgemma_4b.py # MedGemma-4B-IT
├── run_all.py             # 전체 모델 순차 실행
└── requirements.txt
```

## 입력 파일 형식 (xlsx)

| 열 | 내용 |
|----|------|
| 1열 | 나이 |
| 2열 | 성별 |
| 3열 | 수술날짜 |
| 4열 | 병리기록지 (텍스트) |

## 출력 항목 (5열~21열)

TumorLocation, TumorCircumference, TumorSize, Histologic_type, Lauren_type,
Differentiation, GrossType, ProximalMargin, DistalMargin,
T_stage, N_stage, M_stage, Staging,
MetastaticLymphNode, HarvestedLymphNode, LymphovascularInvasion, PerineuralInvasion

## 설치

```bash
pip install -r requirements.txt
```

### Gated 모델 접근 (Llama, MedGemma)

```bash
huggingface-cli login
# 또는
export HF_TOKEN=hf_xxxx
```

## 실행 방법

### 모델별 개별 실행

```bash
# Llama-3.1-8B (A6000 1대)
python extract_llama31_8b.py --input data.xlsx --output results_llama.xlsx --tensor_parallel 1

# Qwen2.5-7B
python extract_qwen25_7b.py --input data.xlsx --output results_qwen.xlsx --tensor_parallel 1

# Gemma3-12B (메모리 여유가 있다면 tp=1로 충분)
python extract_gemma3_12b.py --input data.xlsx --output results_gemma3.xlsx --tensor_parallel 1

# Med42-v2-8B
python extract_med42_8b.py --input data.xlsx --output results_med42.xlsx --tensor_parallel 1

# MedGemma-4B
python extract_medgemma_4b.py --input data.xlsx --output results_medgemma.xlsx --tensor_parallel 1
```

### 전체 모델 한번에 실행

```bash
# 기본 (모든 모델, tp=1)
python run_all.py --input data.xlsx --output_dir results/

# GPU 2대 사용
python run_all.py --input data.xlsx --output_dir results/ --tensor_parallel 2

# 특정 모델만 선택
python run_all.py --input data.xlsx --output_dir results/ --models llama qwen med42
```

## GPU 사용 권장 설정 (A6000 48GB 기준)

| 모델 | VRAM | 권장 tp | batch_size |
|------|------|---------|------------|
| Llama-3.1-8B | ~18GB | 1 | 8 |
| Qwen2.5-7B | ~17GB | 1 | 8 |
| Gemma3-12B | ~26GB | 1 | 4 |
| Med42-v2-8B | ~18GB | 1 | 8 |
| MedGemma-4B | ~10GB | 1 | 8 |

- `--tensor_parallel 2`로 설정 시 A6000 2대를 하나의 모델에 사용 (더 큰 batch_size 가능)
- 각 모델은 subprocess로 순차 실행되어 GPU 메모리가 완전히 해제된 후 다음 모델 실행됨