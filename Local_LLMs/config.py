EXTRACTION_FIELDS = [
    "TumorLocation",
    "TumorCircumference",
    "TumorSize",
    "Histologic_type",
    "Lauren_type",
    "Differentiation",
    "GrossType",
    "ProximalMargin",
    "DistalMargin",
    "T_stage",
    "N_stage",
    "M_stage",
    "Staging",
    "MetastaticLymphNode",
    "HarvestedLymphNode",
    "LymphovascularInvasion",
    "PerineuralInvasion",
]

# ─────────────────────────────────────────────
# (1) Zero-shot
# ─────────────────────────────────────────────
ZERO_SHOT_SYSTEM = """You are a pathology information extraction engine for gastric cancer reports.
Extract ONLY the fields listed below. Return a single valid JSON object with no preamble,
no explanation, and no markdown. If a value cannot be determined from the text, use null.

Fields and allowed values:
- TumorLocation       : free text (e.g., antrum, body, cardia, fundus, pylorus)
- TumorCircumference  : free text (e.g., anterior wall, posterior wall, lesser curvature, greater curvature)
- TumorSize           : number in cm (e.g., 3.5) — numeric only, no unit string
- Histologic_type     : free text (e.g., tubular adenocarcinoma, signet ring cell carcinoma, mucinous carcinoma)
- Lauren_type         : one of [intestinal, diffuse, mixed]
- Differentiation     : one of [well, moderately, poorly, undifferentiated]
- GrossType           : one of [Borrmann Type 1, Borrmann Type 2, Borrmann Type 3, Borrmann Type 4,
                        EGC Type I, EGC Type IIa, EGC Type IIb, EGC Type IIc, EGC Type III]
- ProximalMargin      : one of [free, involved, cannot be assessed] — append distance in cm if stated
- DistalMargin        : one of [free, involved, cannot be assessed] — append distance in cm if stated
- T_stage             : one of [T1a, T1b, T2, T3, T4a, T4b]
- N_stage             : one of [N0, N1, N2, N3a, N3b]
- M_stage             : one of [M0, M1]
- Staging             : one of [IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IV]
- MetastaticLymphNode : integer (number of metastatic lymph nodes)
- HarvestedLymphNode  : integer (number of retrieved/harvested lymph nodes)
- LymphovascularInvasion : one of [present, absent]
- PerineuralInvasion     : one of [present, absent]"""

ZERO_SHOT_USER = """Extract from the following gastric cancer pathology report:

{report}"""


# ─────────────────────────────────────────────
# (2) Few-shot
# ─────────────────────────────────────────────
FEW_SHOT_SYSTEM = """You are a pathology information extraction engine for gastric cancer reports.
Return ONLY a valid JSON object. No explanation, no markdown. Use null if not found."""

FEW_SHOT_EXAMPLES = [
    # Example 1: 완전한 정보, EGC
    {
        "report": """Specimen: Total gastrectomy
Gross type: EGC Type IIc
Location: Antrum, posterior wall
Tumor size: 1.8 x 1.5 cm
Histologic type: Tubular adenocarcinoma, moderately differentiated
Lauren: Intestinal type
Lymphovascular invasion: Not identified
Perineural invasion: Not identified
Resection margins: Proximal margin free (5.2 cm), Distal margin free (3.1 cm)
Lymph node: No metastasis in 0 of 32 regional lymph nodes
pT1b, pN0, pM0, Stage IA""",
        "answer": """{
  "TumorLocation": "antrum",
  "TumorCircumference": "posterior wall",
  "TumorSize": 1.8,
  "Histologic_type": "tubular adenocarcinoma",
  "Lauren_type": "intestinal",
  "Differentiation": "moderately",
  "GrossType": "EGC Type IIc",
  "ProximalMargin": "free, 5.2cm",
  "DistalMargin": "free, 3.1cm",
  "T_stage": "T1b",
  "N_stage": "N0",
  "M_stage": "M0",
  "Staging": "IA",
  "MetastaticLymphNode": 0,
  "HarvestedLymphNode": 32,
  "LymphovascularInvasion": "absent",
  "PerineuralInvasion": "absent"
}"""
    },
    # Example 2: LVI/PNI present, Lauren 미기재 → 조직형으로 추론
    {
        "report": """Gross type: Borrmann Type 3
Location: Body, lesser curvature
Size: 5.5 x 4.0 cm
Histology: Poorly cohesive carcinoma with signet ring cells
Lauren classification: not described
LVI: Present
PNI: Present
Proximal resection margin: free (1.8 cm)
Distal resection margin: free (4.0 cm)
Lymph node metastasis: 7/35
pT3 N2 M0, Stage IIIA""",
        "answer": """{
  "TumorLocation": "body",
  "TumorCircumference": "lesser curvature",
  "TumorSize": 5.5,
  "Histologic_type": "poorly cohesive carcinoma with signet ring cells",
  "Lauren_type": "diffuse",
  "Differentiation": "poorly",
  "GrossType": "Borrmann Type 3",
  "ProximalMargin": "free, 1.8cm",
  "DistalMargin": "free, 4.0cm",
  "T_stage": "T3",
  "N_stage": "N2",
  "M_stage": "M0",
  "Staging": "IIIA",
  "MetastaticLymphNode": 7,
  "HarvestedLymphNode": 35,
  "LymphovascularInvasion": "present",
  "PerineuralInvasion": "present"
}"""
    },
    # Example 3: null 필드 다수, margin 거리 없음
    {
        "report": """Procedure: Subtotal gastrectomy
Tumor site: Cardia
Gross finding: Borrmann Type 2, size 3.2 x 2.8 cm
Differentiation: moderately differentiated
Lauren type: mixed type
Margins: both proximal and distal margins are free
LVI: Focally present
PNI: Not identified
LN: metastasis in 2 of 18 lymph nodes
Pathologic stage: pT2 pN1 pM0, Stage IB""",
        "answer": """{
  "TumorLocation": "cardia",
  "TumorCircumference": null,
  "TumorSize": 3.2,
  "Histologic_type": null,
  "Lauren_type": "mixed",
  "Differentiation": "moderately",
  "GrossType": "Borrmann Type 2",
  "ProximalMargin": "free",
  "DistalMargin": "free",
  "T_stage": "T2",
  "N_stage": "N1",
  "M_stage": "M0",
  "Staging": "IB",
  "MetastaticLymphNode": 2,
  "HarvestedLymphNode": 18,
  "LymphovascularInvasion": "present",
  "PerineuralInvasion": "absent"
}"""
    },
]

def build_few_shot_user(report: str) -> str:
    """Few-shot 예시를 하나의 user 메시지로 조합."""
    parts = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"### Example {i}\n[PATHOLOGY REPORT]\n{ex['report']}\n\n[OUTPUT]\n{ex['answer']}")
    parts.append(f"### Now extract from this report:\n[PATHOLOGY REPORT]\n{report}\n\n[OUTPUT]")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# (3) Chain-of-Thought
# ─────────────────────────────────────────────
COT_SYSTEM = """You are a medical pathology extraction assistant for gastric cancer reports.
Follow this EXACT two-step process:

STEP 1 — ANALYSIS:
Read the report carefully. For each field, note the exact phrase or sentence that supports
your answer. If a field is ambiguous or absent, explicitly state why.
Pay special attention to:
  - Lauren type: may need to be inferred from histologic type
    (signet ring cell / poorly cohesive → diffuse; tubular / papillary → intestinal)
  - Lymph node counts: extract both numbers from formats like "3/28" or "3 of 28"
  - Margins: status (free/involved) and distance may appear separately in the text
  - Staging: if not stated, derive from pT + pN + pM using AJCC 8th edition rules

STEP 2 — EXTRACTION:
Output a single JSON object using exactly these keys.
Use null for any field not determinable from the report.
Numeric fields (TumorSize, MetastaticLymphNode, HarvestedLymphNode) must be numbers, not strings.

Output format (always end your response with the JSON block):
<analysis>
[your field-by-field reasoning here]
</analysis>
<json>
{
  "TumorLocation": ...,
  "TumorCircumference": ...,
  "TumorSize": ...,
  "Histologic_type": ...,
  "Lauren_type": ...,
  "Differentiation": ...,
  "GrossType": ...,
  "ProximalMargin": ...,
  "DistalMargin": ...,
  "T_stage": ...,
  "N_stage": ...,
  "M_stage": ...,
  "Staging": ...,
  "MetastaticLymphNode": ...,
  "HarvestedLymphNode": ...,
  "LymphovascularInvasion": ...,
  "PerineuralInvasion": ...
}
</json>"""

COT_USER = """Extract from the following gastric cancer pathology report:

{report}"""


# ─────────────────────────────────────────────
# Prompt selector
# ─────────────────────────────────────────────
PROMPT_TYPES = ["zero_shot", "few_shot", "cot"]

def get_messages(prompt_type: str, report: str) -> list:
    """prompt_type에 따라 messages 리스트 반환."""
    if prompt_type == "zero_shot":
        return [
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user",   "content": ZERO_SHOT_USER.format(report=report)},
        ]
    elif prompt_type == "few_shot":
        return [
            {"role": "system", "content": FEW_SHOT_SYSTEM},
            {"role": "user",   "content": build_few_shot_user(report)},
        ]
    elif prompt_type == "cot":
        return [
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user",   "content": COT_USER.format(report=report)},
        ]
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}. Choose from {PROMPT_TYPES}")


# ─────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "llama3.1-8b": {
        "model_id": "/data/kude/models/Llama-3.1-8B-Instruct",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    },
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
    },
    "gemma3-12b": {
        "model_id": "google/gemma-3-12b-it",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    },
    "med42-v2-8b": {
        "model_id": "m42-health/med42-v2-8B",
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    },
    "medgemma-4b": {
        "model_id": "google/medgemma-4b-it",
        "gpu_memory_utilization": 0.80,
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
    },
}