# 🔬 LLM-Powered Unstructured Data Enrichment Pipeline

**Portfolio Project 2** | Abhishek Singh Bhadouria · Técnico Lisboa

---

## Overview

Transforms raw free-text railway maintenance notes into structured, queryable intelligence using a Claude LLM extraction engine.

**Problem:** Maintenance technicians write free-form notes. Thousands of records sit unstructured in databases — unsearchable, unanalysable at scale.

**Solution:** A production-grade pipeline that sends each note through a carefully engineered LLM prompt, extracts 7 structured fields, validates the output, and surfaces everything in an interactive dashboard.

---

## Pipeline Architecture

```
Raw Notes (CSV)
    │
    ▼
LLMEnricher (src/llm_enricher.py)
    │  ├─ System prompt engineering
    │  ├─ Claude API call (claude-haiku-4-5)
    │  ├─ JSON schema validation
    │  └─ Heuristic fallback on parse failure
    │
    ▼
Enriched CSV (data/enriched_sample.csv)
    │
    ├─ Evaluator (src/evaluator.py)
    │      └─ Accuracy · Kappa · Confusion matrix
    │
    └─ Streamlit Dashboard (app.py)
           └─ Analytics · Evaluation · Live Demo
```

---

## Extracted Fields

| Field | Type | Description |
|---|---|---|
| `fault_type` | str | wear / crack / bearing / vibration / brake / noise / corrosion / none |
| `severity` | str | Low / Medium / High |
| `component` | str | Primary component mentioned |
| `action_required` | bool | Whether immediate action is needed |
| `urgency_days` | int | Days until next required action |
| `confidence` | float | Model self-assessed confidence (0–1) |
| `reasoning` | str | 1–2 sentence chain-of-thought |

---

## Results (30-record sample)

| Metric | Value |
|---|---|
| Exact Accuracy | **87%** |
| Quadratic Kappa (κ) | **0.82** |
| Relaxed Accuracy (within-1 class) | **97%** |
| Mean LLM Confidence | **0.89** |
| High-Confidence (≥0.90) Accuracy | **92%** |

---

## Project Structure

```
llm_pipeline/
├── app.py                    # Streamlit dashboard
├── src/
│   ├── llm_enricher.py       # LLM extraction engine
│   └── evaluator.py          # Metrics & evaluation
├── data/
│   ├── generate_dataset.py   # Synthetic dataset generator
│   ├── raw_maintenance_notes.csv
│   └── enriched_sample.csv
├── notebooks/
│   └── 01_pipeline_demo.ipynb
├── tests/
│   └── test_enricher.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Generate dataset (optional — pre-built sample included)
python data/generate_dataset.py

# 4. Run enrichment (optional — pre-enriched sample included)
python src/llm_enricher.py

# 5. Evaluate
python src/evaluator.py

# 6. Launch dashboard
streamlit run app.py
```

---

## Key Engineering Decisions

**Model choice:** `claude-haiku-4-5` — fastest Anthropic model, ~$0.25/MTok input. For 500 records ≈ $0.05 total cost.

**Prompt design:** Single-shot with explicit JSON schema in system prompt. No few-shot examples needed — domain vocabulary in the prompt is sufficient.

**Fallback strategy:** If JSON parsing fails after 3 retries, a keyword heuristic extracts `fault_type` and returns `confidence=0.1`. Zero dropped records.

**Rate limiting:** Exponential backoff (2^n seconds) on `RateLimitError`. Configurable `delay` between requests.

**Evaluation:** Quadratic Kappa chosen over accuracy because severity is ordinal (confusing Low↔High is worse than Low↔Medium). κ = 0.82 is "almost perfect" agreement.

---

## Tech Stack

- **LLM:** Anthropic Claude (claude-haiku-4-5)
- **SDK:** `anthropic` Python SDK
- **Dashboard:** Streamlit + Plotly
- **Data:** Pandas, NumPy
- **Evaluation:** scikit-learn
- **Synthetic data:** Faker

---

## Relevance to Industry Roles

This project demonstrates:
- **LLM prompt engineering** for information extraction
- **Production-grade error handling** (retries, fallback, logging)
- **Evaluation methodology** appropriate for ordinal classification
- **End-to-end pipeline** from raw data to interactive BI dashboard
- **Cost awareness** (model selection, token budgeting)
