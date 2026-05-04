"""
LLM Extraction Engine
Uses Claude API to extract structured fields from unstructured maintenance notes.

Extracted fields per record:
  - fault_type     : primary fault category (str)
  - severity       : Low / Medium / High
  - component      : affected component name
  - action_required: True / False
  - urgency_days   : estimated days to next required action (int or null)
  - confidence     : model confidence 0.0–1.0
  - reasoning      : brief chain-of-thought
"""

import json
import time
import logging
import os
from typing import Optional
import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a railway maintenance data analyst. Your job is to extract structured
information from free-text maintenance notes written by field technicians.

Return ONLY a valid JSON object with these exact keys:
{
  "fault_type": "<one of: wear | crack | bearing | vibration | brake | noise | corrosion | none>",
  "severity": "<one of: Low | Medium | High>",
  "component": "<short name of the primary component mentioned, or null>",
  "action_required": <true | false>,
  "urgency_days": <integer days until action needed, or null if not determinable>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<1-2 sentence chain-of-thought>"
}

Severity rules:
- High: asset removed from service, cracks, TADS alarms, immediate replacement
- Medium: approaching limits, elevated temperature, uneven wear, monitoring required
- Low: routine inspection, no defects, within spec

Do not include markdown fences. Return only the JSON object."""

USER_TEMPLATE = """Maintenance note:
\"\"\"{note}\"\"\"

Asset type: {asset_type}
Date: {date}"""


class LLMEnricher:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.token_usage = {"input": 0, "output": 0}

    def extract(self, note: str, asset_type: str, date: str, retries: int = 3) -> dict:
        """Extract structured fields from a single maintenance note."""
        user_msg = USER_TEMPLATE.format(note=note, asset_type=asset_type, date=date)

        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}]
                )
                raw = response.content[0].text.strip()
                self.token_usage["input"] += response.usage.input_tokens
                self.token_usage["output"] += response.usage.output_tokens
                return json.loads(raw)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt+1}: {e} | raw={raw[:120]}")
                if attempt == retries - 1:
                    return self._fallback(note)
                time.sleep(1)
            except anthropic.RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return self._fallback(note)

        return self._fallback(note)

    def _fallback(self, note: str) -> dict:
        """Keyword-based fallback when LLM fails."""
        note_l = note.lower()
        fault_type = "none"
        for ft, kws in {
            "crack": ["crack", "fracture", "ndt"],
            "wear": ["wear", "flange", "tread", "hollow"],
            "bearing": ["bearing", "tads", "axle box"],
            "brake": ["brake", "pad", "disc"],
            "vibration": ["vibration", "flat", "hunting"],
            "noise": ["squeal", "clunk", "noise"],
            "corrosion": ["corrosion", "rust"],
        }.items():
            if any(k in note_l for k in kws):
                fault_type = ft
                break
        return {
            "fault_type": fault_type,
            "severity": "Low",
            "component": None,
            "action_required": False,
            "urgency_days": None,
            "confidence": 0.1,
            "reasoning": "Fallback heuristic used due to extraction failure."
        }

    def enrich_batch(self, records: list[dict], delay: float = 0.15) -> list[dict]:
        """Enrich a list of maintenance records."""
        enriched = []
        for i, rec in enumerate(records):
            logger.info(f"Processing {i+1}/{len(records)} — {rec['record_id']}")
            extraction = self.extract(
                note=rec["maintenance_note"],
                asset_type=rec["asset_type"],
                date=rec["date"]
            )
            enriched_rec = {**rec, **{f"llm_{k}": v for k, v in extraction.items()}}
            enriched.append(enriched_rec)
            if delay:
                time.sleep(delay)

        logger.info(f"Done. Token usage — input: {self.token_usage['input']}, output: {self.token_usage['output']}")
        return enriched


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/raw_maintenance_notes.csv")
    sample = df.head(20).to_dict("records")  # demo with 20 records

    enricher = LLMEnricher()
    results = enricher.enrich_batch(sample, delay=0.2)

    out = pd.DataFrame(results)
    out.to_csv("data/enriched_sample.csv", index=False)
    print(out[["record_id", "llm_fault_type", "llm_severity", "llm_action_required", "llm_confidence"]].to_string())
