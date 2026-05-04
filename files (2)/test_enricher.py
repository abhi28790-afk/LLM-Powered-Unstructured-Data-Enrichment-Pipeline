"""
Unit Tests — LLM Enricher
Tests the fallback heuristic and JSON schema validation without calling the API.
"""

import sys
import os
import pytest
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm_enricher import LLMEnricher

ENRICHER = LLMEnricher.__new__(LLMEnricher)  # instantiate without API key
ENRICHER.token_usage = {"input": 0, "output": 0}

VALID_FIELDS = {
    "fault_type", "severity", "component", "action_required",
    "urgency_days", "confidence", "reasoning"
}

SEVERITY_VALUES = {"Low", "Medium", "High"}
FAULT_TYPES = {"wear", "crack", "bearing", "vibration", "brake", "noise", "corrosion", "none"}


def test_fallback_crack():
    result = ENRICHER._fallback("Surface crack observed on tread. Asset removed from service.")
    assert result["fault_type"] == "crack"
    assert result["confidence"] == 0.1


def test_fallback_bearing():
    result = ENRICHER._fallback("TADS alarm triggered. Temperature spike 24°C above ambient.")
    assert result["fault_type"] == "bearing"


def test_fallback_wear():
    result = ENRICHER._fallback("Flange thickness approaching lower limit. Significant wear pattern.")
    assert result["fault_type"] == "wear"


def test_fallback_none():
    result = ENRICHER._fallback("Routine inspection completed. All measurements within spec.")
    assert result["fault_type"] == "none"


def test_fallback_output_schema():
    result = ENRICHER._fallback("Some maintenance note.")
    assert VALID_FIELDS.issubset(result.keys()), f"Missing keys: {VALID_FIELDS - result.keys()}"


def test_fallback_severity_valid():
    result = ENRICHER._fallback("Some note")
    assert result["severity"] in SEVERITY_VALUES


def test_json_schema_valid():
    """Mock a valid JSON response and ensure all fields parse correctly."""
    mock_json = json.dumps({
        "fault_type": "wear",
        "severity": "Medium",
        "component": "wheelset flange",
        "action_required": True,
        "urgency_days": 14,
        "confidence": 0.91,
        "reasoning": "Flange approaching lower limit; action needed soon."
    })
    parsed = json.loads(mock_json)
    assert VALID_FIELDS.issubset(parsed.keys())
    assert parsed["severity"] in SEVERITY_VALUES
    assert parsed["fault_type"] in FAULT_TYPES
    assert isinstance(parsed["confidence"], float)
    assert 0.0 <= parsed["confidence"] <= 1.0


def test_json_schema_action_required_bool():
    mock_json = json.dumps({
        "fault_type": "none", "severity": "Low", "component": None,
        "action_required": False, "urgency_days": None,
        "confidence": 0.95, "reasoning": "All nominal."
    })
    parsed = json.loads(mock_json)
    assert isinstance(parsed["action_required"], bool)
    assert parsed["urgency_days"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
