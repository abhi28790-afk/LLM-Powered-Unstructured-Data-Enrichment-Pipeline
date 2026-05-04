"""
Synthetic Maintenance Note Generator
Generates realistic unstructured maintenance records for the pipeline demo.
"""

import random
import json
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
random.seed(42)

ASSET_TYPES = [
    "Bogie Frame", "Wheelset", "Axle Box", "Brake System",
    "Pantograph", "Traction Motor", "HVAC Unit", "Coupler Assembly",
    "Door Mechanism", "Suspension Spring"
]

TECHNICIANS = [f"Tech_{i:03d}" for i in range(1, 31)]

COMPLAINT_TEMPLATES = [
    # Normal maintenance
    "Routine inspection completed. No anomalies detected. All measurements within spec.",
    "Scheduled lubrication performed on {part}. Grease level restored.",
    "Visual check passed. Surface condition satisfactory.",
    # Wear issues
    "Noticed unusual wear pattern on {part}. Flange thickness measured at {meas}mm, approaching lower limit.",
    "Significant flange wear detected. Asset flagged for early wheelset replacement.",
    "Uneven tread wear observed. Suspect hunting oscillation or misalignment.",
    "Tread hollowing measured: {meas}mm. Reprofiling scheduled.",
    # Noise / vibration
    "Driver reported excessive vibration above 80km/h. Investigated {part} – bearing clearance within limits but noted micro-pitting.",
    "Intermittent squealing noise from {part}. Lubrication applied; monitoring required.",
    "High-frequency noise traced to {part}. Fastener torque checked and found loose at {meas}Nm.",
    "Rhythmic clunking at low speed. Wheel flat detected: {meas}mm depth.",
    # Cracks / damage
    "Crack indication found on {part} during magnetic particle inspection. Length approx {meas}mm. Sent for NDT.",
    "Surface crack observed on {part} tread. Asset removed from service pending workshop assessment.",
    "Impact damage noted on {part}. Possible ballast strike. Minor gouging, {meas}mm depth.",
    # Brake issues
    "Brake pad thickness below threshold: {meas}mm. Replaced brake block set.",
    "Uneven brake application reported. Slack adjuster inspected and reset.",
    "Brake disc scoring observed on {part}. Score depth {meas}mm. Monitoring.",
    # Bearing
    "Bearing temperature elevated: {meas}°C during run. Checked lubrication – regreased.",
    "Axle box bearing replaced preventively due to grease discolouration.",
    "TADS alarm triggered for {part}. Temperature spike {meas}°C above ambient. Bearing replaced.",
    # OK but watch
    "Minor corrosion on {part} surface. Treated and coated. Monitor at next cycle.",
    "Spring deflection marginally above mean: {meas}mm. Within acceptable range. Note for trend analysis.",
    "All measurements nominal. Component life estimated {meas}% remaining.",
]

FAULT_KEYWORDS = {
    "crack": ["crack", "fracture", "split", "NDT", "indication"],
    "wear": ["wear", "flange", "tread", "hollowing", "reprofiling", "worn"],
    "bearing": ["bearing", "TADS", "axle box", "temperature", "grease"],
    "vibration": ["vibration", "oscillation", "hunting", "rhythmic", "flat"],
    "brake": ["brake", "pad", "disc", "slack adjuster", "scoring"],
    "noise": ["squeal", "clunk", "noise", "intermittent"],
}

def label_severity(text: str) -> str:
    text_l = text.lower()
    if any(w in text_l for w in ["crack", "removed from service", "replaced", "flat detected", "below threshold", "tads"]):
        return "High"
    if any(w in text_l for w in ["wear", "elevated", "approaching", "uneven", "corrosion", "above"]):
        return "Medium"
    return "Low"

def generate_note() -> str:
    template = random.choice(COMPLAINT_TEMPLATES)
    return template.format(
        part=random.choice(ASSET_TYPES).lower(),
        meas=round(random.uniform(0.5, 95.0), 1)
    )

def build_dataset(n: int = 500) -> pd.DataFrame:
    records = []
    base_date = datetime(2023, 1, 1)

    for i in range(n):
        note = generate_note()
        asset_id = f"ASSET_{random.randint(1000, 9999)}"
        date = base_date + timedelta(days=random.randint(0, 700))
        records.append({
            "record_id": f"REC_{i+1:04d}",
            "asset_id": asset_id,
            "asset_type": random.choice(ASSET_TYPES),
            "technician_id": random.choice(TECHNICIANS),
            "date": date.strftime("%Y-%m-%d"),
            "maintenance_note": note,
            "true_severity": label_severity(note)  # ground truth for evaluation
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    df = build_dataset(500)
    df.to_csv("raw_maintenance_notes.csv", index=False)
    print(f"Generated {len(df)} records.")
    print(df["true_severity"].value_counts())
    print(df.head(3)[["record_id", "maintenance_note", "true_severity"]].to_string())
