import pandas as pd
import requests
import json
import re
import time

# -------------------------
# AI Function
# -------------------------

def analyze_po_with_ai(po_dict):

    prompt = f"""
You are a GST Fraud Detection Expert.

Analyze this purchase order and determine fraud risk.

Purchase Order Data:

PO ID: {po_dict.get('po_id')}
Vendor ID: {po_dict.get('vendor_id')}
Rule Score: {po_dict.get('rule_score')}
Severity: {po_dict.get('severity')}
Risk Tier: {po_dict.get('risk_tier')}
Anomaly Rate: {po_dict.get('anomaly_rate')}
Filing Compliance Rate: {po_dict.get('filing_compliance_rate')}

Rules:

- Rule Score below 70 = suspicious
- Filing Compliance below 0.5 = risky
- Anomaly Rate above 0.5 = high risk
- Critical Risk Tier increases fraud probability

Return ONLY valid JSON.

Example:

{{
  "risk_level":"High",
  "anomaly_types_detected":["GST Risk"],
  "reasoning":"Low filing compliance and high anomaly rate detected.",
  "recommended_action":"Manual audit required.",
  "confidence_score":"0.91"
}}

Now analyze the record and return ONLY JSON.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        },
        timeout=180
    )

    return response.json()["response"]


# -------------------------
# Load Suspicious Records
# -------------------------

df = pd.read_csv("suspicious_pos.csv")

# Test ke liye 5 records
# sample_df = df.head(5)

# Final run ke liye
sample_df = df

results = []

# -------------------------
# AI Analysis Loop
# -------------------------

for _, row in sample_df.iterrows():

    try:

        ai_result = analyze_po_with_ai(row)

        match = re.search(r'\{.*\}', ai_result, re.DOTALL)

        if match:

            parsed = json.loads(match.group())

            results.append({
                "po_id": row["po_id"],
                "risk_level": parsed.get("risk_level"),
                "anomaly_types_detected": str(parsed.get("anomaly_types_detected")),
                "reasoning": parsed.get("reasoning"),
                "recommended_action": parsed.get("recommended_action"),
                "confidence_score": parsed.get("confidence_score")
            })

        print(f"Completed: {row['po_id']}")

        time.sleep(1)

    except Exception as e:

        print(f"Error for {row['po_id']}: {e}")


# -------------------------
# Save Output
# -------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "ai_results.csv",
    index=False
)

print("AI Analysis Completed")
print(f"Total Records Processed: {len(results_df)}")