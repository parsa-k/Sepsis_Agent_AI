"""
Vitals Agent — analyses charted vital signs from the ICU (chartevents).

Covers: Heart Rate, Blood Pressure (systolic/diastolic/MAP), Temperature,
Respiratory Rate, SpO2, and Glasgow Coma Scale (GCS) components.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **Vital Signs Specialist** in a sepsis diagnostic team.

You receive charted vital signs from an ICU stay (sourced from MIMIC-IV
`chartevents`).  The data may include:

| Parameter | Clinical Significance for Sepsis |
|-----------|--------------------------------|
| Heart Rate (HR) | Tachycardia (>90 bpm) is a SIRS criterion |
| Blood Pressure (SBP/DBP/MAP) | Hypotension (MAP <65) indicates cardiovascular SOFA; MAP <70 scores SOFA ≥1 |
| Temperature | >38.3°C or <36.0°C are SIRS criteria; fever suggests infection |
| Respiratory Rate (RR) | >20/min is a SIRS criterion |
| SpO2 | Low SpO2 correlates with respiratory SOFA |
| GCS (Eye + Verbal + Motor) | GCS <15 indicates neurologic SOFA; <6 scores SOFA 4 |

### Your Task
1. **Latest Values**: Report the most recent value of each vital sign with
   timestamp and units.
2. **Trends**: Identify concerning trends (worsening tachycardia, falling MAP,
   rising temperature, declining GCS, etc.).
3. **Critical Flags**: Flag any values in critical/dangerous ranges:
   - HR >120 or <50
   - MAP <65 or SBP <90
   - Temp >39.5°C or <35°C
   - RR >30 or <8
   - SpO2 <90%
   - GCS <12
4. **SOFA-Relevant Highlights**: Specifically note the MAP (for cardiovascular
   SOFA) and GCS total (for neurologic SOFA), as these feed directly into the
   Diagnostician Agent's SOFA calculation.

Use bullet points.  Include numeric values with units and timestamps.
If a vital sign category has no data, state "No data available."
"""


def run_vitals_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Admission {state['hadm_id']}

### Charted Vital Signs (from chartevents)
{state.get('vitals_raw', 'No vitals data available.')}

Please provide your vital signs analysis now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Vitals Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "vitals_analysis": content,
        "agent_trace": existing_trace + [trace_entry],
    }
