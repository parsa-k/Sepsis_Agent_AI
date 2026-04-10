"""
Lab Agent — analyses laboratory results from labevents.

Covers all sepsis-critical labs: WBC, Lactate, Creatinine, Bilirubin,
Platelets, PaO2, Blood Gas (pH, pCO2), BUN, Glucose, Electrolytes,
Bands, and Troponin.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **Laboratory Results Specialist** in a sepsis diagnostic team.

You receive laboratory test results from a hospital admission (sourced from
MIMIC-IV `labevents`).  The data includes sepsis-critical labs:

| Lab | SOFA Component | Clinical Significance |
|-----|---------------|----------------------|
| PaO2 | Respiration SOFA | PaO2/FiO2 ratio determines respiratory score |
| Platelets | Coagulation SOFA | <150k scores ≥1; <20k scores 4 |
| Bilirubin | Liver SOFA | ≥1.2 scores ≥1; >12 scores 4 |
| Creatinine | Renal SOFA | ≥1.2 scores ≥1; >5.0 scores 4 |
| Lactate | SEP-1 trigger | ≥2 triggers bundle; ≥4 indicates septic shock |
| WBC | SIRS criterion | >12k or <4k fulfils SIRS |
| Bands | SIRS criterion | >10% immature bands fulfils SIRS |
| pH, pCO2 | Acid-base | pCO2 <32 is SIRS criterion; acidosis suggests shock |
| BUN, Glucose, Electrolytes | Supportive | Organ function and metabolic context |
| Troponin | Supportive | Cardiac involvement / myocardial stress |

### Your Task
1. **Latest Values**: Report the most recent value of each lab with timestamp,
   units, and reference range flag (normal / abnormal / critical).
2. **Trends**: Identify concerning trends (rising lactate, falling platelets,
   worsening creatinine, etc.).
3. **SOFA-Relevant Highlights**: For each SOFA lab component (PaO2, Platelets,
   Bilirubin, Creatinine), explicitly state the value and which SOFA score
   bracket it falls into (0–4).
4. **SIRS-Relevant Highlights**: Flag WBC and Bands values that meet SIRS
   criteria, and note pCO2 if <32 mmHg.
5. **Lactate**: Emphasise the initial lactate, peak lactate, and timing
   relative to admission — this is critical for SEP-1 bundle compliance.

Use bullet points.  Include numeric values with units and timestamps.
If a lab category has no data, state "No data available."
"""


def run_lab_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Admission {state['hadm_id']}

### Laboratory Results (from labevents)
{state.get('labs_raw', 'No lab data available.')}

Please provide your laboratory analysis now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Lab Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "lab_analysis": content,
        "agent_trace": existing_trace + [trace_entry],
    }
