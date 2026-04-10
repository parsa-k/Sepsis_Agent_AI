"""
Diagnostician Agent — applies the academic Sepsis-3 definition:
  "Sepsis = suspected infection + acute organ dysfunction (SOFA >= 2 above baseline)"

Receives processed analyses from all five feature agents and synthesises
them into a SOFA score calculation and Sepsis-3 determination.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are an **expert critical-care diagnostician**.
Your task is to determine whether a patient meets the **Sepsis-3** criteria.

### Sepsis-3 Definition (Singer et al., JAMA 2016)
Sepsis is defined as **life-threatening organ dysfunction caused by a
dysregulated host response to infection**.

Operationally:
- There must be **suspected or documented infection**.
- There must be **acute organ dysfunction** defined as an increase in the
  Sequential Organ Failure Assessment (SOFA) score of **>= 2 points** from
  the patient's baseline.

### SOFA Score Components (0-4 points each, max 24):
| System        | Variable              | 0     | 1       | 2       | 3       | 4       |
|---------------|-----------------------|-------|---------|---------|---------|---------|
| Respiration   | PaO2/FiO2 (mmHg)     | >=400 | <400    | <300    | <200+MV | <100+MV |
| Coagulation   | Platelets (x10^3/uL)  | >=150 | <150    | <100    | <50     | <20     |
| Liver         | Bilirubin (mg/dL)     | <1.2  | 1.2-1.9 | 2.0-5.9 | 6.0-11.9| >12.0   |
| Cardiovascular| MAP or vasopressors   | >=70  | <70     | Dopa<=5 | Dopa>5  | Dopa>15 |
| CNS           | GCS                   | 15    | 13-14   | 10-12   | 6-9     | <6      |
| Renal         | Creatinine (mg/dL)    | <1.2  | 1.2-1.9 | 2.0-3.4 | 3.5-4.9 | >5.0    |

### Input Data
You receive pre-processed analyses from five specialised agents:
- **Vitals Analysis**: HR, BP/MAP, Temp, RR, SpO2, GCS (from Vitals Agent)
- **Lab Analysis**: WBC, Lactate, Creatinine, Bilirubin, Platelets, PaO2, Blood Gas (from Lab Agent)
- **Microbiology Analysis**: Cultures, organisms, infection evidence (from Microbiology Agent)
- **Pharmacy Analysis**: Antibiotics, vasopressors, IV fluids, urine output (from Pharmacy Agent)
- **History Analysis**: Chronic conditions, baseline organ function (from History Agent)

### Instructions
1. Calculate or estimate each SOFA component score using data from the
   appropriate feature agent.
2. Determine baseline SOFA from chronic conditions (History Agent).
3. Compute delta-SOFA = current - baseline.
4. State whether infection is suspected/documented (Microbiology + Pharmacy).
5. Conclude: **Sepsis-3 criteria MET** or **NOT MET**, with clear reasoning.

Output your answer in this format:
## SOFA Score Breakdown
(table or list of each component with source data, score, and baseline)

## Infection Assessment
(yes/no with evidence from Microbiology and Pharmacy agents)

## Conclusion
**Sepsis-3: YES/NO**
Reasoning: ...
"""


def run_diagnostician_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Admission {state['hadm_id']}

### Vitals Analysis (from Vitals Agent)
{state.get('vitals_analysis', 'Not available.')}

### Lab Analysis (from Lab Agent)
{state.get('lab_analysis', 'Not available.')}

### Microbiology Analysis (from Microbiology Agent)
{state.get('microbiology_analysis', 'Not available.')}

### Pharmacy Analysis (from Pharmacy Agent)
{state.get('pharmacy_analysis', 'Not available.')}

### Clinical History Analysis (from History Agent)
{state.get('history_analysis', 'Not available.')}

Please perform your Sepsis-3 assessment now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    content_lower = content.lower()
    sepsis3_met = None
    if "sepsis-3: yes" in content_lower or "sepsis-3 criteria met" in content_lower.replace("**", ""):
        sepsis3_met = True
    elif "sepsis-3: no" in content_lower or "sepsis-3 criteria not met" in content_lower.replace("**", ""):
        sepsis3_met = False

    trace_entry = {"agent": "Diagnostician Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "sofa_score": content,
        "organ_dysfunction": content,
        "sepsis3_met": sepsis3_met,
        "sepsis3_reasoning": content,
        "agent_trace": existing_trace + [trace_entry],
    }
