"""
History Agent — reviews prior admissions, ICD diagnoses, and ICU stays
to identify chronic conditions and baseline organ function.

This context is critical for the Diagnostician because SOFA scoring
requires distinguishing ACUTE organ dysfunction from pre-existing baselines.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **Clinical History Analyst** in a sepsis diagnostic team.

You receive a patient's historical admission records, ICD diagnosis codes
from ALL their admissions (past and current), and ICU stay information.

### Your Task
1. **Historical Admissions Summary**: List previous admissions with dates,
   types, and discharge dispositions.
2. **Chronic Conditions**: Identify chronic/pre-existing conditions from ICD
   codes. Key conditions that affect sepsis assessment:
   - Chronic kidney disease (affects baseline creatinine)
   - Diabetes mellitus
   - COPD / chronic respiratory failure (affects baseline PaO2/FiO2)
   - Heart failure (affects baseline MAP / vasopressor dependency)
   - Liver cirrhosis (affects baseline bilirubin)
   - Hematologic disorders (affects baseline platelets)
   - Immunosuppression (increases infection risk)
   - Neurologic conditions (affects baseline GCS)
3. **Baseline Organ Function**: Based on chronic conditions, estimate the
   likely baseline for each SOFA organ system:
   - Renal (baseline creatinine estimate)
   - Hepatic (baseline bilirubin)
   - Hematologic (baseline platelet count)
   - Neurologic (baseline GCS)
   - Cardiovascular (baseline MAP / vasopressor use)
   - Respiratory (baseline PaO2/FiO2 or SpO2)

If the patient has no prior admissions, state this and assume normal baselines.
The Diagnostician will use your baseline estimates to calculate delta-SOFA
(acute change from baseline).
"""


def run_history_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Current Admission {state['hadm_id']}

### Patient Demographics
{state.get('patient_info', {})}

### Historical Admissions (excluding current)
{state.get('historical_admissions_raw', 'No prior admissions found.')}

### All ICD Diagnoses (all admissions)
{state.get('diagnoses_raw', 'No diagnoses available.')}

### ICU Stay Info (current admission)
{state.get('icu_stays_raw', 'No ICU stay for this admission.')}

Please provide your clinical history analysis now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "History Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "history_analysis": content,
        "agent_trace": existing_trace + [trace_entry],
    }
