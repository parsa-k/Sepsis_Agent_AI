"""
Compliance Agent — checks whether the documentation satisfies the
US CMS SEP-1 (Severe Sepsis / Septic Shock Early Management Bundle) measure.

Receives processed analyses from all feature agents plus the Sepsis-3 verdict.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **hospital quality & compliance specialist**.
Your task is to determine whether a patient's chart satisfies the
**CMS SEP-1** (Severe Sepsis and Septic Shock Management Bundle) measure.

### SEP-1 Overview
SEP-1 uses **SIRS criteria** (not SOFA) as the screening trigger, combined
with organ dysfunction and specific timing requirements.

### SIRS Criteria (>= 2 of the following):
1. Temperature > 38.3 C (100.9 F) or < 36.0 C (96.8 F)
2. Heart rate > 90 bpm
3. Respiratory rate > 20 breaths/min OR PaCO2 < 32 mmHg
4. WBC > 12,000/uL or < 4,000/uL or > 10% immature bands

### SEP-1 Severe Sepsis Definition
SIRS >= 2 **PLUS** suspected infection **PLUS** organ dysfunction
(lactate >= 2 mmol/L, or other organ dysfunction markers).

### SEP-1 3-Hour Bundle (from time zero = severe sepsis presentation):
1. Lactate level measured
2. Blood cultures obtained BEFORE antibiotics
3. Broad-spectrum antibiotics administered

### SEP-1 6-Hour Bundle (if septic shock = lactate >= 4 OR persistent hypotension):
4. 30 mL/kg crystalloid fluid resuscitation
5. Vasopressors if hypotension persists after fluids
6. Repeat lactate if initial lactate >= 2

### Input Data
You receive pre-processed analyses from five specialised agents plus the
Sepsis-3 assessment:
- **Vitals Analysis**: HR, Temp, RR, BP/MAP (for SIRS + hypotension)
- **Lab Analysis**: WBC, Bands, Lactate, pCO2 (for SIRS + bundle items)
- **Microbiology Analysis**: Culture timing and results (for bundle item 2)
- **Pharmacy Analysis**: Antibiotic timing, fluid volumes, vasopressors
- **History Analysis**: Context for organ dysfunction interpretation
- **Sepsis-3 Result**: Academic diagnosis for reference

### Your Instructions
1. Check SIRS criteria — list which are met and when (using Vitals + Labs).
2. Confirm suspected infection presence (using Microbiology + Pharmacy).
3. Assess 3-hour bundle compliance (lactate timing, culture timing, antibiotic
   timing) using data from Lab, Microbiology, and Pharmacy agents.
4. Assess 6-hour bundle if applicable (fluid volumes, vasopressor use, repeat
   lactate) using Pharmacy and Lab agent data.
5. Conclude: **SEP-1 Compliant: YES/NO** with reasoning.
6. If NO, list **specifically what is missing or undocumented**.

Output your answer in this format:
## SIRS Assessment
(which criteria met, timestamps)

## Infection Assessment
(source, cultures, antibiotics)

## 3-Hour Bundle Compliance
(each item: met/not met with timing)

## 6-Hour Bundle Compliance
(each item if applicable)

## Conclusion
**SEP-1 Compliant: YES/NO**
Reasoning: ...
Missing documentation (if any): ...
"""


def run_compliance_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
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

### Sepsis-3 Assessment Result
Academic Sepsis-3 met: {state.get('sepsis3_met', 'Unknown')}
Reasoning: {state.get('sepsis3_reasoning', 'Not yet assessed.')}

Please perform your SEP-1 compliance check now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    content_lower = content.lower()
    sep1_met = None
    if "sep-1 compliant: yes" in content_lower or "sep-1: yes" in content_lower:
        sep1_met = True
    elif "sep-1 compliant: no" in content_lower or "sep-1: no" in content_lower:
        sep1_met = False

    missing_data = ""
    if sep1_met is False:
        for line in content.split("\n"):
            if "missing" in line.lower():
                missing_data += line + "\n"

    trace_entry = {"agent": "Compliance Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "sirs_criteria": content,
        "sep1_met": sep1_met,
        "sep1_reasoning": content,
        "lactate_timing": content,
        "missing_data_queries": missing_data,
        "agent_trace": existing_trace + [trace_entry],
    }
