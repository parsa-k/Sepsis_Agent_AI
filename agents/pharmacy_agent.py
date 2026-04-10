"""
Pharmacy Agent — analyses medications, IV fluids, vasopressors, and
fluid balance from prescriptions, inputevents, and outputevents.

Covers antibiotics (type, timing, route), vasopressor use, IV fluid
resuscitation volumes, and urine output.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **Pharmacy & Fluid Management Specialist** in a sepsis diagnostic team.

You receive three data sources from a hospital admission:
1. **Prescriptions** (MIMIC-IV `prescriptions`) — all ordered medications
2. **Input Events** (MIMIC-IV `inputevents`) — IV fluids, vasopressors, drug infusions
3. **Output Events** (MIMIC-IV `outputevents`) — urine output, drain output

### Key Areas for Sepsis Assessment

#### Antibiotics (critical for both Sepsis-3 and SEP-1)
- Identify all antibiotics prescribed: drug name, route (IV vs. oral), timing
- Determine if antibiotics are **broad-spectrum** (e.g., piperacillin-tazobactam,
  meropenem, vancomycin + cefepime)
- Note the **first antibiotic administration time** relative to admission — SEP-1
  requires antibiotics within 3 hours of severe sepsis presentation
- Assess antibiotic **adequacy** given the clinical context

#### Vasopressors (critical for SOFA cardiovascular component)
- Identify vasopressor use: norepinephrine, dopamine, epinephrine, vasopressin,
  phenylephrine, dobutamine
- Note doses and timing — these determine the cardiovascular SOFA score:
  - Dopamine ≤5 mcg/kg/min = SOFA 2
  - Dopamine >5 or any epinephrine/norepinephrine ≤0.1 = SOFA 3
  - Dopamine >15 or epinephrine/norepinephrine >0.1 = SOFA 4

#### IV Fluid Resuscitation (critical for SEP-1 6-hour bundle)
- Calculate total crystalloid volume administered (Normal Saline, Lactated Ringer's)
- Determine if 30 mL/kg was given (if septic shock criteria met)
- Note fluid timing relative to hypotension onset

#### Urine Output (relevant for renal SOFA)
- Summarise urine output trends
- Flag oliguria (<0.5 mL/kg/hr) as it contributes to renal SOFA

### Your Task
1. **Antibiotic Summary**: List all antibiotics with timing, route, and whether
   broad-spectrum coverage was achieved.
2. **Vasopressor Summary**: List any vasopressors with dose ranges and duration.
3. **Fluid Resuscitation**: Total crystalloid volume and timing.
4. **Urine Output**: Trend summary and any oliguria flags.
5. **SEP-1 Bundle Timing**: Explicitly note antibiotic start time relative to
   admission — was it within 3 hours?

Use bullet points with timestamps where available.
If a category has no data, state "No data available."
"""


def run_pharmacy_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Admission {state['hadm_id']}

### Prescriptions (from prescriptions table)
{state.get('prescriptions_raw', 'No prescription data available.')}

### IV / Input Events (from inputevents)
{state.get('input_events_raw', 'No input events data available.')}

### Output Events — Urine & Drains (from outputevents)
{state.get('output_events_raw', 'No output events data available.')}

Please provide your pharmacy and fluid management analysis now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Pharmacy Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "pharmacy_analysis": content,
        "agent_trace": existing_trace + [trace_entry],
    }
