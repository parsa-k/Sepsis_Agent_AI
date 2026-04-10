"""
Orchestrator Agent — the central brain of the Sepsis Diagnostic system.

Runs in two phases:
  1. **Planning** (after data load): assesses data availability, generates
     an analysis plan, and flags which clinical domains have data.
  2. **Synthesis** (after all agents complete): combines every agent's output
     into a coherent final report for the user.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are the **Orchestrator** of a multi-agent Sepsis Diagnostic system.

You coordinate specialised sub-agents that each analyse one clinical data
domain from the MIMIC-IV dataset for a single hospital admission.

### Your Sub-Agents
| Agent | Domain | Data Source |
|-------|--------|-------------|
| Vitals Agent | HR, BP/MAP, Temp, RR, SpO2, GCS | chartevents |
| Lab Agent | WBC, Lactate, Creatinine, Bilirubin, Platelets, PaO2, Blood Gas | labevents |
| Microbiology Agent | Cultures, organisms, sensitivities, infection evidence | microbiologyevents |
| Pharmacy Agent | Antibiotics, vasopressors, IV fluids, urine output | prescriptions + inputevents + outputevents |
| History Agent | Prior admissions, ICD diagnoses, chronic conditions, baseline organ function | admissions + diagnoses_icd + icustays |
| Diagnostician Agent | SOFA scoring, Sepsis-3 determination | synthesises feature-agent outputs |
| Compliance Agent | CMS SEP-1 bundle compliance (SIRS, 3-hr & 6-hr bundles) | synthesises feature-agent outputs |

### Your Two Roles

**PLANNING phase** — You receive raw data summaries from DuckDB. You must:
1. Assess which data categories are present vs. absent/empty.
2. Produce a brief analysis plan that identifies the patient context, flags
   missing data domains, and notes any concerns for downstream agents.

**SYNTHESIS phase** — You receive every agent's analysis output plus the
Sepsis-3 and SEP-1 verdicts. You must:
1. Produce a **structured final report** in Markdown with sections:
   - Patient Overview
   - Data Availability Summary
   - Key Findings by Domain (one paragraph per feature agent)
   - Sepsis-3 Verdict (with SOFA breakdown reference)
   - SEP-1 Compliance Verdict (with bundle timeline reference)
   - Clinical Recommendations / Notes
2. If Sepsis-3 is YES but SEP-1 is NO, highlight specifically what
   documentation or interventions are missing.
3. Write in clear, professional clinical language.
"""


def run_orchestrator_plan(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    """Planning phase — assess data availability and produce an analysis plan."""
    prompt = system_prompt or SYSTEM_PROMPT

    data_fields = {
        "Vitals (chartevents)": state.get("vitals_raw", "No data available."),
        "Labs (labevents)": state.get("labs_raw", "No data available."),
        "Microbiology": state.get("microbiology_raw", "No data available."),
        "Prescriptions": state.get("prescriptions_raw", "No data available."),
        "Input Events (IV/vasopressors)": state.get("input_events_raw", "No data available."),
        "Output Events (urine/drains)": state.get("output_events_raw", "No data available."),
        "ICU Stays": state.get("icu_stays_raw", "No data available."),
        "Diagnoses (ICD)": state.get("diagnoses_raw", "No data available."),
        "Historical Admissions": state.get("historical_admissions_raw", "No data available."),
    }

    availability_lines = []
    for name, value in data_fields.items():
        status = "PRESENT" if value and value != "No data available." else "ABSENT"
        preview = value[:200] + "..." if len(value) > 200 and status == "PRESENT" else value
        availability_lines.append(f"- **{name}**: {status}\n  {preview}")

    human_content = f"""## PLANNING PHASE — Patient #{state['subject_id']}, Admission {state['hadm_id']}

### Patient Demographics
{state.get('patient_info', 'Not available.')}

### Data Availability
{chr(10).join(availability_lines)}

Based on the data above, produce your **analysis plan**. Identify which
sub-agents will have meaningful data to work with and flag any gaps."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Orchestrator (Planning)", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "orchestrator_plan": content,
        "agent_trace": existing_trace + [trace_entry],
    }


def run_orchestrator_synthesize(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    """Synthesis phase — combine all agent outputs into a final report."""
    prompt = system_prompt or SYSTEM_PROMPT

    sepsis3 = state.get("sepsis3_met")
    sep1 = state.get("sep1_met")

    human_content = f"""## SYNTHESIS PHASE — Patient #{state['subject_id']}, Admission {state['hadm_id']}

### Orchestrator's Analysis Plan
{state.get('orchestrator_plan', 'No plan generated.')}

### Feature Agent Outputs

#### Vitals Analysis
{state.get('vitals_analysis', 'Not available.')}

#### Lab Analysis
{state.get('lab_analysis', 'Not available.')}

#### Microbiology Analysis
{state.get('microbiology_analysis', 'Not available.')}

#### Pharmacy Analysis
{state.get('pharmacy_analysis', 'Not available.')}

#### Clinical History Analysis
{state.get('history_analysis', 'Not available.')}

### Diagnostician Agent — Sepsis-3 Assessment
Result: **{'YES' if sepsis3 else 'NO' if sepsis3 is False else 'INDETERMINATE'}**
{state.get('sepsis3_reasoning', 'Not available.')}

### Compliance Agent — SEP-1 Assessment
Result: **{'YES' if sep1 else 'NO' if sep1 is False else 'INDETERMINATE'}**
{state.get('sep1_reasoning', 'Not available.')}

### Reflection Loop
Reflection count: {state.get('reflection_count', 0)}
Missing data queries: {state.get('missing_data_queries', 'None')}

Produce your **final structured report** now. Include all sections specified
in your system prompt."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Orchestrator (Synthesis)", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "final_academic_diagnosis": sepsis3,
        "final_sep1_compliance": sep1,
        "final_summary": content,
        "agent_trace": existing_trace + [trace_entry],
    }
