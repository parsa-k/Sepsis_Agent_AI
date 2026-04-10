"""
Microbiology Agent — analyses culture results and infection evidence
from microbiologyevents.

Determines whether there is suspected or documented infection, identifies
organisms, and assesses antibiotic sensitivity patterns.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.state import SepsisState

SYSTEM_PROMPT = """You are a **Microbiology & Infection Specialist** in a sepsis diagnostic team.

You receive microbiology culture results from a hospital admission (sourced
from MIMIC-IV `microbiologyevents`).  This data is critical because both
Sepsis-3 and SEP-1 require evidence of **suspected or documented infection**.

### Data Fields You May See
- **spec_type_desc**: Specimen type (e.g., BLOOD CULTURE, URINE, SPUTUM, WOUND)
- **test_name**: Microbiological test performed
- **org_name**: Organism identified (NULL = no growth or pending)
- **ab_name**: Antibiotic tested in sensitivity panel
- **interpretation**: Sensitivity result (S = Sensitive, R = Resistant, I = Intermediate)

### Your Task
1. **Culture Summary**: List all cultures obtained — specimen type, date/time,
   and whether an organism grew.
2. **Positive Cultures**: For each positive culture:
   - Organism identified
   - Specimen source (blood, urine, respiratory, etc.)
   - Antibiotic sensitivity profile (S/R/I for each antibiotic tested)
3. **Infection Assessment**: Based on the cultures, state:
   - Whether infection is **documented** (positive culture with pathogen) or
     **suspected** (cultures obtained but pending/negative, yet clinical
     picture suggests infection).
   - The likely **source** of infection (bacteremia, UTI, pneumonia, etc.).
4. **Blood Culture Timing**: Note when blood cultures were obtained relative
   to admission — this is critical for SEP-1 3-hour bundle compliance
   (cultures must be drawn before antibiotics).
5. **Resistance Patterns**: Flag any multi-drug resistant organisms (MRSA,
   VRE, ESBL, CRE) that may affect antibiotic adequacy.

If no microbiology data is available, explicitly state this — the absence of
cultures is itself clinically significant and may indicate a documentation gap.
"""


def run_microbiology_agent(state: SepsisState, llm, system_prompt: str | None = None) -> dict:
    prompt = system_prompt or SYSTEM_PROMPT

    human_content = f"""## Patient #{state['subject_id']} — Admission {state['hadm_id']}

### Microbiology Results (from microbiologyevents)
{state.get('microbiology_raw', 'No microbiology data available.')}

Please provide your microbiology and infection analysis now."""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=human_content),
    ]
    response = llm.invoke(messages)
    content = response.content

    trace_entry = {"agent": "Microbiology Agent", "content": content}
    existing_trace = state.get("agent_trace", [])

    return {
        "microbiology_analysis": content,
        "agent_trace": existing_trace + [trace_entry],
    }
