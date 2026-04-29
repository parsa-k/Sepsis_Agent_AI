"""Unit tests for the five feature agents and the master Diagnoses Agent."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from agents.memory_manager_agent import MemoryManager
from agents.vitals_agent import run_vitals_agent
from agents.lab_agent import run_lab_agent
from agents.microbiology_agent import run_microbiology_agent
from agents.pharmacy_agent import run_pharmacy_agent
from agents.history_agent import run_history_agent
from agents.diagnoses_agent import run_diagnoses_agent


def two_part(actionable: dict, source: list, reasoning: str) -> str:
    return "```json\n" + json.dumps({
        "part1_payload": {"actionable": actionable, "source_records": source},
        "part2_reasoning": reasoning,
    }) + "\n```"


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    def invoke(self, messages, *args, **kwargs):
        self.last_messages = messages
        return SimpleNamespace(content=self.response)


# ── Common state factory ────────────────────────────────────────────────────

def _state_with_active(active: list, *, history_baseline: dict | None = None,
                        multi: bool = False):
    return {
        "subject_id": 7,
        "selected_hadm_ids": [101, 102] if multi else [101],
        "user_intent": "test intent",
        "history_baseline": history_baseline,
        "visits_data": {
            101: {
                "vitals_raw": "HR 110 at 09:00",
                "labs_raw": "Lactate 3.0 mmol/L at 10:00",
                "microbiology_raw": "BLOOD CULTURE pending",
                "prescriptions_raw": "Ceftriaxone 1g IV at 11:00",
                "input_events_raw": "Norepinephrine 0.1 mcg/kg/min",
                "output_events_raw": "Foley 50 mL/hr",
                "diagnoses_raw": "I50.9 Heart failure",
                "icu_stays_raw": "MICU 2150-05-16",
                "admission_info": {"admittime": "2150-05-14"},
            },
            102: {
                "vitals_raw": "HR 80 at 12:00",
                "labs_raw": "Lactate 1.5 mmol/L at 13:00",
                "microbiology_raw": "No data available.",
                "prescriptions_raw": "Aspirin 81mg PO",
                "input_events_raw": "No data available.",
                "output_events_raw": "No data available.",
                "diagnoses_raw": "N18.6 CKD stage 5",
                "icu_stays_raw": "No ICU stay for this admission.",
                "admission_info": {"admittime": "2149-12-01"},
            },
        },
        "orchestrator_decision": {
            "role": "Sepsis audit",
            "active_agents": active,
            "agent_instructions": {a: f"focus on {a}" for a in active},
        },
        "agent_trace": [],
    }


# ── Feature agents ──────────────────────────────────────────────────────────

class TestFeatureAgentsActive(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fa_test_")
        self.mm = MemoryManager(patient_id="7", base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vitals_active_emits_two_part(self):
        llm = FakeLLM(two_part(
            {"hr_latest": 110, "map_latest": 62},
            ["chartevents 09:00 HR=110"],
            "Tachycardia and hypotension.",
        ))
        state = _state_with_active(["vitals"])
        out = run_vitals_agent(state, llm, memory_manager=self.mm)
        env = out["vitals_output"]
        self.assertEqual(env["part1_payload"]["actionable"]["hr_latest"], 110)
        self.assertEqual(env["agent_name"], "vitals")
        self.assertIn("Tachycardia", env["part2_reasoning"])

        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("focus on vitals", msgs_text)
        self.assertIn("HR 110 at 09:00", msgs_text)

    def test_vitals_skipped_when_not_active(self):
        llm = FakeLLM("(should not be called)")
        state = _state_with_active(["lab"])
        out = run_vitals_agent(state, llm, memory_manager=self.mm)
        self.assertTrue(out["vitals_output"]["skipped"])

    def test_lab_active(self):
        llm = FakeLLM(two_part(
            {"lactate_latest": 3.0, "sirs_criteria_from_labs": ["wbc>12"]},
            ["labevents 10:00 lactate=3.0"],
            "Mildly elevated lactate.",
        ))
        state = _state_with_active(["lab"])
        out = run_lab_agent(state, llm, memory_manager=self.mm)
        self.assertEqual(out["lab_output"]["part1_payload"]["actionable"]["lactate_latest"], 3.0)

    def test_microbiology_active(self):
        llm = FakeLLM(two_part(
            {"infection_status": "suspected", "first_blood_culture_time": "10:25"},
            ["microbiologyevents 10:25 BLOOD"],
            "No growth yet.",
        ))
        state = _state_with_active(["microbiology"])
        out = run_microbiology_agent(state, llm, memory_manager=self.mm)
        self.assertEqual(
            out["microbiology_output"]["part1_payload"]["actionable"]["infection_status"],
            "suspected",
        )

    def test_pharmacy_active_uses_three_data_sections(self):
        llm = FakeLLM(two_part(
            {"first_antibiotic_time": "11:00",
             "vasopressors": [{"drug": "norepinephrine"}]},
            ["prescriptions Ceftriaxone 1g"],
            "First-line broad spectrum.",
        ))
        state = _state_with_active(["pharmacy"])
        out = run_pharmacy_agent(state, llm, memory_manager=self.mm)
        env = out["pharmacy_output"]
        self.assertEqual(env["part1_payload"]["actionable"]["first_antibiotic_time"], "11:00")
        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("Ceftriaxone", msgs_text)
        self.assertIn("Norepinephrine", msgs_text)
        self.assertIn("Foley", msgs_text)

    def test_baseline_block_is_injected_when_provided(self):
        llm = FakeLLM(two_part({}, [], ""))
        baseline = {
            "actionable": {"chronic_conditions": ["CKD stage 5"]},
            "source_records": ["d.icd_code N18.6"],
        }
        state = _state_with_active(["lab"], history_baseline=baseline)
        run_lab_agent(state, llm, memory_manager=self.mm)
        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("CKD stage 5", msgs_text)
        self.assertIn("Baseline context (from History Agent", msgs_text)


# ── History agent ───────────────────────────────────────────────────────────

class TestHistoryAgent(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hist_test_")
        self.mm = MemoryManager(patient_id="7", base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_history_returns_baseline_envelope(self):
        llm = FakeLLM(two_part(
            {"chronic_conditions": ["CKD"], "baseline_organ_function": {"renal": "elevated baseline creatinine"}},
            ["diagnoses_icd N18.6"],
            "Patient with chronic kidney disease.",
        ))
        state = _state_with_active(["history", "vitals"], multi=True)
        out = run_history_agent(state, llm, memory_manager=self.mm)
        env = out["history_output"]
        self.assertIn("chronic_conditions", env["part1_payload"]["actionable"])
        self.assertEqual(out["history_baseline"], env["part1_payload"])

    def test_history_message_includes_all_selected_visits(self):
        llm = FakeLLM(two_part({"chronic_conditions": []}, [], ""))
        state = _state_with_active(["history"], multi=True)
        run_history_agent(state, llm, memory_manager=self.mm)
        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("Visit 101", msgs_text)
        self.assertIn("Visit 102", msgs_text)


# ── Diagnoses agent ─────────────────────────────────────────────────────────

class TestDiagnosesAgent(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="diag_test_")
        self.mm = MemoryManager(patient_id="7", base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mk_envelope(self, actionable: dict) -> dict:
        return {
            "agent_name": "x",
            "part1_payload": {
                "actionable": actionable,
                "source_records": ["row 1"],
            },
            "part2_reasoning": "long-form should NOT be visible to diagnoses",
        }

    def test_only_part1_payloads_reach_diagnoses_prompt(self):
        canned = json.dumps({
            "summary": "Likely septic shock.",
            "patient_score": 5,
            "final_diagnosis": "Septic shock with multi-organ dysfunction.",
            "details": "SOFA 8...",
            "sepsis3_met": True,
            "sep1_compliant": False,
        })
        llm = FakeLLM(canned)
        state = {
            "subject_id": 7,
            "user_intent": "audit",
            "orchestrator_decision": {
                "role": "audit",
                "user_intent": "audit",
                "active_agents": ["vitals", "lab", "microbiology", "pharmacy"],
            },
            "vitals_output": self._mk_envelope({"map_latest": 55}),
            "lab_output": self._mk_envelope({"lactate_latest": 4.5}),
            "microbiology_output": self._mk_envelope({"infection_status": "documented"}),
            "pharmacy_output": self._mk_envelope({"first_antibiotic_time": "11:00"}),
            "agent_trace": [],
        }
        out = run_diagnoses_agent(state, llm, memory_manager=self.mm)
        diag = out["diagnoses_output"]
        self.assertEqual(diag["patient_score"], 5)
        self.assertTrue(diag["sepsis3_met"])
        self.assertFalse(diag["sep1_compliant"])
        self.assertEqual(diag["summary"], "Likely septic shock.")

        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("map_latest", msgs_text)
        self.assertIn("lactate_latest", msgs_text)
        self.assertNotIn("long-form should NOT be visible", msgs_text)

    def test_skipped_agents_are_not_included_in_prompt(self):
        canned = json.dumps({"summary": "ok", "patient_score": 2,
                             "final_diagnosis": "no sepsis", "details": "."})
        llm = FakeLLM(canned)
        state = {
            "subject_id": 7,
            "orchestrator_decision": {
                "active_agents": ["vitals"],
            },
            "vitals_output": self._mk_envelope({"hr_latest": 80}),
            "lab_output": {"agent_name": "lab", "skipped": True,
                            "part1_payload": {"actionable": {"status": "skipped"},
                                              "source_records": []},
                            "part2_reasoning": ""},
            "agent_trace": [],
        }
        out = run_diagnoses_agent(state, llm, memory_manager=self.mm)
        msgs_text = "\n".join(m.content for m in llm.last_messages)
        self.assertNotIn("\"status\": \"skipped\"", msgs_text)
        self.assertEqual(out["diagnoses_output"]["patient_score"], 2)

    def test_score_clamped_to_1_to_5(self):
        for raw_score, expected in [("9", 5), ("0", 1), ("nope", 3), ("3", 3)]:
            with self.subTest(raw_score=raw_score):
                canned = json.dumps({
                    "summary": "x", "patient_score": raw_score,
                    "final_diagnosis": "x", "details": ".",
                })
                llm = FakeLLM(canned)
                state = {
                    "subject_id": 1,
                    "orchestrator_decision": {"active_agents": []},
                    "agent_trace": [],
                }
                out = run_diagnoses_agent(state, llm, memory_manager=self.mm)
                self.assertEqual(out["diagnoses_output"]["patient_score"], expected)


if __name__ == "__main__":
    unittest.main()
