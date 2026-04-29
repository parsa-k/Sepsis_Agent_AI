"""Unit tests for the Evaluator Agent (the final quality gate)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from agents.memory_manager_agent import MemoryManager
from agents.evaluator_agent import (
    run_evaluator_agent,
    _normalise_evaluation,
    _parse_evaluation,
    _gather_part1_payloads,
)


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    def invoke(self, messages, *a, **k):
        self.last_messages = messages
        return SimpleNamespace(content=self.response)


def _state(active=None, with_diagnoses=True, all_skipped=False):
    if active is None:
        active = ["vitals", "lab", "microbiology", "pharmacy"]

    def env(actionable, skipped=False):
        return {
            "agent_name": "x",
            "skipped": skipped,
            "part1_payload": {"actionable": actionable, "source_records": ["r"]},
            "part2_reasoning": "PART2_LEAK_SENTINEL_should_not_reach_evaluator",
        }

    state = {
        "subject_id": 7,
        "selected_hadm_ids": [101],
        "user_intent": "Audit Sepsis-3 + SEP-1",
        "orchestrator_decision": {
            "role": "audit",
            "active_agents": active,
            "agent_instructions": {a: f"focus on {a}" for a in active},
            "history_first": False,
            "rationale": "single-visit audit",
            "user_intent": "Audit Sepsis-3 + SEP-1",
        },
        "visits_data": {
            101: {
                "vitals_raw": "HR 110",
                "labs_raw": "Lactate 3.0",
                "microbiology_raw": "BLOOD CULTURE pending",
                "prescriptions_raw": "Ceftriaxone",
                "input_events_raw": "No data available.",
                "output_events_raw": "No data available.",
                "diagnoses_raw": "I50.9 Heart failure",
                "icu_stays_raw": "MICU 2150-05-16",
                "admission_info": {"admittime": "2150-05-14"},
            },
        },
        "vitals_output":       env({"hr_latest": 110},      skipped=all_skipped),
        "lab_output":          env({"lactate_latest": 3.0}, skipped=all_skipped),
        "microbiology_output": env({"infection_status": "suspected"}, skipped=all_skipped),
        "pharmacy_output":     env({"first_antibiotic_time": "11:00"}, skipped=all_skipped),
        "agent_trace": [],
    }
    if with_diagnoses:
        state["diagnoses_output"] = {
            "summary": "Likely septic shock.",
            "patient_score": 5,
            "final_diagnosis": "Septic shock",
            "details": "SOFA 8.",
            "sepsis3_met": True, "sep1_compliant": False,
            "next_steps": "- Resuscitate.",
            "short_term_treatment": "Vasopressors + AB.",
            "mid_term_plan": "Day 7 oral switch.",
        }
    return state


# ── parsing & normalisation ────────────────────────────────────────────────

class TestNormalisation(unittest.TestCase):

    def test_invalid_flag_normalised_to_yellow(self):
        out = _normalise_evaluation(
            {"flag": "purple"}, decision={"active_agents": []}, raw="",
        )
        self.assertEqual(out["flag"], "yellow")

    def test_valid_flag_kept(self):
        for f in ("green", "yellow", "red"):
            out = _normalise_evaluation(
                {"flag": f}, decision={"active_agents": []}, raw="",
            )
            self.assertEqual(out["flag"], f)

    def test_confidence_clamped(self):
        out = _normalise_evaluation(
            {"flag": "green", "confidence": 9999},
            decision={"active_agents": []}, raw="",
        )
        self.assertEqual(out["confidence"], 100)
        out = _normalise_evaluation(
            {"flag": "green", "confidence": "garbage"},
            decision={"active_agents": []}, raw="",
        )
        self.assertEqual(out["confidence"], 85)  # default for green

    def test_missing_agent_reports_filled(self):
        out = _normalise_evaluation(
            {"flag": "green"},
            decision={"active_agents": ["vitals", "lab"]}, raw="",
        )
        for name in (
            "orchestrator", "history", "vitals", "lab",
            "microbiology", "pharmacy", "diagnoses",
        ):
            self.assertIn(name, out["agent_reports"])
            self.assertIn(out["agent_reports"][name]["verdict"],
                          ("ok", "warn", "fail"))

    def test_history_skipped_marked_ok_when_not_active(self):
        out = _normalise_evaluation(
            {"flag": "green"},
            decision={"active_agents": ["vitals"]}, raw="",
        )
        self.assertEqual(out["agent_reports"]["history"]["verdict"], "ok")
        self.assertIn("Skipped", out["agent_reports"]["history"]["notes"])

    def test_parser_recovers_from_garbage(self):
        self.assertEqual(_parse_evaluation(""), {})
        self.assertEqual(_parse_evaluation("nope"), {})
        ok = _parse_evaluation('```json\n{"flag":"green"}\n```')
        self.assertEqual(ok.get("flag"), "green")


# ── end-to-end with FakeLLM ─────────────────────────────────────────────────

class TestRunEvaluator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eval_test_")
        self.mm = MemoryManager(patient_id="7", base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _canned(self, flag="green"):
        return json.dumps({
            "flag": flag,
            "task_executed": flag != "red",
            "confidence": 90 if flag == "green" else 40,
            "overall_summary": "Pipeline behaved as expected.",
            "agent_reports": {
                "orchestrator": {"verdict": "ok",   "notes": "Plan clear."},
                "history":      {"verdict": "ok",   "notes": "N/A single visit."},
                "vitals":       {"verdict": "ok",   "notes": "HR found."},
                "lab":          {"verdict": "warn", "notes": "Lactate sparse."},
                "microbiology": {"verdict": "ok",   "notes": "Cultures pending."},
                "pharmacy":     {"verdict": "ok",   "notes": "AB on file."},
                "diagnoses":    {"verdict": "ok",   "notes": "Verdict supported."},
            },
            "missing_data": [],
            "improvement_recommendations": "- Repeat lactate in 4 h.",
        })

    def test_evaluator_emits_full_envelope(self):
        llm = FakeLLM(self._canned("green"))
        out = run_evaluator_agent(_state(), llm, memory_manager=self.mm)
        ev = out["evaluator_output"]
        self.assertEqual(ev["flag"], "green")
        self.assertTrue(ev["task_executed"])
        self.assertEqual(ev["agent_reports"]["lab"]["verdict"], "warn")

    def test_evaluator_prompt_contains_part1_and_intent(self):
        llm = FakeLLM(self._canned("green"))
        run_evaluator_agent(_state(), llm, memory_manager=self.mm)
        msgs = "\n".join(m.content for m in llm.last_messages)
        self.assertIn("Audit Sepsis-3 + SEP-1", msgs)
        self.assertIn("hr_latest", msgs)
        self.assertIn("Septic shock", msgs)
        # Part-2 reasoning must NOT leak into the Evaluator's context.
        self.assertNotIn("PART2_LEAK_SENTINEL", msgs)

    def test_red_flag_when_llm_says_red(self):
        llm = FakeLLM(self._canned("red"))
        out = run_evaluator_agent(_state(), llm, memory_manager=self.mm)
        self.assertEqual(out["evaluator_output"]["flag"], "red")
        self.assertFalse(out["evaluator_output"]["task_executed"])

    def test_garbage_response_yields_yellow_default(self):
        llm = FakeLLM("totally not json")
        out = run_evaluator_agent(_state(), llm, memory_manager=self.mm)
        ev = out["evaluator_output"]
        self.assertEqual(ev["flag"], "yellow")

    def test_gather_part1_payloads_marks_skipped(self):
        state = _state(all_skipped=True)
        out = _gather_part1_payloads(state)
        self.assertTrue(out["vitals"]["skipped"])
        self.assertTrue(out["lab"]["skipped"])


if __name__ == "__main__":
    unittest.main()
