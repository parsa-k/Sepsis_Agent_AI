"""End-to-end LangGraph wiring tests with a fake LLM and a stubbed DuckDB.

Exercises both routing branches:
  * single visit  → orchestrator → vitals → lab → microbiology → pharmacy → diagnoses
  * multi visit   → orchestrator → history → propagate → vitals → … → diagnoses
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from agents.memory_manager_agent import MemoryManager
from agents.graph import build_graph, run_pipeline


# ── Fake LLM that responds based on the SYSTEM message ─────────────────────

class RoutedFakeLLM:
    """Pick a canned response based on a key the system message contains."""

    def __init__(self, responses: dict[str, str], default: str = ""):
        self.responses = responses
        self.default = default
        self.calls: list[list] = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        sys_text = ""
        for m in messages:
            if getattr(m, "type", "") == "system" or m.__class__.__name__ == "SystemMessage":
                sys_text = m.content
                break
        for key, response in self.responses.items():
            if key in sys_text:
                return SimpleNamespace(content=response)
        return SimpleNamespace(content=self.default)


def two_part_envelope(actionable: dict, reasoning: str = "x") -> str:
    return "```json\n" + json.dumps({
        "part1_payload": {"actionable": actionable, "source_records": ["src"]},
        "part2_reasoning": reasoning,
    }) + "\n```"


# ── DuckDB stub ─────────────────────────────────────────────────────────────

def _adm_df(hadm_id: int, subject_id: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "admittime": "2150-05-14 19:51:00",
        "dischtime": "2150-05-22 16:25:00",
        "admission_type": "EW EMER.",
        "admission_location": "ED",
        "discharge_location": "HOME",
        "hospital_expire_flag": 0,
        "gender": "M",
        "anchor_age": 57,
        "dod": None,
    }])


def _vitals_df(hadm_id: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"charttime": "2150-05-16 09:00:00", "itemid": 220045,
         "value": "110", "valuenum": 110.0, "valueuom": "bpm",
         "label": "Heart Rate"},
    ])


def _labs_df(hadm_id: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"charttime": "2150-05-16 10:00:00", "itemid": 50813,
         "value": "3.0", "valuenum": 3.0, "valueuom": "mmol/L",
         "flag": "abnormal", "label": "Lactate"},
    ])


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _patch_db(*, with_pharmacy_data: bool = True):
    """Install monkey-patches on agents.graph.db for the duration of a test."""

    def find_patient(conn, subject_id=None, hadm_id=None):
        if hadm_id is None:
            return _adm_df(101, subject_id or 99)
        return _adm_df(hadm_id, 99)

    def get_vitals(conn, s, h): return _vitals_df(h)
    def get_labs(conn, s, h): return _labs_df(h)
    def get_microbiology(conn, s, h): return _empty_df()
    def get_prescriptions(conn, s, h):
        if with_pharmacy_data:
            return pd.DataFrame([{"starttime": "2150", "drug": "Ceftriaxone"}])
        return _empty_df()
    def get_icu_stays(conn, s, h): return _empty_df()
    def get_diagnoses(conn, s, h=None):
        return pd.DataFrame([{"icd_code": "N18.6", "long_title": "CKD"}])
    def get_input_events(conn, s, h): return _empty_df()
    def get_output_events(conn, s, h): return _empty_df()
    def get_historical_admissions(conn, s, current_hadm_id=None): return _empty_df()

    class _StubConn:
        def close(self):
            pass

    def get_conn(): return _StubConn()

    patches = [
        patch("agents.graph.db.get_conn", get_conn),
        patch("agents.graph.db.find_patient", find_patient),
        patch("agents.graph.db.get_vitals", get_vitals),
        patch("agents.graph.db.get_labs", get_labs),
        patch("agents.graph.db.get_microbiology", get_microbiology),
        patch("agents.graph.db.get_prescriptions", get_prescriptions),
        patch("agents.graph.db.get_icu_stays", get_icu_stays),
        patch("agents.graph.db.get_diagnoses", get_diagnoses),
        patch("agents.graph.db.get_input_events", get_input_events),
        patch("agents.graph.db.get_output_events", get_output_events),
        patch("agents.graph.db.get_historical_admissions", get_historical_admissions),
    ]
    return patches


# ── Tests ──────────────────────────────────────────────────────────────────

class TestGraphFlow(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="graph_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _llm(self, multi_visit: bool):
        feature_active = ["vitals", "lab", "microbiology", "pharmacy"]

        # Phase 1 — pre-plan
        preplan_resp = json.dumps({
            "role": "Sepsis-3 + SEP-1 audit",
            "multi_visit": multi_visit,
            "history_first": multi_visit,
            "agent_instructions": (
                {"history": "Extract baseline organ function and chronic burden."}
                if multi_visit else {}
            ),
            "rationale": "Phase-1 gating.",
        })
        # Phase 2 — re-plan
        replan_resp = json.dumps({
            "role": "Refined audit",
            "active_agents": feature_active,
            "agent_instructions": {a: f"focus on {a}" for a in feature_active},
            "rationale": "Phase-2 picks.",
        })
        diagnoses_resp = json.dumps({
            "summary": "Likely sepsis.",
            "patient_score": 4,
            "final_diagnosis": "Sepsis-3 met; SEP-1 partial.",
            "details": "SOFA 5, lactate 3.0",
            "sepsis3_met": True,
            "sep1_compliant": False,
            "next_steps": "- Repeat lactate.",
            "short_term_treatment": "Broad-spectrum AB x 7 days.",
            "mid_term_plan": "Outpatient follow-up day 14.",
        })
        evaluator_resp = json.dumps({
            "flag": "green",
            "task_executed": True,
            "confidence": 88,
            "overall_summary": "Pipeline executed cleanly.",
            "agent_reports": {
                "orchestrator": {"verdict": "ok", "notes": "Plan was clear."},
                "history":      {"verdict": "ok", "notes": "Skipped or ran appropriately."},
                "vitals":       {"verdict": "ok", "notes": "HR captured."},
                "lab":          {"verdict": "ok", "notes": "Lactate captured."},
                "microbiology": {"verdict": "ok", "notes": "No evidence."},
                "pharmacy":     {"verdict": "ok", "notes": "Antibiotic on file."},
                "diagnoses":    {"verdict": "ok", "notes": "Verdict consistent."},
            },
            "missing_data": [],
            "improvement_recommendations": "_None._",
        })
        return RoutedFakeLLM({
            "Vital Signs Specialist": two_part_envelope({"hr_latest": 110}),
            "Laboratory Results Specialist": two_part_envelope({"lactate_latest": 3.0}),
            "Microbiology & Infection Specialist": two_part_envelope({"infection_status": "no_evidence"}),
            "Pharmacy & Fluid-Management Specialist": two_part_envelope({"first_antibiotic_time": "11:00"}),
            "Clinical History Analyst": two_part_envelope({"chronic_conditions": ["CKD"]}),
            "**Diagnoses Agent**": diagnoses_resp,
            "**Evaluator Agent**": evaluator_resp,
            # Phase-2 must be matched BEFORE Phase-1 because the substring
            # "Orchestrator" appears in both prompts; dict iteration is
            # insertion-ordered in Py3.7+ and the routed mock returns the
            # first key found in the system message.
            "Orchestrator (Phase 2 — re-plan)": replan_resp,
            "Orchestrator (Phase 1 — pre-plan)": preplan_resp,
        })

    def test_single_visit_skips_history(self):
        llm = self._llm(multi_visit=False)
        mm = MemoryManager(patient_id="99", base_dir=self.tmp)
        with self._apply_patches():
            result = run_pipeline(
                llm=llm, subject_id=99, selected_hadm_ids=[101],
                user_intent="audit single visit",
                memory_manager=mm,
            )
        decision = result.get("orchestrator_decision") or {}
        self.assertNotIn("history", decision["active_agents"])
        self.assertIsNone(result.get("history_output"))
        self.assertEqual(
            result["diagnoses_output"]["patient_score"], 4,
        )

    def test_multi_visit_runs_history_first(self):
        llm = self._llm(multi_visit=True)
        mm = MemoryManager(patient_id="99", base_dir=self.tmp)
        with self._apply_patches():
            result = run_pipeline(
                llm=llm, subject_id=99, selected_hadm_ids=[101, 102],
                user_intent="audit multi visit",
                memory_manager=mm,
            )
        self.assertIn("history_output", result)
        self.assertEqual(
            result["history_baseline"]["actionable"]["chronic_conditions"],
            ["CKD"],
        )
        decision = result.get("orchestrator_decision") or {}
        self.assertEqual(decision["active_agents"][0], "history")

        msgs_after_history = []
        for messages in llm.calls:
            sys_text = next(
                (m.content for m in messages if m.__class__.__name__ == "SystemMessage"),
                "",
            )
            if "Vital Signs Specialist" in sys_text:
                msgs_after_history.append(
                    next(m.content for m in messages if m.__class__.__name__ == "HumanMessage")
                )
        self.assertTrue(msgs_after_history, "Vitals agent should have been called")
        self.assertIn("Baseline context (from History Agent", msgs_after_history[0])
        self.assertIn("CKD", msgs_after_history[0])

    def test_pipeline_writes_session_files(self):
        llm = self._llm(multi_visit=False)
        mm = MemoryManager(patient_id="99", base_dir=self.tmp)
        with self._apply_patches():
            run_pipeline(
                llm=llm, subject_id=99, selected_hadm_ids=[101],
                user_intent="x", memory_manager=mm,
            )
        import os
        self.assertTrue(os.path.exists(mm.summary_path))
        with open(mm.summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertIn("vitals", summary["agent_outputs"])
        self.assertIn("diagnoses", summary["agent_outputs"])
        self.assertIn("evaluator", summary["agent_outputs"])
        self.assertIn("final_state", summary)

    def test_pipeline_emits_evaluator_flag(self):
        """Pipeline must produce evaluator_output with a valid flag."""
        llm = self._llm(multi_visit=False)
        mm = MemoryManager(patient_id="99", base_dir=self.tmp)
        with self._apply_patches():
            result = run_pipeline(
                llm=llm, subject_id=99, selected_hadm_ids=[101],
                user_intent="audit", memory_manager=mm,
            )
        ev = result.get("evaluator_output") or {}
        self.assertIn(ev.get("flag"), {"green", "yellow", "red"})
        self.assertIsInstance(ev.get("agent_reports"), dict)
        self.assertIn("vitals", ev["agent_reports"])
        self.assertIn("diagnoses", ev["agent_reports"])

    def test_replan_runs_after_history_in_multi_visit(self):
        """After history Phase-2 must run and pick feature agents."""
        llm = self._llm(multi_visit=True)
        mm = MemoryManager(patient_id="99", base_dir=self.tmp)
        with self._apply_patches():
            result = run_pipeline(
                llm=llm, subject_id=99, selected_hadm_ids=[101, 102],
                user_intent="multi visit audit", memory_manager=mm,
            )
        decision = result.get("orchestrator_decision") or {}
        # Phase-2 fills active_agents with feature agents.
        self.assertIn("vitals", decision.get("active_agents", []))
        self.assertIn("lab",    decision.get("active_agents", []))
        # Phase-2 should produce a `replan` trace entry.
        kinds = {e.get("kind") for e in (result.get("agent_trace") or [])
                 if e.get("agent") == "Orchestrator"}
        self.assertIn("preplan", kinds)
        self.assertIn("replan",  kinds)

    # ── helper ─────────────────────────────────────────────────────────────

    def _apply_patches(self):
        patches = _patch_db()

        class _Multi:
            def __enter__(self):
                for p in patches:
                    p.start()

            def __exit__(self, exc_type, exc_val, exc_tb):
                for p in reversed(patches):
                    p.stop()

        return _Multi()


class TestGraphCompiles(unittest.TestCase):
    """Sanity-check the LangGraph itself builds with no missing nodes."""

    def test_build_graph(self):
        mm = MemoryManager(patient_id="0", base_dir=tempfile.mkdtemp())
        llm = SimpleNamespace(invoke=lambda *a, **k: SimpleNamespace(content=""))
        graph = build_graph(llm, mm)
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
