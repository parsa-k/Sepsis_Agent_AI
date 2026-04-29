"""Tests for the dynamic Orchestrator: rule enforcement, parsing, routing."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from agents.memory_manager_agent import MemoryManager
from agents.orchestrator_agent import (
    run_orchestrator,
    needs_history_first,
    propagate_history_baseline,
    _enforce_rules,
)


class FakeLLM:
    """Returns canned responses; records every prompt for assertions."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return SimpleNamespace(content=self.response)


# ── _enforce_rules ──────────────────────────────────────────────────────────

class TestEnforceRules(unittest.TestCase):

    def _flags(self, **overrides):
        base = {
            "vitals": True, "labs": True, "microbiology": True,
            "pharmacy": True, "icu": True, "diagnoses": True,
        }
        base.update(overrides)
        return base

    def test_single_visit_strips_history_even_if_llm_added_it(self):
        decision = {
            "active_agents": ["history", "vitals", "lab"],
            "history_first": True,
        }
        out = _enforce_rules(
            decision, multi_visit=False,
            aggregated_flags=self._flags(),
            user_intent="evaluate sepsis", raw="",
        )
        self.assertNotIn("history", out["active_agents"])
        self.assertFalse(out["history_first"])

    def test_multi_visit_forces_history_first(self):
        decision = {
            "active_agents": ["vitals"],  # LLM forgot history
        }
        out = _enforce_rules(
            decision, multi_visit=True,
            aggregated_flags=self._flags(),
            user_intent="multi-visit audit", raw="",
        )
        self.assertEqual(out["active_agents"][0], "history")
        self.assertTrue(out["history_first"])

    def test_unknown_agents_are_filtered(self):
        decision = {"active_agents": ["vitals", "wizardry", "lab"]}
        out = _enforce_rules(
            decision, multi_visit=False,
            aggregated_flags=self._flags(),
            user_intent="x", raw="",
        )
        self.assertNotIn("wizardry", out["active_agents"])

    def test_data_present_agents_are_added_when_missing(self):
        decision = {"active_agents": []}
        out = _enforce_rules(
            decision, multi_visit=False,
            aggregated_flags=self._flags(),
            user_intent="x", raw="",
        )
        for agent in ("vitals", "lab", "microbiology", "pharmacy"):
            self.assertIn(agent, out["active_agents"])

    def test_default_instructions_filled(self):
        decision = {"active_agents": ["vitals"]}
        out = _enforce_rules(
            decision, multi_visit=False,
            aggregated_flags=self._flags(microbiology=False, pharmacy=False, labs=False),
            user_intent="lactate trend", raw="",
        )
        self.assertIn("vitals", out["agent_instructions"])
        self.assertIn("lactate trend", out["agent_instructions"]["vitals"])


# ── run_orchestrator end-to-end with fake LLM ───────────────────────────────

class TestRunOrchestrator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="orch_test_")
        self.mm = MemoryManager(patient_id="999", base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _state(self, multi: bool):
        if multi:
            sel = [10001, 10002]
            flags = {
                10001: {"vitals": True, "labs": True, "microbiology": True,
                        "pharmacy": True, "icu": True, "diagnoses": True},
                10002: {"vitals": True, "labs": False, "microbiology": False,
                        "pharmacy": True, "icu": False, "diagnoses": True},
            }
        else:
            sel = [10001]
            flags = {
                10001: {"vitals": True, "labs": True, "microbiology": False,
                        "pharmacy": True, "icu": True, "diagnoses": True},
            }
        return {
            "subject_id": 999,
            "selected_hadm_ids": sel,
            "available_data_flags": flags,
            "user_intent": "Audit sepsis & SEP-1",
        }

    def test_single_visit_excludes_history(self):
        canned = json.dumps({
            "role": "Sepsis-3 audit",
            "multi_visit": False, "history_first": False,
            "active_agents": ["vitals", "lab", "pharmacy"],
            "agent_instructions": {"vitals": "v", "lab": "l", "pharmacy": "p"},
        })
        llm = FakeLLM(canned)
        out = run_orchestrator(self._state(multi=False), llm, memory_manager=self.mm)
        decision = out["orchestrator_decision"]
        self.assertNotIn("history", decision["active_agents"])
        self.assertFalse(decision["history_first"])
        self.assertEqual(needs_history_first({"orchestrator_decision": decision}), "vitals")

    def test_multi_visit_forces_history_first(self):
        canned = json.dumps({
            "role": "Multi-visit audit",
            "active_agents": ["history", "vitals", "lab", "pharmacy"],
            "history_first": True,
            "agent_instructions": {
                "history": "h", "vitals": "v", "lab": "l", "pharmacy": "p",
            },
        })
        llm = FakeLLM(canned)
        out = run_orchestrator(self._state(multi=True), llm, memory_manager=self.mm)
        decision = out["orchestrator_decision"]
        self.assertEqual(decision["active_agents"][0], "history")
        self.assertTrue(decision["history_first"])
        self.assertEqual(needs_history_first({"orchestrator_decision": decision}), "history")

    def test_propagate_history_baseline_copies_part1(self):
        state = {
            "history_output": {
                "part1_payload": {
                    "actionable": {"chronic_conditions": ["CKD"]},
                    "source_records": ["d.icd_code N18.6"],
                },
                "part2_reasoning": "long story",
            },
        }
        out = propagate_history_baseline(state)
        self.assertIn("history_baseline", out)
        baseline = out["history_baseline"]
        self.assertEqual(
            baseline["actionable"], {"chronic_conditions": ["CKD"]},
        )

    def test_recovers_when_llm_returns_garbage(self):
        llm = FakeLLM("totally not json")
        out = run_orchestrator(self._state(multi=False), llm, memory_manager=self.mm)
        decision = out["orchestrator_decision"]
        self.assertIn("active_agents", decision)
        self.assertGreaterEqual(len(decision["active_agents"]), 1)


if __name__ == "__main__":
    unittest.main()
