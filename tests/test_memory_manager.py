"""Unit tests for ``agents.memory_manager_agent.MemoryManager``."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from agents.memory_manager_agent import MemoryManager, empty_output


class TestMemoryManagerParsing(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="mm_test_")
        self.mm = MemoryManager(patient_id="42", base_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── parsing ────────────────────────────────────────────────────────────

    def test_parses_strict_fenced_json(self):
        raw = """```json
        {
          "part1_payload": {
            "actionable": {"hr_latest": 110},
            "source_records": ["chartevents 2150-05-16 09:01 HR=110 bpm"]
          },
          "part2_reasoning": "Tachycardia trend persists."
        }
        ```"""
        out = self.mm._parse_two_part(raw)
        self.assertEqual(out["part1_payload"]["actionable"], {"hr_latest": 110})
        self.assertIn("Tachycardia", out["part2_reasoning"])
        self.assertNotIn("parse_error", out)

    def test_parses_raw_json_object(self):
        raw = json.dumps({
            "part1_payload": {"actionable": {"x": 1}, "source_records": []},
            "part2_reasoning": "raw object",
        })
        out = self.mm._parse_two_part(raw)
        self.assertEqual(out["part1_payload"]["actionable"], {"x": 1})
        self.assertEqual(out["part2_reasoning"], "raw object")

    def test_parses_markdown_headings_with_json_part1(self):
        raw = (
            "## Part 1: Actionable\n"
            "```json\n"
            '{"actionable": {"map_latest": 62}, "source_records": ["row 1"]}\n'
            "```\n\n"
            "## Part 2: Reasoning\n"
            "MAP is borderline; consistent with vasoplegia."
        )
        out = self.mm._parse_two_part(raw)
        self.assertEqual(
            out["part1_payload"]["actionable"], {"map_latest": 62},
        )
        self.assertIn("vasoplegia", out["part2_reasoning"])

    def test_total_failure_records_parse_error(self):
        raw = "I am just prose with no structure at all."
        out = self.mm._parse_two_part(raw)
        self.assertIn("parse_error", out)
        self.assertIn("raw_response", out["part1_payload"]["actionable"])
        self.assertEqual(out["part2_reasoning"], raw)

    def test_coerce_part1_normalises_shape(self):
        weird = {"actionable": "string-not-dict", "source_records": "single"}
        out = MemoryManager._coerce_part1(weird)
        self.assertIsInstance(out["actionable"], dict)
        self.assertIsInstance(out["source_records"], list)

    # ── persistence ────────────────────────────────────────────────────────

    def test_standardize_output_writes_summary_and_jsonl(self):
        raw = json.dumps({
            "part1_payload": {"actionable": {"a": 1}, "source_records": []},
            "part2_reasoning": "ok",
        })
        env = self.mm.standardize_output("vitals", raw)
        self.assertEqual(env["agent_name"], "vitals")

        self.assertTrue(os.path.exists(self.mm.summary_path))
        with open(self.mm.summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertIn("vitals", summary["agent_outputs"])
        self.assertEqual(
            summary["agent_outputs"]["vitals"]["part1_payload"]["actionable"],
            {"a": 1},
        )

        self.assertTrue(os.path.exists(self.mm.log_path))
        with open(self.mm.log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(any(line["kind"] == "output" for line in lines))

    def test_record_skipped_writes_skipped_flag(self):
        env = self.mm.record_skipped("microbiology", "no microbiology data")
        self.assertTrue(env["skipped"])
        self.assertEqual(env["agent_name"], "microbiology")

    def test_finalise_dumps_trimmed_state(self):
        state = {
            "subject_id": 42,
            "vitals_raw": "huge text" * 1000,
            "visits_data": {"x": "y"},
            "diagnoses_output": {"summary": "ok"},
        }
        path = self.mm.finalise(state)
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("final_state", saved)
        self.assertNotIn("vitals_raw", saved["final_state"])
        self.assertNotIn("visits_data", saved["final_state"])
        self.assertEqual(saved["final_state"]["subject_id"], 42)

    def test_empty_output_helper(self):
        out = empty_output("lab", "no labs")
        self.assertTrue(out["skipped"])
        self.assertEqual(out["part1_payload"]["actionable"]["status"], "skipped")
        self.assertEqual(out["agent_name"], "lab")


if __name__ == "__main__":
    unittest.main()
