"""Smoke test: every refactored module must import without error."""

from __future__ import annotations

import importlib
import unittest


MODULES = [
    "agents.state",
    "agents.memory_manager_agent",
    "agents.orchestrator_agent",
    "agents.history_agent",
    "agents.vitals_agent",
    "agents.lab_agent",
    "agents.microbiology_agent",
    "agents.pharmacy_agent",
    "agents.diagnoses_agent",
    "agents.graph",
    "agents._agent_utils",
    "app.workspace",
    "app.history",
    "app.controller",
]


class TestImports(unittest.TestCase):

    def test_modules_import_clean(self):
        for mod_name in MODULES:
            with self.subTest(module=mod_name):
                importlib.import_module(mod_name)


class TestObsoleteRemoved(unittest.TestCase):

    def test_diagnostician_module_gone(self):
        with self.assertRaises(ImportError):
            importlib.import_module("agents.diagnostician_agent")

    def test_compliance_module_gone(self):
        with self.assertRaises(ImportError):
            importlib.import_module("agents.compliance_agent")


if __name__ == "__main__":
    unittest.main()
