#!/usr/bin/env python3
"""Numerical and boundary checks; no device or network required."""

import copy
import json
import subprocess
import sys
import unittest
from decimal import Decimal
from pathlib import Path

from estimate_memory import calculate, quantity

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "references/kimi-k3-calculations.json").read_text())


def q(value, unit="GiB"):
    return {"value": value, "unit": unit}


class CalculatorTests(unittest.TestCase):
    def test_binary_and_decimal_units(self):
        self.assertEqual(quantity(q(65536, "MiB")), quantity(q(64)))
        self.assertEqual(quantity(q(1, "GB")), Decimal(10**9))
        self.assertNotEqual(quantity(q(1, "GB")), quantity(q(1)))
        with self.assertRaises(ValueError):
            quantity(q(64, "G"))

    def test_invalid_numeric_values(self):
        for value in (-1, True, float("nan"), float("inf"), "64"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantity(q(value))

    def test_small_state_and_checkpoints(self):
        self.assertAlmostEqual(calculate(CASES[0])["state_per_device_gib"], 14.528632164001465)
        self.assertEqual(calculate(CASES[1])["checkpoint_gib"], 0.24609375)
        self.assertEqual(calculate(CASES[2])["checkpoint_gib"], 1.9072265625)

    def test_different_state_shards_and_formats(self):
        case = dict(CASES[0], public_parameters=2**30, expert_parameters=2**30,
                    public_shards=2, ep=4, efsdp=2,
                    public_bytes_per_param=16, expert_bytes_per_param=8)
        result = calculate(case)
        self.assertEqual(result["total_state_gib"], 24)
        self.assertEqual(result["state_per_device_gib"], 9)

    def test_residual_and_mixed_units(self):
        result = calculate(CASES[3])
        self.assertEqual(result["residual_small_gib"], 5.01)
        self.assertEqual(result["residual_large_gib"], 10.02)
        case = dict(CASES[3], measured_small=q(2), known_small=q(2**30, "B"))
        self.assertEqual(calculate(case)["residual_large_gib"], 2)

    def test_negative_residual(self):
        with self.assertRaisesRegex(ValueError, "negative residual"):
            calculate(dict(CASES[3], measured_small=q(1)))

    def test_metric_mismatch_and_reserved(self):
        for metric in ("device", "reserved"):
            with self.assertRaises(ValueError):
                calculate(dict(CASES[3], measurement_metric=metric))

    def test_training_case(self):
        result = calculate(CASES[4])["candidates"][0]
        self.assertEqual(result["min_shards_for_this_overhead"], 854)
        self.assertEqual(result["nodes"], 56)
        self.assertEqual(round(result["estimated_peak_gib"], 1), 58.5)
        self.assertTrue(result["passes_supplied_checks"])

    def test_nonpositive_denominator(self):
        case = copy.deepcopy(CASES[4])
        case["candidates"][0]["remaining"] = q(64)
        result = calculate(case)["candidates"][0]
        self.assertIsNone(result["min_shards_for_this_overhead"])
        self.assertFalse(result["passes_supplied_checks"])

    def test_exact_budget_boundary_and_ceiling(self):
        case = dict(CASES[4], state_total=q(16), capacity=q(1), checkpoint=q(0),
                    budget_fraction=1, experts=16,
                    candidates=[{"ep": 16, "efsdp": 1, "remaining": q(0)}])
        self.assertTrue(calculate(case)["candidates"][0]["passes_supplied_checks"])
        case["state_total"] = q(16.000001)
        result = calculate(case)["candidates"][0]
        self.assertEqual(result["min_shards_for_this_overhead"], 17)
        self.assertFalse(result["passes_supplied_checks"])

    def test_expert_count_not_inferred_from_efsdp(self):
        case = copy.deepcopy(CASES[4])
        case["candidates"] = [{"ep": 448, "efsdp": 2, "remaining": q(10.02)}]
        result = calculate(case)["candidates"][0]
        self.assertEqual(result["experts_per_device"], 2)
        self.assertEqual(result["devices"], 896)

    def test_training_grid_constraints(self):
        case = copy.deepcopy(CASES[4])
        case["candidates"] = [{"ep": 15, "efsdp": 1, "remaining": q(10.02)}]
        result = calculate(case)["candidates"][0]
        self.assertFalse(result["passes_supplied_checks"])
        self.assertGreaterEqual(len(result["reasons"]), 2)
        with self.assertRaises(ValueError):
            calculate(dict(case, tp=2))

    def test_inference_topology_and_unknown_dynamic(self):
        rows = calculate(CASES[5])["candidates"]
        self.assertFalse(rows[0]["passes_supplied_checks"])
        self.assertIn("exceeds single-instance device limit", rows[0]["reasons"])
        self.assertEqual(rows[1]["dp"], 28)
        self.assertAlmostEqual(rows[1]["static_gib"], 45.32727611541748)
        self.assertIsNone(rows[1]["estimated_peak_gib"])
        self.assertIsNone(rows[1]["passes_supplied_checks"])
        self.assertAlmostEqual(rows[2]["physical_headroom_after_static_gib"], 7.35241138458252)

    def test_dynamic_peak_changes_decision(self):
        case = copy.deepcopy(CASES[5])
        case["candidates"] = [case["candidates"][1]]
        case["candidates"][0]["temporary"] = q(10)
        self.assertTrue(calculate(case)["candidates"][0]["passes_supplied_checks"])
        case["candidates"][0]["temporary"] = q(20)
        self.assertFalse(calculate(case)["candidates"][0]["passes_supplied_checks"])

    def test_unknown_topology_remains_unknown(self):
        case = copy.deepcopy(CASES[5])
        case["max_instance_devices"] = None
        case["candidates"] = [case["candidates"][1]]
        case["candidates"][0]["temporary"] = q(0)
        self.assertIsNone(calculate(case)["candidates"][0]["passes_supplied_checks"])

    def test_invalid_inference_grid(self):
        case = copy.deepcopy(CASES[5])
        case["candidates"] = [dict(case["candidates"][0], experts_per_device=3)]
        self.assertFalse(calculate(case)["candidates"][0]["passes_supplied_checks"])
        case = dict(CASES[5], tp=15)
        self.assertFalse(calculate(case)["candidates"][1]["passes_supplied_checks"])

    def test_kda_and_mla_arithmetic(self):
        kda = (2*1536*(7168+128+10*96*128+2*96*64)+4*1536*5*96)/2**30
        self.assertEqual(kda, 0.41033935546875)
        self.assertEqual(24*(512+64)*2, 27648)
        self.assertEqual(27648*16*1536/2**30, 0.6328125)

    def test_cli_success_and_failure(self):
        script = str(ROOT / "scripts/estimate_memory.py")
        run = subprocess.run([sys.executable, script, "-"], input=json.dumps(CASES),
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(len(json.loads(run.stdout)), len(CASES))
        run = subprocess.run([sys.executable, script, "-"], input='{"operation":"unknown"}',
                             text=True, capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("Input error", run.stderr)


if __name__ == "__main__":
    unittest.main()
