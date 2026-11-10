#!/usr/bin/env python3
"""Deterministic memory ledger; Python standard library only.

See references/calculator.md. This tool validates supplied arithmetic and a
limited parallel layout, not model compatibility or actual device execution.
"""

import argparse
import json
import sys
from decimal import Decimal, ROUND_CEILING
from pathlib import Path


UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30,
         "KB": 10**3, "MB": 10**6, "GB": 10**9}
GIB = Decimal(2**30)


def number(value, name, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{name}: expected a JSON number")
    value = Decimal(str(value))
    if not value.is_finite() or value < 0 or (positive and value == 0):
        raise ValueError(f"{name}: expected a finite {'positive' if positive else 'nonnegative'} value")
    if integer and value != value.to_integral_value():
        raise ValueError(f"{name}: expected an integer")
    return value


def count(data, key, zero=False):
    return int(number(data[key], key, positive=not zero, integer=True))


def quantity(value):
    if not isinstance(value, dict) or set(value) != {"value", "unit"}:
        raise ValueError('memory quantity must be {"value": number, "unit": "GiB"}')
    unit = value["unit"]
    if unit not in UNITS:
        raise ValueError(f"unsupported or ambiguous unit: {unit!r}")
    return number(value["value"], "memory value") * UNITS[unit]


def gib(value):
    return float(value / GIB)


def ceil(value):
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def budget(data):
    capacity = quantity(data["capacity"])
    fraction = number(data["budget_fraction"], "budget_fraction", positive=True)
    if capacity == 0 or fraction > 1:
        raise ValueError("capacity must be positive and budget_fraction in (0, 1]")
    return capacity, capacity * fraction


def nonempty_candidates(data):
    rows = data["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidates must be a nonempty list")
    return rows


def state(data):
    pd = count(data, "public_parameters", zero=True)
    pe = count(data, "expert_parameters", zero=True)
    bd = number(data["public_bytes_per_param"], "public_bytes_per_param", positive=True)
    be = number(data["expert_bytes_per_param"], "expert_bytes_per_param", positive=True)
    fd, ep, ef = (count(data, key) for key in ("public_shards", "ep", "efsdp"))
    return {"total_state_gib": gib(pd * bd + pe * be),
            "public_state_per_device_gib": gib(pd * bd / fd),
            "expert_state_per_device_gib": gib(pe * be / (ep * ef)),
            "state_per_device_gib": gib(pd * bd / fd + pe * be / (ep * ef))}


def checkpoint(data):
    elements = count(data, "boundaries", zero=True)
    for key in ("batch", "sequence", "hidden"):
        elements *= count(data, key)
    size = elements * number(data["bytes_per_element"], "bytes_per_element", positive=True)
    return {"checkpoint_gib": gib(size)}


def extrapolate(data):
    metric = data["measurement_metric"]
    if metric not in ("allocated", "device") or metric != data["known_metric"]:
        raise ValueError("use matching allocated/device measurement and known-item metrics; reserved is not an additive tensor ledger")
    residual = quantity(data["measured_small"]) - quantity(data["known_small"])
    if residual < 0:
        raise ValueError("negative residual: check units, stage, residency, offload and state format")
    ratio = Decimal(count(data, "expert_tokens_large")) / count(data, "expert_tokens_small")
    return {"residual_small_gib": gib(residual), "expert_token_ratio": float(ratio),
            "residual_large_gib": gib(residual * ratio),
            "basis": "first-order extrapolation; scaling applicability must be checked separately"}


def training_grid(data):
    # This mode deliberately supports only the uniform-state TP1/CP1/PP1 grid.
    for key in ("tp", "cp", "pp"):
        if count(data, key) != 1:
            raise ValueError("training_grid supports TP1/CP1/PP1 only; use a stage/rank-specific ledger for other layouts")
    capacity, allowed = budget(data)
    total = quantity(data["state_total"])
    ckpt = quantity(data["checkpoint"])
    experts = count(data, "experts")
    node = count(data, "devices_per_node")
    available = data.get("available_devices")
    if available is not None:
        available = count(data, "available_devices")
    results = []
    for row in nonempty_candidates(data):
        ep, ef = count(row, "ep"), count(row, "efsdp")
        devices = ep * ef
        remaining = quantity(row["remaining"])
        denominator = allowed - ckpt - remaining
        peak = total / devices + ckpt + remaining
        reasons = []
        if experts % ep:
            reasons.append("EP does not evenly divide experts")
        if devices % node:
            reasons.append("device count is not a whole number of nodes")
        if available is not None and devices > available:
            reasons.append("exceeds available devices")
        if denominator <= 0:
            reasons.append("non-state overhead exhausts budget")
        elif peak > allowed:
            reasons.append("estimated peak exceeds budget")
        results.append({"ep": ep, "efsdp": ef, "fsdp": devices,
                        "devices": devices, "nodes": devices // node if devices % node == 0 else None,
                        "experts_per_device": experts // ep if experts % ep == 0 else None,
                        "min_shards_for_this_overhead": ceil(total / denominator) if denominator > 0 else None,
                        "estimated_peak_gib": gib(peak), "budget_headroom_gib": gib(allowed - peak),
                        "physical_headroom_gib": gib(capacity - peak),
                        "passes_supplied_checks": not reasons, "reasons": reasons})
    return {"scope": "uniform state sharding, TP1/CP1/PP1; supplied overhead per candidate",
            "candidates": results}


def inference_grid(data):
    capacity, allowed = budget(data)
    experts, tp, node = (count(data, key) for key in ("experts", "tp", "devices_per_node"))
    pe = count(data, "expert_parameters", zero=True)
    pd = count(data, "public_tp_parameters", zero=True)
    prep = count(data, "replicated_parameters", zero=True)
    be = number(data["expert_weight_bytes"], "expert_weight_bytes", positive=True)
    bd = number(data["public_weight_bytes"], "public_weight_bytes", positive=True)
    limit = data["max_instance_devices"]
    if limit is not None:
        limit = count(data, "max_instance_devices")
    results = []
    for row in nonempty_candidates(data):
        per_rank = count(row, "experts_per_device")
        if experts % per_rank:
            results.append({"experts_per_device": per_rank, "passes_supplied_checks": False,
                            "reasons": ["experts_per_device does not divide experts"]})
            continue
        ep = experts // per_rank
        weight = be * pe / ep + bd * (Decimal(pd) / tp + prep)
        static = weight + quantity(row["resident"]) + quantity(row["cache"])
        dynamic = row["temporary"]
        peak = static + quantity(dynamic) if dynamic is not None else None
        reasons, unknowns = [], []
        if ep % tp:
            reasons.append("EP is not divisible by TP in the DP*TP layout")
        if ep % node:
            reasons.append("instance does not occupy whole nodes")
        if limit is None:
            unknowns.append("instance topology limit not supplied")
        elif ep > limit:
            reasons.append("exceeds single-instance device limit")
        if dynamic is None:
            unknowns.append("prefill/decode temporary peak not supplied")
        if (peak if peak is not None else static) > allowed:
            reasons.append("memory estimate exceeds budget")
        results.append({"experts_per_device": per_rank, "ep": ep, "tp": tp,
                        "dp": ep // tp if ep % tp == 0 else None, "devices": ep,
                        "nodes": ep // node if ep % node == 0 else None,
                        "weight_gib": gib(weight), "static_gib": gib(static),
                        "estimated_peak_gib": gib(peak) if peak is not None else None,
                        "budget_headroom_after_static_gib": gib(allowed - static),
                        "physical_headroom_after_static_gib": gib(capacity - static),
                        "budget_headroom_after_peak_gib": gib(allowed - peak) if peak is not None else None,
                        "passes_supplied_checks": False if reasons else (None if unknowns else True),
                        "reasons": reasons, "unknowns": unknowns})
    return {"scope": "EP=DP*TP; experts sharded by EP, public weights by TP or replicated; no PP",
            "candidates": results}


OPERATIONS = {"state": state, "checkpoint": checkpoint, "extrapolate": extrapolate,
              "training_grid": training_grid, "inference_grid": inference_grid}


def calculate(data):
    operation = data["operation"]
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation!r}")
    return {"operation": operation, **OPERATIONS[operation](data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON object or list of independent calculation objects; '-' for stdin")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        data = json.loads(raw, parse_float=Decimal)
        result = [calculate(item) for item in data] if isinstance(data, list) else calculate(data)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
