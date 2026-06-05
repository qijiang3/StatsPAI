#!/usr/bin/env python3
"""Coverage-campaign helper for the core-module ≥95% initiative.

This tool exists because StatsPAI's coverage can only be measured reliably with
the *whole-package* source (``--cov=statspai``). Scoping ``--cov`` to a
sub-package (e.g. ``--cov=statspai.iv``) reorders imports relative to the
``scipy.optimize`` PyO3/pybind11 stabiliser in ``tests/conftest.py`` and
crashes with ``generic_type: type "ObjSense" is already registered``. So we
always measure with ``--cov=statspai`` and slice per-module numbers out of the
Cobertura XML here.

Usage
-----
    # Per-module % for the six core modules, from any coverage XML:
    python scripts/coverage_campaign.py report [--xml PATH]

    # Exact uncovered line ranges for one module (what still needs a test):
    python scripts/coverage_campaign.py gaps MODULE [--xml PATH] [--per-file]

The default XML is ``.coverage_campaign/baseline_fullsuite.xml`` if present,
else the repo-root committed ``coverage.xml``.

This script is read-only over coverage data; it never runs pytest itself so it
stays fast and side-effect free. Generate a fresh XML with, e.g.::

    pytest tests/ -q --cov-report=xml:.coverage_campaign/latest.xml --cov-report=

Note the authoritative metric is the *full-suite* number: a module is exercised
by many cross-module tests, so running only ``tests/test_<mod>*.py`` understates
its true coverage substantially (iv: 38% module-only vs 87% full-suite).
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

CORE_MODULES = ("did", "iv", "rd", "synth", "dml", "panel")
TARGET = 95.0


def _default_xml() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(here, ".coverage_campaign", "baseline_fullsuite.xml")
    if os.path.exists(cand):
        return cand
    return os.path.join(here, "coverage.xml")


def _module_of(filename: str) -> str | None:
    parts = filename.replace("\\", "/").split("/")
    if "statspai" not in parts:
        return None
    i = parts.index("statspai")
    if i + 1 >= len(parts):
        return None
    name = parts[i + 1]
    return name[:-3] if name.endswith(".py") else name


def _iter_classes(xml_path: str):
    root = ET.parse(xml_path).getroot()
    yield from root.iter("class")


def cmd_report(args) -> int:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # mod -> [covered, total]
    for cls in _iter_classes(args.xml):
        mod = _module_of(cls.get("filename", ""))
        if mod is None:
            continue
        lines = cls.find("lines")
        if lines is None:
            continue
        for ln in lines.findall("line"):
            agg[mod][1] += 1
            if int(ln.get("hits", "0")) > 0:
                agg[mod][0] += 1

    print(f"# coverage source: {args.xml}\n")
    print(f"{'module':10}{'cov%':>8}{'covered':>10}{'total':>8}{'+to 95%':>10}{'':>4}")
    print("-" * 50)
    for m in CORE_MODULES:
        c, tot = agg[m]
        pct = 100 * c / tot if tot else 0.0
        need = max(0, int(__import__("math").ceil(TARGET / 100 * tot)) - c)
        flag = "OK" if pct >= TARGET else ""
        print(f"{m:10}{pct:8.1f}{c:10}{tot:8}{need:10}{flag:>4}")
    return 0


def cmd_gaps(args) -> int:
    mod = args.module
    per_file: dict[str, list[int]] = defaultdict(list)
    for cls in _iter_classes(args.xml):
        fn = cls.get("filename", "")
        if _module_of(fn) != mod:
            continue
        lines = cls.find("lines")
        if lines is None:
            continue
        miss = [
            int(ln.get("number"))
            for ln in lines.findall("line")
            if int(ln.get("hits", "0")) == 0
        ]
        if miss:
            per_file[fn] = sorted(miss)

    def _ranges(nums: list[int]) -> str:
        out, start, prev = [], None, None
        for n in nums:
            if start is None:
                start = prev = n
            elif n == prev + 1:
                prev = n
            else:
                out.append(f"{start}" if start == prev else f"{start}-{prev}")
                start = prev = n
        if start is not None:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
        return ", ".join(out)

    print(f"# uncovered lines for module '{mod}'  (source: {args.xml})\n")
    total_missing = 0
    for fn in sorted(per_file, key=lambda f: -len(per_file[f])):
        miss = per_file[fn]
        total_missing += len(miss)
        short = fn.replace("\\", "/").split("statspai/", 1)[-1]
        if args.per_file:
            print(f"{len(miss):5}  statspai/{short}\n        {_ranges(miss)}")
        else:
            print(f"{len(miss):5}  statspai/{short}")
    print(f"\n  total uncovered: {total_missing} lines across {len(per_file)} files")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="per-module coverage for the 6 core modules")
    pr.add_argument("--xml", default=_default_xml())
    pr.set_defaults(func=cmd_report)

    pg = sub.add_parser("gaps", help="uncovered line ranges for one module")
    pg.add_argument("module", choices=CORE_MODULES)
    pg.add_argument("--xml", default=_default_xml())
    pg.add_argument(
        "--per-file",
        action="store_true",
        help="also print the exact line ranges per file",
    )
    pg.set_defaults(func=cmd_gaps)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
