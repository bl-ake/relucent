#!/usr/bin/env python3
"""Refresh golden metadata in ``tests/integration/fixtures/manifest.json``.

Uses bundled witness checkpoints only — no dependency on the poly ``analysis`` tree.

Run from the relucent package root::

    uv run --python 3.13 --with torch --extra dev python scripts/refresh_integration_golden.py

Or with the experiments env::

    cd relucent && python scripts/refresh_integration_golden.py --nworkers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_RELUCENT_ROOT = Path(__file__).resolve().parents[1]
if str(_RELUCENT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT / "src"))
if str(_RELUCENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT))

from relucent import Complex  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    MANIFEST_PATH,
    WitnessSpec,
    boundary_shi_for_spec,
    load_manifest,
    load_witness_model,
    run_bfs_ambient,
    tag_set,
    truncated_betti,
    witness_by_id,
)


def _compute_golden(
    model: Any,
    spec: WitnessSpec,
    *,
    nworkers: int,
) -> dict[str, Any]:
    os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")
    ambient = run_bfs_ambient(model, spec, nworkers=nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary_full = ambient.get_boundary_complex(shi, verbose=False)
    boundary_disc = Complex(model).discover_boundary_complex(shi, verbose=False, nworkers=nworkers)
    tags_full = tag_set(boundary_full)
    tags_disc = tag_set(boundary_disc)
    betti_full = truncated_betti(boundary_full)
    betti_disc = truncated_betti(boundary_disc)
    return {
        "n_regions": len(ambient),
        "n_boundary_cells": len(tags_full),
        "truncated_betti": betti_full,
        "methods_agree": tags_full == tags_disc and betti_full == betti_disc,
        "boundary_shi": shi,
        "n_boundary_discover": len(tags_disc),
        "truncated_betti_discover": betti_disc,
    }


def _apply_golden(doc: dict[str, Any], golden: dict[str, Any]) -> None:
    doc["expected"] = {
        "n_regions": golden["n_regions"],
        "n_boundary_cells": golden["n_boundary_cells"],
        "truncated_betti": {str(k): v for k, v in golden["truncated_betti"].items()},
        "methods_agree": golden["methods_agree"],
    }
    doc["boundary_shi"] = int(golden["boundary_shi"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-id", type=str, default=None, help="Refresh one witness only.")
    parser.add_argument("--nworkers", type=int, default=1)
    args = parser.parse_args()

    raw = json.loads(MANIFEST_PATH.read_text())
    witnesses: list[dict[str, Any]] = list(raw.get("witnesses") or [])
    by_id = {str(w["id"]): w for w in witnesses}

    if args.witness_id:
        target_ids = [args.witness_id]
        try:
            witness_by_id(args.witness_id)
        except KeyError as exc:
            raise SystemExit(f"unknown witness id: {args.witness_id!r}") from exc
    else:
        target_ids = [s.id for s in load_manifest()]

    for wid in target_ids:
        spec = witness_by_id(wid)
        print(f"Refreshing {wid} (nworkers={args.nworkers}) ...", flush=True)
        model = load_witness_model(spec)
        golden = _compute_golden(model, spec, nworkers=args.nworkers)
        if not golden["methods_agree"]:
            print(
                "  WARNING: methods disagree:"
                + f" full={golden['n_boundary_cells']} disc={golden['n_boundary_discover']}"
                + f" betti_full={golden['truncated_betti']} disc={golden['truncated_betti_discover']}",
                flush=True,
            )
        doc = by_id.setdefault(wid, {"id": wid, "file": spec.file, "widths": spec.widths})
        _apply_golden(doc, golden)

    MANIFEST_PATH.write_text(json.dumps({"witnesses": witnesses}, indent=2) + "\n")
    print(f"Wrote {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
