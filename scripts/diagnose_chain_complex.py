#!/usr/bin/env python3
"""Diagnose ∂²=0 on untruncated vs truncated meta-graphs for an integration witness.

Run from the relucent package root::

    python scripts/diagnose_chain_complex.py --witness-id meta_graph_large --nworkers 8
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
from relucent.topology import ChainComplexInconsistent, get_betti_numbers  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)


def _check(label: str, meta: Any) -> dict[str, Any]:
    try:
        get_betti_numbers(meta, verify_chain_complex=True, verbose=False)
        return {"label": label, "ok": True}
    except ChainComplexInconsistent as exc:
        return {"label": label, "ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-id", type=str, default="meta_graph_large")
    parser.add_argument("--nworkers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON report here.")
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")
    spec = witness_by_id(args.witness_id)
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=args.nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary = ambient.get_boundary_complex(shi, verbose=False)

    meta = boundary.get_meta_graph(verbose=False)
    meta_untrunc = meta.copy()
    meta_trunc = meta.copy()
    Complex.truncate_meta_graph(meta_trunc)

    report = {
        "witness_id": spec.id,
        "n_regions": len(ambient),
        "n_boundary": len(boundary),
        "meta_nodes": meta.number_of_nodes(),
        "meta_edges": meta.number_of_edges(),
        "trunc_nodes": meta_trunc.number_of_nodes(),
        "trunc_edges": meta_trunc.number_of_edges(),
        "checks": [
            _check("untruncated", meta_untrunc),
            _check("truncated", meta_trunc),
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
