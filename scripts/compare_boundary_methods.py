#!/usr/bin/env python3
"""Compare get_boundary_complex vs discover_boundary_complex for one witness."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_RELUCENT_ROOT = Path(__file__).resolve().parents[1]
if str(_RELUCENT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT / "src"))
if str(_RELUCENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT))

from relucent import Complex  # noqa: E402
from relucent.topology import ChainComplexInconsistent  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    boundary_shi_for_spec,
    load_witness_model,
    run_bfs_ambient,
    tag_set,
    truncated_betti,
    witness_by_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-id", type=str, required=True)
    parser.add_argument("--nworkers", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")
    spec = witness_by_id(args.witness_id)
    print(f"=== {spec.id} (nworkers={args.nworkers}) ===", flush=True)

    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=args.nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    print(f"ambient regions: {len(ambient)}", flush=True)

    boundary_full = ambient.get_boundary_complex(shi, verbose=False)
    boundary_disc = Complex(model).discover_boundary_complex(shi, verbose=False, nworkers=args.nworkers)

    tags_full = tag_set(boundary_full)
    tags_disc = tag_set(boundary_disc)
    betti_full = truncated_betti(boundary_full)
    betti_disc = truncated_betti(boundary_disc)

    only_full = tags_full - tags_disc
    only_disc = tags_disc - tags_full

    print(f"boundary full: {len(tags_full)} discover: {len(tags_disc)}", flush=True)
    print(f"only_full: {len(only_full)} only_discover: {len(only_disc)}", flush=True)
    print(f"betti_full: {betti_full}", flush=True)
    print(f"betti_disc: {betti_disc}", flush=True)
    print(f"methods_agree: {tags_full == tags_disc and betti_full == betti_disc}", flush=True)

    try:
        boundary_full.get_betti_numbers(compactify=False, verify_chain_complex=True, verbose=False)
        print("chain_complex full: OK", flush=True)
    except ChainComplexInconsistent as exc:
        print(f"chain_complex full: FAIL {exc}", flush=True)

    try:
        boundary_disc.get_betti_numbers(compactify=False, verify_chain_complex=True, verbose=False)
        print("chain_complex discover: OK", flush=True)
    except ChainComplexInconsistent as exc:
        print(f"chain_complex discover: FAIL {exc}", flush=True)


if __name__ == "__main__":
    main()
