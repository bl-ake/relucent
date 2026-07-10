#!/usr/bin/env python3
"""Diagnose SHI bound sensitivity for integration witness ``shi_bound_5303``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_RELUCENT_ROOT = Path(__file__).resolve().parents[1]
if str(_RELUCENT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT / "src"))
if str(_RELUCENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RELUCENT_ROOT))

import numpy as np  # noqa: E402

from relucent.calculations import get_shis  # noqa: E402
from relucent.poly import Polyhedron  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    boundary_shi_for_spec,
    default_bound,
    load_witness_model,
    run_bfs_ambient,
    witness_by_id,
)

_LARGE_BOUND = 1e8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-id", type=str, default="shi_bound_5303")
    parser.add_argument("--nworkers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("DISABLE_RESEARCH_WARNING", "1")
    spec = witness_by_id(args.witness_id)
    model = load_witness_model(spec)
    ambient = run_bfs_ambient(model, spec, nworkers=args.nworkers, verify=True)
    shi = boundary_shi_for_spec(ambient, spec)
    boundary = ambient.get_boundary_complex(shi, verbose=False)
    net_bound = default_bound(model)

    hits = []
    on_boundary = 0
    for poly in boundary:
        ss = np.asarray(poly.ss_np, dtype=np.int8).reshape(1, -1)
        if ss.ravel()[shi] != 0:
            continue
        on_boundary += 1
        ss_pos = ss.copy()
        ss_pos.ravel()[shi] = 1
        ppos = ambient[ss_pos]
        for escalate in (False, True):
            sh_large = get_shis(
                Polyhedron(model, ppos.ss_np, bound=_LARGE_BOUND),
                bound=_LARGE_BOUND,
                escalate_bound=escalate,
            )
            sh_net = get_shis(
                Polyhedron(model, ppos.ss_np, bound=net_bound),
                bound=net_bound,
                escalate_bound=escalate,
            )
            if not escalate and shi not in sh_large and shi in sh_net:
                hits.append({"tag": poly.tag.hex(), "sh_large": sh_large, "sh_net": sh_net})

    report = {
        "witness_id": spec.id,
        "boundary_shi": shi,
        "net_bound": net_bound,
        "large_bound": _LARGE_BOUND,
        "n_boundary_on_shi": on_boundary,
        "hits_fixed_bound": len(hits),
        "hits": hits[:20],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
