"""Shared helpers for Relucent integration tests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from relucent import Complex
from relucent._network_scale import default_polyhedron_bound
from relucent.exploration import finalize_ambient_search
from relucent.utils import TorchMLP

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
_HYPERPLANE_START_ERROR = "Start point must not be on a hyperplane"


@dataclass(frozen=True)
class WitnessSpec:
    id: str
    file: str
    widths: list[int]
    add_last_relu: bool
    seed: int
    experiment_key: int | None
    bfs_geometry_properties: list[str]
    bfs_max_polys: float
    expected_n_regions: int | None
    expected_n_boundary_cells: int | None
    expected_truncated_betti: dict[int, int] | None
    methods_agree: bool
    boundary_shi: int | None = None


def load_manifest() -> list[WitnessSpec]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"integration manifest missing: {MANIFEST_PATH}")
    raw = json.loads(MANIFEST_PATH.read_text())
    witnesses = raw.get("witnesses") or []
    out: list[WitnessSpec] = []
    for entry in witnesses:
        bfs = entry.get("bfs") or {}
        expected = entry.get("expected") or {}
        betti_raw = expected.get("truncated_betti")
        betti = {int(k): int(v) for k, v in betti_raw.items()} if betti_raw else None
        out.append(
            WitnessSpec(
                id=str(entry["id"]),
                file=str(entry["file"]),
                widths=[int(w) for w in entry["widths"]],
                add_last_relu=bool(entry.get("add_last_relu", True)),
                seed=int(entry.get("seed", 0)),
                experiment_key=int(entry["experiment_key"]) if entry.get("experiment_key") is not None else None,
                bfs_geometry_properties=[str(x) for x in bfs.get("geometry_properties", ["finite"])],
                bfs_max_polys=float(bfs.get("max_polys", 0) or 0),
                expected_n_regions=int(expected["n_regions"]) if expected.get("n_regions") is not None else None,
                expected_n_boundary_cells=int(expected["n_boundary_cells"])
                if expected.get("n_boundary_cells") is not None
                else None,
                expected_truncated_betti=betti,
                methods_agree=bool(expected.get("methods_agree", True)),
                boundary_shi=int(entry["boundary_shi"]) if entry.get("boundary_shi") is not None else None,
            )
        )
    return out


def witness_by_id(witness_id: str) -> WitnessSpec:
    for spec in load_manifest():
        if spec.id == witness_id:
            return spec
    raise KeyError(f"unknown witness id: {witness_id!r}")


def _fc_indices_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> list[int]:
    indices: set[int] = set()
    for key in state_dict:
        match = re.fullmatch(r"fc(\d+)\.weight", key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _build_torch_mlp_from_bundle(state_dict: Mapping[str, torch.Tensor], widths: list[int]) -> TorchMLP:
    """Rebuild an experiment-style ``TorchMLP`` (``fc{i}_relu`` on the last block)."""
    from collections import OrderedDict

    fc_indices = _fc_indices_from_state_dict(state_dict)
    if not fc_indices:
        raise ValueError("fixture state_dict has no fc*.weight keys")

    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in fc_indices:
        weight = state_dict[f"fc{i}.weight"]
        in_f, out_f = int(weight.shape[1]), int(weight.shape[0])
        layers[f"fc{i}"] = nn.Linear(in_f, out_f, dtype=torch.float64)
        if i < fc_indices[-1]:
            layers[f"relu{i}"] = nn.ReLU()
        else:
            layers[f"fc{i}_relu"] = nn.ReLU()

    model = TorchMLP(layers, list(widths))
    model.load_state_dict(dict(state_dict))
    model.eval()
    return model


def load_witness_model(spec: WitnessSpec) -> TorchMLP:
    path = FIXTURES_DIR / spec.file
    if not path.is_file():
        raise FileNotFoundError(f"witness fixture missing: {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or "state_dict" not in bundle:
        raise ValueError(f"fixture {path} must be a dict with state_dict")
    state_dict = bundle["state_dict"]
    widths = [int(w) for w in bundle.get("widths", spec.widths)]
    return _build_torch_mlp_from_bundle(state_dict, widths)


def _model_device_dtype(model: TorchMLP) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


def _infer_input_shape(model: TorchMLP) -> tuple[int, int]:
    return (1, int(model.widths[0]))


def run_bfs_ambient(
    model: TorchMLP,
    spec: WitnessSpec,
    *,
    nworkers: int,
    verify: bool = True,
    max_attempts: int = 128,
) -> Complex:
    """BFS the full ambient complex with seeded random-start retry."""
    model_device, model_dtype = _model_device_dtype(model)
    input_shape = _infer_input_shape(model)
    bfs_kwargs: dict[str, Any] = {
        "verbose": False,
        "geometry_properties": spec.bfs_geometry_properties,
        "nworkers": nworkers,
        "verify": verify,
    }
    if spec.bfs_max_polys > 0:
        bfs_kwargs["max_polys"] = int(spec.bfs_max_polys)

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = torch.Generator(device=model_device)
    rng.manual_seed(spec.seed)

    last_err: ValueError | None = None
    for _attempt in range(max_attempts):
        cplx = Complex(model)
        start = torch.randn(input_shape, device=model_device, dtype=model_dtype, generator=rng)
        try:
            cplx.bfs(start=start, **bfs_kwargs)
        except ValueError as exc:
            if _HYPERPLANE_START_ERROR not in str(exc):
                raise
            last_err = exc
            continue
        finalize_ambient_search(cplx, complete=bool(cplx.complete), verify=verify)
        return cplx

    raise ValueError(
        f"BFS failed after {max_attempts} random starts for witness {spec.id}: {last_err}",
    ) from last_err


def output_neuron_shi(cplx: Complex) -> int:
    last_layer = max(cplx.ss_layers)
    for shi, (layer_idx, _) in enumerate(cplx.ssi2maski):
        if layer_idx == last_layer:
            return int(shi)
    raise RuntimeError("could not resolve output-neuron SHI")


def boundary_shi_for_spec(cplx: Complex, spec: WitnessSpec) -> int:
    if spec.boundary_shi is not None:
        return int(spec.boundary_shi)
    return output_neuron_shi(cplx)


def tag_set(cplx: Complex) -> set[bytes]:
    return {p.tag for p in cplx}


def truncated_betti(boundary: Complex) -> dict[int, int]:
    raw = boundary.get_betti_numbers(compactify=False, verbose=False)
    return {int(k): int(v) for k, v in raw.items()}


def snapshot_shis_by_tag(cplx: Complex) -> dict[bytes, list[int]]:
    return {p.tag: [int(s) for s in p.shis] for p in cplx}


def restore_shis_by_tag(cplx: Complex, shis_by_tag: Mapping[bytes, list[int]]) -> None:
    for poly in cplx:
        shis = shis_by_tag.get(poly.tag)
        if shis is not None:
            poly._shis = list(shis)


def export_dual_graph_payload(cplx: Complex) -> dict[str, Any]:
    graph = cplx.get_dual_graph(relabel=True, verbose=False)
    source = 0
    initial_ss = np.asarray(cplx.index2poly[source].ss_np, dtype=np.int8)
    return {
        "graph": graph,
        "initial_ss": initial_ss,
        "source": int(source),
        "shis_by_tag": snapshot_shis_by_tag(cplx),
    }


def recover_from_dual_payload(model: TorchMLP, payload: Mapping[str, Any]) -> Complex:
    cplx = Complex(model)
    cplx.recover_from_dual_graph(
        payload["graph"],
        initial_ss=payload["initial_ss"],
        source=int(payload.get("source", 0)),
    )
    shis = payload.get("shis_by_tag")
    if shis:
        restore_shis_by_tag(cplx, shis)
    return cplx


def write_failure_artifacts(
    outdir: str | Path,
    witness_id: str,
    *,
    only_full: set[bytes] | None = None,
    only_discover: set[bytes] | None = None,
    betti_full: dict[int, int] | None = None,
    betti_discover: dict[int, int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    root = Path(outdir) / witness_id
    root.mkdir(parents=True, exist_ok=True)

    def _write_tags(name: str, tags: set[bytes]) -> None:
        (root / name).write_text("\n".join(t.hex() for t in sorted(tags)) + ("\n" if tags else ""))

    if only_full is not None:
        _write_tags("only_full.txt", only_full)
    if only_discover is not None:
        _write_tags("only_discover.txt", only_discover)
    if betti_full is not None:
        (root / "betti_full.json").write_text(json.dumps(betti_full, indent=2, sort_keys=True) + "\n")
    if betti_discover is not None:
        (root / "betti_discover.json").write_text(json.dumps(betti_discover, indent=2, sort_keys=True) + "\n")
    if extra:
        (root / "extra.json").write_text(json.dumps(extra, indent=2, sort_keys=True, default=str) + "\n")
    return root


def default_bound(model: TorchMLP) -> float:
    from relucent import convert

    return float(default_polyhedron_bound(convert(model)))
