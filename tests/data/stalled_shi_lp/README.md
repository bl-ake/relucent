# SHI LPs that Gurobi never finishes

Two cells whose SHI LP for one row ran for hours in a real search, recorded so a test can
exercise `GUROBI_TIME_LIMIT` and the recovery after it. Both come from trained networks of the
`mlp_topology` experiments (input dimension 4, width 8, ReLU, float64); every search that reached
the cell stopped there.

| Case | Network | Cell rows | Stuck row | Facets |
|---|---|---|---|---|
| `blobs_d4_w8_L5` | 4 → 8×5 → 1 | 41 | 27 | 19, 22, 27, 33, 34, 35 |
| `cifar_one_d4_w8_L4` | 4 → 8×4 → 1 | 33 | 19 | 2, 9, 11, 26, 31, 32 |

## Files in each case

- `network.npz`: `W0, b0, W1, b1, …` of the `nn.Linear` layers, float64, with a ReLU between
  consecutive layers and none after the last. relucent saw `add_output_relu(that network)`.
- `cell.json`: the cell's `sign_sequence` (one entry per ReLU, the output ReLU last), the
  `bound` the search passed to `calculations.shis`, the row whose LP stalls
  (`stuck_halfspace`), and the cell's `facets` (its SHIs).
- `stuck_lp.mps`, `stuck_lp.prm`, `stuck_lp_start.bas`: that LP as Gurobi saw it: the model, its
  non-default parameters (`ScaleFlag 2`, `OptimalityTol 1e-9`, `BestObjStop 0.5`) and the
  warm-start basis left by the previous row's solve.

## Reproducing

relucent, with the network rebuilt from `network.npz`:

```python
cell = json.load(open("cell.json"))
net = Complex(add_output_relu(rebuilt_sequential))._net
poly = Polyhedron(net, np.array([cell["sign_sequence"]], dtype=np.int8), bound=cell["bound"])
calculations.shis(poly, bound=cell["bound"])  # == cell["facets"]
```

With `GUROBI_TIME_LIMIT = 0` (no limit) the call does not return: the warm-started solve of the
stuck row runs the simplex for millions of iterations. With the default 10 s it returns the facets
above after about 10 s: that solve stops at the limit, and the cold re-solve with `ScaleFlag 0`
(`_cold_retry`) finishes in under 0.1 s.

Gurobi alone (13.0.3), on the 4-variable LP:

```python
m = gp.read("stuck_lp.mps")
m.read("stuck_lp.prm")
m.read("stuck_lp_start.bas")
m.optimize()
```

| Start | ScaleFlag | `blobs_d4_w8_L5` | `cifar_one_d4_w8_L4` |
|---|---|---|---|
| saved basis | 2 (relucent's) | no end (8.2 M iterations in 30 s) | no end (10.5 M in 30 s) |
| saved basis | 0 | optimal, 12 iterations | no end |
| saved basis | -1, 1, 3 | optimal | optimal |
| cold | 2 | optimal, 15 iterations | no end |
| cold | 0 | optimal | optimal, 0.06 s |
| cold | -1, 1, 3 | optimal | `NUMERIC` |

So the stall is the simplex cycling, and only on some (start, scaling) pairs; relucent's sequence
(warm under `ScaleFlag 2`, then cold under `ScaleFlag 0`) recovers both once the first solve is
cut off.
