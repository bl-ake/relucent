## Test that the README.md example code works

import numpy as np
import torch.nn as nn

import relucent


def test_readme():
    # Create Model
    network = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )  ## or conveniently, relucent.mlp(widths=[2, 10, 5, 1])

    ## Initialize a Complex to track calculations
    cplx = relucent.Complex(network)

    ## Calculate the activation regions via local search
    cplx.bfs()

    ## Plotting functions return Plotly figures
    _ = cplx.plot()

    assert len(cplx) > 0
    assert sum(len(p.shis) for p in cplx) / len(cplx) > 0

    input_point = np.random.random((1, 2))
    p = cplx.point2poly(input_point)

    _ = p.halfspaces[p.shis]
    _ = p.center
    _ = p.inradius
    _ = cplx.get_dual_graph()


def test_readme_numpy():
    # Create Model

    network = [(np.random.randn(10, 2), np.random.randn(10)), (np.random.randn(5, 10), np.random.randn(5))]

    cplx = relucent.Complex(network)

    cplx.bfs()

    _ = cplx.plot()

    assert len(cplx) > 0
    assert sum(len(p.shis) for p in cplx) / len(cplx) > 0

    input_point = np.random.random((1, 2))
    p = cplx.point2poly(input_point)
    _ = p.halfspaces[p.shis]
