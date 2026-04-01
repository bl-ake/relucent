## Test that the README.md example code works

import numpy as np
import torch.nn as nn

import relucent

# Create Model
network = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
)  ## or conveniently, relucent.get_mlp_model(widths=[2, 10, 5, 1])

## Initialize a Complex to track calculations
cplx = relucent.Complex(network)

## Calculate the activation regions via local search
if __name__ == "__main__":
    cplx.bfs()

## Plotting functions return Plotly figures
fig = cplx.plot_cells()

input_point = np.random.random((1, 2))
p = cplx.point2poly(input_point)

_ = sum(len(p.shis) for p in cplx) / len(cplx)

_ = p.halfspaces[p.shis]
_ = p.center
_ = p.inradius
_ = cplx.get_dual_graph()
