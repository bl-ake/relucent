

![Relucent](/docs/title.svg)

<div align="center">


[![Usable](https://github.com/bl-ake/relucent/actions/workflows/python-package.yml/badge.svg)](https://github.com/bl-ake/relucent/actions/workflows/python-package.yml)
[![Latest Release](https://img.shields.io/github/v/tag/bl-ake/relucent?label=Latest%20Release)](https://github.com/bl-ake/relucent/releases)

</div>

Relucent is a Python package for computing the polyhedra of ReLU networks! Its main features include:
- Distributed calculation of the activation regions of ReLU networks via local search
- Visualization of ReLU complexes in two or three dimensions with [Plotly](https://plotly.com/python/)
- Automatic compatibility with existing [PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) networks
- Computation of the complex's dual as a [NetworkX](https://networkx.org/documentation/stable/tutorial.html) Graph
- Various calculations for individual activation regions, decision boundaries, and affine splines

## Environment Setup 
1. Install Python >= 3.11
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
3. Run `pip install relucent`

## Getting Started
To see if the installation has been successful, try plotting the complex of a randomly initialized network in 2 dimensions like this:

```python
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
fig.show()
```

Given some input point, you could get a minimal H-representation of the polyhedral region containing it like this:
```
input_point = np.random.random((1, 2))
p = cplx.point2poly(input_point)
print(p.halfspaces[p.shis])
```
Attributes like `p.halfspaces` (halfspaces of the form Ax + b <= 0, in format [A; b], induced by each neuron), `p.shis` (the indices of the non-redundant halfspaces), and `p.center` (the Chebyshev Center) are computed lazily.

You could also check the average number of faces of all polyhedrons with:
```
sum(len(p.shis) for p in cplx) / len(cplx)
```
Or, get the adjacency graph of top-dimensional cells in the complex with:
```
print(cplx.get_dual_graph())
```

You can view the full documentation for this library at https://bl-ake.github.io/relucent/

## Obtaining a Gurobi License
This package will work for most applications without a [license](https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license). However, without one, Gurobi will only work with a limited feature set. This includes a limit on the number of decision variables in the models it can solve, which limits the size of the networks this code is able to analyze. There are multiple ways to install the software, but we recommend the following steps to those eligible for an academic license:
0. Create a fresh Python environment using a distribution of [Anaconda](https://mamba.readthedocs.io/en/latest/index.html).
1. Install the [Gurobi Python library](https://pypi.org/project/gurobipy/) using `conda install -c gurobi gurobi`.
2. [Obtain a Gurobi license](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license) (Note: a WLS license will limit the number of concurrent sessions across multiple devices, which can result in slowdowns when using this library on different machines simultaneously.)
3. In your Conda environment, run `grbgetkey` followed by your license key
4. Complete the remaining steps in [Getting Started](#getting-started)

## Citing this Package
If you run into any problems or have any feature requests, please create an issue on the project's [Github](https://github.com/bl-ake/relucent). If you want to credit its use in your research, please cite our [paper](https://openreview.net/forum?id=TgLW2DiRDG).

```
@inproceedings{
  gaines2026characterizing,
  title={Characterizing the Discrete Geometry of Re{LU} Networks},
  author={Blake B. Gaines and Jinbo Bi},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=TgLW2DiRDG}
}
```

## Related Software:
Please check out the amazing software created by others working in this area. Depending on your goal, some of these could be even better!
- [GoL Toolbox](https://github.com/cglrtrgy/GoL_Toolbox) by Turgay Caglar ([Paper](https://doi.org/10.3389/fdata.2023.1274831))
- [CanonicalPoly 2.0](https://github.com/mmasden/canonicalpoly2.0) by Marissa Maden ([Paper](https://doi.org/10.1137/24M1646996))
- [ReLU Edge Subdivision](https://github.com/arturs-berzins/relu_edge_subdivision) by Arturs Berzins ([Paper](https://proceedings.mlr.press/v202/berzins23a.html))
- [SplineCam](https://github.com/AhmedImtiazPrio/SplineCAM) by Ahmed Imtiaz Humayun ([Paper](https://doi.org/10.48550/arXiv.2302.12828))
- [Neural Network Elements](https://github.com/gtri/neural-network-elements) by Andrew Tawfeek ([Paper](https://doi.org/10.48550/arXiv.2510.12700))

## Bibliography
This package was made possible by the following work:
  - Fukuda, K. (2004, August 26). Frequently Asked Questions in Polyhedral Computation. https://people.inf.ethz.ch/~fukudak/polyfaq/
  - Grigsby, J. E., & Lindsey, K. (2022). On Transversality of Bent Hyperplane Arrangements and the Topological Expressiveness of ReLU Neural Networks. SIAM Journal on Applied Algebra and Geometry, 6(2), 216–242. https://doi.org/10.1137/20M1368902
  - Liu, Y., Caglar, T., Peterson, C., & Kirby, M. (2023). Integrating geometries of ReLU feedforward neural networks. Frontiers in Big Data, 6, 1274831. https://doi.org/10.3389/fdata.2023.1274831
  - Masden, M. (2025). Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks. SIAM Journal on Applied Algebra and Geometry, 9(2), 374–404. https://doi.org/10.1137/24M1646996
  - Xu, S., Vaughan, J., Chen, J., Zhang, A., & Sudjianto, A. (2022). Traversing the Local Polytopes of ReLU Neural Networks. The AAAI-22 Workshop on Adversarial Machine Learning and Beyond. https://openreview.net/forum?id=EQjwT2-Vaba
  - Yajing Liu, Christina M Cole, Chris Peterson, & Michael Kirby. (2023). ReLU Neural Networks, Polyhedral Decompositions, and Persistent Homolog. TAG-ML.
  - Zhang, X., & Wu, D. (2019, September 25). Empirical Studies on the Properties of Linear Regions in Deep Neural Networks. International Conference on Learning Representations. https://openreview.net/forum?id=SkeFl1HKwr
