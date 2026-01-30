[![Usable](https://github.com/bl-ake/relucent/actions/workflows/python-package.yml/badge.svg)](https://github.com/bl-ake/relucent/actions/workflows/python-package.yml)
[![Latest Release](https://img.shields.io/github/v/tag/bl-ake/relucent?label=Latest%20Release)](https://github.com/bl-ake/relucent/releases)

# Relucent
Explore the polyhedral complexes of ReLU neural networks

This package differs from existing software (for example, check out the amazing [SplineCam](https://github.com/AhmedImtiazPrio/SplineCAM) by Ahmed Imtiaz Humayun) because it avoids computing vertices of the complex. This tradeoff changes which computations are efficient, in particular speeding up local search of the complex in higher dimensions.

If you run into any problems or have any feature requests, please create an issue on the project's [Github](https://github.com/bl-ake/relucent/issues).

## Environment Setup 
1. Install Python 3.13
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
3. Run `pip install relucent`

## Getting Started
To see if the installation has been successful, try plotting the complex of a randomly initialized network in 2 dimensions like this:
```
from relucent import Complex, get_mlp_model

network = get_mlp_model(widths=[2, 10, 5, 1])
cplx = Complex(network)
cplx.bfs()
fig = cplx.plot(bound=10000)
fig.show()
```

The "NN" object returned by get_mlp_model inherits from torch.nn.Module, so you can train and manipulate it just like you're used to :)

Given some input point, you could get a minimal H-representation of the polyhedron containing it like this:
```
import numpy as np

input_point = np.random.random(network.input_shape)
p = cplx.point2poly(input_point)
print(p.halfspaces[p.shis, :])
```

You could also check the average number of faces of all polyhedrons with:
```
sum(len(p.shis) for p in cplx) / len(cplx)
```
Or, get the adjacency graph of top-dimensional cells in the complex as a [NetworkX Graph](https://networkx.org/documentation/stable/tutorial.html) with:
```
print(cplx.get_dual_graph())
```

You can view the full documentation for this library at https://bl-ake.github.io/relucent/

## Obtaining a Gurobi License
Without a [license](https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license), Gurobi will only work with a limited feature set. This includes a limit on the number of decision variables in the models it can solve, which limits the size of the networks this code is able to analyze. There are multiple ways to install the software, but we recommend the following steps to those eligible for an academic license:
1. Install the [Gurobi Python library](https://pypi.org/project/gurobipy/), for example using `pip install gurobipy`
2. [Obtain a Gurobi license](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license) (Note: a WLS license will limit the number of concurrent sessions across multiple devices, which can result in slowdowns when using this library on different machines simultaneously.)
3. In your Conda environment, run `grbgetkey` followed by your license key

## Bibliography
This package was made possible by the following works:
  - Fukuda, K. (2004, August 26). Frequently Asked Questions in Polyhedral Computation. https://people.inf.ethz.ch/~fukudak/polyfaq/
  - Grigsby, J. E., & Lindsey, K. (2022). On Transversality of Bent Hyperplane Arrangements and the Topological Expressiveness of ReLU Neural Networks. SIAM Journal on Applied Algebra and Geometry, 6(2), 216–242. https://doi.org/10.1137/20M1368902
  - Liu, Y., Caglar, T., Peterson, C., & Kirby, M. (2023). Integrating geometries of ReLU feedforward neural networks. Frontiers in Big Data, 6, 1274831. https://doi.org/10.3389/fdata.2023.1274831
  - Masden, M. (2025). Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks. SIAM Journal on Applied Algebra and Geometry, 9(2), 374–404. https://doi.org/10.1137/24M1646996
  - Xu, S., Vaughan, J., Chen, J., Zhang, A., & Sudjianto, A. (2022). Traversing the Local Polytopes of ReLU Neural Networks. The AAAI-22 Workshop on Adversarial Machine Learning and Beyond. https://openreview.net/forum?id=EQjwT2-Vaba
  - Yajing Liu, Christina M Cole, Chris Peterson, & Michael Kirby. (2023). ReLU Neural Networks, Polyhedral Decompositions, and Persistent Homolog. TAG-ML.
  - Zhang, X., & Wu, D. (2019, September 25). Empirical Studies on the Properties of Linear Regions in Deep Neural Networks. International Conference on Learning Representations. https://openreview.net/forum?id=SkeFl1HKwr