# Relucent
Explore polyhedral complexes associated with ReLU networks

## Environment Setup 
1. Install Python 3.13
2. Install [PyTorch 2.3.0](https://pytorch.org/get-started/previous-versions/#:~:text=org/whl/cpu-,v2.3.0)
3. Install the remaining dependencies with `pip install -r requirements.txt`

## Code Structure
* [model.py](model.py): Model class
* [poly.py](poly.py): Class for calculations involving individual polyhedrons (e.g. computing boundaries, neighbors, volume)
* [complex.py](complex.py): Class for calculations involving the polyhedral cplx (e.g. polyhedron search, dual graph calculation)
* [convert_model.py](convert_model.py): Utilities for converting various PyTorch.nn layers to Linear layers
* [bvs.py](bvs.py): Data structures for storing large numbers of sign vectors

## Obtaining a Gurobi License

**The following steps are not necessary when replicating the experiments from the paper.** 

Without a [license](https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license), Gurobi will only work with a limited feature set. This includes a limit on the number of decision variables in the models it can solve, which limits the size of the networks this code is able to analyze. There are multiple ways to install the software, but we recommend the following steps
1. Install Gurobi through Conda with `conda install -c gurobi gurobi`
2. [Obtain a Gurobi license](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license) (free for academics)
3. In your Conda environment, run `grbgetkey` followed by your license key