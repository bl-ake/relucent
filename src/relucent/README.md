## Source Code Structure
* [model.py](src/relucent/model.py): PyTorch Module that acts as an interface between the model and the rest of the code
* [poly.py](src/relucent/poly.py): Class for calculations involving individual polyhedrons (e.g. computing boundaries, neighbors, volume)
* [complex.py](src/relucent/complex.py): Class for calculations involving the polyhedral cplx (e.g. polyhedron search, connectivity graph calculation)
* [convert_model.py](src/relucent/convert_model.py): Utilities for converting various PyTorch.nn layers to Linear layers
* [bvs.py](src/relucent/bvs.py): Data structures for storing large numbers of sign vectors