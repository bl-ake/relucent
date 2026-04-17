Quick Start
===========

This guide mirrors the setup and first-use flow from ``relucent/README.md`` so
you can get a working run quickly.

Requirements
------------

1. Install Python 3.11 or newer.
2. Install PyTorch (see the `PyTorch install guide <https://pytorch.org/get-started/locally/>`_).
3. Install relucent:

.. code-block:: bash

   pip install relucent

First Run Example
-----------------

The following script builds a small random ReLU MLP, computes activation
regions with local search, and opens a Plotly figure of the resulting complex.

.. code-block:: python

   import numpy as np
   import torch.nn as nn
   import relucent

   if __name__ == "__main__":
       # Create model
       network = nn.Sequential(
           nn.Linear(2, 10),
           nn.ReLU(),
           nn.Linear(10, 5),
           nn.ReLU(),
           nn.Linear(5, 1),
       )  # or relucent.mlp(widths=[2, 10, 5, 1])

       # Initialize a Complex to track calculations
       cplx = relucent.Complex(network)

       # Calculate activation regions via local search
       cplx.bfs()

       # Plotting functions return Plotly figures
       fig = cplx.plot()
       fig.show()

Search and Geometry Guide
-------------------------

For detailed guidance on search-time geometry computation and caching strategy,
see :doc:`search_geometry`.

Working With Regions
--------------------

Get a minimal H-representation for the region containing an input point:

.. code-block:: python

   input_point = np.random.random((1, 2))
   p = cplx.point2poly(input_point)
   print(p.halfspaces[p.shis])

Useful lazily computed attributes include:

* ``p.halfspaces``: halfspaces of the form :math:`Ax + b \le 0`.
* ``p.shis``: indices of non-redundant supporting halfspaces.
* ``p.center``: Chebyshev center.

Two additional quick checks:

.. code-block:: python

   # Average number of faces over computed polyhedra
   sum(len(p.shis) for p in cplx) / len(cplx)

   # Adjacency graph of top-dimensional cells in the complex
   print(cplx.get_dual_graph())

Gurobi License Note
-------------------

Relucent works for many tasks without a Gurobi license, but larger models may
hit solver feature/size limits. For academic users, a typical path is:

1. Create a fresh Conda environment.
2. Install ``gurobi`` (for example, ``conda install -c gurobi gurobi``).
3. Obtain a license key from Gurobi.
4. Run ``grbgetkey`` in that environment.

For detailed and current license instructions, see the
`Gurobi setup documentation <https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license>`_.
