Network Definition Options
==========================

``Complex`` accepts multiple formats for defining the ReLU network whose
activation regions you want to explore.

At a high level, all inputs are converted to relucent's internal
``ReLUNetwork`` representation.

1) Pass A PyTorch Model
-----------------------

You can pass common PyTorch model structures directly.

.. code-block:: python

   import torch.nn as nn
   import relucent

   model = nn.Sequential(
       nn.Linear(4, 8),
       nn.ReLU(),
       nn.Linear(8, 2),
   )

   cplx = relucent.Complex(model)

This path is convenient when your model is already in PyTorch.

2) Pass A ReLUNetwork
------------------------------

You can construct the canonical network explicitly with NumPy arrays.

.. code-block:: python

   import numpy as np
   from collections import OrderedDict
   import relucent
   from relucent.model import ReLUNetwork, LinearLayer, ReLULayer

   layers = OrderedDict(
       [
           ("fc0", LinearLayer(weight=np.random.randn(8, 4), bias=np.random.randn(1, 8))),
           ("relu0", ReLULayer()),
           ("fc1", LinearLayer(weight=np.random.randn(2, 8), bias=np.random.randn(1, 2))),
       ]
   )
   net = ReLUNetwork(layers=layers, input_shape=(4,))
   cplx = relucent.Complex(net)

This is the most explicit, torch-free way to define your network.

3) Pass An Iterable Of ``(W_i, b_i)`` Tuples
--------------------------------------------

You can pass only affine layers; relucent inserts ReLU layers between them.

.. code-block:: python

   import numpy as np
   import relucent

   W0 = np.random.randn(8, 4)
   b0 = np.random.randn(8)      # also accepts shape (1, 8)
   W1 = np.random.randn(2, 8)
   b1 = np.random.randn(2)

   cplx = relucent.Complex([(W0, b0), (W1, b1)])

Rules:

* Each ``W_i`` must be 2D with shape ``(out_i, in_i)``.
* Each ``b_i`` must have shape ``(out_i,)`` or ``(1, out_i)``.
* ReLU is inserted after every affine layer except the last one.

4) Use ``relucent.convert(...)`` Explicitly
-------------------------------------------

If you want to inspect or reuse the canonical network before building a
``Complex``, call ``convert`` directly.

.. code-block:: python

   from relucent import convert, Complex

   canonical = convert(model_or_layer_spec)
   cplx = Complex(canonical)

