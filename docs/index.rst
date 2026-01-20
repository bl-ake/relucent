.. relucent documentation master file, created by
   sphinx-quickstart on Tue Jan 20 13:48:03 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Relucent
======================

Welcome to the Relucent documentation!

.. _complex:

.. Complex
.. -------

.. autoclass:: relucent.Complex
   :members:
   :undoc-members:
   :show-inheritance:

.. _polyhedron:

.. Polyhedron
.. ----------

.. autoclass:: relucent.Polyhedron
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: MAX_RADIUS

.. _neural-network-models:

.. Neural Network Models
.. ---------------------

.. _neural-network-class:

Neural Network Class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: relucent.NN
   :members:
   :undoc-members:
   :show-inheritance:

.. _model-functions:

Model Functions
~~~~~~~~~~~~~~~

.. autofunction:: relucent.get_mlp_model

.. autofunction:: relucent.convert

.. _utility-functions:

.. Utility Functions
.. -----------------

.. autofunction:: relucent.get_env

.. autofunction:: relucent.set_seeds

.. toctree::
   :maxdepth: 3
   :caption: Navigation:
   :hidden:


Indices and tables
==================
* :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`