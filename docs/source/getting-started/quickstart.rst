Quick start
===========

Setting the backend
"""""""""""""""""""
QiboTN supports two backends cutensornet (using CuQuantum library) and Quimbbackend (using Quimb library) for tensor network based simulations. The backend can be set using the following command line.

For CuQuantum library,

.. testcode::
   qibo.set_backend(backend="qibotn", platform="cutensornet", runcard=computation_settings)
..

and for Quimb library

.. testcode::
   qibo.set_backend(
       backend="qibotn", platform="QuimbBackend", runcard=computation_settings
   )
..

Setting the runcard
""""""""""""""""""""
Basic structure of runcard is

.. testcode::
   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": {
           "pauli_string_pattern": "IXZ",
       },
   }
..

Basic example
""""""""""""
.. testcode::
   # Construct the circuit
   c = Circuit(2)
   # Add some gates
   c.add(gates.H(0))
   c.add(gates.H(1))

   # Execute the circuit and obtain the final state
   result = c()

   print(result.state())

..
