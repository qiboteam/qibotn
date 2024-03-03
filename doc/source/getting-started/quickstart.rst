Quick start
===========

Setting the backend
"""""""""""""""""""

QiboTN supports two backends: cutensornet (using cuQuantum library) and QuimbBackend (using Quimb library) for tensor network based simulations. The backend can be set using the following command line.

To use cuQuantum library, cutensornet can be specified as follows:

.. code-block:: bash

   qibo.set_backend(
      backend="qibotn", platform="cutensornet", runcard=computation_settings
   )

Similarly, to use Quimb library, QuimbBackend can be as follows:

.. code-block:: bash

   qibo.set_backend(
       backend="qibotn", platform="QuimbBackend", runcard=computation_settings
   )

Setting the runcard
"""""""""""""""""""

The basic structure of the runcard is as follows:

.. code-block:: bash

   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": {
           "pauli_string_pattern": "IXZ",
       },
   }

Basic example
"""""""""""""

The following is a basic example:

.. code-block:: bash

   # Construct the circuit
   c = Circuit(2)

   # Add some gates
   c.add(gates.H(0))
   c.add(gates.H(1))

   # Execute the circuit and obtain the final state
   result = c()

   print(result.state())
