Quick start
===========

In this section, we provide an example of two qubit ciruit simulation using qibotn package in Qibo simulator. First, the backend is to be set with appropriate run card settings, followed by the circuit simulation using Qibo documentation.

Setting the backend
"""""""""""""""""""

QiboTN offers two backends: cutensornet (using cuQuantum library) and qutensornet (using Quimb library) for tensor network based simulations. At present, cutensornet backend works only for GPUs whereas qutensornet for CPUs.  The backend can be set using the following command line.

To use cuQuantum library, cutensornet can be specified as follows::

   qibo.set_backend(
      backend="qibotn", platform="cutensornet", runcard=computation_settings
   )

Similarly, to use Quimb library, qutensornet can be set as follows::

   qibo.set_backend(
       backend="qibotn", platform="qutensornet", runcard=computation_settings
   )

Setting the runcard
"""""""""""""""""""

The basic structure of the runcard is as follows::

   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": {
           "pauli_string_pattern": "IXZ",
       },
   }


**MPI_enabled:** Setting this option *True* results in parallel execution of circuit using MPI (Message Passing Interface). At present, only works for cutensornet platform.

**MPS_enabled:** This option is set *True* for Matrix Product State (MPS) based calculations where as general tensor network structure is used for *False* value.

**NCCL_enabled:** This is set *True* for cutensoret interface for further acceleration while using Nvidia Collective Communication Library (NCCL).

**expectation_enabled:** This option is set *True* while calculating expecation value of the circuit. Observable whose expectation value is to be calculated is passed as a string in the dict format as {"pauli_string_pattern": "observable"}. When the option is set *False*, the dense vector state of the circuit is calculated.


Basic example
"""""""""""""

The following is a basic example to execute a two qubit circuit and print the final state in dense vector form using quimb backend::

   import qibo
   from qibo import Circuit, gates

   # Set the runcard
   computation_settings = {
       "MPI_enabled": False,
       "MPS_enabled": False,
       "NCCL_enabled": False,
       "expectation_enabled": False,
   }


   # Set the quimb backend
   qibo.set_backend(
       backend="qibotn", platform="qutensornet", runcard=computation_settings
   )


   # Construct the circuit with two qubits
   c = Circuit(2)

   # Apply Hadamard gates on first and second qubit
   c.add(gates.H(0))
   c.add(gates.H(1))

   # Execute the circuit and obtain the final state
   result = c()

   # Print the final state
   print(result.state())
