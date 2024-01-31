Qibotn is the tensor-network translation module for Qibo to support large-scale simulation of quantum circuits and acceleration.

To get started, `python setup.py install` to install the tools and dependencies.

# Sample Codes
## Single Node

<pre>
```
import numpy as np
from qibo import Circuit, gates
import qibo

# Below shows 3 ways of setting the computation_settings
# Note that for MPS_enabled and expectation_enabled parameters the accepted inputs are boolean or a dictionary with the format shown below.
# If computation_settings is not specified, the default setting is used in which all booleans will be False. 
# This will trigger the dense vector computation of the tensornet.

computation_settings = {
    'MPI_enabled': False,
    'MPS_enabled': {
                "qr_method": False,
                "svd_method": {
                    "partition": "UV",
                    "abs_cutoff": 1e-12,
                },
            } ,
    'NCCL_enabled': False,
    'expectation_enabled': False
}

# computation_settings = {
#    'MPI_enabled': False,
#    'MPS_enabled': False,
#    'NCCL_enabled': False,
#    'expectation_enabled': {
#        'pauli_string_pattern': "IXZ"
# }

# computation_settings = {
#     'MPI_enabled': False,
#     'MPS_enabled': True,
#     'NCCL_enabled': False,
#     'expectation_enabled': False
# }

qibo.set_backend(backend="qibotn", runcard=computation_settings)

# Construct the circuit
c = Circuit(2)
# Add some gates
c.add(gates.H(0))
c.add(gates.H(1))

# Execute the circuit and obtain the final state
result = c()

print(result.state())
```
</pre>