from qibo.backends.numpy import NumpyBackend


class CuTensorNet(NumpyBackend):  # pragma: no cover
    # CI does not test for GPU

    def __init__(self):
        super().__init__()
        import cuquantum  # pylint: disable=import-error
        from cuquantum import cutensornet as cutn  # pylint: disable=import-error

        self.cuquantum = cuquantum
        self.cutn = cutn
        self.platform = "cutensornet"
        self.versions["cuquantum"] = self.cuquantum.__version__
        self.supports_multigpu = True
        self.handle = self.cutn.create()

    def __del__(self):
        if hasattr(self, "cutn"):
            self.cutn.destroy(self.handle)

    def set_precision(self, precision):
        if precision != self.precision:
            super().set_precision(precision)

    def get_cuda_type(self, dtype="complex64"):
        if dtype == "complex128":
            return (
                self.cuquantum.cudaDataType.CUDA_C_64F,
                self.cuquantum.ComputeType.COMPUTE_64F,
            )
        elif dtype == "complex64":
            return (
                self.cuquantum.cudaDataType.CUDA_C_32F,
                self.cuquantum.ComputeType.COMPUTE_32F,
            )
        else:
            raise TypeError("Type can be either complex64 or complex128")
