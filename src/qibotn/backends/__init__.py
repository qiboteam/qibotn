from typing import Union

from qibo.config import raise_error

from qibotn.backends.cutensornet import CuTensorNet  # pylint: disable=E0401
from qibotn.backends.quimb import QuimbBackend  # pylint: disable=E0401

QibotnBackend = Union[CuTensorNet, QuimbBackend]

PLATFORMS = ("cutensornet", "qutensornet")


class MetaBackend:
    """Meta-backend class which takes care of loading the qibotn backends."""

    @staticmethod
    def load(platform: str, runcard: dict = None) -> QibotnBackend:
        """Loads the backend.

        Args:
            platform (str): Name of the backend to load: either `cutensornet` or `qutensornet`.
            runcard (dict): Dictionary containing the simulation settings.
        Returns:
            qibo.backends.abstract.Backend: The loaded backend.
        """

        if platform == "cutensornet":  # pragma: no cover
            return CuTensorNet(runcard)
        elif platform == "qutensornet":  # pragma: no cover
            return QuimbBackend(runcard)
        else:
            raise_error(
                NotImplementedError,
                f"Unsupported platform {platform}, please pick one in (`cutensornet`, `qutensornet)",
            )

    def list_available(self) -> dict:
        """Lists all the available qibotn backends."""
        available_backends = {}
        for platform in PLATFORMS:
            try:
                MetaBackend.load(platform=platform)
                available = True
            except:
                available = False
            available_backends[platform] = available
        return available_backends
