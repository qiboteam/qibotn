"""conftest.py.

Pytest fixtures.
"""

import os
import sys

import pytest

# backends to be tested
# TODO: add cutensornet and quimb here as well
BACKENDS = ["cutensornet"]
# BACKENDS = ["qmatchatea"]


def get_backend(backend_name):

    from qibotn.backends.cutensornet import CuTensorNet
    from qibotn.backends.qmatchatea import QMatchaTeaBackend

    NAME2BACKEND = {"qmatchatea": QMatchaTeaBackend, "cutensornet": CuTensorNet}

    return NAME2BACKEND[backend_name]()


AVAILABLE_BACKENDS = []
for backend_name in BACKENDS:
    try:
        _backend = get_backend(backend_name)
        AVAILABLE_BACKENDS.append(backend_name)
    except (ModuleNotFoundError, ImportError):
        pass


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip(f"Cannot run test on platform {plat}.")


@pytest.fixture
def backend(backend_name):
    yield get_backend(backend_name)


def pytest_runtest_setup(item):
    ALL = {"darwin", "linux"}
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    plat = sys.platform
    if supported_platforms and plat not in supported_platforms:  # pragma: no cover
        # case not covered by workflows
        pytest.skip(f"Cannot run test on platform {plat}.")


def pytest_configure(config):
    config.addinivalue_line("markers", "linux: mark test to run only on linux")
    if os.getenv("OMPI_COMM_WORLD_SIZE"):
        if hasattr(config.option, "no_cov"):
            config.option.no_cov = True
        cov_plugin = config.pluginmanager.get_plugin("_cov")
        if cov_plugin is not None:
            config.pluginmanager.unregister(cov_plugin)


def pytest_addoption(parser):
    # Keep pyproject's [tool.pytest.ini_options].env valid even when pytest-env is not installed.
    parser.addini("env", type="linelist", help="Environment variables for tests.")


def pytest_generate_tests(metafunc):
    module_name = metafunc.module.__name__

    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)
