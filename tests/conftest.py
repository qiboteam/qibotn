"""To skip the pytests marked with gpu"""
import pytest
import os

tt = os.system('nvidia-smi')

def pytest_runtest_setup(item):
    if (tt!=0):
        for marker in item.iter_markers(name="gpu"):
            pytest.skip(f"test requires gpu")

"""Pytest fixtures.
"""


