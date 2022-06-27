# Installation script for python
from setuptools import setup, find_packages
import os
import re

PACKAGE = "qibotn"

# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qibotn",
    version="0.1",
    description="A tensor-network translation module for quantum computing",
    author="The Qibo team",
    author_email="",
    url="https://github.com/qiboteam/qibotn",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.out", "*.yml"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "scipy",
        "numba",
        "cytoolz", 
        "tqdm", 
        "psutil", 
        "opt_einsum", 
        "autoray",
        "quimb",
        "qibo"
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinxcontrib-bibtex", "sphinx_markdown_tables", "nbsphinx", "IPython"],
        "tests": ["pytest", "cirq", "ply", "sklearn", "dill"],
    },
    python_requires=">=3.7.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
