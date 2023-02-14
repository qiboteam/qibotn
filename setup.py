from setuptools import setup, find_packages
import os
import re
import pathlib

PACKAGE = "qibotn"


# Returns the qibotn version
def get_version():
    """ Gets the version from the package's __init__ file
    if there is some problem, let it happily fail """
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


# load long description from README
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="qibotn",
    version=get_version(),
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
        "qibo>=0.1.10",
        "qibojit>=0.0.7",
        "quimb[tensor]>=1.4.0",
    ],
    extras_require={
        "docs": [],
        "tests": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "pytest-env>=0.8.1",
        ],
        "analysis": [
            "pylint>=2.16.0",
        ],
    },
    python_requires=">=3.7.0",
    long_description=long_description,
    long_description_content_type='text/markdown',
)
