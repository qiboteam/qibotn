import pathlib
import re

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent.absolute()
PACKAGE = "qibotn"


# Returns the qibotn version
def version():
    """Gets the version from the package's __init__ file if there is some
    problem, let it happily fail."""
    version_file = HERE / "src" / PACKAGE / "__init__.py"
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"

    initfile = version_file.read_text(encoding="utf-8")
    matched = re.search(version_regex, initfile, re.M)

    if matched is not None:
        return matched.group(1)
    return "0.0.0"


# load long description from README
setup(
    name="qibotn",
    version=version(),
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
        "quimb[tensor]>=1.6.0",
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
        "cuda": [
            "cupy>=11.6.0",
            "cuquantum-python-cu11>=23.3.0",
        ],
    },
    python_requires=">=3.8.0",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
