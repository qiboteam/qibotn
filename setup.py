from setuptools import setup, find_packages
import pathlib

PACKAGE = "qibotn"

HERE = pathlib.Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="qibotn",
    version="0.0.1",
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
        "quimb[tensor]>=1.4.0",
    ],
    extras_require={
        "docs": [],
        "tests": ["pytest>=7.2.0", "pytest-cov>=4.0.0"],
    },
    python_requires=">=3.7.0",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
)
