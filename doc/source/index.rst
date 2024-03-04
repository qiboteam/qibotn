.. title::
   QiboTN

What is QiboTN?
===============

QiboTN  is the dedicated `Qibo <https://github.com/qiboteam/qibo>`_ backend to support large-scale simulation of quantum circuits and acceleration.

Supported Computation:

- Tensornet (TN)
- Matrix Product States (MPS)

Tensor Network contractions to:

- dense vectors
- expecation values of given Pauli string

The supported HPC configurations are:

- single-node CPU
- single-node GPU or GPUs
- multi-node multi-GPU with Message Passing Interface (MPI)
- multi-node multi-GPU with NVIDIA Collective Communications Library (NCCL)

Currently, the supported tensor network libraries are:

- `cuQuantum <https://github.com/NVIDIA/cuQuantum>`_, an NVIDIA SDK of optimized libraries and tools for accelerating quantum computing workflows.
- `quimb <https://quimb.readthedocs.io/en/latest/>`_, an easy but fast python library for ‘quantum information many-body’ calculations, focusing primarily on tensor networks.

How to Use the Documentation
============================

Welcome to the comprehensive documentation for QiboTN! This guide will help you navigate through the various sections and make the most of the resources available.


1. **Getting started**: Begin by referring to the
   :doc:`/getting-started/installation/` guide to set up the ``QiboTN`` library in your environment.

2. **Tutorials**: Explore the :doc:`getting-started/quickstart/` section for basic usage examples


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Introduction

   getting-started/index

.. toctree::
    :maxdepth: 1
    :caption: Main documentation

    api-reference/qibotn
    Developer guides <https://qibo.science/qibo/stable/developer-guides/index.html>

.. toctree::
    :maxdepth: 1
    :caption: Documentation links

    Qibo docs <https://qibo.science/qibo/stable/>
    Qibolab docs <https://qibo.science/qibolab/stable/>
    Qibocal docs <https://qibo.science/qibocal/stable/>
    Qibosoq docs <https://qibo.science/qibosoq/stable/>
    Qibochem docs <https://qibo.science/qibochem/stable/>
    Qibotn docs <https://qibo.science/qibotn/stable/>
    Qibo-cloud-backends docs <https://qibo.science/qibo-cloud-backends/stable/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
