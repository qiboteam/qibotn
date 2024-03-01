# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path

from recommonmark.transform import AutoStructify
from sphinx.ext import apidoc

import qibotn

# sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Qibotn"
copyright = "The Qibo team"
author = "The Qibo team"

# The full version, including alpha/beta/rc tags
release = qibotn.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "recommonmark",
    "sphinxcontrib.katex",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_show_sourcelink = False

# Adapted this from
# https://github.com/readthedocs/recommonmark/blob/ddd56e7717e9745f11300059e4268e204138a6b1/docs/conf.py
# app setup hook


def run_apidoc(_):
    """Extract autodoc directives from package structure."""
    source = Path(__file__).parent
    docs_dest = source / "api-reference"
    package = source.parents[1] / "src" / "qibotn"
    apidoc.main(["--no-toc", "--module-first", "-o", str(docs_dest), str(package)])


def setup(app):
    app.add_config_value("recommonmark_config", {"enable_eval_rst": True}, True)
    app.add_transform(AutoStructify)
    app.add_css_file("css/style.css")

    app.connect("builder-inited", run_apidoc)
