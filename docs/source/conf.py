from dunamai import Pattern, Version, serialize_pep440
from sphinx_pyproject import SphinxConfig

version = Version.from_git(
    pattern=Pattern.DefaultUnprefixed,
)

if version.distance == 0:
    out = serialize_pep440(version.base)
else:
    out = serialize_pep440(
        version.base, post=version.distance, dev=version.timestamp.strftime("%Y%m%d%H%M%S")
    )

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

config = SphinxConfig(
    "../../pyproject.toml",
    globalns=globals(),
    config_overrides={"version": out, "release": out},
)
project = config.name
author = config.author

smv_remote_whitelist = None

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_sidebars = {"**": ["versioning.html"]}
