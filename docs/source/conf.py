# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QAT"
copyright = "2023, Oxford Quantum Circuits Ltd"
author = (
    "Hamid El Maazouz <helmaazouz@oxfordquantumcircuits.com>, "
    "Harry Waring <hwaring@oxfordquantumcircuits.com>, "
    "Jamie Friel <jfriel@oxfordquantumcircuits.com>, "
    "John Dumbell <jdumbell@oxfordquantumcircuits.com>, "
    "Kajsa Eriksson Rosenqvist <keriksson.rosenqvist@oxfordquantumcircuits.com>, "
    "Norbert Deak <ndeak@oxfordquantumcircuits.com>, "
    "Owen Arnold <oarnold@oxfordquantumcircuits.com>, "
    "Benjamin sach <bsach@oxfordquantumcircuits.com>, "
    "Daria Van Hende <dvanhende@oxfordquantumcircuits.com>, "
    "Luke Causer <lcauser@oxfordquantumcircuits.com>"
)
release = version = "2.4.2"
add_module_names = False
autoclass_content = "both"
smv_remote_whitelist = None

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_multiversion",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx_paramlinks",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "qat-logo.png"
html_sidebars = {"**": ["versioning.html"]}

html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 6,
    "includehidden": True,
    "titles_only": False,
}
