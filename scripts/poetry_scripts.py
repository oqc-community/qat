import os


def format_code():
    os.system("poetry run black .")
    os.system("poetry run isort .")
    os.system("poetry run autoflake .")


def jupytext_sync():
    os.system("poetry run jupytext --sync --pipe black notebooks/ipynb/*.ipynb")


def build_docs():
    from importlib.util import find_spec

    if find_spec("sphinx") is None:
        print(
            "Buliding docs requires the optional group 'docs' to be installed.\n"
            "Please run\n\tpoetry install --with docs\nthen try again."
        )
        exit(1)
    # Requires the package to have been installed with the --with docs flag
    os.system("poetry run sphinx-apidoc -f -M -e -o docs/source/ src")
    os.system("poetry run sphinx-build docs/source/ docs/build/")
