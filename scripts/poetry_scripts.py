import os
from pathlib import Path


def format_code():
    os.system("poetry run ruff check --fix")
    os.system("poetry run ruff format")


def jupytext_sync():
    ipynb_files = Path("notebooks/ipynb").rglob("*.ipynb")
    for file in ipynb_files:
        os.system(f"poetry run nbstripout --extra-keys='metadata.kernelspec' {file}")
    os.system(
        "poetry run jupytext --sync --pipe 'ruff check --fix {}' "
        "--pipe 'ruff format {}' 'notebooks/ipynb/**/*.ipynb'"
    )


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
