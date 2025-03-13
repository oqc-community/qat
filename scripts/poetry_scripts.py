import os


def format_code():
    os.system("poetry run black .")
    os.system("poetry run isort .")
    os.system("poetry run autoflake .")


def jupytext_sync():
    os.system("poetry run jupytext --sync --pipe black notebooks/ipynb/*.ipynb")
