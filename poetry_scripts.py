import os


def build_docs():
    os.system(
        'poetry run sphinx-apidoc -f -d 12 -M -e -o docs/sphinx/source/api/generated src/QAT "**/tests/**"'
    )
    os.system("poetry run sphinx-build docs/sphinx/source docs/sphinx/build")


def format_code():
    os.system("poetry run black .")
    os.system("poetry run isort src/QAT")
    os.system("poetry run isort src/tests")
    os.system("poetry run autoflake .")
