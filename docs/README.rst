--------------------------
Building the docs locally
--------------------------

To run the docs locally, make sure you have the correct dependencies installed, use
:code:`poetry install --all-groups`. 

Then you can build the docs by running :code:`poetry run sphinx-apidoc -f -M -e -o docs/source/ src`
and :code:`poetry run sphinx-build docs/source/ docs/build/`.