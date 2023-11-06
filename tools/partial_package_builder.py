# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import argparse
from pathlib import Path
from shutil import copy2, copytree
from subprocess import run as run_cmd
from tempfile import TemporaryDirectory

import toml

parser = argparse.ArgumentParser(
    "Partial Package builder",
    description="Easily package parts of the source code into a separate distribution."
)
parser.add_argument(
    "--toml-file",
    required=True,
    help="The pyproject.toml file to use for the patial package."
)


class PartialPackageBuilder:
    def __init__(self, toml_file: str):
        self.file_dir = Path(toml_file).parent
        self.toml = toml.load(toml_file)
        self.package_dict = {
            p["include"]: p for p in self.toml["tool"]["poetry"]["packages"]
        }
        self.include_paths = self.toml["tool"]["poetry"].pop("package_includes")

    def build_partial_package(self):
        with TemporaryDirectory() as tempdir:
            for path_string in self.include_paths:
                self._include_path(path_string, tempdir)
            readme = self.toml["tool"]["poetry"].get("readme", None)
            if readme is not None:
                README = f"README{Path(readme).suffix}"
                copy2(Path(self.file_dir, readme), Path(tempdir, README))
                self.toml["tool"]["poetry"]["readme"] = README
            self.toml["tool"]["poetry"]["version"] = run_cmd([
                "poetry", "version", "--short"
            ],
                                                             capture_output=True,
                                                             text=True).stdout.rstrip("\n")
            with open(Path(tempdir, "pyproject.toml"), "w+") as pyproject_file:
                toml.dump(self.toml, pyproject_file)
            run_cmd(["poetry", "build"], cwd=tempdir)
            copytree(Path(tempdir, "dist"), "dist", dirs_exist_ok=True)

    def _include_path(self, path_string: str, tempdir: TemporaryDirectory):
        rel_path = Path(path_string)
        pkg = self.package_dict.get(rel_path.parts[0], None)
        if pkg is None:
            raise ValueError(
                f"Include path '{path_string}' is not part of any included package."
            )
        from_path = Path(pkg.get("from", "."))
        path = Path(from_path, rel_path)
        self._build_parent_tree(path, from_path, tempdir)
        if path.is_dir():
            copytree(path, Path(tempdir, path))
        elif path.is_file():
            copy2(path, Path(tempdir, path))

    def _build_parent_tree(self, path: Path, from_path: Path, tempdir: TemporaryDirectory):
        parents = path.parents
        Path(tempdir, parents[0]).mkdir(parents=True, exist_ok=True)
        for parent in parents:
            if parent == from_path:
                break
            Path(tempdir, parent, "__init__.py").touch(exist_ok=True)


if __name__ == "__main__":
    args = parser.parse_args()
    ppb = PartialPackageBuilder(args.toml_file)
    ppb.build_partial_package()
