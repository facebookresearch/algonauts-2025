# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import subprocess
import typing as tp
from glob import glob
from pathlib import Path

import pydantic
from tqdm import tqdm

class Wildcard(pydantic.BaseModel):
    folder: str


class Datalad(pydantic.BaseModel):
    dset_dir: str | Path
    folder: str = "download"

    _dl_dir: Path = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        dset_dir = Path(self.dset_dir).resolve()
        if not dset_dir.parent.exists():
            raise ValueError(f"Parent folder must exist for {dset_dir}")
        dset_dir.mkdir(exist_ok=True)
        self._dl_dir = dset_dir / self.folder
        self._dl_dir.mkdir(exist_ok=True, parents=True)

    def get_success_file(self) -> Path:
        cls_name = self.__class__.__name__.lower()
        return self._dl_dir / f"{cls_name}_Algonauts2025_success_download.txt"

    @tp.final
    def download(self, overwrite: bool = False) -> None:
        if self.get_success_file().exists() and not overwrite:
            return
        print(f"Downloading Algonauts2025 to {self._dl_dir}...")
        self._download()
        self.get_success_file().write_text("success")

    folders: list[str | Wildcard] = []

    def install_requirements(cls) -> None:
        subprocess.run(
            [
                "datalad-installer",
                "datalad",
                "git-annex",
            ]
        )

    @pydantic.computed_field
    @property
    def repo_name(self) -> str:

        repo_name = Path(
            "https://github.com/courtois-neuromod/algonauts_2025.competitors.git",
        ).name
        if Path(repo_name).suffix == ".git":
            repo_name = repo_name[:-4]
        return repo_name

    def _datalad(self, cmd: str, path: Path | str) -> None:

        proc = subprocess.run(
            cmd, cwd=str(path), capture_output=True, text=True, shell=True
        )
        if "install(error)" in proc.stdout:
            logging.warning("Potential error in datalad clone:\n> %s", proc.stdout)
        if proc.stderr:

            logging.warning("Potential error in datalad clone:\n> %s", proc.stderr)

    def _dl_item(self, cur_path: Path | str) -> None:
        cmd = f'datalad get "{cur_path}"'
        self._datalad(cmd, self._dl_dir / self.repo_name)

    def _download(self) -> None:

        self._datalad(
            "datalad clone https://github.com/courtois-neuromod/algonauts_2025.competitors.git",
            self._dl_dir,
        )

        folders = self.folders if self.folders else [Wildcard(folder="*")]

        all_folders: list[Path] = []
        for folder in folders:
            if isinstance(folder, Wildcard):
                all_folders += [
                    Path(str(p))
                    for p in glob(str(self._dl_dir / self.repo_name / folder.folder))
                ]
            else:
                all_folders += [self._dl_dir / self.repo_name / folder]

        print(f"Loading {len(all_folders)} folders: ", all_folders)

        for item in tqdm(all_folders, desc="Downloading Algonauts2025", ncols=100):
            if not item.is_dir():
                continue
            self._dl_item(item)

        print("\nDownloaded Dataset")
