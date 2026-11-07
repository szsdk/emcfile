from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        rust_manifest = root / "rust" / "Cargo.toml"
        pkg_dir = root / "src" / "emcfile"
        target_so = pkg_dir / "_parquet_rs.so"

        if rust_manifest.is_file() and shutil.which("cargo") is not None:
            env = os.environ.copy()
            link_arg = (
                "-Wl,-undefined,dynamic_lookup"
                if sys.platform == "darwin"
                else "-Wl,--unresolved-symbols=ignore-in-object-files"
            )
            env.setdefault("RUSTFLAGS", f"-C target-cpu=native -C link-arg={link_arg}")
            subprocess.run(
                ["cargo", "build", "--release", "--manifest-path", str(rust_manifest)],
                env=env,
                check=True,
            )
            ext = ".dylib" if sys.platform == "darwin" else ".so"
            built = rust_manifest.parent / "target" / "release" / f"lib_parquet_rs{ext}"
            if built.is_file():
                shutil.copy2(built, target_so)

        if target_so.is_file():
            build_data["pure_python"] = False
            build_data["infer_tag"] = True
            build_data.setdefault("force_include", {})[str(target_so)] = (
                "emcfile/_parquet_rs.so"
            )
