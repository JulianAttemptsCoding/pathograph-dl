from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable


def _gcloud_bin() -> str:
    # Windows often needs gcloud.cmd when called from Python
    for name in ("gcloud.cmd", "gcloud"):
        p = shutil.which(name)
        if p:
            return name
    return "gcloud"


GCLOUD = _gcloud_bin()


def run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out


def ls(uri: str) -> str:
    return run([GCLOUD, "storage", "ls", uri])


def ls_any(uri: str) -> bool:
    # Works for objects and prefixes
    tries = [uri]
    if uri.startswith("gs://") and not uri.endswith("/"):
        tries.append(uri + "/")
    tries.append(uri.rstrip("/") + "/**")
    for u in tries:
        try:
            ls(u)
            return True
        except Exception:
            pass
    return False


def cp(src: str, dst: str) -> None:
    run([GCLOUD, "storage", "cp", src, dst])


def rsync(src: str, dst: str) -> None:
    run([GCLOUD, "storage", "rsync", "-r", src, dst])


@dataclass
class GCSPath:
    uri: str

    def exists(self) -> bool:
        return ls_any(self.uri)
