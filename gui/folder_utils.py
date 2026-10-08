from __future__ import annotations

import os
import subprocess
import sys


def open_folder(path: str) -> None:
    """
    Open a folder using the operating system's default file manager.

    Supported platforms:
    - Windows -> File Explorer
    - macOS   -> Finder
    - Linux   -> default file manager through xdg-open

    Raises:
        ValueError: if path is empty.
        FileNotFoundError: if path does not exist or is not a directory.
        RuntimeError: if the current operating system is unsupported.
    """
    if not path or not path.strip():
        raise ValueError("Folder path is empty.")

    folder_path = os.path.abspath(os.path.expanduser(path.strip()))

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder does not exist: {folder_path}"
        )

    if sys.platform.startswith("win"):
        os.startfile(folder_path)

    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder_path])

    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", folder_path])

    else:
        raise RuntimeError(
            f"Opening folders is not supported on platform '{sys.platform}'."
        )