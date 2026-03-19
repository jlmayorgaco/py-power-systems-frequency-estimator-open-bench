from __future__ import annotations
import os
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(path_no_ext: str) -> None:
    """
    Saves both PDF and PNG. Use path without extension.
    """
    d = os.path.dirname(path_no_ext)
    if d:
        ensure_dir(d)
    plt.tight_layout()
    plt.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.savefig(path_no_ext + ".png", bbox_inches="tight")
    plt.close()
