"""Allow `python -m openfreqbench` as an alternative to the `ofb` script."""

import os
import sys

# Force UTF-8 on Windows terminals (cp1252 can't handle Rich unicode output)
os.environ.setdefault("PYTHONUTF8", "1")
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from openfreqbench.cli.app import main

if __name__ == "__main__":
    main()
