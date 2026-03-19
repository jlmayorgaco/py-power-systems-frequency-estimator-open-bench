import subprocess
import sys
from pathlib import Path


def main():
    out = Path("requirements_frozen.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=out.open("w"))
    print(f"[OK] Environment frozen to {out}")


if __name__ == "__main__":
    main()
