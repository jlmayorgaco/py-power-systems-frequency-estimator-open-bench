from __future__ import annotations
import argparse


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--no-mc", action="store_true", help="Disable Monte Carlo")
    parser.add_argument("--out", type=str, default=".", help="Base output directory")
    return parser
