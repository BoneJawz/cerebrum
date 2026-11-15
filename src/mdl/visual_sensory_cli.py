from __future__ import annotations

import argparse

from .visual_sensory_experiment import run


def main():
    parser = argparse.ArgumentParser(description="Validate audio associations using a binary visual sensory path")
    parser.add_argument("--outdir", default="results/visual-sensory-pathway")
    run(parser.parse_args().outdir)


if __name__ == "__main__":
    main()
