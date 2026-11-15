import argparse
from .experiment import run

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--outdir", default="results/association_baseline")
    run(p.parse_args().outdir)

if __name__ == "__main__": main()
