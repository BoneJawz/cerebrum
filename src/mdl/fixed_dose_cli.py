import argparse
from .fixed_dose import run

def main():
    parser = argparse.ArgumentParser(description="Fixed-dose, no-feedback association validation")
    parser.add_argument("--outdir", default="results/fixed_dose_validation")
    run(parser.parse_args().outdir)

if __name__ == "__main__":
    main()
