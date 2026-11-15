import argparse
from .continuous_experiment import run

def main():
    parser = argparse.ArgumentParser(description="Compare continuous and teaching-gated STDP")
    parser.add_argument("--outdir", default="results/continuous_learning")
    run(parser.parse_args().outdir)

if __name__ == "__main__":
    main()
