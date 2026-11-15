import argparse
from .diagnostic import run
def main():
    p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results/association_diagnostics'); run(p.parse_args().outdir)
if __name__=='__main__': main()
