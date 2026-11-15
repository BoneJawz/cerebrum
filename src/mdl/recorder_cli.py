import argparse
from .recorder import run
def main():
 p=argparse.ArgumentParser();p.add_argument('--outdir',default='results/neural_activity');run(p.parse_args().outdir)
if __name__=='__main__':main()
