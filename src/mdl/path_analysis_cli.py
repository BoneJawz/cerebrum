import argparse
from .path_analysis import run
def main():
 p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results/association_path_analysis'); run(p.parse_args().outdir)
if __name__=='__main__': main()
