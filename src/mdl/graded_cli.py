import argparse
from .graded import run
def main():
 p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results/graded_recognition'); run(p.parse_args().outdir)
if __name__=='__main__': main()
