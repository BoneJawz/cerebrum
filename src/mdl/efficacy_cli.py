import argparse
from .efficacy import run
def main():
 p=argparse.ArgumentParser(); p.add_argument('--outdir',default='results/association_efficacy'); run(p.parse_args().outdir)
if __name__=='__main__': main()
