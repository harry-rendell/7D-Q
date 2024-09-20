import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, color_transform
from module.modelling import fitting
from module.assets import load_vac
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--frame",   type=str, required=True, help=("OBS or REST to specify rest frame or observer frame time. \n"
                                                                    "Defaults to rest frame for Quasars and observer time for Stars.\n"))
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
    parser.add_argument("--best_phot", action='store_true', help="Use this flag to only save the best subset of the data")
    parser.add_argument("--survey",   type=str, help="name of surveys (separated with a space) to restrict data to. If left blank, then all surveys are used.")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
    else:
        ID = 'uid_s'
    
    if args.best_phot:
        phot_str = 'best_phot'
    else:
        phot_str = 'clean'

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    if args.survey == 'ztf':
        bounds = {'tau16':(1.6,4.5), 'tau50':(2,5), 'tau84':(2,6), 'sig16':(-1.1,0.5), 'sig50':(-1,0.5), 'sig84':(-0.9,0.5)}
    else:
        bounds = {'tau16':(2,  4.5), 'tau50':(2,5), 'tau84':(2,6), 'sig16':(-1.1,0.5), 'sig50':(-1,0.5), 'sig84':(-0.9,0.5)}
    for band in args.band:
        print(f'band: {band}')
        sig = pd.read_csv(cfg.D_DIR + f"computed/{OBJ}/mcmc_fits/mcmc_drw_fits_{band}_{args.survey.replace(' ','_')}_{args.nobs_min}_{phot_str}_{args.frame.lower()}_frame.csv", index_col=ID)
        sig = parse.filter_data(sig, bounds=bounds, dropna=False, verbose=True)
        tau_z = sig[['tau16','tau50','tau84']].dropna()
        
        print('applying skewfit')
        start = time.time()
        skewfits = data_io.dispatch_function(fitting.apply_fit_skewnnorm, np.array_split(tau_z, args.n_cores), max_processes=args.n_cores)
        skewfits = skewfits.rename(columns={0:'a',1:'loc',2:'scale'})
        skewfits = parse.filter_data(skewfits, bounds={'a':(0,30), 'loc':(0,10), 'scale':(0,5)}, dropna=True, verbose=True)
        skewfits.to_csv(cfg.D_DIR + f"computed/{OBJ}/skewfits/skewfit_{band}_{args.survey.replace(' ','_')}_{args.nobs_min}.csv")
        print('finished skewfit')
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)),'\n')