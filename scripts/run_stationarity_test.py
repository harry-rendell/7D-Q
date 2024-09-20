import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.assets import load_n_tot
from module.preprocessing import data_io, lightcurve_statistics
from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
    parser.add_argument("--survey",   type=str, help="name of surveys (separated with a space) to restrict data to. If left blank, then all surveys are used.")
    parser.add_argument("--best_phot", action='store_true', help="Use this flag to only save the best subset of the data")
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

    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.lc_dtypes,
              'usecols': [ID,'mag','band','sid'],
              'ID':ID}

    if args.survey:
        kwargs['sid'] = [cfg.PREPROC.SURVEY_IDS[s] for s in args.survey.split()]
    else:
        args.survey = 'all'

    if args.nobs_min:
        # We select a subset of quasars which have a minimum of args.nobs_min observations in each of the bands 
        #   specified in args.band
        n_tot = load_n_tot(OBJ)
        if args.survey != 'ztf':
            # Leave out ZTF for n_obs threshold because it has many observations, and we mostly care about SDSS/PS
            survey_str = args.survey.replace('ztf','').split()
        else:
            survey_str = args.survey.split()

    else:
        args.nobs_min = 0

    for band in args.band:

        mask = (n_tot[[f'n_{s}_{band}' for s in survey_str]].sum(axis=1) > args.nobs_min)
        kwargs['subset'] = n_tot[mask].index

        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/{phot_str}/'

        start = time.time()
        print('band:',band)

        f = partial(data_io.groupby_apply_dispatcher, lightcurve_statistics.ad_fuller)
        adf_prob = data_io.dispatch_function(f, max_processes=cfg.USER.N_CORES, concat_output=True, **kwargs)

        adf_prob.to_csv(cfg.D_DIR + f"computed/{OBJ}/adf_prob_{args.survey.replace(' ','_')}_{band}_{phot_str}_{args.nobs_min}.csv")
        # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))