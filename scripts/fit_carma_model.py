import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io
from module.modelling import carma
from module.assets import load_n_tot
from functools import partial

"""
This script fits a CARMA model to the photometric data for a given object and band.
It is analagous to the fit_drw_mcmc.py script, but does not do an MCMC fit.
Can be used to fit either a DRW or DHO model.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--frame",   type=str, required=True, help=("OBS or REST to specify rest frame or observer frame time. \n"
                                                                    "Defaults to rest frame for Quasars and observer time for Stars.\n"))
    parser.add_argument("--n_rows",  type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--survey",   type=str, help="name of survey to restrict data to. If left blank, then all surveys are used.")
    parser.add_argument("--model",  type=str, required=True, help ="Model to fit to the data. Options are drw or dho.")
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
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
    
    if args.frame.lower() == 'obs':
        mjd_key = 'mjd'
    elif args.frame.lower() == 'rest':
        mjd_key = 'mjd_rf'
    else:
        raise ValueError('frame must be either OBS or REST')

    if args.best_phot:
        phot_str = 'best_phot'
    else:
        phot_str = 'clean'

    nrows = args.n_rows
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'usecols': [ID,mjd_key,'mag','magerr','band','sid'],
              'ID':ID,
              'mjd_key':mjd_key}

    if args.survey:
        kwargs['sid'] = [cfg.PREPROC.SURVEY_IDS[s] for s in args.survey.split()]
    else:
        args.survey = 'all'

    if args.nobs_min:
        # We select a subset of quasars which have a minimum of args.nobs_min observations in each of the bands 
        #   specified in args.band
        # ignore ztf when using n_tot as a threshold if args.survey is not ztf
        n_tot = load_n_tot(OBJ)
        if args.survey != 'ztf':
            survey_str = args.survey.replace('ztf','').split()
        else:
            survey_str = args.survey.split()

        # mask = np.all([(n_tot[[f'n_{s}_{b}' for s in survey_str]].sum(axis=1) > args.nobs_min) for b in args.band], axis=0)
        # kwargs['subset'] = n_tot[mask].index
    else:
        args.nobs_min = 0

    for band in args.band:
        if args.nobs_min > 0:
            mask = (n_tot[[f'n_{s}_{band}' for s in survey_str]].sum(axis=1) > args.nobs_min)
            kwargs['subset'] = n_tot[mask].index

        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/{phot_str}/'

        start = time.time()
        print('band:',band)

        if args.model == 'drw':
            f = partial(data_io.groupby_apply_dispatcher, carma.apply_drw_fit)
        elif args.model == 'dho':
            f = partial(data_io.groupby_apply_dispatcher, carma.apply_dho_fit)
        else:
            raise ValueError('Invalid model selected. Options are drw or dho.')

        results = data_io.dispatch_function(f, max_processes=cfg.USER.N_CORES, concat_output=True, **kwargs)
        output_dir = cfg.D_DIR + f'computed/{OBJ}/{args.model}_fits/{args.frame}/'
        os.makedirs(output_dir, exist_ok=True)
        results.to_csv(output_dir + f"{band}_{args.survey.replace(' ','_')}_{args.nobs_min}_{phot_str}.csv")
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))