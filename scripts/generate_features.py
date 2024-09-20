import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, lightcurve_statistics
from module.preprocessing.binning import construct_T_edges
from functools import partial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_bins",  type=int, required=True, help="Number of time bins to use")
    parser.add_argument("--n_rows",  type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--frame",   type=str, help=("OBS or REST to specify rest frame or observer frame time. \n"
                                                     "Defaults to rest frame for Quasars and observer time for Stars.\n"))
    parser.add_argument("--inner", action='store_true', default=False, help="Apply pairwise analysis to points only within a survey")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
        mjd_key = 'mjd_rf'
    else:
        ID = 'uid_s'
        mjd_key = 'mjd'

    nrows = args.n_rows
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores
    
    n_points = args.n_bins

    if args.inner:
        MAX_DTS = cfg.PREPROC.MAX_DT_INNER
    else:
        MAX_DTS = cfg.PREPROC.MAX_DT
    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.lc_dtypes,
              'nrows': nrows,
              'usecols': [...], # TODO: decide on columns
              'ID':ID,
              'mjd_key':mjd_key,
              'inner':args.inner,
              'features':['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF c', 'SF', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis'],
              'n_points':n_points}
    
    for band in args.band:
        # set the maximum time to use for this band
        if args.frame:
            max_t = MAX_DTS[args.frame][OBJ][band]
        elif OBJ == 'qsos':
            max_t = MAX_DTS['REST']['qsos'][band]
        elif OBJ == 'calibStars':
            max_t = MAX_DTS['OBS']['calibStars'][band]

        mjd_edges = construct_T_edges(t_max=max_t, n_edges=n_points+1)

        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean'

        start = time.time()
        print('band:',band)
        print('max_t',max_t)
        # create output directories

        # output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_stats/all/{log_or_lin}/{band}')
        # print(f'creating output directory if it does not exist: {output_dir}')
        # os.makedirs(output_dir, exist_ok=True)
        f = partial(data_io.groupby_apply_dispatcher, lightcurve_statistics...) # TODO: add relevant function
        results = data_io.dispatch_function(f, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=True, **kwargs)
        results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/..._{band}.csv') # TODO: decide on a filename

        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))