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
    parser.add_argument("--name", type=str, default='', help="Name to append to the output file")
    parser.add_argument("--dsid", type=int, nargs="+", help="list of dsids to use for pairs")
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
              'dtypes': cfg.PREPROC.dtdm_dtypes,
              'nrows': nrows,
              'usecols': [ID,'dt','dm','de','dsid'],
              'ID':ID,
              'mjd_key':mjd_key,
              'inner':args.inner,
              'features':['n', 'SF2_cw', 'SF2_w', 'SF_err'],
              'n_points':n_points
              }
    
    if args.dsid:
        # [25, 49, 35, 15, 21, 33] # sdss-sdss, ps-ps, sdss-ps, ssa-sdss, ssa-ps, ssa-ztf
        kwargs['dsid'] = args.dsid
    
    for band in args.band:
        # set the maximum time to use for this band
        if args.frame:
            max_t = MAX_DTS[args.frame][OBJ][band]
        elif OBJ == 'qsos':
            # max_t = MAX_DTS['REST']['qsos'][band]
            max_t = MAX_DTS['REST']['qsos']['r'] # fix this so all bands have the same mjd bin edges
        elif OBJ == 'calibStars':
            max_t = MAX_DTS['OBS']['calibStars'][band]
        
        mjd_edges = construct_T_edges(t_max=max_t, n_edges=n_points+1) # These bins have integer edges by design
        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'
        kwargs['mjd_edges'] = mjd_edges

        start = time.time()
        print('band:',band)
        print('max_t',max_t)
        # create output directories

        # output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_stats/all/{log_or_lin}/{band}')
        # print(f'creating output directory if it does not exist: {output_dir}')
        # os.makedirs(output_dir, exist_ok=True)

        # define keys to save columns in the right order, otherwise non-ordered dictionary means they are saved in random order
        # the features here MUST match the features in lightcurve_statistics.calculate_sf_per_qso
        f = partial(data_io.groupby_apply_dispatcher, lightcurve_statistics.calculate_sf_per_qso)
        results = data_io.dispatch_function(f, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=True, **kwargs)

        columns = [f'{key}_{mjd_edges[i]}_{mjd_edges[i+1]}' for key in kwargs['features'] for i in range(n_points)]
        output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/features/{band}')
        os.makedirs(output_dir, exist_ok=True)
        # change dtype of columns that start with n to integer
        for col in columns:
            if col.startswith('n'):
                results[col] = results[col].astype(int)
        results[columns].to_csv(os.path.join(output_dir, f'SF_{args.n_bins}_bins_{args.name}.csv'))
        np.savetxt(os.path.join(output_dir, f'mjd_edges_{args.n_bins}_bins_{args.name}.csv'), mjd_edges, delimiter=',', fmt='%d')
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))