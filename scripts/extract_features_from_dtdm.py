import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, pairwise
from module.preprocessing.binning import construct_T_edges
from module.assets import load_grouped_tot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--name",    type=str, required=True, help="name for folder of output files, should start with log or lin to indicate whether to use log or linear time bins")
    parser.add_argument("--n_bins",  type=int, required=True, help="Number of time bins to use")
    parser.add_argument("--n_rows",  type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Whether to do a dry run (i.e. don't save the output)")
    parser.add_argument("--frame",   type=str, default='REST', help="OBS or REST to specify rest frame or observer frame time for Quasars")
    parser.add_argument("--inner", action='store_true', default=False, help="Apply pairwise analysis to points only within a survey")
    parser.add_argument("--mag_max", type=float, help="Maximum magnitude to use for the analysis")
    
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
        if args.frame == 'OBS':
            mjd_key = 'mjd'
        elif args.frame == 'REST':
            mjd_key = 'mjd_rf'
        else:
            raise ValueError(f'frame should be OBS or REST, not {args.frame}')
    elif OBJ == 'calibStars':
        ID = 'uid_s'
        mjd_key = 'mjd'
        args.frame = 'OBS'
    elif OBJ == 'sim':
        ID = 'uid'
        mjd_key = 'mjd'    

    nrows = args.n_rows
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores
    
    n_points = args.n_bins
    log_or_lin = args.name

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
              'log_or_lin':log_or_lin,
              'inner':args.inner,
              'features':['n', 'mean weighted a', 'mean weighted b', 'median a', 'median b', 'SF cwf a', 'SF c', 'SF', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis', 'SF iqr'],
              'n_points':n_points}

    if args.mag_max:
        log_or_lin += f'_mag_max_{args.mag_max}'
    
    for band in args.band:
        # set the maximum time to use for this band
        if args.frame:
            max_t = MAX_DTS[args.frame][OBJ][band]
        elif OBJ == 'qsos':
            max_t = MAX_DTS['REST']['qsos'][band]
        elif OBJ == 'calibStars':
            max_t = MAX_DTS['OBS']['calibStars'][band]
        elif OBJ == 'sim':
            max_t = MAX_DTS['OBS']['sim'][band]

        if args.mag_max:
            mag_med = load_grouped_tot(OBJ, band, usecols=['mag_med']).squeeze()
            uid_subset = mag_med[(mag_med < args.mag_max).values].index
            kwargs['subset'] = uid_subset

        assert not np.isnan(max_t), 'max_t is nan'
        # create time bins given the maximum time
        if log_or_lin.startswith('log'):
            mjd_edges = construct_T_edges(t_max=max_t, n_edges=n_points+1)
        elif log_or_lin.startswith('lin'):
            mjd_edges = np.linspace(0, max_t, n_points+1)
        else:
            raise ValueError(f'name should start with log or lin')

        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'
        kwargs['mjd_edges'] = mjd_edges

        start = time.time()
        print('band:',band)
        print('max_t',max_t)
        # create output directories
        output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_stats/all/{log_or_lin}/{band}')
        print(f'creating output directory if it does not exist: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        results = data_io.dispatch_function(pairwise.calculate_stats_looped, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)
        n_chunks = len(results)
        # repack results
        results = {key:np.array([result[key] for result in results]) for key in kwargs['features']}

        pooled_results = pairwise.calculate_pooled_statistics(results, n_points)

        if not args.dry_run:
            for key in pooled_results.keys():
                np.savetxt(os.path.join(output_dir, f"pooled_{key.replace(' ','_')}.csv"), pooled_results[key])
            np.savetxt(os.path.join(output_dir, 'mjd_edges.csv'), mjd_edges)

        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
