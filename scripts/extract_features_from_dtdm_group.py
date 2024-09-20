import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, pairwise, binning
from module.assets import load_vac

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--property",type=str, required=True, help="QSO property to use for splitting features")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--name",    type=str, required=True, help="name for folder of output files, should start with log or lin to indicate whether to use log or linear time bins")
    parser.add_argument("--n_bins",  type=int, required=True, help="Number of time bins to use")
    parser.add_argument("--n_rows",  type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Whether to do a dry run (i.e. don't save the output)")
    parser.add_argument("--frame",   type=str, default='REST', help="OBS or REST to specify rest frame or observer frame time for Quasars")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = 'qsos'
    ID = 'uid'
    if args.frame == 'OBS':
        mjd_key = 'mjd'
    elif args.frame == 'REST':
        mjd_key = 'mjd_rf'
    else:
        raise ValueError(f'frame should be OBS or REST, not {args.frame}')
    nrows = args.n_rows
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    n_points = args.n_bins
    log_or_lin = args.name

    vac = load_vac(OBJ, 'dr16q_vac')
    vac = parse.filter_data(vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)

    bounds_z = np.array([-3.5,-1.5,-1,-0.5,0,0.5,1,1.5,3.5])
    groups, bounds_values = binning.calculate_groups(vac[args.property], bounds = bounds_z)
    n_groups = len(groups)
    
    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.dtdm_dtypes,
              'nrows': nrows,
              'usecols': [ID,'dt','dm','de','dsid'],
              'ID':ID,
              'mjd_key':mjd_key,
              'log_or_lin':log_or_lin,
              'inner':False, # too few observations per bin if we try this
              'features':['n', 'mean weighted a', 'mean weighted b', 'SF cwf a', 'SF cwf b', 'SF cwf p', 'SF cwf n', 'skewness', 'kurtosis'],
              'n_points':n_points,
              'groups':groups,
              'property':args.property}
    
    for band in args.band:
        # set the maximum time to use for this band
        max_ts = cfg.PREPROC.MAX_DT_VAC[args.property][band]
        
        # create time bins given the maximum time
        if log_or_lin.startswith('log'):
            mjd_edges = [np.logspace(0, np.log10(max_t), n_points+1) for max_t in max_ts]
        elif log_or_lin.startswith('lin'):
            mjd_edges = [np.linspace(0, max_t, n_points+1) for max_t in max_ts]
        else:
            raise ValueError(f'name should start with log or lin')
       
        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'
        kwargs['mjd_edges'] = mjd_edges

        start = time.time()
        print('band:',band)

        # create output directories
        output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_stats/{args.property}/{log_or_lin}/{band}')
        print(f'creating output directory if it does not exist: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        results = data_io.dispatch_function(pairwise.calculate_stats_looped_key, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)
        n_chunks = len(results)
        # repack results
        results = {key:np.array([result[key] for result in results]) for key in kwargs['features']}

        pooled_results = pairwise.calculate_pooled_statistics(results, n_points, n_groups)
        
        if not args.dry_run:
            for key in pooled_results.keys():
                for group_idx in range(n_groups):
                    np.savetxt(os.path.join(output_dir, f"pooled_{key.replace(' ','_')}_{group_idx}.csv"), pooled_results[key][:, group_idx])
            for i, mjd_edge in enumerate(mjd_edges):
                np.savetxt(os.path.join(output_dir, f'mjd_edges_{i}.csv'), mjd_edge)
            np.savetxt(os.path.join(output_dir, 'bounds_values.csv'), bounds_values)

        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
