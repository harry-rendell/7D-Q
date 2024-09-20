"""
Save (∆t, ∆m) pairs from lightcurves. 
dtdm defined as: ∆m = (m2 - m1), ∆t = (t2 - t1) where (t1, m1) is the first obs and (t2, m2) is the second obs.
Thus a negative ∆m corresponds to a brightening of the object

time_key : str
    either mjd or mjd_rf for regular and rest frame respectively

Output
-------
DataFrame(columns=[self.ID, 'dt', 'dm', 'de', 'dsid'])
    dt : time interval between pair
    dm : magnitude difference between pair
    de : error on dm, calculated by combining individual errors in quadrature as sqrt(err1^2 + err2^2)
    dsid : an ID representing which catalogue this pair was created from, calculated as survey_id_1*survey_id_2
        where survey_ids are: 
            3 = SSS
            5 = SDSS
            7 = PS1
            11 = ZTF
"""

import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, pairwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--n_skiprows", type=int, help="Number of chunks of n_rows to skip when reading in photometric data")
    parser.add_argument("--overwrite", action='store_true', default=False, help="Overwrite existing files")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
        mjd_key = 'mjd_rf'
    elif OBJ == 'calibStars':
        ID = 'uid_s'
        mjd_key = 'mjd'
    elif OBJ == 'sim':
        ID = 'uid'
        mjd_key = 'mjd_rf'

    nrows = args.n_rows
    skiprows = args.n_skiprows
    if args.n_skiprows and nrows is not None:
        skiprows = nrows * args.n_skiprows
    
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # keyword arguments to pass to our reading function
    kwargs = {'dtypes': {**cfg.PREPROC.lc_dtypes, **cfg.PREPROC.dtdm_dtypes},
              'nrows': nrows,
              'skiprows': skiprows,
              'basepath': cfg.D_DIR + f'merged/{OBJ}/clean/',
              'na_filter': False,
              'usecols': [ID,mjd_key,'mag','magerr','band','sid'],
              'ID':ID,
              'mjd_key':mjd_key,
              'overwrite':args.overwrite}
              
    start = time.time()
    
    for band in args.band:
        print(f'band: {band}')
        if (OBJ == 'qsos') | (OBJ == 'calibStars'):
            bool_arr = pd.read_csv(cfg.D_DIR + f'catalogues/{OBJ}/sets/clean_{band}.csv', index_col=ID, comment='#')
            if OBJ == 'qsos':
                mask = bool_arr['vac'].values & np.any(bool_arr[['sdss','ps']].values, axis=1)
            else:
                mask = np.any(bool_arr[['sdss','ps']].values, axis=1)
            restricted_set = bool_arr.index[mask]
            print(f'size of restricted set: {len(restricted_set)}')
            del bool_arr
            kwargs['subset'] = restricted_set
        kwargs['band'] = band

        data_io.dispatch_function(pairwise.groupby_save_pairwise, chunks=None, max_processes=cfg.USER.N_CORES, **kwargs)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
