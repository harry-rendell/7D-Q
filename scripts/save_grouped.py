import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse, lightcurve_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="one or more filterbands (ie g or gri) to calculated grouped data for")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF or some combination (separated by a single space)")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--n_skiprows", type=int, help="Number of chunks of n_rows to skip when reading in photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Use this flag to print output instead of saving")
    parser.add_argument("--cleaned", action='store_true', help="Use this flat to point to cleaned data")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'

    nrows = args.n_rows
    skiprows = args.n_skiprows
    if args.n_skiprows and nrows is not None:
        skiprows = nrows * args.n_skiprows
    
    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    if args.cleaned:
        is_clean_str = 'clean'
    else:
        is_clean_str = 'unclean'
    
    for survey in args.survey.lower().split(' '):
        for band in args.band.lower():
            start = time.time()
            print('\nsurvey: {}, band: {}'.format(survey, band))
            # keyword arguments to pass to our reading function
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'skiprows': skiprows,
                      'basepath': cfg.D_DIR + 'surveys/{}/{}/{}/{}_band'.format(survey, OBJ, is_clean_str, band),
                      'ID':ID}

            df = data_io.dispatch_reader(kwargs)
            # Remove obviously bad data
            bounds={'mag':(15,25),'magerr':(0,2)}
            df = parse.filter_data(df, bounds=bounds, dropna=True)

            chunks = parse.split_into_non_overlapping_chunks(df, args.n_cores)
            kwargs = {'dtypes':cfg.PREPROC.stats_dtypes}
            grouped = data_io.dispatch_function(lightcurve_statistics.groupby_apply_stats, chunks=chunks, max_processes=args.n_cores, kwargs=kwargs)
            output_fpath = os.path.join(cfg.D_DIR, 'surveys/{}/{}/{}/{}_band'.format(survey, OBJ, is_clean_str, band), 'grouped.csv')
            if args.dry_run:
                print(grouped)
                print('output will be saved to:',output_fpath)
            else:
                grouped.to_csv(output_fpath)
                print('output has been saved to:',output_fpath)

            print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
    print('Finished')
