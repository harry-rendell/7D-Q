import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF")
    parser.add_argument("--n_chunks", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Use this flag to print output instead of saving")
    parser.add_argument("--best_phot", action='store_true', help="Use this flag to only save the best subset of the data")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'

    if args.best_phot:
        phot_str = 'best_phot'
    else:
        phot_str = 'clean'

    if OBJ == 'qsos':
        SAVE_COLS = [ID,'mjd','mjd_rf','mag','magerr','band','sid']
        redshift = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr14q/dr14q_redshift.csv').set_index(ID)
    else:
        SAVE_COLS = [ID,'mjd','mag','magerr','band','sid']
    uid_ranges, _ = parse.split_into_non_overlapping_chunks(None, 106, bin_size=5000, return_bin_edges=True)
    # uid_ranges, _ = parse.split_into_non_overlapping_chunks(None, 53, bin_size=10000, return_bin_edges=True)
    for uid_range in uid_ranges:
        output_name = cfg.D_DIR + f'merged/{OBJ}/{phot_str}/lc_{uid_range}.csv'
        with open(output_name, 'w') as file:
            file.write(','.join(SAVE_COLS) + '\n')

    start = time.time()
    for survey in args.survey.lower().split(' '):
        print(survey)
        for band in args.band.lower():
            print(band)
            nrows  = None
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'ID':ID,
                      'basepath': cfg.D_DIR + f'surveys/{survey}/{OBJ}/clean/{band}_band/',
                      'usecols':[ID,'mjd','mag','magerr']}
            
            df = data_io.dispatch_reader(kwargs, multiproc=True, max_processes=32).dropna()
            if args.best_phot:
                # File below is generated in grouped_analysis-NB.ipynb
                subset = np.loadtxt(cfg.D_DIR + f'merged/{OBJ}/{phot_str}/best_phot_uids.csv', dtype=np.uint32)
                df = df[df.index.isin(subset)]
            df['band'] = np.array(band, dtype=np.dtype(('U',1)))
            df['sid'] = np.array(cfg.PREPROC.SURVEY_IDS[survey], dtype=np.uint8)

            if OBJ == 'qsos':
                df = df.join(redshift, on=ID)
                df['mjd_rf'] = df['mjd']/(1+df['z'])

            uid_ranges, chunks = parse.split_into_non_overlapping_chunks(df, 106, bin_size=5000, return_bin_edges=True)
            # uid_ranges, chunks = parse.split_into_non_overlapping_chunks(df, 53, bin_size=10000, return_bin_edges=True)
            kwargs = {'basepath':cfg.D_DIR + f'merged/{OBJ}/{phot_str}/',
                      'mode':'a',
                      'savecols':SAVE_COLS[1:]}
            data_io.dispatch_writer(chunks, kwargs=kwargs, max_processes=32, fname_suffixes=uid_ranges)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
