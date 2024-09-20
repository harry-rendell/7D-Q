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
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--survey", type=str, required=True, help="SDSS, PS or ZTF")
    parser.add_argument("--n_cores", type=int, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--n_rows", type=int, help="Number of rows to read in from the photometric data")
    parser.add_argument("--n_skiprows", type=int, help="Number of chunks of n_rows to skip when reading in photometric data")
    parser.add_argument("--dry_run", action='store_true', help="Number of chunks of n_rows to skip when reading in photometric data")
    args = parser.parse_args()
    # Print the arguments and time for the log
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

    for survey in args.survey.lower().split(' '):
        for band in args.band.lower():
            print('\n'+'='*34+f' {survey}, {band} '+'='*35)
            # keyword arguments to pass to our reading function
            kwargs = {'dtypes': cfg.PREPROC.lc_dtypes,
                      'nrows': nrows,
                      'skiprows': skiprows,
                      'usecols': [ID,'mjd','mag','mag_orig','magerr'],
                      'basepath': cfg.D_DIR + 'surveys/{}/{}/unclean/{}_band/'.format(survey, OBJ, band),
                      'ID':ID}

            df = data_io.dispatch_reader(kwargs, multiproc=True)

            # Read in grouped data
            grouped = pd.read_csv(cfg.D_DIR + 'surveys/{}/{}/unclean/{}_band/grouped.csv'.format(survey, OBJ, band), usecols=[ID, 'mag_med','mag_std']).set_index(ID)
            mask = (grouped['mag_med']<cfg.PREPROC.LIMIT_MAG[survey.upper()][band]).values
            print(f'Removing {(~mask).sum():,} objects due to being fainter than limiting mag ({cfg.PREPROC.LIMIT_MAG[survey.upper()][band]}) of survey:')
            uids = grouped.index[mask]
            df = df[df.index.isin(uids)]

            # Remove obviously bad data and uids that should not be present.
            valid_uids = pd.read_csv(cfg.D_DIR + 'catalogues/{}/{}_subsample_coords.csv'.format(OBJ, OBJ), usecols=[ID], index_col=ID, comment='#')
            df = parse.filter_data(df, bounds=cfg.PREPROC.FILTER_BOUNDS, dropna=True, valid_uids=valid_uids)

            # Discretise mjd
            df['mjd_floor'] = np.floor(df['mjd']).astype('uint32')

            # Separate observations into ones which share another observation on the same night (multi_obs) and those that do not (single_obs)
            mask = df.reset_index()[[ID,'mjd_floor']].duplicated(keep=False).values
            if mask.sum() != 0:
                single_obs = df[~mask].drop(columns='mjd_floor')
                multi_obs = df[mask]
                del df

                # Join lightcurve median and std onto dataframe
                multi_obs = multi_obs.join(grouped[['mag_med','mag_std']], on=ID)
                
                chunks = parse.split_into_non_overlapping_chunks(multi_obs, args.n_cores)
                kwargs = {'dtypes':cfg.PREPROC.lc_dtypes}

                # make header for output
                print('Unable to average the following:\n'+ID+', mjd')

                start = time.time()
                averaged = data_io.dispatch_function(lightcurve_statistics.groupby_apply_average, chunks=chunks, max_processes=args.n_cores, kwargs=kwargs)
                end   = time.time()
                print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(end-start)))

                # Add the data from nights with multiple observations back on
                df = pd.concat([single_obs,averaged])
                df.index = df.index.astype('uint32')
            else:
                print('No observations on the same night to average.')
            
            if args.dry_run:
                print(df)
            else:
                # Add comment to start of csv file
                comment = ( "# CSV of photometry with preprocessing and cleaning.\n"
                            "# mag : photometry in PanSTARRS photometric system")
                # keyword arguments to pass to our writing function
                kwargs = {'comment':comment,
                          'basepath':cfg.D_DIR + 'surveys/{}/{}/clean/{}_band/'.format(survey, OBJ, band),
                          'savecols':['mjd','mag','mag_orig','magerr']}
                
                df = df.sort_values([ID,'mjd'])
                chunks = parse.split_into_non_overlapping_chunks(df, 4)
                data_io.dispatch_writer(chunks, kwargs)
    print('Finished')
