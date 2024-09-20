import pandas as pd
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
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

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.dtdm_dtypes,
              'usecols': [ID,'dt','dm','de','dsid'],
              'savecols': ['dt','dm','de','dsid'],
              'ID':ID,
              'dsid':args.dsid
              }
    
    def save_dtdm(df, args):
        mask = df['dsid'].isin(args['dsid']).values
        if mask.sum() > 0:
            fname = args['fname']
            df[mask].to_csv(os.path.join(output_dir, fname), columns=args['savecols'])

    for band in args.band:
        start = time.time()

        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'

        start = time.time()
        print('band:',band)

        # create output directories
        output_dir = os.path.join(cfg.D_DIR, f'merged/{OBJ}/dsid_{"_".join(map(str,args.dsid))}/dtdm_{band}')
        print(f'creating output directory if it does not exist: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        results = data_io.dispatch_function(save_dtdm, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)

        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
