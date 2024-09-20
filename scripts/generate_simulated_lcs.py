import pandas as pd
import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, parse
from module.modelling.carma import save_mock_dataset
from module.assets import load_grouped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band", type=str, required=True, help="filterband for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    ID = 'uid' if (OBJ == 'qsos') else 'uid_s'

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    # Make a script to do this, but do it properly. write it out on paper first.
    survey_dict = load_grouped(OBJ, bands=args.band)
    surveys = ['ssa', 'sdss', 'ps', 'ztf']
    survey_features = {s:{key:(survey_dict[s][key].mean(), survey_dict[s][key].std()) for key in ['mjd_min', 'mjd_max', 'n_tot']} for s in surveys}
    
    kwargs = {'survey_features':survey_features,
              'band':args.band}
    
    start = time.time()
    
    chunks = np.array_split(np.arange(1,1000*args.n_cores+1, dtype='int'), args.n_cores)
    suffixes, _ = parse.split_into_non_overlapping_chunks(None, args.n_cores, bin_size=5000, return_bin_edges=True)
    # chunks_and_suffixes = list(zip(chunks, suffixes))
    data_io.dispatch_function(save_mock_dataset, chunks=list(zip(chunks,suffixes)), max_processes=cfg.USER.N_CORES, **kwargs)

    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
