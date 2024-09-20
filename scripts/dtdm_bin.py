import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import data_io, binning
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="one or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--name",    type=str, required=True, help="name for folder of output files")
    parser.add_argument("--n_bins_T",  type=int, required=True, help="Number of time bins to use. Default is 19")
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
    
    bin_dict1 = {'n_bins_t': 200,
                 'n_bins_T': args.n_bins_T,
                 'n_bins_m': 200,
                 'n_bins_m2': 200,
                 'width': 2,
                 'steepness': 0.1,
                 'leftmost_bin': -0.244}

    # keyword arguments to pass to our reading function
    kwargs = {'obj':OBJ,
              'dtypes': cfg.PREPROC.dtdm_dtypes,
              'nrows': nrows,
              'usecols': [ID,'dt','dm','de','dsid'],
              'ID':ID,
              'mjd_key':mjd_key,
              'inner':args.inner}

    if args.inner:
        MAX_DTS = cfg.PREPROC.MAX_DT_COMBINED['inner']
    else:
        MAX_DTS = cfg.PREPROC.MAX_DT_COMBINED['outer']

    for band in args.band:
        # set the maximum time to use for this band
        # if args.frame:
        #     t_max = MAX_DTS[args.frame][OBJ][band]
        # elif OBJ == 'qsos':
        #     t_max = MAX_DTS['REST']['qsos'][band]
        # elif OBJ == 'calibStars':
        #     t_max = MAX_DTS['OBS']['calibStars'][band]
        bin_dict1['t_max'] = MAX_DTS[band]

        bin_dict2 = binning.create_bins(bin_dict1)

        # create output directories
        output_dir = os.path.join(cfg.D_DIR, f'computed/{OBJ}/dtdm_binned/{args.name}/')
        print(f'creating output directory if it does not exist: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)

        # add these back into the kwargs dictionary
        kwargs['band'] = band
        kwargs['basepath'] = cfg.D_DIR + f'merged/{OBJ}/clean/dtdm_{band}'
        kwargs['output_dir'] = output_dir
        kwargs['bin_dict'] = {**bin_dict1, **bin_dict2}

        start = time.time()
        print('band:',band)
    
        results = data_io.dispatch_function(binning.create_binned_data_from_dtdm, chunks=None, max_processes=cfg.USER.N_CORES, concat_output=False, **kwargs)

        # save results
        dts_binned     = np.array([a[0] for a in results]).sum(axis=0)
        dms_binned     = np.array([a[1] for a in results]).sum(axis=0)
        des_binned     = np.array([a[2] for a in results]).sum(axis=0)
        dm2_de2_binned = np.array([a[3] for a in results]).sum(axis=0)
        dsids_binned   = np.array([a[4] for a in results]).sum(axis=0)

        # np.savetxt(output_dir + 'dts.csv', dts_binned, fmt='%i', delimiter=',')
        # np.savetxt(output_dir + 'dms.csv', dms_binned, fmt='%i', delimiter=',')
        # np.savetxt(output_dir + 'des.csv', des_binned, fmt='%i', delimiter=',')
        # np.savetxt(output_dir + 'dm2_de2.csv', dm2_de2_binned, fmt='%i', delimiter=',')
        # np.savetxt(output_dir + 'dcs.csv', dsids_binned, fmt='%i', delimiter=',')
        
        # save bin_dict as a json
        # np.save(output_dir + 'bin_dict.npy', kwargs['bin_dict'])
        np.savez(output_dir + f'binned_{band}.npz', dts_binned=dts_binned, dms_binned=dms_binned, des_binned=des_binned, dm2_de2_binned=dm2_de2_binned, dsids_binned=dsids_binned, bin_dict=kwargs['bin_dict'])
        
        print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)))
