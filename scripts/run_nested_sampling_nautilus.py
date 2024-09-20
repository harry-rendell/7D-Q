import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import parse, binning, color_transform
from module.assets import load_vac
import pandas as pd
from scipy.stats import skewnorm, norm
from nautilus import Prior, Sampler

def likelihood(param_dict):
    x = [param_dict[p] for p in params]
    models = [x[i+1] + x[0]*np.log10(1+fits[i][:,3]) for i in range(len(x)-1)]
    t_pdfs = [np.sum(skewnorm.logpdf(models[i], a=fits[i][:,0], loc=fits[i][:,1], scale=fits[i][:,2])) for i in range(len(models))]
    return sum(t_pdfs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
    parser.add_argument("--survey",   type=str, help="name of surveys (separated with a space) to restrict data to. If left blank, then all surveys are used.")
    parser.add_argument("--output_dir",   type=str, help="Directory to save the output to.")
    parser.add_argument("--threshold", type=int, required=True, help="Threshold for masking")
    parser.add_argument("--qsos_per_bin", type=int, required=True, help="Threshold for masking")
    args = parser.parse_args()
    # Print the arguments for the log
    print(time.strftime('%H:%M:%S %d/%m/%y'))
    print('args:',args)
    
    OBJ = args.object
    if OBJ == 'qsos':
        ID = 'uid'
    else:
        ID = 'uid_s'

    if args.n_cores:
        cfg.USER.N_CORES = args.n_cores

    vac = load_vac('qsos', usecols=['z','Lbol'])
    skewfits = []
    for band in args.band:
        # TODO: we should use args.frame
        s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits_old/obs/{band}_{args.survey.replace(' ','_')}_{args.nobs_min}.csv", index_col=ID)
        s['band'] = band
        vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
        s = s.join(vac, on=ID)
        skewfits.append(s)
    skewfits = pd.concat(skewfits).dropna().sort_index()

    # TODO: put tau/sig bounds in here too.
    skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(1.5,5),'scale':(0.1,1), 'z':(0.2,5)}, verbose=True)

    mask_dict = binning.create_mask_lambda_lbol(skewfits, threshold=args.threshold, n = 15, l_low = 1000, l_high = 5000, L_low = 45.2, L_high = 47.2)
    fits = [skewfits[['a','loc','scale','z']][mask].sample(args.qsos_per_bin).values for mask in mask_dict.values()]
    
    start = time.time()
    os.makedirs(cfg.W_DIR + f'temp/{args.output_dir}/', exist_ok=True)
    os.chdir(cfg.W_DIR + f'temp/{args.output_dir}/')

    params = ['n'] + [f'c_{1+i}' for i in range(len(fits))]
    prior = Prior()
    for key in params:
        if key == 'n':
            # prior.add_parameter(key, dist=norm(loc=1.0, scale=0.5))
            prior.add_parameter(key, dist=norm(loc=1.0, scale=0.5))
        else:
            prior.add_parameter(key, dist=(2,5))

    print(f'dimensionality of mcmc: {len(params)}')

    sampler = Sampler(prior,
                      likelihood,
                      n_live=5000,
                      n_dim=len(params),
                      pool=args.n_cores,
                      filepath='checkpoint.hdf5',
                      resume=False)
    sampler.run(verbose=True)
    points, log_w, log_l = sampler.posterior()
    
    np.savez('posterior.npz', points=points, log_w=log_w, log_l=log_l)
    # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)),'\n')