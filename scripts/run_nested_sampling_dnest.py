import numpy as np
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module.config import cfg
from module.preprocessing import parse, binning, color_transform
from module.modelling import fitting
from module.assets import load_vac
import pandas as pd
import dnest4
import numpy.random as rng
from scipy.stats import skewnorm
import dnest4.classic as dn4

def prior_transform(u):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[1, 5)`."""

    # x = 1. + 4*u 
    x = 4*u 

    # This is n set to 1. Change the first term here to set it to another
    # value, or comment out this line to let n be free.

    # x[0] = 1 + 1e-6*(u[0]-0.5)

    return x

def logprob(x):
    models = [x[i+1] + x[0]*np.log10(1+fits[i][:,3]) for i in range(len(x)-1)]
    t_pdfs = [np.sum(skewnorm.logpdf(models[i], a=fits[i][:,0], loc=fits[i][:,1], scale=fits[i][:,2])) for i in range(len(models))]
    return sum(t_pdfs)

class Model(object):
    """
    DNest4 model.
    """

    def __init__(self):
        self.ndim = len(fits)+1

    def from_prior(self):
        return rng.rand(self.ndim)

    def perturb(self, coords):
        i = np.random.randint(self.ndim)
        coords[i] += dnest4.randh()
        # Note: use the return value of wrap, unlike in C++
        coords[i] = dnest4.wrap(coords[i], 0.0, 1.0)
        return 0.0

    def log_likelihood(self, coords):
        try:
            logl = logprob(prior_transform(coords))
        except:
            logl = -np.Inf
        return logl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",  type=str, required=True, help ="qsos or calibStars")
    parser.add_argument("--band",    type=str, required=True, help="One or more filterbands for analysis")
    parser.add_argument("--n_cores", type=int, required=True, help="Number of cores to use. If left blank, then this value is taken from N_CORES in the config file.")
    parser.add_argument("--nobs_min",type=int, help="Minimum number of observations required to fit a model.")
    parser.add_argument("--survey",   type=str, help="name of surveys (separated with a space) to restrict data to. If left blank, then all surveys are used.")

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
        s = pd.read_csv(cfg.D_DIR + f"computed/qsos/mcmc_fits/skewfit_{band}_{args.survey.replace(' ','_')}_{args.nobs_min}.csv", index_col=ID)
        s['band'] = band
        vac['wavelength'] = color_transform.calculate_wavelength(band, vac['z'])
        s = s.join(vac, on=ID)
        skewfits.append(s)
    skewfits = pd.concat(skewfits).dropna().sort_index()
    skewfits = parse.filter_data(skewfits, bounds={'a':(0,0.01),'loc':(2,5),'scale':(0.1,1), 'z':(0.2,5)}, verbose=True)

    mask_dict = binning.create_mask_lambda_lbol(skewfits, n = 15, l_low = 1000, l_high = 5000, L_low = 45.2, L_high = 47.2)
    fits = [skewfits[['a','loc','scale','z']][mask].sample(100).values for mask in mask_dict.values()]
    print(f'mcmc dimensionality: {len(fits)+1}')

    # Cannot parallelise this? :(
    start = time.time()
    os.makedirs(cfg.W_DIR + 'temp/dnest4_logs_parse_free_n2', exist_ok=True)
    os.chdir(cfg.W_DIR + 'temp/dnest4_logs_parse_free_n2/')
    model = Model()
    sampler = dnest4.DNest4Sampler(model,backend=dnest4.backends.CSVBackend(".", sep=" "))
    gen = sampler.sample(100, num_steps=100000, new_level_interval=5000,
                         num_per_step=50, thread_steps=100,
                         num_particles=5, lam=1, beta=100, seed=1234)
    for i, sample in enumerate(gen):
        if (i + 1) % 100 == 0:
            print(f"Sample {i+1}.", flush=True)
            dn4.postprocess(plot=False, temperature=1)
            # Convert uniform coordinates to parameter space
            posterior_sample = np.atleast_2d(np.loadtxt("posterior_sample.txt"))
            for i in range(posterior_sample.shape[0]):
                posterior_sample[i, :] = prior_transform(posterior_sample[i, :])
            np.savetxt("posterior_sample.txt", posterior_sample)
    
    # results.to_csv(cfg.D_DIR + f'computed/{OBJ}/features/{args.model}_fits_{band}_{args.survey}_{args.nobs_min}_{phot_str}.csv')
    print('Elapsed:',time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start)),'\n')