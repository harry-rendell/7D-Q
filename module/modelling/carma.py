import os
import pandas as pd
import numpy as np
from ..config import cfg
from eztao.ts import drw_fit, dho_fit,  neg_param_ll
from eztao.carma import DRW_term
from eztao.ts import gpSimRand
from celerite import GP
import emcee
from .models import piecewise_exponential

def apply_drw_fit(group, kwargs):
    """
    Find best DRW parameters for a single lightcurve using whole lightcurve and just ZTF
    """
    t, y, yerr = group[[kwargs['mjd_key'], 'mag', 'magerr']].values.T
    
    try:
        sig, tau = drw_fit(t, y, yerr)
    except Exception as e:
        print('error with uid:', group.index[0],'\n',e,flush=True)
        sig, tau = np.nan, np.nan
    
    try:
        if (tau > np.ptp(t)) | (tau < np.diff(t).min()):
            # condition imposted by Yu 2022
            sig, tau = np.nan, np.nan
    except:
        sig, tau = np.nan, np.nan

    return {'sig':sig, 'tau':tau, 'mjd_ptp':np.ptp(t), 'n':len(t)}

def apply_fit_drw_mcmc(group, kwargs):
    """
    Use MCMC to find 16th, 50th and 84th percentiles for tau and sigma DRW parameters
    TODO: Add hyperparams of MCMC fit into kwargs
    """
    t, y, yerr = group[[kwargs['mjd_key'], 'mag', 'magerr']].values.T
    sid_counts = group['sid'].value_counts()
    n_sss = sid_counts.get(3, default=0)
    n_sdss = sid_counts.get(5, default=0)
    n_ps = sid_counts.get(7, default=0)
    n_ztf = sid_counts.get(11, default=0)
    
    nan_result = {'sig16':np.nan, 'sig50':np.nan, 'sig84':np.nan, 'tau16':np.nan, 'tau50':np.nan, 'tau84':np.nan, 'mjd_ptp':np.ptp(t),
                  'n':len(t), 'nsss':n_sss, 'nsdss':n_sdss, 'nps':n_ps, 'nztf':n_ztf}
    # obtain best-fit 
    try:
        best_drw = drw_fit(t, y, yerr)
    except Exception as e:
        print('error with eztao fit for uid:', group.index[0],'\n',e, flush=True)
        return nan_result
    # define celerite GP model
    drw_gp = GP(DRW_term(*np.log(best_drw)), mean=np.median(y))
    drw_gp.compute(t, yerr)

    # define log prob function
    def param_ll(*args):
        return -neg_param_ll(*args)

    # initialize the walker, specify number of walkers, prob function, args and etc.
    initial = np.array(np.log(best_drw))
    ndim, nwalkers = len(initial), 16
    sampler_drw = emcee.EnsembleSampler(nwalkers, ndim, param_ll, args=[y, drw_gp])

    # run a burn-in surrounding the best-fit parameters obtained above
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    try:
        p0, _, _ = sampler_drw.run_mcmc(p0, 500, skip_initial_state_check=False)
    except Exception as e:
        print('error with burn-in for uid:', group.index[0],'\n',e, flush=True)
        return nan_result
    # clear up the stored chain from burn-in, rerun the MCMC
    sampler_drw.reset()
    try:
        sampler_drw.run_mcmc(p0, 1000, skip_initial_state_check=False)
    except Exception as e:
        print('error with MCMC for uid:', group.index[0],'\n',e, flush=True)
        return nan_result

    # remove points with low prob (ie less than 5%) for the sake of making good corner plot
    prob_threshold_drw = np.percentile(sampler_drw.flatlnprobability, 5)
    clean_chain_drw = sampler_drw.flatchain[sampler_drw.flatlnprobability > prob_threshold_drw, :]
    
    if len(clean_chain_drw) == 0:
        print(f'clean chain (n={len(t)}) is empty for uid:', group.index[0], flush=True)
        return nan_result
    
    p = np.percentile(clean_chain_drw, q=[16,50,84], axis=0) * 0.4342944819032518 # multiply to convert from ln to log10

    return {'sig16':p[0,0], 'sig50':p[1,0], 'sig84':p[2,0], 'tau16':p[0,1], 'tau50':p[1,1], 'tau84':p[2,1], 'mjd_ptp':np.ptp(t),
            'n':len(t), 'nsss':n_sss, 'nsdss':n_sdss, 'nps':n_ps, 'nztf':n_ztf}

def apply_dho_fit(group, kwargs):
    """
    Find best DHO parameters for a single lightcurve
    """

    t, y, yerr = group[[kwargs['mjd_key'], 'mag', 'magerr']].values.T
    
    try:
        alpha1, alpha2, beta1, beta2  = dho_fit(t, y, yerr)
    except Exception as e:
        print('error with uid:', group.index[0],'\n',e, flush=True)
        alpha1, alpha2, beta1, beta2 = np.nan, np.nan, np.nan, np.nan

    return {'alpha1':alpha1, 'alpha2':alpha2, 'beta1':beta1, 'beta2':beta2, 'mjd_ptp':np.ptp(t), 'n':len(t)}

def generate_mask(t, survey_features):
    # np.random.seed(6)
    indices = np.array([], dtype='int')
    sid = np.array([], dtype='int')
    for s in survey_features.keys():
        idx_lower = np.argmin(abs(survey_features[s]['mjd_min'][0] - t))
        idx_upper = np.argmin(abs(survey_features[s]['mjd_max'][0] - t))
        if idx_lower == idx_upper:
            continue
        n_tot = int(max(1,3*np.random.normal(survey_features[s]['n_tot'][0], survey_features[s]['n_tot'][1])))
        indices = np.append(indices, np.unique(np.random.randint(idx_lower, idx_upper, size=n_tot)))
        sid = np.append(sid, np.full(n_tot, cfg.PREPROC.SURVEY_IDS[s]))
        # idxs = np.random.randint(0, len(t), size=np.random.randint(0,len(t)), replace=False)
        idxs = np.argsort(indices)

    return indices[idxs], sid[idxs]

def generate_simulated_lc(survey_features=None, superposed_model=None, frac=0.1):
    # np.random.seed(6)
    amp = -1
    while amp < 0:
        amp = np.random.normal(loc=0.6, scale=0.14)
    tau = -1
    while tau < 0:
        tau = np.random.normal(loc=50000, scale=5000)
    DRW_kernel = DRW_term(np.log(amp), np.log(tau))
    t,y,yerr = gpSimRand(DRW_kernel, SNR=10, duration=365*70, N=int(365*70*frac), nLC=1, log_flux=True, season=False, full_N=10000)
    # t,y,yerr = np.linspace(0,365*70,int(365*70*frac)), np.ones(int(365*70*frac)), np.ones(int(365*70*frac))*0.1 # uncomment this to see what the flares look like by themselves
    # np.random.seed(6)
    y += np.random.normal(20, 1)
    if superposed_model == 'piecewise_exponential':
        # a = np.random.uniform(5, 50)
        # b = np.random.uniform(1, a//2)
        # y /= (1+piecewise_exponential(t, a/t[-1], b/t[-1], t[-1]//np.random.randint(2,8))*5e2)
        # Add 5 flares
        N_FLARES = 8
        t0 = np.random.uniform(-t[-1],2*t[-1], size=N_FLARES)
        multiplier = sum([piecewise_exponential(t, 100/t[-1], 15/t[-1], t0[i]) for i in range(N_FLARES)])
        # N = 10
        # k1 = np.linspace(100, 1000, N)
        # k2 = k1/70
        # multiplier = sum([piecewise_exponential(t, k1[i]/t[-1], k2[i]/t[-1], np.random.uniform(0,t[-1]*0.9)) for i in range(N)])

        # scale them by a factor of 50
        y /= (1+50*multiplier)

    t += 33238 # shift lcs to first observation of SSS

    if survey_features is None:
        idxs = np.sort(np.unique(np.random.randint(0, len(t), size=np.random.randint(len(t)/2,len(t)-1))))
        # idxs = np.sort(np.unique(np.random.randint(len(t)//4, len(t)//2, size=np.random.randint(0,len(t)//2)))) # half the length of the light curve so that our exponentials cover the span of the light curve
        sid = np.full(len(idxs), 0)
    else:
        idxs, sid = np.sort(generate_mask(t, survey_features))
    
    return t[idxs], y[idxs], yerr[idxs], sid

def generate_mock_dataset(uids, kwargs):
    # np.random.seed(6)
    # Use os.urandom to generate a random seed, required if we are running this function in parallel
    if 'seed' in kwargs:
        np.random.seed(kwargs['seed'])
    else:
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    survey_features = kwargs['survey_features']
    superposed_model = kwargs['superposed_model'] if 'superposed_model' in kwargs else None
    frac = kwargs['frac'] if 'frac' in kwargs else 0.1
    b = []
    for uid in uids:
        mjd, mag, magerr, sid = generate_simulated_lc(survey_features, superposed_model=superposed_model, frac=frac)
        c = pd.DataFrame({'mjd':mjd, 'mag':mag, 'magerr':magerr, 'sid':sid})
        c['uid'] = uid
        # c['mjd_rf'] = c['mjd']/(1+abs(np.random.uniform(1.83, 0.78)))
        c['mjd_rf'] = c['mjd']/(1+abs(np.random.uniform(1, 4)))
        b.append(c)

    b = pd.concat(b)
    b = b.astype({col:dtype for col, dtype in cfg.PREPROC.lc_dtypes.items() if col in b.columns})
    b['band'] = kwargs['band']
    return b
    
def save_mock_dataset(uids_and_suffix, kwargs):
    uids, suffix = uids_and_suffix
    df = generate_mock_dataset(uids, kwargs)
    df.to_csv(cfg.D_DIR + f'merged/sim/clean/lc_{suffix}.csv', index=False)
