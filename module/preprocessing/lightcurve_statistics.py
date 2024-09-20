import numpy as np
import pandas as pd
from .data_io import groupby_apply_dispatcher
from statsmodels.tsa.stattools import adfuller

def groupby_apply_average(df, kwargs):
    s = df.groupby([df.index.name, 'mjd_floor']).apply(average_nightly_obs)
    return pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32').reset_index('mjd_floor', drop=True)

def groupby_apply_stats(df, kwargs):
    """
    TODO replace with partial(groupby_apply_dispatcher, stats)
    """
    return groupby_apply_dispatcher(stats, df, kwargs)

def groupby_apply_welch_stetson(df, kwargs):
    return groupby_apply_dispatcher(welch_stetson_JK, df, kwargs)


def ad_fuller(group, kwargs):
    if len(group) < 4:
        prob = np.nan
    else:
        prob = adfuller(group['mag'].values)[1]
    return {'ad_fuller':prob}

def welch_stetson_JK(group):
    """
    Welch-Stetson variability index J and K (Stetson 1996)
    
    Might be better to make this a function of ∆m, ∆t

    mag : array of magnitudes
    magerr : array of magnitude errors
    mag_mean : float of magnitude mean
    n : int of number of observations
    """
    n = len(group)
    if n==1:
        J, K = np.nan, np.nan
    else:
        mag = group['mag'].values
        mag_mean = np.mean(mag)
        delta = (n/(n-1)) ** 0.5 * (mag-mag_mean)/np.std(mag)

        K = n**-0.5 * np.abs(delta).sum() / ((delta**2).sum()**0.5)

        p = delta * delta[:, np.newaxis]
        k = int((n*(n-1))/2)
        unique_pair_indicies = np.triu_indices(n,1)
        p = p[unique_pair_indicies]

        J = ( np.sign(p)*(np.abs(p)**0.5) ).sum()/k

    return {'welch_stetson_j':J, 'welch_stetson_k':K}

def calculate_sf_per_qso(group, kwargs):
    mjd_edges = kwargs['mjd_edges']
    n_points = kwargs['n_points']
    features = kwargs['features']

    dm, dt, de = group[['dm','dt','de']].values.T
    n_features = len(features)
    a = np.zeros(shape=(n_points, n_features), dtype=np.float32)
    for i in range(n_points):
        mask = (mjd_edges[i] < dt) & (dt <= mjd_edges[i+1])
        if mask.sum()==0:
            a[i,1:] = np.nan
        else:
            a[i,0] = int(mask.sum())
            weights = 0.5*de[mask]**-4
            a[i,1] = np.average(dm[mask]**2 - de[mask]**2, weights=weights) # SF2_cw
            a[i,2] = np.average(dm[mask]**2, weights=weights) # SF2_w
            a[i,3] = 1/weights.sum() # SF2_err

    return {f'{key}_{mjd_edges[i]}_{mjd_edges[i+1]}':a[i,j] for i in range(n_points) for j, key in enumerate(features)}



def average_nightly_obs(group):
    """
    Parameters
    ----------
    group : pandas.DataFrame
        A group of observations for a single object on a single night.
    Returns
    -------
    pandas.Series
        A series containing the average magnitude, error, and number of observations for the group.
    """
    n = len(group)
    # bear in mind this will fail on PS data which does not have mag_orig
    mjd, mag, magerr, mag_median, mag_std = group[['mjd','mag','magerr','mag_med','mag_std']].values.T
    if 'mag_orig' in group:
        mag_native = group['mag_orig'].values
    
    if np.ptp(mag) > 0.5:
        mask = abs(mag-mag_median) < 5*mag_std
        if mask.sum()==0:
            mag_mean = np.nan
            mag_mean_native = np.nan
            uid = group.index[0]
            err_uids = str(uid)+', '+str(int(mjd[0]))
            print(err_uids, flush=True)
                
        else:
            mag = mag[mask] # remove points that are 1mag away from the median of the group
            mag_native = mag_native[mask]
            magerr = magerr[mask]
            mjd = mjd[mask]
            
    mjd_mean  = np.mean(mjd)
    magerr_mean = ((magerr ** 2).sum()/n) ** 0.5 # sum errors in quadrature. Do not use 'error on optimal average' since it makes the errors unphysically small.
    mag_mean  = -2.5*np.log10(np.average(10**(-(mag-8.9)/2.5), weights = magerr**-2)) + 8.9
    mag_mean_native  = -2.5*np.log10(np.average(10**(-(mag_native-8.9)/2.5), weights = magerr**-2)) + 8.9
    # we don't really care about mag_orig, and if we want to compare mag vs mag_orig we can look at the unclean data. Let's leave it out.
    return {'mjd':mjd_mean, 'mag':mag_mean, 'mag_orig':mag_mean_native, 'magerr':magerr_mean}

def stats(group, kwargs):
    """
    Parameters
    ----------
    group : pandas.DataFrame
        A group of observations for a single object.
    Returns
    -------
    pandas.Series
        A series containing the statistics for the group.
    """
    # assign pandas columns to numpy arrays
    mjd, mag, magerr = group[['mjd','mag','magerr']].values.T

    # number of observations
    n_tot   = len(group)

    if n_tot == 1:
        mjd_min = mjd[0]
        mjd_max = mjd[0]
        mjd_ptp = 0
        mag_min = np.nan
        mag_max = np.nan
        mag_mean = mag[0]
        mag_med = mag[0]
        mag_opt_mean = np.nan
        mag_opt_mean_flux = np.nan
        mag_std = np.nan
        magerr_max = np.nan
        magerr_mean = magerr[0]
        magerr_med = magerr[0]
        magerr_opt_std = np.nan

        if 'mag_orig' in group:
            mag_mean_native = group['mag_orig'].values[0]
            mag_med_native = group['mag_orig'].values[0]
            mag_stats_native = {'mag_mean_native':mag_mean_native,
                                'mag_med_native':mag_med_native}
        else:
            mag_stats_native = {}
            
    else:
        # time
        mjd_min =  mjd.min()
        mjd_max =  mjd.max()
        mjd_ptp =  np.ptp(group['mjd'])

        # magnitudes, using PS system
        mag_min = min(mag)
        mag_max = max(mag)
        mag_med = -2.5*np.log10(np.median(10**(-(mag-8.9)/2.5))) + 8.9
        mag_mean= -2.5*np.log10(np.mean  (10**(-(mag-8.9)/2.5))) + 8.9
        mag_std = np.std(mag)
        
        # native (untransformed) magnitudes
        if 'mag_orig' in group:
            mag_native = group['mag_orig'].values
            mag_med_native  = -2.5*np.log10(np.median(10**(-(mag_native-8.9)/2.5))) + 8.9
            mag_mean_native = -2.5*np.log10(np.mean  (10**(-(mag_native-8.9)/2.5))) + 8.9
            mag_stats_native = {'mag_mean_native':mag_mean_native,
                                'mag_med_native':mag_med_native}
        else:
            mag_stats_native = {}
        
        # magnitude errors
        magerr_max = magerr.max()
        magerr_med = np.median(magerr)
        magerr_mean= np.mean(magerr)
        
        # using flux flux
        flux = 10**(-(mag-8.9)/2.5)
        fluxerr = 0.921*flux*magerr # ln(10)/2.5 ~ 0.921
        fluxerr_mean_opt = ( flux*(fluxerr**-2) ).sum()/(fluxerr**-2).sum()
        # calculate the optimal flux average then convert back to mags
        mag_opt_mean_flux = -2.5*np.log10(fluxerr_mean_opt) + 8.9
        # magerr_opt_std_flux = not clear how to calculate this. magerr_opt_std should suffice.

        # optimal (inverse-variance weighted) averages (see aaa04)
        mag_opt_mean   = ( mag*(magerr**-2) ).sum()/(magerr**-2).sum()
        magerr_opt_std = (magerr**-2).sum()**-0.5

    return {**{'n_tot':n_tot,
            'mjd_min':mjd_min,
            'mjd_max':mjd_max,
            'mjd_ptp':mjd_ptp,
            'mag_min':mag_min,
            'mag_max':mag_max,
            'mag_mean':mag_mean,
            'mag_med':mag_med,
            'mag_opt_mean':mag_opt_mean,
            'mag_opt_mean_flux':mag_opt_mean_flux,
            'mag_std':mag_std,
            'magerr_max':magerr_max,
            'magerr_mean':magerr_mean,
            'magerr_med':magerr_med,
            'magerr_opt_std':magerr_opt_std}, **mag_stats_native}
