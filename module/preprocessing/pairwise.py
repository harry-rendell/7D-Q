import numpy as np
import pandas as pd
import os
from ..config import cfg
from .parse import split_into_non_overlapping_chunks, create_mask_from_bounds, filter_data
from scipy.stats import skew, kurtosis, iqr

def groupby_dtdm_between(df, args):
    """
    NOTE: Unused
    """
    for index, group in df.groupby(df.index.name):
        return calculate_dtdm_between_surveys(group, *args)
    
def groupby_dtdm_within(df, args):
    """
    NOTE: Unused
    """
    df_list = []
    for index, group in df.groupby(df.index.name):
        df_list.append(calculate_dtdm_within_surveys(group))
    return pd.concat(df_list, ignore_index=True)

def calculate_dtdm_between_surveys(group, sid_1, sid_2):
    """
    NOTE: Unused
    """
    mask = (group['sid']==sid_1).values
    n = mask.sum()*(~mask).sum()

    mjd_mag = group[['mjd','mag']].values
    magerr  = group['magerr'].values
    
    dtdm = mjd_mag[mask, np.newaxis] - mjd_mag[~mask]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr[mask, np.newaxis]**2 + magerr[~mask]**2 )**0.5
    uid = np.full(n,group.index[0],dtype='uint32')

    return pd.DataFrame({'uid':uid, 'dt':dtdm[:,:,0].ravel(), 'dm':dtdm[:,:,1].ravel(), 'de':dmagerr.ravel(), 'dm2_de2': (dtdm[:,:,1]**2 - dmagerr**2).ravel()})

def calculate_dtdm_within_surveys(group):
    """
    NOTE: Unused
    """
    mjd_mag = group[['mjd','mag']].values
    magerr = group['magerr'].values
    n = len(mjd_mag)

    unique_pair_indicies = np.triu_indices(n,1)

    dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
    dtdm = dtdm[unique_pair_indicies]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
    dmagerr = dmagerr[unique_pair_indicies]
    
    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={'uid':uid,'dt':dtdm[:,0],'dm':dtdm[:,1], 'de':dmagerr})

def groupby_save_pairwise(df, kwargs):
    if not (('basepath' in kwargs) and ('fname' in kwargs)):
        raise Exception('Both basepath and fname must be provided')
    if 'overwrite' in kwargs:
        overwrite = kwargs['overwrite']
    else:
        overwrite = False
    
    # Use appropriate time (rest frame or observer frame)
    mjd_key = kwargs['mjd_key'] if ('mjd_key' in kwargs) else 'mjd'

    # If a subset of objects has been provided, restrict our DataFrame to that subset.
    if 'subset' in kwargs:
        df = df[df.index.isin(kwargs['subset'])]

    # Remove observations that are not in the specified band
    df = df[df['band']==kwargs['band']]

    output_dir = os.path.join(kwargs['basepath'], 'dtdm_'+kwargs['band'])
    os.makedirs(output_dir, exist_ok=True)
    output_fpath = os.path.join(output_dir, kwargs['fname'].replace('lc','dtdm'))
    
    if os.path.exists(output_fpath) and not overwrite:
        raise Exception(f'File already exists: {output_fpath}')
    else:
        with open(output_fpath, 'w') as file:
            file.write(','.join([kwargs['ID'],'dt','dm','de','dsid']) + '\n')

    n_chunks = len(df)//30000 + 1 # May need to reduce this down to, say, 30,000 if the memory usage is too large.
    for i, chunk in enumerate(split_into_non_overlapping_chunks(df, n_chunks)):
        # If multiple bands are provided, iterate through them.
        s = chunk.groupby(kwargs['ID']).apply(calculate_dtdm, mjd_key)
        s = pd.DataFrame(s.values.tolist(), columns=s.columns, dtype='float32').astype({k:v for k,v in kwargs['dtypes'].items() if k in s.columns})
        s.dropna(axis=0, how='any').to_csv(output_fpath, index=False, header=False, mode='a')
        del s
    print('finished processing file:',kwargs['fname'], flush=True)

def calculate_dtdm(group, mjd_key):
    """
    Calculate ∆t, ∆m, ∆e, ∆sid for all pairs of observations in a group.
    ∆m = (m2 - m1)
    ∆t = (t2 - t1)
    ∆e = sqrt(e1^2 + e2^2)
    ∆sid = sid1*sid2
        where 1 and 2 are the first and second observations in the pair, respectively.
    ∆m < 0 corresponds to brightening
    ∆m > 0 corresponds to dimming
    """
    mjd_mag = group[[mjd_key,'mag']].values
    magerr = group['magerr'].values
    sid = group['sid'].values
    n = len(mjd_mag)

    unique_pair_indicies = np.triu_indices(n,1)
    
    dsid = sid*sid[:,np.newaxis]
    dsid = dsid[unique_pair_indicies]

    dtdm = mjd_mag - mjd_mag[:,np.newaxis,:]
    dtdm = dtdm[unique_pair_indicies]
    dtdm = dtdm*np.sign(dtdm[:,0])[:,np.newaxis]

    dmagerr = ( magerr**2 + magerr[:,np.newaxis]**2 )**0.5
    dmagerr = dmagerr[unique_pair_indicies]
    
    uid = np.full(n*(n-1)//2,group.index[0],dtype='uint32')
    
    return pd.DataFrame(data={group.index.name:uid, 'dt':dtdm[:,0], 'dm':dtdm[:,1], 'de':dmagerr, 'dsid':dsid})

def calculate_stats_looped(df, kwargs):
    """
    Loop over dtdm files and calculate stats of each file. Append to dictionary.
    Make sure to include name of desired quantites in names.
    
    Parameters
    ----------
    n_chunks : int
        how many files to read in of files to read in.
        maximum value: stars = 200/4 = 50 (though 46 seems to be better, 50 runs of out memory), qsos = 52/4 = 13

    log_or_lin : str

    save : bool

    Returns
    -------
    results : dict of nd_arrays, shape (n_chunk, n_points)
    """
    
    inner = kwargs['inner']
    features = kwargs['features']
    n_points = kwargs['n_points'] # number of points to plot
    mjd_edges = kwargs['mjd_edges']

    # names = ['n','SF 1', 'SF 2', 'SF 3', 'SF 4', 'SF weighted', 'SF corrected', 'SF corrected weighted', 'SF corrected weighted fixed', 'SF corrected weighted fixed 2', 'mean', 'mean weighted']
    results = {feature:np.zeros(shape=(n_points, 2)) for feature in features}
    results['n'] = np.zeros(shape=(n_points), dtype='uint64')

    if 'subset' in kwargs:
        df = df[df.index.isin(kwargs['subset'])]

    if inner:
        # df = df[df['dsid'].isin([25, 49, 121]).values] # Allow sdss-sdss, ps-ps, ztf-ztf only
        # [25,49,121,35,55,77] # all except ssa
        df = df[df['dsid'].isin([35, 25, 49, 121]).values] # Allow sdss-ps, sdss-sdss, ps-ps, ztf-ztf only

    mask1 = df['dm'].notna().values & df['de'].notna().values

    # bounds={'dm':df['dm'].quantile([0.001,0.999]).values, 'de':(0, 2)}
    # lower, upper = df['dm'].quantile([0.0001,0.9999]).values
    # lower = max(lower, -5)
    # upper = min(upper, 5)
    # bounds={'dm':(lower,upper), 'de':(0, 2)}

    # bounds={'dm':(-3,3), 'de':(0, 2)}
    # mask2 = create_mask_from_bounds(df, cfg.PREPROC.dtdm_bounds[kwargs['obj']])
    # print(bounds)
    # mask2 = create_mask_from_bounds(df, bounds)
    # df = df[mask1 & mask2]

    df = df[mask1]
    for j, edges in enumerate(zip(mjd_edges[:-1], mjd_edges[1:])):
        mjd_lower, mjd_upper = edges
        boolean = ((mjd_lower < df['dt']) & (df['dt']<mjd_upper)).values# & (df['dm2_de2']>0) # include last condition to remove negative SF values
        subset = df[boolean]

        lower, upper = subset['dm'].quantile([0.0001,0.9999]).values
        lower = max(lower, -5)
        upper = min(upper, 5)
        mask = create_mask_from_bounds(subset, {'dm':(lower,upper), 'de':(0, 2)})
        subset = subset[mask]

        n = len(subset)
        results['n'][j] = n
        # print('\t\tnumber of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))
        if n>0:
            weights = subset['de']**-2
            mean = np.average(subset['dm'], weights = weights)
            results['mean weighted a'][j,(0,1)] = mean, 1/weights.sum()
            results['mean weighted b'][j,(0,1)] = mean, subset['dm'].var()

            median = np.median(subset['dm'])
            results['median a'][j,(0,1)] = median, 1/weights.sum()
            results['median b'][j,(0,1)] = median, subset['dm'].var()

            if n>8:
                skew_err = (6*n*(n-1)/((n-2)*(n+1)*(n+3)))**0.5
                results['skewness'][j,(0,1)] = skew(subset['dm'], nan_policy='omit'), skew_err
            if n>20:
                kurt_err = 2*skew_err*( (n**2-1)/((n-3)*(n+5)) )**0.5
                results['kurtosis'][j,(0,1)] = kurtosis(subset['dm'], nan_policy='omit'), kurt_err
            
            weights = 0.5*subset['de']**-4

            dm2_de2 = subset['dm']**2 - subset['de']**2

            results['SF cwf a'][j, 0] = np.average(dm2_de2, weights = weights)
            results['SF cwf a'][j, 1] = 1/weights.sum()

            results['SF c'][j, 0] = np.average(dm2_de2)
            results['SF c'][j, 1] = dm2_de2.var()

            results['SF iqr'][j, 0] = (0.741*iqr(dm2_de2**0.5, nan_policy='omit'))**2
            results['SF iqr'][j, 1] = 1/weights.sum()

            results['SF'][j, 0] = np.average(subset['dm']**2)
            results['SF'][j, 1] = subset['dm'].var()
            
            mask_p = (subset['dm']>0).values
            mask_n = (subset['dm']<0).values
            
            try:
                if mask_p.sum()>0:
                    SF_p = np.average(dm2_de2[mask_p], weights = weights[mask_p])
                    if SF_p < 0:
                        SF_p = 0
                    results['SF cwf p'][j,0] = SF_p
                    results['SF cwf p'][j,1] = 1/weights[mask_p].sum()
                
                if mask_n.sum()>0:
                    SF_n = np.average(dm2_de2[mask_n], weights = weights[mask_n])
                    if SF_n < 0:
                        SF_n = 0
                    results['SF cwf n'][j,0] = SF_n
                    results['SF cwf n'][j,1] = 1/weights[mask_n].sum()
            except:
                print(f'weights cannot be normalized, no points in bin: {mjd_lower:.1f} < ∆t < {mjd_upper:.1f}', flush=True)
            
    del df
    return results

def calculate_stats_looped_key(df, kwargs):
    """
    Loop over dtdm files and calculate stats of each file. Append to dictionary.

    Parameters
    ----------
    n_chunks : int
        how many files to read in of files to read in.
        maximum value: stars = 200/4 = 50, qsos = 52/4 = 13

    log_or_lin : str

    save : bool

    Returns
    -------
    results : dict of nd_arrays, shape (n_chunk, n_points)
    """
    features = kwargs['features']
    n_points = kwargs['n_points'] # number of points to plot
    mjd_edges = kwargs['mjd_edges']
    groups = kwargs['groups']
    n_groups = len(groups)

    results = {feature:np.zeros(shape=(n_points, n_groups, 2)) for feature in features}
    results['n'] = np.zeros(shape=(n_points, n_groups), dtype='uint64')

    mask1 = df['dm'].notna().values & df['de'].notna().values
    mask2 = create_mask_from_bounds(df, cfg.PREPROC.dtdm_bounds[kwargs['obj']])
    df = df[mask1 & mask2]
    for group_idx in range(n_groups):
        subgroup = df[df.index.isin(groups[group_idx])]
        for j, edges in enumerate(zip(mjd_edges[group_idx][:-1], mjd_edges[group_idx][1:])):
            mjd_lower, mjd_upper = edges
            boolean = ((mjd_lower < subgroup['dt']) & (subgroup['dt'] < mjd_upper)).values# & (subgroup['dm2_de2']>0) # include last condition to remove negative SF values
            subset = subgroup[boolean]
            n = len(subset)
            results['n'][j, group_idx] = n
            # print('\t\tnumber of points in {:.1f} < ∆t < {:.1f}: {}'.format(mjd_lower, mjd_upper, boolean.sum()))
            if n>0:

                weights = subset['de']**-2

                results['mean weighted a'][j,group_idx,(0,1)] = np.average(subset['dm'], weights = weights), 1/weights.sum()
                results['mean weighted b'][j,group_idx,(0,1)] = np.average(subset['dm'], weights = weights), subset['dm'].var() 
                if n>8:
                    skew_err = (6*n*(n-1)/((n-2)*(n+1)*(n+3)))**0.5
                    results['skewness'][j,group_idx,(0,1)] = skew(subset['dm']), skew_err
                if n>10:
                    kurt_err = 2*skew_err*( (n**2-1)/((n-3)*(n+5)) )**0.5
                    results['kurtosis'][j,group_idx,(0,1)] = kurtosis(subset['dm']), kurt_err
                    
                weights = 0.5*subset['de']**-4

                dm2_de2 = subset['dm']**2 - subset['de']**2

                results['SF cwf a'][j,group_idx,0] = np.average(dm2_de2, weights = weights)
                results['SF cwf a'][j,group_idx,1] = 1/weights.sum()

                results['SF cwf b'][j,group_idx,0] = np.average(dm2_de2, weights = weights)
                results['SF cwf b'][j,group_idx,1] = dm2_de2.var()
                
                mask_p = (subset['dm']>0).values
                mask_n = (subset['dm']<0).values
                
                try:
                    if mask_p.sum()>0:
                        SF_p = np.average(dm2_de2[mask_p], weights = weights[mask_p])
                        if SF_p < 0:
                            SF_p = 0
                        results['SF cwf p'][j,group_idx,0] = SF_p
                        results['SF cwf p'][j,group_idx,1] = 1/weights[mask_p].sum()
                    
                    if mask_n.sum()>0:
                        SF_n = np.average(dm2_de2[mask_n], weights = weights[mask_n])
                        if SF_n < 0:
                            SF_n = 0
                        results['SF cwf n'][j,group_idx,0] = SF_n
                        results['SF cwf n'][j,group_idx,1] = 1/weights[mask_n].sum()
                except:
                    print(f'weights cannot be normalized, no points in bin: {mjd_lower:.1f} < ∆t < {mjd_upper:.1f}', flush=True)
                
    del df
    return results

# calculate pooled statistics
def calculate_pooled_statistics(results, n_points, n_groups=None):
    """
    Calculate pooled statistics from the results of calculate_stats... functions above
    """
    if n_groups is None:
        shape = (n_points, 2)
    else:
        shape = (n_points, n_groups, 2)

    pooled_results = {key:np.zeros(shape=shape) for key in results.keys()}
    pooled_results['n'] = np.zeros(shape=shape[:-1], dtype='uint64')

    for key in results.keys():
        if key == 'n':
            # Simply sum the bin counts
            pooled_results[key] = results[key].sum(axis=0)
        else:
            # Pooling statistics: see below.
            # https://en.wikipedia.org/wiki/Law_of_total_variance
            # https://arxiv.org/pdf/1007.1012.pdf
            # https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
            
            mask = results['n'].sum(axis=0) == 0
            if mask.sum()>0:
                print(f'Warning: {mask.sum()} bins have zero points', flush=True)
                results['n'][..., mask] = 1 # Avoid ZeroDivisionError in np.average
            
            # Pooled mean is the weighted average of the means
            pooled_mean = np.average(results[key][...,0], weights=results['n'], axis=0)
            # Pooled variance is the mean of the variances plus the variance of the means
            pooled_var  = np.average(results[key][...,1], weights=results['n'], axis=0) + np.average((results[key][...,0]-pooled_mean)**2, weights=results['n'], axis=0)
            
            if mask.sum()>0:
                # Set the bins with zero points to NaN 
                pooled_mean[mask] = np.nan
                pooled_var[mask]  = np.nan
            
            if key.startswith('SF'):
                pooled_results[key][...,0] = np.sign(pooled_mean) * (abs(pooled_mean) ** 0.5) # Square root to get SF instead of SF^2
            else:
                pooled_results[key][...,0] = pooled_mean
            
            pooled_results[key][...,1] = np.sign(pooled_var ) * (abs(pooled_var ) ** 0.5) # Square root to get sig instead of var

            
    return pooled_results