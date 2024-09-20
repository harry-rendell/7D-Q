import numpy as np
import pandas as pd

def construct_T_edges(t_max, n_edges):
    # note n_edges = n_bins + 1
    # Note that bins created this way have integer edges
    all_edges = np.ceil(np.logspace(1,np.log10(26062), 100)).astype('uint16') # 26062 is the max dt in the data, except qsos observer time frame which isnt used. 26702 is for qsos obs frame
    idxs = np.digitize(np.logspace(1, np.log10(t_max),n_edges-2), bins=all_edges, right=True)
    edges = all_edges[idxs]
    return np.concatenate(([0,5],edges))

def construct_m_edges(n_bins_m_, steepness, width, leftmost_bin=None):
    """
    Produce a set of bins with max extent 'width'.
    Note that by keeping the steepness the same, the bin spacing is linear with width.
    if leftmost_bin is not None, then first bin will begin from leftmost_bin. This is used for dm2_de2 which has an asymmetrical distribution about zero. 
        qsos:  bins=248 for dm2_de2, leftmost_bin = -0.244
        stars: bins=235 for dm2_de2, leftmost_bin = -0.21
    """
    steepness = steepness*width
    start = np.log10(steepness)
    stop = np.log10(steepness+width)
    bin_edges = np.concatenate((-np.logspace(start,stop,int(n_bins_m_/2+1))[:0:-1]+steepness,np.logspace(start,stop,int(n_bins_m_/2+1))-steepness))
    if leftmost_bin is not None:
        bin_edges = np.logspace(start,stop,int(n_bins_m_+1))-steepness
    # bin_centres = np.sign(bin_edges[:-1])*(bin_edges[1:]*bin_edges[:-1]) ** 0.5 # geometric mean. Note, difference between geometric and arithmetic mean is only 0.0006 mag at most (at the edges) so it is not necessary.s
    return bin_edges

def create_bins(bin_dict):
    n_bins_t = bin_dict['n_bins_t']
    n_bins_T = bin_dict['n_bins_T']
    n_bins_m = bin_dict['n_bins_m']
    n_bins_m2 = bin_dict['n_bins_m2']
    t_max = bin_dict['t_max']
    steepness = bin_dict['steepness']
    width = bin_dict['width']
    leftmost_bin = bin_dict['leftmost_bin']

    #compute edges of bins
    m_bin_edges = construct_m_edges(n_bins_m, steepness, width)
    m2_bin_edges = construct_m_edges(n_bins_m2, steepness, width, leftmost_bin)
    T_bin_edges = construct_T_edges(t_max=t_max, n_edges=n_bins_T+1)
    
    t_bin_edges = np.linspace(0,t_max,(n_bins_t+1))
    e_bin_edges = np.linspace(0,0.75,201) # This assumes that the max error pair is sqrt(0.5**2+0.5**2)

    # t denotes smaller time bins, T denotes larger time bins
    T_dict = dict(enumerate(['{0:1.0f}<∆t<{1:1.0f}'.format(T_bin_edges[i],T_bin_edges[i+1]) for i in range(len(T_bin_edges)-1)]))

    # only bin_edges need to be returned since:
    # n_bins = len(edges)-1
    # width = edges[1:] - edges[:-1]
    # centres = (edges[1:] + edges[:-1])/2

    bin_dict = {'t_bin_edges':t_bin_edges,
                'T_bin_edges':T_bin_edges,
                'm_bin_edges':m_bin_edges,
                'm2_bin_edges':m2_bin_edges,
                'e_bin_edges':e_bin_edges,
                'T_dict':T_dict}
    
    return bin_dict

def bin_data(dtdm, kwargs):
    args = kwargs['bin_dict']
    dts_binned = np.zeros((args['n_bins_T'],args['n_bins_t']), dtype = 'int64')
    dms_binned = np.zeros((args['n_bins_T'],args['n_bins_m']), dtype = 'int64')
    des_binned = np.zeros((args['n_bins_T'],args['n_bins_m']), dtype = 'int64')
    dm2_de2_binned = np.zeros((args['n_bins_T'],args['n_bins_m2']), dtype = 'int64') # This should stay as n_bins_m since we cut down the full array of length n_bins_m2
    dsid           = np.zeros((args['n_bins_T'],122), dtype = 'int64')

    dm2_de2 = dtdm[:,1]**2 - dtdm[:,2]**2
    # First decide which larger bin chunk the data should be in
    # start = time()
    idxs = np.digitize(dtdm[:,0], args['T_bin_edges'])-1
    for i in np.unique(idxs): #Can we vectorize this?
        mask = (idxs == i)
        dts_binned[i]     = np.histogram(dtdm[mask,0], args['t_bin_edges'])[0]
        dms_binned[i]     = np.histogram(dtdm[mask,1], args['m_bin_edges'])[0]
        des_binned[i]     = np.histogram(dtdm[mask,2], args['e_bin_edges'])[0]
        dm2_de2_binned[i] = np.histogram(dm2_de2[mask], args['m2_bin_edges'])[0]
        dsid[i]           = np.bincount( dtdm[mask,3].astype('int64'), minlength=122) # 121 = 11^2, since 11 is surveyID of ZTF

    del dtdm
    
    return dts_binned, dms_binned, des_binned, dm2_de2_binned, dsid

def create_binned_data_from_dtdm(df, kwargs):
    """
    TODO: This could be put in the function above
    """
    if not (('basepath' in kwargs) and ('fname' in kwargs)):
        raise Exception('Both basepath and fname must be provided')
    print(f"processing file: {kwargs['fname']}", flush=True)
    # If a subset of objects has been provided, restrict our DataFrame to that subset.
    if 'subset' in kwargs:
        df = df[df.index.isin(kwargs['subset'])]
    if 'inner' in kwargs:
        if kwargs['inner']:
            df = df[(np.sqrt(df['dsid'])%1==0).values]

    return bin_data(df[['dt','dm','de','dsid']].values, kwargs)

def calculate_groups(x, bounds):
    """
    Compute z score of key for each object

    Parameters
    ----------
    key : string
            property from VAC

    Returns
    -------
    bounds : array_like
            array of bounds to be used
    z_score : pandas.DataFrame
            z value of property column (value-mean / std) and original value
    self.bounds_values : values of property for each value in bounds
    """
    bounds_tuple = list(zip(bounds[:-1],bounds[1:]))
    mean = x.mean()
    std  = x.std()
    z_score = (x-mean)/std
    bounds_values = bounds * std + mean
    groups = [z_score[(lower <= z_score).values & (z_score < upper).values].index.values for lower, upper in bounds_tuple]
    # label_range_val = {i:'{:.1f} < {} < {:.1f}'.format(bounds_values[i],key,bounds_values[i+1]) for i in range(len(bounds_values)-1)}
    return groups, bounds_values

def assign_groups(df_, property_='Lbol'):
    import pandas as pd
    df = df_.copy()
    vac = pd.read_csv(cfg.D_DIR + 'catalogues/qsos/dr16q/dr16q_vac_shen_matched.csv', index_col=ID)
    vac = parse.filter_data(vac, cfg.PREPROC.VAC_BOUNDS, dropna=False)
    bounds_z = np.array([5,-3.5,-1,0,1,4])
    groups, bounds_values = calculate_groups(vac[property_], bounds = bounds_z)
    df['group'] = np.nan
    for i, group in enumerate(groups):
        df.loc[df.index.isin(group),'group'] = i
    print(df['group'].value_counts().sort_index())
    return df.join(vac, on='uid')

# def create_mask_lambda_lbol(df, threshold=100, n_l=15, n_L=15, l_low=1000, l_high=5000, L_low=45.2, L_high=47.2, gap=(0,0), return_edges=False, verbose=False):
#     """
#     Create a mask for each bin in the lambda-Lbol plane.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Dataframe containing at least uid, Lbol, wavelength
#     threshold : int
#         Minimum number of objects in a bin to be included in the mask
#     n : int
#         Number of bins in each dimension
#     l_low : float
#         Minimum wavelength
#     l_high : float
#         Maximum wavelength
#     L_low : float
#         Minimum Lbol
#     L_high : float
#         Maximum Lbol
#     gap : int
#         Number of bins to skip between each mask
#     verbose : bool
#         Print number of objects in each bin
    
#     Returns
#     -------
#     mask_dict : dict
#         Dictionary of masks, with keys (l,L) and values boolean arrays
#     """
#     import itertools

#     lambda_edges = np.linspace(l_low, l_high, n_l)
#     Lbol_edges   = np.linspace(L_low, L_high, n_L)

#     # create a series of 2d bins from the edges
#     Lbol_bins = pd.cut(df['Lbol'], Lbol_edges, labels=False)
#     lambda_bins = pd.cut(df['wavelength'], lambda_edges, labels=False)

#     # masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))]
#     masks_full = {(l,L):(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n_l-1), range(n_L-1))}
#     mask_dict = {key:value for key,value in masks_full.items() if (value.sum() > threshold) and (key[0] % (gap[0]+1) == 0) and (key[1] % (gap[1]+1) == 0)}
    
#     if verbose:
#         for key, mask in mask_dict.items():
#             print(f"Number of objects in bin {key}: {np.sum(mask)}")

#     if return_edges:
#         return mask_dict, lambda_edges, Lbol_edges
#     return mask_dict

def create_mask_lambda(df, threshold=100, n_l=15, n_p=15, l_low=1000, l_high=5000, gap=0, return_edges=False, verbose=False):
    """
    Create a mask for each bin in the lambda-Lbol plane.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least uid, Lbol, wavelength
    property_name : str
        Name of the property to bin
    threshold : int
        Minimum number of objects in a bin to be included in the mask
    n : int
        Number of bins in each dimension
    l_low : float
        Minimum wavelength
    l_high : float
        Maximum wavelength
    p_low : float
        Minimum property
    p_high : float
        Maximum property
    gap : int
        Number of bins to skip between each mask
    verbose : bool
        Print number of objects in each bin
    
    Returns
    -------
    mask_dict : dict
        Dictionary of masks, with keys (l,L) and values boolean arrays
    """
    import itertools

    lambda_edges = np.linspace(l_low, l_high, n_l)

    # create a series of 2d bins from the edges
    lambda_bins = pd.cut(df['wavelength'], lambda_edges, labels=False)

    # masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))]
    masks_full = {l:(lambda_bins == l).values for l in range(n_l-1)}
    mask_dict = {key:value for key,value in masks_full.items() if (value.sum() > threshold) and (key % (gap+1) == 0)}
    
    if verbose:
        for key, mask in mask_dict.items():
            print(f"Number of objects in bin {key}: {np.sum(mask)}")

    if return_edges:
        return mask_dict, lambda_edges
    return mask_dict

def create_mask_lambda_prop(df, property_name, threshold=100, n_l=15, n_p=15, l_low=1000, l_high=5000, p_low=45.2, p_high=47.2, gap=(0,0), return_edges=False, verbose=False):
    """
    Create a mask for each bin in the lambda-Lbol plane.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least uid, Lbol, wavelength
    property_name : str
        Name of the property to bin
    threshold : int
        Minimum number of objects in a bin to be included in the mask
    n : int
        Number of bins in each dimension
    l_low : float
        Minimum wavelength
    l_high : float
        Maximum wavelength
    p_low : float
        Minimum property
    p_high : float
        Maximum property
    gap : int
        Number of bins to skip between each mask
    verbose : bool
        Print number of objects in each bin
    
    Returns
    -------
    mask_dict : dict
        Dictionary of masks, with keys (l,L) and values boolean arrays
    """
    import itertools

    lambda_edges = np.linspace(l_low, l_high, n_l)
    prop_edges   = np.linspace(p_low, p_high, n_p)

    # create a series of 2d bins from the edges
    prop_bins = pd.cut(df[property_name], prop_edges, labels=False)
    lambda_bins = pd.cut(df['wavelength'], lambda_edges, labels=False)

    # masks = [(Lbol_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n-1), range(n-1))]
    masks_full = {(l,L):(prop_bins == L).values & (lambda_bins == l).values for l,L in itertools.product(range(n_l-1), range(n_p-1))}
    mask_dict = {key:value for key,value in masks_full.items() if (value.sum() > threshold) and (key[0] % (gap[0]+1) == 0) and (key[1] % (gap[1]+1) == 0)}
    
    if verbose:
        for key, mask in mask_dict.items():
            print(f"Number of objects in bin {key}: {np.sum(mask)}")

    if return_edges:
        return mask_dict, lambda_edges, prop_edges
    return mask_dict