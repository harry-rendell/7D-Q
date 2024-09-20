import pandas as pd
import numpy as np
from multiprocessing import Pool
import os
from .import parse

def to_ipac(df, save_as, columns=None):
    from astropy.table import Table
    from astropy.io import ascii
    if columns is None:
        columns = df.columns
    t = Table.from_pandas(df[columns], index=True)
    ascii.write(t, save_as, format='ipac', overwrite=True)

def reader(fname, kwargs):
    """
    Generalised pandas csv reader.
    dtypes		: dict of column names and dtypes
    nrows 		: number of rows to read
    usecols		: list of columns to read
    skiprows	: number of rows to skip
    delimiter	: delimiter to use for parsing
    na_filter	: whether to check for NaNs when parsing.
        set to False if it is known that there are no NaNs in the file
        as it will improve performance.
    basepath	: path to the folder containing the file
    """
    dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else None
    nrows  = kwargs['nrows']  if 'nrows'  in kwargs else None
    usecols = kwargs['usecols'] if 'usecols' in kwargs else None
    skiprows = kwargs['skiprows'] if 'skiprows' in kwargs else None
    delimiter = kwargs['delimiter'] if 'delimiter' in kwargs else ','
    na_filter = kwargs['na_filter'] if 'na_filter' in kwargs else True
    basepath = kwargs['basepath']
    
    if 'ID' in kwargs:
        ID = kwargs['ID']
    else:
        raise Exception('ID must be provided in keyword args')

    # Open the file and skip any comments. Leave the file object pointed to the header.
    # Pass in the header in case we decide to skip rows.
    with open(os.path.join(basepath,fname)) as file:
        ln = 0
        for line in file:
            ln += 1
            if not line.strip().startswith(('#','|','\\')):
                names = line.replace('\n','').split(',')
                break
        df = pd.read_csv(file,
                         usecols=usecols,
                         dtype=dtypes,
                         nrows=nrows,
                         names=names,
                         skiprows=skiprows,
                         delimiter=delimiter,
                         na_filter=na_filter).set_index(ID)
        if 'uids' in kwargs:
            df = df[df.index.isin(kwargs['uids'])]
        
        return df

def dispatch_reader(kwargs, multiproc=True, i=None, max_processes=64, concat=True, fnames=None, uids=None):
    """
    Dispatching function for reader
    """
    if uids is None:
        if fnames is None:
            # If we have not passed in a list of filenames, then read all files in basepath
            fnames = sorted([f for f in os.listdir(kwargs['basepath']) if ((f.startswith('lc_') or f.startswith('dtdm_')) and f.endswith('.csv'))])
        elif i is not None:
            raise Exception('Cannot specify both i and fnames')
                
        if isinstance(i, int):
            fnames = fnames[i]
            multiproc = False
        elif isinstance(i, np.ndarray):
            fnames = fnames[i]
        elif isinstance(i, list):
            fnames = np.array(fnames)[i]
    else:
        fnames = find_lcs_containing_uids(uids, kwargs['basepath'])
        kwargs['uids'] = uids
    
    n_files = len(fnames)
    if __name__ == 'module.preprocessing.data_io':
        # Make as many tasks as there are files, unless we have set max_processes
        n_tasks = min(n_files, max_processes)
        with Pool(n_tasks) as pool:
            df_list = pool.starmap(reader, [(fname, kwargs) for fname in fnames])
        # sorting is required as we cannot guarantee that starmap returns dataframes in the order we expect.
        if concat:
            return pd.concat(df_list, sort=True)
        else:
            return df_list


def writer(i, chunk, kwargs):
    """
    Writing function for multiprocessing
    """
    mode = kwargs['mode'] if 'mode' in kwargs else 'w'
    savecols = kwargs['savecols'] if 'savecols' in kwargs else None

    if 'basepath' in kwargs:
        basepath = kwargs['basepath']
    else:
        raise Exception('user must provide path for saving output')

    # if folder does not exist, create it
    os.makedirs(basepath, exist_ok=True)

    with open(os.path.join(basepath,'lc_{}.csv'.format(i)), mode) as f:
        if 'comment' in kwargs:
            newline = '' if kwargs['comment'].endswith('\n') else '\n'
            f.write(kwargs['comment']+newline)
        chunk.to_csv(f, columns=savecols, header=(not mode.startswith('a')))
        if mode.startswith('w'):
            print('output saved to:',f.name)

def to_csv(df, save_as, columns=None, comment=None):
    """
    Generalised pandas csv writer, allowing user to pass comments, 
        written at the top of the file
    """
    if columns is None:
        columns = df.columns
    # overwrite file if it exists
    with open(save_as, 'w') as f:
        pass
    with open(save_as, 'a') as f:
        if comment is not None:
            newline = '' if comment.endswith('\n') else '\n'
            f.write(comment+newline)
        df.to_csv(f, columns=columns)

def dispatch_writer(chunks, kwargs, max_processes=64, fname_suffixes=None):
    """
    Dispatching function for writer
    if fnames is provided, use them to name each file for saving (list must be in the same order as chunks)
        otherwise name each file lc_0, lc_1, ...#
    NOTE fname is actually the middle part of lc_{}.csv
    TODO: Is it bad that we sometimes spawn more processes than needed?
    """
    # If we are passed a DataFrame rather than a list of DataFrames, wrap it in a list.
    if isinstance(chunks, pd.DataFrame): chunks = [chunks]

    if fname_suffixes is None:
        iterable = enumerate(chunks)
    else:
        iterable = zip(fname_suffixes, chunks)

    if __name__ == 'module.preprocessing.data_io':
        n_tasks = min(len(chunks), max_processes)
        with Pool(n_tasks) as pool:
            pool.starmap(writer, [(i, chunk, kwargs) for i, chunk in iterable])

def process_input(function, df_or_fname, kwargs):
    """
    Handles inputs for dispatch_function
    """
    if isinstance(df_or_fname, str):
        # In this case, are provided a filename. Read it with reader() then pass to function()
        # Add the fname to the dictionary so function() can access it if necessary.
        kwargs['fname'] = df_or_fname
        return function(reader(df_or_fname, kwargs), kwargs)
    else:
        # In this case, df_or_fname is a DataFrame chunk.
        return function(df_or_fname, kwargs)

def dispatch_function(function, chunks=None, max_processes=64, concat_output=True, **kwargs):
    """
    Parameters
    ----------
    function : function for which we use to dispatch on files or DataFrame object via multiprocessing
    chunks : DataFrame, list of DataFrames or None
        if a DataFrame is provided, it will be split automatically into number of max_processes
        if a list of DataFrames are provided, they are unpacked and passed to 'function'
        if left as None, then basepath may be provided in kwargs. files that match basepath/lc_{.*}.csv are read and passed
            to 'function'

    Note, we may provide kwargs as usual or pass a dictionary as **{...}

    Returns
    -------
    returns the output of 'function'
    """
    if chunks is None:
        if 'basepath' in kwargs:
            chunks = sorted([f for f in os.listdir(kwargs['basepath']) if ((f.startswith('lc_') or f.startswith('dtdm_')) and f.endswith('.csv'))])
        else:
            raise Exception('Either one of chunks or basepath (in kwargs) must be provided')
    elif isinstance(chunks, pd.DataFrame):
        chunks = parse.split_into_non_overlapping_chunks(chunks, max_processes)
    elif not isinstance(chunks, list):
        raise Exception('Invalid input')

    if __name__ == 'module.preprocessing.data_io':
        # Make as many processes as there are files/chunks.
        # There may be more elements in chunks than there are processes,
        #   in this case the tasks will do them in turn.
        n_tasks = min(len(chunks), max_processes)
        with Pool(n_tasks) as pool:
            output = pool.starmap(process_input, [(function, chunk, kwargs) for chunk in chunks])
        
        if not all(o is None for o in output):
            # If pool.starmap returns some output:
            # If it is better to save chunks rather than concatenate result into one DataFrame
            #    (eg in case of calculate dtdm) then only run this block if a result is returned.
            if concat_output:
                output = pd.concat(output, sort=True) # overwrite immediately for prevent holding unnecessary dataframes in memory
                if 'dtypes' in kwargs:
                    dtypes = {k:v for k,v in kwargs['dtypes'].items() if k in output.columns}
                else:
                    dtypes={}
                return output.astype(dtypes)
            else:
                return output

def groupby_apply_dispatcher(func, df, kwargs):
    if 'fname' in kwargs:
        print(f"processing file: {kwargs['fname']}", flush=True)
    
    masks = []
    if 'subset' in kwargs:
        masks.append(df.index.isin(kwargs['subset']))
    
    if 'sid' in kwargs:
        assert (type(kwargs['sid']) is not int), 'sid must be a list or array, not an int'
        masks.append(df['sid'].isin(kwargs['sid']))

    if 'dsid' in kwargs:
        assert (type(kwargs['dsid']) is not int), 'dsid must be a list or array, not an int'
        masks.append(df['dsid'].isin(kwargs['dsid']))
    
    if ('band' in kwargs) & ('band' in df.columns):
        masks.append(df['band'] == kwargs['band'])
    
    if len(masks)>0:
        masks = np.all(masks, axis=0)
        df = df[masks]

    s = df.groupby(df.index.name, group_keys=False).apply(func, kwargs)
    return pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32')

    # if 'subset' in kwargs:
    #     df = df[df.index.isin(kwargs['subset'])]
    # if 'sid' in kwargs:
    #     df = df[df['sid'].isin(kwargs['sid'])]
    # if ('band' in kwargs) & ('band' in df.columns):
    #     s = df[df['band'] == kwargs['band']].groupby(df.index.name).apply(func, kwargs)
    # else:
    #     s = df.groupby(df.index.name).apply(func, kwargs)
    # return pd.DataFrame(s.values.tolist(), index=s.index, dtype='float32')

def find_lcs_containing_uids(uids, basepath):
    file_bounds = {f:(int(f[3:9]),int(f[10:16])) for f in sorted(os.listdir(basepath)) if f.startswith('lc_')}
    fnames = []
    for f, bounds in file_bounds.items():
        for uid in uids:
            if bounds[0] <= uid <= bounds[1]:
                if f not in fnames:
                    fnames.append(f)
    return fnames

def load_lcs_containing_uids(uids, basepath, max_processes=4):
    """
    Load lightcurves containing the uids provided.
    uids : list of uids to load
    basepath : path to the folder containing the lightcurves
    max_processes : maximum number of processes (ie cpus)
    """
    fnames = find_lcs_containing_uids(uids, basepath)
    n_files = len(fnames)
    kwargs = {'basepath':basepath, 'uids':uids, 'ID':'uid'}
    if __name__ == 'module.preprocessing.data_io':
        # Make as many tasks as there are files, unless we have set max_processes
        n_tasks = min(n_files, max_processes)
        with Pool(n_tasks) as pool:
            df_list = pool.starmap(reader, [(fname, kwargs) for fname in fnames])

        # sorting is required as we cannot guarantee that starmap returns dataframes in the order we expect.
        return pd.concat(df_list, sort=True)
    