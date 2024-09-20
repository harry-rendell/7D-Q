import numpy as np
# Note, all these are defined in a way such that the first argument is the
#   lowest order term, and the last argument is the highest order term.

def power_law(x,a,b):
    """
    Basic power law, y = ax^b
    """
    return a*x**b

def power_law_offset(x, a, b, c):
    """
    Power law with linear offset, y = a + bx^c
    """
    return a + b*x**c

def linear(x, a, b):
    """
    Linear function, y = a + bx
    """
    return a + b*x

def DRW_SF(t, tau, SF_inf):
    """
    Damped random walk structure function
    """
    return SF_inf*((1 - np.exp(-t/tau))**0.5)

def mod_DRW_SF(t, tau, SF_inf, beta):
    """
    Modified damped random walk structure function
    """
    return SF_inf*(1 - np.exp(-t/tau))**beta

def bkn_pow_smooth(x, A, x_b, a_1, a_2, delta=1):
    """
    Smooth broken power law
    https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.SmoothlyBrokenPowerLaw1D.html
    x : x-values
    A : amplitude
    x_b : break point
    a_1 : power law index for x < x_b
    a_2 : power law index for x > x_b
    delta : smoothness parameter
    """
    return A*(x/x_b)**(a_1) * (0.5*(1+(x/x_b)**(1/delta)))**((a_2-a_1)*delta)

def bkn_pow(xvals,breaks,alphas):
    """
    Broken power law for multiple breaks. Takes a list of break points and a list of alphas.
    Note, the derivative of this function is not continuous at the break points.
    """
    try:
        if len(breaks) != len(alphas) - 1:
            raise ValueError("Dimensional mismatch. There should be one more alpha than there are breaks.")
    except TypeError:
        raise TypeError("Breaks and alphas should be array-like.")
    if any(breaks < np.min(xvals)) or any(breaks > np.max(xvals)):
        raise ValueError("One or more break points fall outside given x bounds.")
    
    breakpoints = [np.min(xvals)] + breaks + [np.max(xvals)] # create a list of all the bounding x-values
    chunks = [np.array([x for x in xvals if x >= breakpoints[i] and x <= breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
    
    all_y = []
    
    #alpha = pd.cut(pd.Series(xvals),breakpoints,labels=alphas,include_lowest=True).to_numpy()

    for idx,xchunk in enumerate(chunks):
        yvals = xchunk**alphas[idx]
        all_y.append(yvals) # add this piece to the output
    
    for i in range(1,len(all_y)):
        all_y[i] *= np.abs(all_y[i-1][-1]/all_y[i][0]) # scale the beginning of each piece to the end of the last so it is continuous
    
    return(np.array([y for ychunk in all_y for y in ychunk])) # return flattened list

def piecewise_exponential(t, k1, k2, t0):
    """
    Piecewise exponential function with two decay constants and a break point

    Parameters
    ----------
    t : array-like
        Time array
    k1 : float
        First decay constant
    k2 : float
        Second decay constant
    t0 : float
        Break point
    """
    mask = t<t0
    N1 = (np.exp(k1*t0)-1)/k1 - t0 # integral of y1 from 0 to t0
    N2 = np.exp(-k2*t0)/k2 * np.exp(k2*t0) * ( np.exp(k1*t0) - 1 ) # integral of N2 from t0 to inf
    norm = N1 + N2 # Total normalisation factor
    y1 = np.exp(k1*t[mask])-1
    y2 = np.exp(k2*t0) * ( np.exp(k1*t0)-1 ) * np.exp(-k2*t[~mask])
    unnormed = np.concatenate((y1, y2))
    return abs(unnormed/norm)