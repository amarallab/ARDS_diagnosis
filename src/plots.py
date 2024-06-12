import seaborn as sns


def stdfigsize(scale=1, nx=1, ny=1, ratio=1.3):
    """
    Returns a tuple to be used as figure size.
    -------
    returns (7*ratio*scale*nx, 7.*scale*ny)
    By default: ratio=1.3
    If ratio<0 them ratio = golden ratio
    """
    
    if ratio < 0:
        ratio = 1.61803398875
        
    return((7*ratio*scale*nx, 7*scale*ny))


def stdrcparams1():
    """
    Returns a dictionary of matplotlib rcParams for standardizing the plot style.
    
    This function sets the rcParams for various plot elements
    such as font type, font size, tick size, etc.
    
    These rcParams can be used to create consistent and visually appealing plots.
    
    Returns:
        dict: A dictionary containing the matplotlib rcParams.
    """
    
    sns.set_style("white")
    
    rcparams = {
        'pdf.fonttype' : 42,  # Only accepts 3 or 42
        'ps.fonttype' : 42,   # Only accepts 3 or 42
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 26,
        'axes.labelsize': 26,
        'axes.titlesize': 26,
        'legend.fontsize': 20,
        'ytick.right': 'off',
        'xtick.top': 'off',
        'ytick.left': 'on',
        'xtick.bottom': 'on',
        'xtick.labelsize': '21',
        'ytick.labelsize': '21',
        'axes.linewidth': 2,
        'xtick.major.width': 1.2,
        'xtick.minor.width': 1.2,
        'xtick.major.size': 10,
        'xtick.minor.size': 5,
        'xtick.major.pad': 10,
        'xtick.minor.pad': 10,
        'ytick.major.width': 1.2,
        'ytick.minor.width': 1.2,
        'ytick.major.size': 10,
        'ytick.minor.size': 5,
        'ytick.major.pad': 10,
        'ytick.minor.pad': 5,
        'axes.labelpad': 9,
        'axes.titlepad': 9,
        'axes.spines.right': True,
        'axes.spines.top': True
        }
    
    return rcparams