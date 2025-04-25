"""
This module provides utility functions for generating publication-quality plots 
styled for Nature Communications. It includes functions for calculating figure 
sizes based on caption length and layout type, as well as standardizing matplotlib 
rcParams for consistent and visually appealing plots.

Author(s): Félix L. Morales, Ritika Giri, Luiz G.A. Alves
"""
from typing import Tuple
import seaborn as sns


def stdfigsize(
    len_caption: int,
    n_rows: int = 1,
    n_cols: int = 1,
    ratio: float = 1.3,
    layout: str = "double",
) -> Tuple[float, float]:
    """
    Calculate the figure size for plots styled for Nature Communications.

    Parameters:
        len_caption (int): Length of the caption in characters. Must be in the range [0, 300).
        n_rows (int): Number of rows in the plot grid. Defaults to 1.
        n_cols (int): Number of columns in the plot grid. Defaults to 1.
        ratio (float): Aspect ratio of the figure (width/height). Defaults to 1.3.
        layout (str): Layout type, either "single" or "double". Defaults to "double".

    Returns:
        Tuple[float, float]: A tuple containing the width and height of the figure in inches.

    Raises:
        ValueError: If len_caption is not in the range [0, 300)
                    or if layout is not "single" or "double".
    """

    if layout == "single":
        width = 88 / 25.4

        if 0 <= len_caption < 50:
            max_height = 220 / 25.4
        elif 50 <= len_caption < 150:
            max_height = 180 / 25.4
        elif 150 <= len_caption < 300:
            max_height = 130 / 25.4
        else:
            raise ValueError("len_caption must be a number in the range [0, 300)")

        height = min(max_height, (width / (ratio * n_cols)) * n_rows)

    elif layout == "double":
        width = 180 / 25.4

        if 0 <= len_caption < 50:
            max_height = 225 / 25.4
        elif 50 <= len_caption < 150:
            max_height = 210 / 25.4
        elif 150 <= len_caption < 300:
            max_height = 185 / 25.4
        else:
            raise ValueError("len_caption must be a number in the range [0, 300)")

        height = min(max_height, (width / (ratio * n_cols)) * n_rows)

    else:
        raise ValueError("layout must be either 'single' or 'double'")

    return (width, height)


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
        "pdf.fonttype": 42,  # Only accepts 3 or 42
        "ps.fonttype": 42,  # Only accepts 3 or 42
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "legend.fontsize": 7,
        "ytick.right": "off",
        "xtick.top": "off",
        "ytick.left": "on",
        "xtick.bottom": "on",
        "xtick.labelsize": "7",
        "ytick.labelsize": "7",
        "lines.linewidth": 1,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "legend.frameon": False,
        "savefig.dpi": 1000,
        "figure.dpi": 1000,
    }

    return rcparams
