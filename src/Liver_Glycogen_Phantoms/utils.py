import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
import matplotlib.colors as mcolors


def load_dictionary(dict_filename):
    """
    Helper function to load a dictionary file.
    
    Args:
        dict_filename (str): The filename of the dictionary to load.
    
    Returns:
        dict: The loaded dictionary if the file exists, otherwise None.
    """
    dict_path = os.path.join(os.getcwd(), 'src', 'Liver_Glycogen_Phantoms', 'dicts', dict_filename)
    if not os.path.isfile(dict_path):
        print(f'Dictionary file {dict_filename} not found')
        return None
    return sio.loadmat(dict_path)

def preprocessing(args):
    """
    Preprocess the data by loading the necessary dictionary and acquired data files.

    Args:
        args: Arguments containing paths to acquired data.

    Returns:
        tuple: Contains loaded dictionaries and data for MT and NOE.
    """
    # Load MT dictionary
    # Dict = load_dictionary('dict.mat')
    # if Dict is None:
    #     return

    # Load acquired data for MT
    full_path_acquired_data = os.path.join(os.getcwd(), 'src', 'Liver_Glycogen_Phantoms', 'acquired_data', args.path_to_acquired_data)
    if not os.path.isfile(full_path_acquired_data):
        print(f'Acquired data file {args.path_to_acquired_data} not found')
        return
    DATA = sio.loadmat(full_path_acquired_data)['data']


    return DATA

def generate_plots(quant_maps):
    """
    Generate plots for the given quantitative maps.

    Args:
        quant_maps (dict): Dictionary containing quantitative maps with keys 'fs', 'ksw'.
    """
    # Flip and process the quantitative maps
    fs = quant_maps['fs']
    ksw = quant_maps['ksw']

    # Create a customized viridis colormap with a black background for minimum values
    original_map = plt.cm.get_cmap('viridis')
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0  # Set the minimum value to black
    b_viridis = mcolors.LinearSegmentedColormap.from_list('b_viridis', color_mat)

    # Set unified font sizes
    unified_font_size = 25
    unified_colorbar_label_size = 15

    # Create a figure for the plots
    fig = plt.figure(figsize=(20, 20))
    plt.subplots_adjust(wspace=0.3)

    # Titles for the subplots
    plots_title = ['Conc. (mM)', 'ksw (1/s)']

    # Helper function to create each subplot
    def create_subplot(ax, data, cmap, clim, title, ylabel, ticks):
        plot = ax.imshow(data, cmap=cmap)
        if clim:
            plot.set_clim(clim)
        ax.set_title(title, fontsize=unified_font_size, loc='left')
        ax.set_ylabel(ylabel, fontsize=unified_colorbar_label_size)
        cb = plt.colorbar(plot, ticks=ticks, orientation='vertical', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=unified_colorbar_label_size)
        # set clim (0,100)
        ax.set_axis_off()

    # Create subplots with precomputed ranges and steps
    fs_range = (0, 340)
    ksw_range = (0, 100)

    fs_ticks = np.arange(0,340 + 20, 20)
    ksw_ticks = np.arange(0,100 + 10,10)

    ax_fs = fig.add_subplot(1, 2, 1)
    create_subplot(ax_fs, fs, b_viridis, fs_range, plots_title[0], '%', fs_ticks)

    ax_ksw = fig.add_subplot(1, 2, 2)
    create_subplot(ax_ksw, ksw, 'magma', ksw_range, plots_title[1], 'Hz', ksw_ticks)
    
    
    # Save the figure to a PDF file
    plt.savefig('Glycogen_phantoms.pdf', format='pdf')
    plt.show()






     





