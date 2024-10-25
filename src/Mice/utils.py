import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import scipy.io as sio
import torch

def check_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU found and will be used")
    else:
        print("GPU was not found. Using CPU")
    return device

def normalize_range(original_array, original_min, original_max, new_min, new_max):
    a, b, c, d = original_min, original_max, new_min, new_max
    return (original_array - a) / (b - a) * (d - c) + c

def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    a, b, c, d = original_min, original_max, new_min, new_max
    return (normalized_array - c) / (d - c) * (b - a) + a

def load_dictionary(dict_filename):
    """
    Helper function to load a dictionary file.
    
    Args:
        dict_filename (str): The filename of the dictionary to load.
    
    Returns:
        dict: The loaded dictionary if the file exists, otherwise None.
    """
    dict_path = os.path.join(os.getcwd(), 'src', 'Mice', 'dicts', dict_filename)
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
    # dict_mt = load_dictionary('dict_mt.mat')
    # if dict_mt is None:
    #     return

    # Load acquired data for MT
    full_path_acquired_data_mt = os.path.join(os.getcwd(), 'src', 'Mice', 'acquired_data', 'MT', args.path_to_acquired_data_mt)
    if not os.path.isfile(full_path_acquired_data_mt):
        print(f'Acquired data file {args.path_to_acquired_data_mt} for MT not found')
        return
    DATA_mt = sio.loadmat(full_path_acquired_data_mt)

    # # Load NOE dictionary
    # dict_noe = load_dictionary('dict_noe.mat')
    # if dict_noe is None:
    #     return

    # Load acquired data for NOE
    full_path_acquired_data_noe = os.path.join(os.getcwd(), 'src', 'Mice', 'acquired_data', 'NOE', args.path_to_acquired_data_noe)
    if not os.path.isfile(full_path_acquired_data_noe):
        print(f'Acquired data file {args.path_to_acquired_data_noe} for NOE not found')
        return
    DATA_noe = sio.loadmat(full_path_acquired_data_noe)

    return DATA_mt, DATA_noe

def define_model_path(args):
    """
    Define paths for model checkpoints and parameter tensors, then load them.

    Args:
        args: Arguments containing paths and names for the required files.

    Returns:
        dict: Contains the loaded checkpoint and parameter tensors for both MT and NOE,
              as well as paths to quant maps.
    """
    try:
        # Base paths
        base_dir = os.getcwd()
        checkpoint_dir = os.path.join(base_dir, 'src', 'Mice', 'checkpoint')
        param_minmax_dir = os.path.join(base_dir, 'src', 'Mice', 'param_minmax')
        quant_maps_dir = os.path.join(base_dir, 'src', 'Mice', 'Quant_maps')
        
        # Ensure quant maps directory exists
        os.makedirs(quant_maps_dir, exist_ok=True)

        # Paths for MT
        checkpoint_name_mt = os.path.join(checkpoint_dir, 'checkpoint_mt.pth')
        quant_maps_path_mt = os.path.join(quant_maps_dir, args.name_of_quant_maps_mt)
        
        # Load MT tensors
        checkpoint_mt = torch.load(checkpoint_name_mt)

        # Paths for NOE
        checkpoint_name_noe = os.path.join(checkpoint_dir, 'checkpoint_noe.pth')
        quant_maps_path_noe = os.path.join(quant_maps_dir, args.name_of_quant_maps_noe)
        
        # Load NOE tensors
        checkpoint_noe = torch.load(checkpoint_name_noe)
        # Define the outputs as a dictionary
        param = {
            'checkpoint_mt': checkpoint_mt,
            'quant_maps_path_mt': quant_maps_path_mt,
            'checkpoint_noe': checkpoint_noe,
            'quant_maps_path_noe': quant_maps_path_noe
        }

        return param
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    



def generate_plots(quant_maps):
    """
    Generate plots for the given quantitative maps.

    Args:
        quant_maps (dict): Dictionary containing quantitative maps with keys 'fs', 'ksw', 'kssw', and 'fss'.
    """
    # Flip and process the quantitative maps
    fs = np.flip(quant_maps['fs'] * 100)
    ksw = np.flip(quant_maps['ksw'])
    kssw = np.flip(quant_maps['kssw'])
    fss = np.flip(quant_maps['fss'] * 100)

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
    plots_title = ['A - rnoe fs (%)', 'B - rnoe ksw (1/s)', 'C - mt fss (%)', 'D - mt kssw (1/s)']

    # Helper function to create each subplot
    def create_subplot(ax, data, cmap, clim, title, ylabel, ticks):
        plot = ax.imshow(data, cmap=cmap)
        if clim:
            plot.set_clim(clim)
        ax.set_title(title, fontsize=unified_font_size, loc='left')
        ax.set_ylabel(ylabel, fontsize=unified_colorbar_label_size)
        cb = plt.colorbar(plot, ticks=ticks, orientation='vertical', fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=unified_colorbar_label_size)
        ax.set_axis_off()

    # Create subplots with precomputed ranges and steps
    fs_range = (0.8 , 1.8)
    ksw_range = (0 , 100)
    fss_range = (0 , 25)
    kssw_range = (0 , 60)

    fs_ticks = np.arange(0.8, 1.8 + 0.2 , 0.2)
    ksw_ticks = np.arange(0, 100 + 20, 20)
    fss_ticks = np.arange(0, 25 + 5, 5)
    kssw_ticks = np.arange(0, 60 + 10, 10)

    ax_fs = fig.add_subplot(1, 4, 1)
    create_subplot(ax_fs, fs, b_viridis, fs_range, plots_title[0], '%', fs_ticks)

    ax_ksw = fig.add_subplot(1, 4, 2)
    create_subplot(ax_ksw, ksw, 'magma', ksw_range, plots_title[1], 'Hz', ksw_ticks)

    ax_fss = fig.add_subplot(1, 4, 3)
    create_subplot(ax_fss, fss, b_viridis, fss_range, plots_title[2], '%', fss_ticks)

    ax_kssw = fig.add_subplot(1, 4, 4)
    create_subplot(ax_kssw, kssw, 'magma', kssw_range, plots_title[3], 'Hz', kssw_ticks)

    # Save the plot as a PDF file
    plt.savefig('mice.pdf', format='pdf')

    plt.show()

def get_significance_symbol(p_value):
    if p_value > 0.05:
        return 'ns'
    elif p_value <= 0.05 and p_value > 0.01:
        return '*'
    elif p_value <= 0.01 and p_value > 0.001:
        return '**'
    elif p_value <= 0.001 and p_value > 0.0001:
        return '***'
    else:
        return '****'






