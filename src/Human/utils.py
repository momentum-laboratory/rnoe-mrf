import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, pearsonr
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as sio
import torch
import sys
import importlib

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
    # dict_mt = load_dict('dict_mt.mat')
    # if dict_mt is None:
    #     return

    # Load acquired data for MT
    full_path_acquired_data_mt = os.path.join(os.getcwd(), 'src', 'Human', 'acquired_data', 'MT', args.path_to_acquired_data_human_mt)
    if not os.path.isfile(full_path_acquired_data_mt):
        print(f'Acquired data file {args.path_to_acquired_data_mt} for MT not found')
        return
    DATA_mt = load_acquired_data_mt(full_path_acquired_data_mt)

    # Load NOE dictionary
    # dict_noe = load_dict_noe('dict_noe.mat')
    # if dict_noe is None:
    #     return

    # Load acquired data for NOE
    full_path_acquired_data_noe = os.path.join(os.getcwd(), 'src', 'Human', 'acquired_data', 'NOE', args.path_to_acquired_data_human_noe)
    if not os.path.isfile(full_path_acquired_data_noe):
        print(f'Acquired data file {args.path_to_acquired_data_noe} for NOE not found')
        return
    
    DATA_noe = load_acquired_data_noe(full_path_acquired_data_noe, args.dict_info)

    return  DATA_mt, DATA_noe


# def load_dict(dict_fn):
#     main_path = os.getcwd()
#     dict_mt_path = os.path.join(main_path, 'src/Human/dicts/', dict_fn)
#     dict_mt = sio.loadmat(dict_mt_path)['dict']

#     # synt_dict = ['T1w','T1s','T1ss', 'T2w', 'T2s' ,'T2ss','M0s', 'M0ss','Ksw','Kssw','B0shift','waterSignalDict']
#     dict_result = {}
#     for name in dict_mt.dtype.names:
#         dict_result[name] = dict_mt[name][0, 0].flatten()
#     dict_result['waterSignalDict'] = dict_result['waterSignalDict'].reshape((30, 531300))

#     dict_result['waterSignalDict'] = dict_result['waterSignalDict'].T  # e.g. 30 x 665,873
#     dict_result['T1w'] = dict_result['T1w'][np.newaxis, :]
#     dict_result['T2w'] = dict_result['T2w'][np.newaxis, :]
#     dict_result['T1s'] = dict_result['T1s'][np.newaxis, :]
#     dict_result['T2s'] = dict_result['T2s'][np.newaxis, :]
#     dict_result['M0s'] = dict_result['M0s'][np.newaxis, :]
#     dict_result['Ksw'] = dict_result['Ksw'][np.newaxis, :]
#     dict_result['M0ss'] = dict_result['M0ss'][np.newaxis, :]
#     dict_result['Kssw'] = dict_result['Kssw'][np.newaxis, :]

#     return dict_result

# def load_dict_noe(dict_fn):
#     main_path = os.getcwd()
#     dict_noe_path = os.path.join(main_path, 'src/Human/dicts/' + dict_fn)
#     dict_noe = sio.loadmat(dict_noe_path)
#     # cut the first params of the dict
#     dict_noe['sig'] = dict_noe['sig'][:,1:]

#     return dict_noe


def load_acquired_data_mt(path_to_aquire_data_mt):
    acquired_data_mt_path = os.path.join(path_to_aquire_data_mt)
    acquired_data_mt = sio.loadmat(acquired_data_mt_path)['mt_mat'][:, :, :, 1:]
    acquired_data_mt = np.nan_to_num(acquired_data_mt)
    return acquired_data_mt

def load_acquired_data_noe(path_to_aquire_data_noe, dict_info):
    acquired_data_noe_path = os.path.join(path_to_aquire_data_noe)
    acquired_data_noe = sio.loadmat(acquired_data_noe_path)[dict_info][:, :, :, 1:]
    acquired_data_noe = np.nan_to_num(acquired_data_noe)
    return acquired_data_noe




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
        checkpoint_dir = os.path.join(base_dir, 'src', 'Human', 'checkpoint')
        quant_maps_dir = os.path.join(base_dir, 'src', 'Human', 'Quant_maps')
        
        # Ensure quant maps directory exists
        os.makedirs(quant_maps_dir, exist_ok=True)

        # Paths for MT
        checkpoint_name_mt = os.path.join(checkpoint_dir, 'checkpoint_mt.pth')
        quant_maps_path_mt = os.path.join(quant_maps_dir, args.name_of_quant_maps_human_mt)
        
        module_name = 'deep_net'
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        dummy_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = dummy_module

        # Define a dummy Network class in the dummy module
        class Network:
            def __init__(self):
                pass

        setattr(dummy_module, 'Network', Network)

        # Load MT tensors
        checkpoint_mt = torch.load(checkpoint_name_mt)

        # Paths for NOE
        checkpoint_name_noe = os.path.join(checkpoint_dir, 'checkpoint_noe.pth')
        quant_maps_path_noe = os.path.join(quant_maps_dir, args.name_of_quant_maps_human_noe)
        
        # Load NOE tensors
        module_name = 'deep_net_noe_combination'
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        dummy_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = dummy_module

        class Network:
            def __init__(self):
                pass

        setattr(dummy_module, 'Network', Network)


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

def plot_and_annotate(ax, data_gray_before, data_white, title, sign):
    # Box plot
    ax.boxplot([data_gray_before, data_white], 
               labels=['Gray Matter', 'White Matter'])
    
    # Set fontsize of the x-axis labels
    ax.set_xticklabels(['Gray Matter', 'White Matter'], fontsize=24)
    
    legend_handles = []

    # Paired t-test and Pearson's r for White Before vs Gray Before
    if len(data_white) == len(data_gray_before):
        t_test_before, p_value_before = ttest_rel(data_white, data_gray_before)
        r_value_before, _ = pearsonr(data_white, data_gray_before)
        legend_handles.append(mpatches.Patch(color='none', 
                                             label='t-test = {:.3f}, p-value = {:.3f}, r-pearson = {:.3f}'.format(
                                                 t_test_before, p_value_before, r_value_before)))

    # Set title, y-label, and legend with double fontsize
    ax.set_title(title, loc='left', fontsize=24)  # Doubling the fontsize from 12 to 24
    ax.set_ylabel(sign, fontsize=24)  # Doubling the fontsize from 12 to 24
    
    # Also double the fontsize for the legend
    ax.legend(handles=legend_handles, loc='upper right', fontsize=12)
    
    # Make sure the y-axis limits are updated to ensure that the larger text does not get cut off
    ylimit = ax.get_ylim()
    ax.set_ylim(ylimit[0], ylimit[1] + (1*ylimit[1])/10)

    # print the mean the std for all the mise and the statistic papms
    print('mean_gray_matter', np.mean(data_gray_before))
    print('std_gray_matter', np.std(data_gray_before))
    print('mean_white_matter', np.mean(data_white))
    print('std_white_matter', np.std(data_white))
    if len(data_white) == len(data_gray_before):
        print('t-test = {:.3f}, p-value = {:.3f}, r-pearson = {:.3f}'.format(t_test_before, p_value_before, r_value_before))



def generate_plots(quant_maps, dot = False):
    fs_roi = quant_maps['fs'] * 100
    ksw_roi = quant_maps['ksw']
    fss_roi = quant_maps['fss'] * 100
    kssw_roi = quant_maps['kssw']

    # Arrange viridis with black background
    original_map = plt.cm.get_cmap('viridis')
    color_mat = original_map(np.arange(original_map.N))
    color_mat[0, 0:3] = 0 
    b_viridis = mcolors.LinearSegmentedColormap.from_list('b_viridis', color_mat)
    unified_font_size = 15
    unified_colorbar_label_size = 10 
    # dict_noe = load_dict_noe('dict_noe.mat')
    # dict_mt = load_dict('dict_mt.mat')

    # step_size_fss = (sorted(np.unique(dict_mt['M0ss'].flatten()))[1] - sorted(np.unique(dict_mt['M0ss'].flatten()))[0])*100
    # fss_down = sorted(np.unique(dict_mt['M0ss'].flatten()))[0] * 100
    # fss_up = sorted(np.unique(dict_mt['M0ss'].flatten()))[-1] * 100

    # step_size_kssw = sorted(np.unique(dict_mt['Kssw'].flatten()))[1] - sorted(np.unique(dict_mt['Kssw'].flatten()))[0]
    # kssw_down = sorted(np.unique(dict_mt['Kssw'].flatten()))[0]
    # kssw_up = sorted(np.unique(dict_mt['Kssw'].flatten()))[-1]

    # step_size_fs = (sorted(np.unique(dict_noe['fs_0'].flatten()))[1] - sorted(np.unique(dict_noe['fs_0'].flatten()))[0])*100
    # fs_down = sorted(np.unique(dict_noe['fs_0'].flatten()))[0] * 100
    # fs_up = sorted(np.unique(dict_noe['fs_0'].flatten()))[-1] * 100

    # step_size_ksw = sorted(np.unique(dict_noe['ksw_0'].flatten()))[1] - sorted(np.unique(dict_noe['ksw_0'].flatten()))[0]
    # ksw_down = sorted(np.unique(dict_noe['ksw_0'].flatten()))[0]
    # ksw_up = sorted(np.unique(dict_noe['ksw_0'].flatten()))[-1]
    
    fig = plt.figure(figsize=(20, 20))
    # Adjust subplot spacing to make more room for labels
    plt.subplots_adjust(wspace=0.3)

    plots_title  = ['A', 'B', 'C', 'D']

    fs_down = 0.8
    fs_up = 1.8
    step_size_fs = 0.1
    ksw_down = 30
    ksw_up = 60
    step_size_ksw = 5
    kssw_down = 5
    kssw_up = 70
    step_size_kssw = 5

    fs_ax = fig.add_subplot(2, 2, 1)
    fs_plot = plt.imshow(np.rot90(fs_roi[70,:,:]), cmap= b_viridis)
    fs_plot.set_clim(fs_down, fs_up + step_size_fs)
    fs_ax.set_title(plots_title[0], fontsize=unified_font_size, loc = 'left')
    fs_ax.set_ylabel('%', fontsize=unified_colorbar_label_size)
    cb = plt.colorbar(ticks=np.arange(fs_down, fs_up + step_size_fs, 2 * step_size_fs), orientation='vertical', fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=unified_colorbar_label_size)
    fs_ax.set_axis_off()

    ksw_ax = fig.add_subplot(2, 2, 2)
    ksw_plot = plt.imshow(np.rot90(ksw_roi[70,:,:]), cmap='magma')
    ksw_plot.set_clim(ksw_down, ksw_up + step_size_ksw)
    ksw_ax.set_title(plots_title[1], fontsize=unified_font_size, loc = 'left')
    ksw_ax.set_ylabel('Hz', fontsize=unified_colorbar_label_size)
    cb = plt.colorbar(ticks=np.arange(ksw_down, ksw_up + step_size_ksw, 2 * step_size_ksw), orientation='vertical', fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=unified_colorbar_label_size)
    ksw_ax.set_axis_off()

    fss_ax = fig.add_subplot(2, 2, 3)
    fss_plot = plt.imshow(fss_roi[:,70,:], cmap= b_viridis)
    fss_plot.set_clim(0, 15 + 1)
    fss_ax.set_title(plots_title[2], fontsize=unified_font_size, loc = 'left')
    fss_ax.set_ylabel('%', fontsize=unified_colorbar_label_size)
    cb = plt.colorbar(ticks=np.arange(0 , 15 + 1), orientation='vertical', fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=unified_colorbar_label_size)
    fss_ax.set_axis_off()

    kssw_ax = fig.add_subplot(2, 2, 4)
    kssw_plot = plt.imshow(kssw_roi[:,70,:], cmap='magma')
    kssw_plot.set_clim(kssw_down, kssw_up + step_size_kssw)
    kssw_ax.set_title(plots_title[3], fontsize=unified_font_size, loc = 'left')
    kssw_ax.set_ylabel('Hz', fontsize=unified_colorbar_label_size)
    cb = plt.colorbar(ticks=np.arange(kssw_down, kssw_up + step_size_kssw, 2 * step_size_kssw), orientation='vertical', fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=unified_colorbar_label_size)
    kssw_ax.set_axis_off()

    plt.show()

# Function to get significance symbol
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

# Function to plot and annotate
def plot_and_annotate(ax, data_gray, data_white, title, sign, ylim=None):
    data_gray = np.ravel(data_gray)
    data_white = np.ravel(data_white)
    
    box = ax.boxplot([data_gray, data_white], labels=['Gray Matter', 'White Matter'], patch_artist=True, showfliers=False)
    ax.set_xticklabels(['Gray Matter', 'White Matter'], fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_title(title, loc='left', fontsize=22, fontweight='bold')
    ax.set_ylabel(sign, fontsize=22)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if len(data_white) == len(data_gray):
        t_test_before, p_value_before = ttest_rel(data_white, data_gray)
        significance_symbol = get_significance_symbol(p_value_before)
        if title == 'A':
            x1, x2 = 1, 2
            y, h, col = max(np.mean(data_gray), np.mean(data_white)) + 0.5, 0.000005, 'k'
        elif title == 'C':
            x1, x2 = 1, 2
            y, h, col = max(np.mean(data_gray), np.mean(data_white)) + 3, 0.000005, 'k'
        elif title == 'B':
            x1, x2 = 1, 2
            y, h, col = max(np.mean(data_gray), np.mean(data_white)) + 6, 0.000005, 'k'
        else:
            x1, x2 = 1, 2
            y, h, col = max(np.mean(data_gray), np.mean(data_white)) + 6, 0.000005, 'k'
        ax.text((x1+x2)*.5, y+h, significance_symbol, ha='center', va='bottom', color=col, fontsize=12)
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c=col)
    else:
        if len(data_white) > len(data_gray):
            data_white = data_white[:len(data_gray)]
        else:
            data_gray = data_gray[:len(data_white)]
        t_test_before, p_value_before = ttest_rel(data_white, data_gray)
        significance_symbol = get_significance_symbol(p_value_before)

    for patch, color in zip(box['boxes'], ['lightgray', 'lightblue']):
        patch.set_facecolor(color)

# Function to plot main plot
def main_plot(mean_fs_gray, mean_fs_white, mean_fss_gray, mean_fss_white, mean_ksw_gray, mean_ksw_white, mean_kssw_gray, mean_kssw_white):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))  
    ylim_fs = (0.4, 2.1)
    ylim_ksw = (0, 60)
    ylim_fss = (0, 23)
    ylim_kssw = (0, 55)

    plot_and_annotate(axs[0,0], mean_fs_gray, mean_fs_white, 'A', r'rNOE $f_{s}$ (%)', ylim=ylim_fs)
    plot_and_annotate(axs[0,1], mean_ksw_gray, mean_ksw_white, 'B', r'rNOE $k_{sw}$ (s$^{-1}$)', ylim=ylim_ksw)
    plot_and_annotate(axs[1,0], mean_fss_gray, mean_fss_white, 'C', r'MT $f_{ss}$ (%)', ylim=ylim_fss)
    plot_and_annotate(axs[1,1], mean_kssw_gray, mean_kssw_white, 'D', r'MT $k_{ssw}$ (s$^{-1}$)', ylim=ylim_kssw)
    
    plt.tight_layout()  # Adjust the layout to prevent overlap
    plt.show()


def extract_data(data):
    # Initialize lists to store the means
    mean_fs_gray = []
    mean_fs_white = []
    mean_fss_gray = []
    mean_fss_white = []
    mean_ksw_gray = []
    mean_ksw_white = []
    mean_kssw_gray = []
    mean_kssw_white = []

    # Define the prefixes and the list of lists
    prefixes = ['mean_fs_gray', 'mean_fs_white', 'mean_fss_gray', 'mean_fss_white',
                'mean_ksw_gray', 'mean_ksw_white', 'mean_kssw_gray', 'mean_kssw_white']
    mean_lists = [mean_fs_gray, mean_fs_white, mean_fss_gray, mean_fss_white,
                  mean_ksw_gray, mean_ksw_white, mean_kssw_gray, mean_kssw_white]

    # Loop over each prefix and extract means
    for prefix, mean_list in zip(prefixes, mean_lists):
        for i in range(1, 6):
            mean_value = np.nanmean(data[f'{prefix}_{i}'].values)
            mean_list.append(mean_value)
    
    return mean_fs_gray, mean_fs_white, mean_fss_gray, mean_fss_white, mean_ksw_gray, mean_ksw_white, mean_kssw_gray, mean_kssw_white