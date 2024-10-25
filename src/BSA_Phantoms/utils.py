import numpy as np
import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import re

def generate_param_for_main(args):

    main = os.getcwd()
    dict_fn = os.path.join(main, 'src/BSA_Phantoms/Additional_Data', 'dict_bsa.mat')
    Reco_Net_name = os.path.join(main, 'src/BSA_Phantoms/Checkpoints', 'checkpoints.pth')
    checkpoint = torch.load(Reco_Net_name)
    acquired_data_name = os.path.join(main, 'src/BSA_Phantoms/Additional_Data', args.path_to_acquired_data_bsa)
    acquired_data = sio.loadmat(acquired_data_name)['acquired_data']

    return acquired_data, checkpoint

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
    plots_title = ['fs (%)', 'ksw (1/s)']

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
    fs_range = (0, 22)
    ksw_range = (0, 40)

    fs_ticks = np.arange(0, 22 + 2, 2)
    ksw_ticks = np.arange(0, 40 + 5, 5)

    ax_fs = fig.add_subplot(1, 2, 1)
    create_subplot(ax_fs, fs, b_viridis, fs_range, plots_title[0], '%', fs_ticks)

    ax_ksw = fig.add_subplot(1, 2, 2)
    create_subplot(ax_ksw, ksw, 'magma', ksw_range, plots_title[1], 'Hz', ksw_ticks)
    
    
    # Save the figure to a PDF file
    plt.savefig('bsa_phantoms.pdf', format='pdf')
    plt.show()
    



