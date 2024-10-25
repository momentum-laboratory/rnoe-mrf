import os
import scipy.io as sio

from src.BSA_Phantoms.utils import *
from src.BSA_Phantoms.src_nn.main import NOEMRFReconstruction


def main(args):
    acquired_data, checkpoint = generate_param_for_main(args)
    model_phantoms = NOEMRFReconstruction()
    checkpoint = torch.load(os.path.join(os.getcwd() ,'src/BSA_Phantoms/Checkpoints/checkpoints.pth'))
    quant_map_fs, quant_map_ksw = model_phantoms.test(acquired_data, checkpoint)
    quant_maps = {'fs': quant_map_fs * 1100 / 1, 'ksw': quant_map_ksw}
    if not os.path.exists(os.path.join(os.getcwd(),'src/BSA_Phantoms/quant_maps')):
        os.makedirs(os.path.join(os.getcwd(),'src/BSA_Phantoms/quant_maps'))
    sio.savemat(os.path.join(os.getcwd(),'src/BSA_Phantoms/quant_maps', args.name_of_quant_maps_bsa), quant_maps)
    generate_plots(quant_maps)
