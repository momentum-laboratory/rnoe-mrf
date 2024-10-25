import numpy as np
import os
import torch
from src.Liver_Glycogen_Phantoms.utils import *
from src.Liver_Glycogen_Phantoms.src_nn.deep_reco_py import NOEMRFReconstruction



def main(args):

    acquired_data = preprocessing(args)
    
    model_phantoms = NOEMRFReconstruction()
    checkpoint = torch.load(os.path.join(os.getcwd() ,'src/Liver_Glycogen_Phantoms/checkpoint/checkpoint.pth'))
    quant_map_fs, quant_map_ksw = model_phantoms.test(acquired_data, checkpoint)
    quant_maps = {'fs': quant_map_fs * 110000 / 1, 'ksw': quant_map_ksw}
    if not os.path.exists(os.path.join(os.getcwd(),'src/Liver_Glycogen_Phantoms/quant_maps')):
        os.makedirs(os.path.join(os.getcwd(),'src/Liver_Glycogen_Phantoms/quant_maps'))
    sio.savemat(os.path.join(os.getcwd(),'src/Liver_Glycogen_Phantoms/quant_maps', args.name_of_quant_maps), quant_maps)

    generate_plots(quant_maps)
