import os
import scipy.io as sio
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..')
from src.Mice.utils import *
from src.Mice.src_nn.main import NOEMRFNetMT ,NOEMRFNetNOE

def main(args):
     acquired_data_mt, acquired_data_noe = preprocessing(args)
     params = define_model_path(args)

     ######################   MT - 2 pools ############################

     model_mt  = NOEMRFNetMT()
     quant_map_fss_mt, quant_map_kssw_mt = model_mt.test(params['checkpoint_mt'], acquired_data_mt)
     quant_maps_mt = {'fss': quant_map_fss_mt, 'kssw': quant_map_kssw_mt}
     if not os.path.exists(params['quant_maps_path_mt']):
          sio.savemat(params['quant_maps_path_mt'], quant_maps_mt)

     ######################   NOE - 3 pools ############################

     model_noe  = NOEMRFNetNOE()
     quant_map_fs, quant_map_ksw, quant_map_fss, quant_map_kssw  = model_noe.test(params['checkpoint_noe'], quant_maps_mt, acquired_data_noe)
     quant_maps = {'fs': quant_map_fs, 'ksw': quant_map_ksw, 'fss': quant_map_fss, 'kssw': quant_map_kssw}
     if not os.path.exists(params['quant_maps_path_noe']):
          sio.savemat(params['quant_maps_path_noe'], quant_maps)

     generate_plots(quant_maps)

     del model_mt, quant_map_fss_mt, quant_map_kssw_mt, quant_maps_mt
     del model_noe, quant_map_fs, quant_map_ksw, quant_map_fss, quant_map_kssw, quant_maps
     del acquired_data_mt, acquired_data_noe


