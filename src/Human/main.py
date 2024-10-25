import os
import scipy.io as sio
import pandas as pd

from src.Human.src_nn.main import CestMRFNetMT, CestMRFNetNOE
from src.Human.utils import *



def main(args):

     acquired_data_mt, acquired_data_noe = preprocessing(args)
     params = define_model_path(args)

     ######################   MT - 2 pools ############################

     model_mt  = CestMRFNetMT()
     quant_map_fss_mt, quant_map_kssw_mt = model_mt.test(params['checkpoint_mt'], acquired_data_mt)
     quant_maps_mt = {'fss': quant_map_fss_mt, 'kssw': quant_map_kssw_mt}
     if not os.path.exists(params['quant_maps_path_mt']):
          sio.savemat(params['quant_maps_path_mt'], quant_maps_mt)

     ######################   NOE - 3 pools ############################

     model_noe  = CestMRFNetNOE()
     quant_map_fs, quant_map_ksw, quant_map_fss, quant_map_kssw  = model_noe.test(params['checkpoint_noe'], quant_maps_mt, acquired_data_noe)
     quant_maps = {'fs': quant_map_fs, 'ksw': quant_map_ksw, 'fss': quant_map_fss, 'kssw': quant_map_kssw}
     if not os.path.exists(params['quant_maps_path_noe']):
          sio.savemat(params['quant_maps_path_noe'], quant_maps)

     generate_plots(quant_maps)

     if not args.paper_example:
               
          # Load the data from Human.txt
          data_file_path = os.path.join(os.getcwd(), 'src/Human/additional_data/Human.xlsx')
          data = pd.read_excel(data_file_path)
          # Extract the data
          mean_fs_gray, mean_fs_white, mean_fss_gray, mean_fss_white, mean_ksw_gray, mean_ksw_white, mean_kssw_gray, mean_kssw_white = extract_data(data)


          # Plot the data
          main_plot(mean_fs_gray, mean_fs_white, mean_fss_gray, mean_fss_white, mean_ksw_gray, mean_ksw_white, mean_kssw_gray, mean_kssw_white)

     del dict_mt, model_mt, quant_map_fss_mt, quant_map_kssw_mt, quant_maps_mt
     del dict_noe, model_noe, quant_map_fs, quant_map_ksw, quant_map_fss, quant_map_kssw, quant_maps
     del acquired_data_mt, acquired_data_noe




