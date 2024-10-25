import torch
import numpy as np
import time
from src.BSA_Phantoms.src_nn.network import Network 
from src.BSA_Phantoms.src_nn.utilities import *
from torch.autograd import Variable


class NOEMRFReconstruction:
    def __init__(self):
        self.device = check_cuda()
        self.dtype = torch.DoubleTensor
        self.prepare_data()
        self.model = None

    def prepare_data(self):
        # Prepare the min and max parameters for normalization
        min_fs = 0.00045454545454545455
        min_ksw = 10.0
        max_fs = 0.022727272727272728
        max_ksw = 50.0

        self.min_param_tensor = torch.tensor(np.hstack((min_fs, min_ksw)), requires_grad=False).type(self.dtype)
        self.max_param_tensor = torch.tensor(np.hstack((max_fs, max_ksw)), requires_grad=False).type(self.dtype)

    def initialize_model(self):
        self.model = Network().to(self.device)
    
    def test(self, DATA, reco_net):
            sched_iter = 30
            device = check_cuda()
            model = Network().to(device)
            model.load_state_dict(reco_net)
            acquired_data = DATA.astype(float)
            dtype = torch.DoubleTensor
            [_, c_acq_data, w_acq_data] = np.shape(acquired_data)
            acquired_data = np.reshape(acquired_data, (sched_iter, c_acq_data * w_acq_data), order='F')
            acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))
            acquired_data = acquired_data.T
            acquired_data = Variable(torch.from_numpy(acquired_data).type(dtype), requires_grad=False).to(device)
            t0 = time.time()
            prediction = model(acquired_data)

            RunTime = time.time() - t0
            print("")
            if RunTime < 60:  # if less than a minute
                print('Prediction time: ' + str(RunTime) + ' sec')
            elif RunTime < 3600:  # if less than an hour
                print('Prediction time: ' + str(RunTime / 60.0) + ' min')
            else:  # If took more than an hour
                print('Prediction time: ' + str(RunTime / 3600.0), ' hour')

            prediction = un_normalize_range(prediction, original_min=self.min_param_tensor.to(device),
                                        original_max=self.max_param_tensor.to(device), new_min=0, new_max=1)
            quant_map_fs = prediction.cpu().detach().numpy()[:, 0]
            quant_map_fs = quant_map_fs.T
            quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

            quant_map_ksw = prediction.cpu().detach().numpy()[:, 1]
            quant_map_ksw = quant_map_ksw.T
            quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

            return quant_map_fs, quant_map_ksw




