import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import time

from src.Mice.utils import normalize_range, un_normalize_range, check_cuda
from src.Mice.src_nn.Dataset import Dataset
from src.Mice.src_nn.Network import Network

class NOEMRFNetMT:
    def __init__(self):
        self.device = check_cuda()
        self.model = Network(sched_iter=30).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.min_param_tensor_mt, self.max_param_tensor_mt = self.get_min_max_params()
        # self.train_loader = DataLoader(Dataset(temp_data_mt), batch_size=256, shuffle=False, num_workers=3)
        self.loss_per_epoch = []
        self.no_improvement_count = 0
        self.patience = 5
        self.look_back = 4
        self.min_epochs = 10
        self.min_improvement_train = 0.035
        self.min_improvement_eval = 0.0007

    def get_min_max_params(self):
        min_fs = 0.01818181818181818
        min_ksw = 5
        max_fs = 0.2727272727272727
        max_ksw = 100
        min_params = torch.tensor([min_fs, min_ksw], dtype=torch.float64, requires_grad=False)
        max_params = torch.tensor([max_fs, max_ksw], dtype=torch.float64, requires_grad=False)
        return min_params.to(self.device), max_params.to(self.device)
        
    def test(self,reco_net_mt, acquired_data):
        self.model.load_state_dict(reco_net_mt)
        dtype = torch.DoubleTensor
        device = check_cuda()
        sched_iter = 30   

        acquired_data = acquired_data['DATA'].astype(np.float64)
        [_, c_acq_data, w_acq_data] = np.shape(acquired_data)
        acquired_data = np.reshape(acquired_data, (sched_iter, c_acq_data * w_acq_data), order='F')
        acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))
        acquired_data = acquired_data.T
        acquired_data = Variable(torch.from_numpy(acquired_data).type(dtype), requires_grad=False).to(device)
        t0 = time.time()
        prediction = self.model(acquired_data)
        RunTime = time.time() - t0
        print("")
        if RunTime < 60:  # if less than a minute
            print('Prediction time: ' + str(RunTime) + ' sec')
        elif RunTime < 3600:  # if less than an hour
            print('Prediction time: ' + str(RunTime / 60.0) + ' min')
        else:  # If took more than an hour
            print('Prediction time: ' + str(RunTime / 3600.0), ' hour')
        prediction = un_normalize_range(prediction, original_min=self.min_param_tensor_mt.to(device),
                                        original_max=self.max_param_tensor_mt.to(device), new_min=0, new_max=1)
        quant_map_fss = prediction.detach().cpu().numpy()[:, 0]
        quant_map_fss = quant_map_fss.T
        quant_map_fss = np.reshape(quant_map_fss, (c_acq_data, w_acq_data), order='F')

        quant_map_kssw = prediction.detach().cpu().numpy()[:, 1]
        quant_map_kssw = quant_map_kssw.T
        quant_map_kssw = np.reshape(quant_map_kssw, (c_acq_data, w_acq_data), order='F')


        return quant_map_fss, quant_map_kssw

    def process_prediction(self, prediction):
        prediction = un_normalize_range(prediction, self.min_param_tensor_mt, self.max_param_tensor_mt, 0, 1)
        quant_map_fss = prediction[:, 0].cpu().numpy().reshape((self.c_acq_data, self.w_acq_data))
        quant_map_kssw = prediction[:, 1].cpu().numpy().reshape((self.c_acq_data, self.w_acq_data))
        return quant_map_fss, quant_map_kssw


class NOEMRFNetNOE:
    def __init__(self):
        self.device = check_cuda()
        self.model = Network().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.min_param_tensor_mt, self.max_param_tensor_mt, self.min_param_tensor_noe, self.max_param_tensor_noe = self.get_min_max_params()
        # self.train_loader = DataLoader(Dataset(temp_data_noe), batch_size = 256, shuffle=False, num_workers = 3)
        self.loss_per_epoch = []
        self.no_improvement_count = 0
        self.patience = 5
        self.look_back = 4
        self.min_epochs = 10
        self.min_improvement_train = 0.035
        self.min_improvement_eval = 0.0007

    def get_min_max_params(self):

        min_fss = 0.0009090909090909091
        min_kssw = 5.0
        max_fss = 0.2727272727272727
        max_kssw = 100.0

        min_fs = 0.0009090909090909091
        min_ksw = 5.0
        max_fs = 0.01818181818181818
        max_ksw = 100.0

        min_params_mt = torch.tensor([min_fss, min_kssw], dtype=torch.float64, requires_grad=False)
        max_params_mt = torch.tensor([max_fss, max_kssw], dtype=torch.float64, requires_grad=False)

        min_params = torch.tensor([min_fs, min_ksw], dtype=torch.float64, requires_grad=False)
        max_params = torch.tensor([max_fs, max_ksw], dtype=torch.float64, requires_grad=False)

        return min_params_mt.to(self.device),max_params_mt.to(self.device), min_params.to(self.device), max_params.to(self.device)

    def test(self,reco_net, quant_maps_mt, acquired_data):
        self.model.load_state_dict(reco_net)
        dtype = torch.DoubleTensor
        device = check_cuda()
        sched_iter = 30   
        acquired_data = acquired_data['DATA'].astype(np.float64)
        [_, c_acq_data, w_acq_data] = np.shape(acquired_data)
        acquired_data = np.reshape(acquired_data, (sched_iter, c_acq_data * w_acq_data), order='F')
        acquired_data = acquired_data / np.sqrt(np.sum(acquired_data ** 2, axis=0))
        MT_fs = quant_maps_mt['fss']
        MT_ksw = quant_maps_mt['kssw']
        MT_fs = np.reshape(MT_fs, (1, c_acq_data * w_acq_data), order='F')
        MT_fs_tensor = torch.tensor(MT_fs, requires_grad=False).type(dtype).to(device)
        min_param_tensor_fss = self.min_param_tensor_mt[0]
        max_param_tensor_fss = self.max_param_tensor_mt[0]
        MT_ksw = np.reshape(MT_ksw, (1, c_acq_data * w_acq_data), order='F')
        # to tensor
        MT_ksw_tensor = torch.tensor(MT_ksw, requires_grad=False).type(dtype).to(device)
        min_param_tensor_kssw = self.min_param_tensor_mt[1]
        max_param_tensor_kssw = self.max_param_tensor_mt[1]
        MT_fs_norm = normalize_range(original_array=MT_fs_tensor, original_min=min_param_tensor_fss,
                                    original_max=max_param_tensor_fss, new_min=0, new_max=1)
        
        MT_ksw_norm = normalize_range(original_array=MT_ksw_tensor , original_min=min_param_tensor_kssw,
                                    original_max=max_param_tensor_kssw, new_min=0, new_max=1)
        MT_params = np.stack((MT_fs_norm.cpu(), MT_ksw_norm.cpu()), axis=0).squeeze()
        acquired_data = np.concatenate((acquired_data, MT_params), axis=0)
        acquired_data = acquired_data.T
        acquired_data = Variable(torch.from_numpy(acquired_data).type(dtype), requires_grad=False).to(device)
        t0 = time.time()
        prediction = self.model(acquired_data)
        RunTime = time.time() - t0
        print("")
        if RunTime < 60:  # if less than a minute
            print('Prediction time: ' + str(RunTime) + ' sec')
        elif RunTime < 3600:  # if less than an hour
            print('Prediction time: ' + str(RunTime / 60.0) + ' min')
        else:  # If took more than an hour
            print('Prediction time: ' + str(RunTime / 3600.0), ' hour')
        prediction = un_normalize_range(prediction, original_min=self.min_param_tensor_noe.to(device),
                                        original_max=self.max_param_tensor_noe.to(device), new_min=0, new_max=1)
        quant_map_fs = prediction.detach().cpu().numpy()[:, 0]
        quant_map_fs = quant_map_fs.T
        quant_map_fs = np.reshape(quant_map_fs, (c_acq_data, w_acq_data), order='F')

        quant_map_ksw = prediction.detach().cpu().numpy()[:, 1]
        quant_map_ksw = quant_map_ksw.T
        quant_map_ksw = np.reshape(quant_map_ksw, (c_acq_data, w_acq_data), order='F')

        return quant_map_fs, quant_map_ksw, quant_maps_mt['fss'] ,quant_maps_mt['kssw'] 


        
