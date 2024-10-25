from torch.utils.data import Dataset
import numpy as np

class DatasetNOE(Dataset):
    def __init__(self, temp_data, noe_flag=False):
        self.kssw_list = temp_data['ksw_1'].transpose()[:, 0]
        self.fss_list = temp_data['fs_1'].transpose()[:, 0]
        self.len = temp_data['ksw_1'].transpose().size
        self.noe_flag = noe_flag

        if noe_flag:

            self.fs_list = temp_data['fs_0'].transpose()[:, 0]
            self.ksw_list = temp_data['ksw_0'].transpose()[:, 0]
            self.len = temp_data['ksw_0'].transpose().size

        sig = temp_data['sig'].transpose()
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))

        print(f"There are {self.len} entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fss = self.fss_list[index]
        kssw = self.kssw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        if self.noe_flag:
            fs = self.fs_list[index]
            ksw = self.ksw_list[index]
            return fs, ksw, fss, kssw, norm_sig
        else:
            return fss, kssw, norm_sig
        

class Dataset(Dataset):
    def __init__(self, temp_data, noe_flag=False):
        self.kssw_list = temp_data['Kssw'].transpose()[:, 0]
        self.fss_list = temp_data['M0ss'].transpose()[:, 0]
        self.len = temp_data['Kssw'].transpose().size
        self.noe_flag = noe_flag

        if noe_flag:

            self.fs_list = temp_data['M0s'].transpose()[:, 0]
            self.ksw_list = temp_data['Ksw'].transpose()[:, 0]
            self.len = temp_data['Ksw'].transpose().size

        sig = temp_data['waterSignalDict'].transpose()
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))

        print(f"There are {self.len} entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fss = self.fss_list[index]
        kssw = self.kssw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        if self.noe_flag:
            fs = self.fs_list[index]
            ksw = self.ksw_list[index]
            return fs, ksw, fss, kssw, norm_sig
        else:
            return fss, kssw, norm_sig