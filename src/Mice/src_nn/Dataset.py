from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, temp_data):
        self.kssw_list = temp_data['ksw_1'].transpose()[:, 0]
        self.fss_list = temp_data['fss_1'].transpose()[:, 0]
        self.len = temp_data['ksw_1'].transpose().size

        if 'fss_0' in temp_data.keys():

            self.fs_list = temp_data['fss_0'].transpose()[:, 0]
            self.ksw_list = temp_data['ksw_0'].transpose()[:, 0]
            self.len = temp_data['ksw_0'].transpose().size

        sig = temp_data['sig'].transpose()
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))

        print(f"There are {self.len} entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if not self.fs_list is None:
            fs = self.fs_list[index]
            ksw = self.ksw_list[index]
        fss = self.fss_list[index]
        kssw = self.kssw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        return fs, ksw, fss, kssw, norm_sig