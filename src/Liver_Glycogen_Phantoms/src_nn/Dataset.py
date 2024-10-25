import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self,training_data):

        self.fs_list = training_data['fss_0'].transpose()[:, 0]
        self.ksw_list = training_data['ksw_0'].transpose()[:, 0]
        sig = training_data['sig'].transpose()
        self.norm_sig_list = sig / np.sqrt(np.sum(sig ** 2, axis=0))
        self.len = training_data['ksw_0'].transpose().size
        print("There are " + str(self.len) + " entries in the training dictionary")

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        fs = self.fs_list[index]
        ksw = self.ksw_list[index]
        norm_sig = self.norm_sig_list[:, index]
        return fs, ksw, norm_sig
