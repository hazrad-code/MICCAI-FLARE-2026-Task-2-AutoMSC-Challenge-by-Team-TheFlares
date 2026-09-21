import torch.multiprocessing as mp

from nnunetv2.training.nnUNetTrainer.nnUNetCLSTrainer import nnUNetCLSTrainerMTL

# /dev/shm is only 64 MB in this container; share tensors via the filesystem instead
mp.set_sharing_strategy("file_system")


class nnUNetCLSTrainerMTL_100epochs(nnUNetCLSTrainerMTL):
    def initialize(self):
        self.num_epochs = 100
        super().initialize()


class nnUNetCLSTrainerMTL_250epochs(nnUNetCLSTrainerMTL):
    def initialize(self):
        self.num_epochs = 250
        super().initialize()


class nnUNetCLSTrainerMTL_500epochs(nnUNetCLSTrainerMTL):
    def initialize(self):
        self.num_epochs = 500
        super().initialize()