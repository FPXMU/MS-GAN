import torch.utils.data as data
from torch import Tensor
from os import listdir
from os.path import join
import numpy as np
import h5py
import torchvision.transforms as transforms
import torch.nn.functional as F
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".hdf5", ".h5"])


def load_img(filepath, stage, max_stage):
    img = None
    with h5py.File(filepath, "r") as f:
        img = f['data'][()]
        print('h5py',img.shape)
    img = np.expand_dims(img, axis=0)
    torch_img = Tensor(img)
    torch_img = torch_img.div(255).sub(0.5).div(0.5)
    #torch_img = torch_img.sub(0.5).div(0.5)
    # print(torch_img)
    return torch_img

class HDF5Dataset(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None,
                 stage=None, max_stage=None):
        super(HDF5Dataset, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.stage = stage
        self.max_stage = max_stage

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index], self.stage, self.max_stage)
        #print('HDF5---》', input.size)
        target = None

        return input

    def __len__(self):
        return len(self.image_filenames)
