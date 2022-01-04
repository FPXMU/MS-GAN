import numpy as np
import h5py
import tifffile



f = h5py.File("/home1/Usr/xiapengfei/ProGAN/test/fake_samples_0.hdf5", 'r')
my_array = f['data'][()]
img = my_array[0, 0, :, :, :].astype(np.float32)
tifffile.imsave("./berea_7.tiff",  img)
print(img.shape)