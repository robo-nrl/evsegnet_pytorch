import imageio
#import numpy as np
import h5py
import os
#import sys



# f = h5py.File('the_file.h5', 'r')
# dset = f['key']
# data = np.array(dset[:,:,:])
# file = 'test.png' # or .jpg
# imageio.imwrite(file, data)

#@sys.path.insert(0, '/local/a/datasets/mvsec-hdf5')
# data_path = '/local/a/datasets/mvsec-hdf5/indoor_flying1_data.hdf5'
# save_path = '/local/a/datasets/mvsec-png/indoor_flying1_data/'
hf = h5py.File('indoor_flying1_seglabel.hdf5', 'w')
gray_image = hf['davis']['left']['image_raw']
#hf = h5py.File('indoor_flying1_data.hdf5', 'r')
# for key in hf.keys:
#     print(key)
# #print(hf.keys())
# group = hf[key]
# for key in group.keys:
#     print(key)
# idx = 0

for idx in range(len(gray_image)):
    #filename = gray_image[idx].split('/')[-1]
    path=os.path.join(data_path, filename, ".png")
    #file = 'test.png' # or .jpg
    imageio.imwrite(path, gray_image[idx])
    idx += 1
