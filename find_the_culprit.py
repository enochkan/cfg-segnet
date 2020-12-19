import glob
import h5py

samples = [x for x in glob.glob("./data/leftkidney_3d/train" + '/*.im')]
for sample in samples:
    image = h5py.File(sample, 'r').get('data')[()]
    # print(image.shape)
    if image.shape[0] != 128:
        print(sample)
