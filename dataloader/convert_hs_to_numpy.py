import os
from ios import envi
import numpy as np
import tqdm 


def convert_hs_dataset_to_numpy(dataset_path, output_path):
    '''
    Convert hyperspectral dataset to numpy format
    '''
    # Load the dataset
    hs_files = list(filter(lambda x: x.endswith('.hdr'), os.listdir(dataset_path)))
    for hs_file in tqdm.tqdm(hs_files):
        hdr_name = os.path.join(dataset_path, hs_file)
        bin_name = hdr_name.replace('.hdr', '.bin')
        _header, _data = envi.load_envi(hdr_name, bin_name)
        _data = np.array(_data)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(os.path.join(output_path, hs_file + '.npy'), _data)

if __name__=='__main__':
    folders = os.listdir('dataset/Debris/CORNING_HSI')
    for folder in folders:
        convert_hs_dataset_to_numpy('dataset/Debris/CORNING_HSI/' + folder, 'dataset/Debris_numpy/CORNING_HSI/' + folder)
