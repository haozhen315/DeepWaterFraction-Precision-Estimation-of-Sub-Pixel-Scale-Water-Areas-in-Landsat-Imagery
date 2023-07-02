import cv2
import numpy as np
import torch


def prepare_input_data(blue, green, red, nir, swir1, swir2):
    ''' 
    Prepare input data for the model 
    '''
    sample_ = {}

    sample_['blue'] = torch.from_numpy(np.array(blue)).type(torch.float32)
    sample_['green'] = torch.from_numpy(np.array(green)).type(torch.float32)
    sample_['red'] = torch.from_numpy(np.array(red)).type(torch.float32)
    sample_['nir'] = torch.from_numpy(np.array(nir)).type(torch.float32)
    sample_['swir1'] = torch.from_numpy(np.array(swir1)).type(torch.float32)
    sample_['swir2'] = torch.from_numpy(np.array(swir2)).type(torch.float32)

    sample_['ndvi'] = (sample_['nir'] - sample_['red']) / (sample_['nir'] + sample_['red'] + 1e-10)
    sample_['ndwi'] = (sample_['green'] - sample_['nir']) / (sample_['green'] + sample_['nir'] + 1e-10)
    sample_['mndwi'] = (sample_['green'] - sample_['swir2']) / (sample_['green'] + sample_['swir2'] + 1e-10)
    sample_['lswi'] = (sample_['nir'] - sample_['swir2']) / (sample_['nir'] + sample_['swir2'] + 1e-10)

    sample_['valid_mask'] = (blue != 0) | (green != 0) | (red != 0) | (nir != 0) | (swir1 != 0) | (swir2 != 0)
    sample_['valid_mask'] = torch.from_numpy(sample_['valid_mask']).type(torch.float32)

    # unexpand dim
    for key in sample_:
        if type(sample_[key]) == torch.Tensor:
            sample_[key] = torch.unsqueeze(torch.unsqueeze(sample_[key], 0), 0)

    # initialize sample
    sample = {}
    for key in sample_:
        sample[key] = sample_[key].to('cuda')

    # normalize
    for key in sample:
        if key in ['ndvi', 'ndwi', 'mndwi', 'lswi', 'valid_mask']:
            continue
        sample[key] = (sample[key] - sample[key].min()) / (sample[key].max() - sample[key].min() + 1e-10)

    return sample


def get_cloud_mask_landsat_toa(qa_arr, flags=[1, 3, 4], buffer=0):
    '''
    Bit 0: Fill
    Bit 1: Dilated Cloud
    Bit 2: Unused
    Bit 3: Cloud
    Bit 4: Cloud Shadow
    Bit 5: Snow
    '''
    cloud_mask = np.bitwise_and(qa_arr, 1 << 1) > 0
    for flag in flags:
        cloud_mask = cloud_mask | np.bitwise_and(qa_arr, 1 << flag)
        cloud_mask = cloud_mask > 0
    if buffer > 0:
        cloud_mask = cv2.dilate(cloud_mask.astype(np.uint8), np.ones((buffer, buffer)))

    snow_mask = np.bitwise_and(qa_arr, 1 << 5) > 0
    return cloud_mask, snow_mask
