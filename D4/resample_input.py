'''
Data-Dependent Depth Discretization (D4)
Resample & mask images for training
- Hang Chu
'''
import os
import numpy as np
import scipy.misc as m
from get_depth_levels import *

'''
Function: resample_images_KITTI
    -Generate resampled images for training
-in:
    -dataset_path: KITTI directory
    -num_of_levels: SurfConv levels
    -gamma: importance index of SurfConv
-out:
    -depth_levels: colums are min/max/avg
'''
def resample_images_KITTI(dataset_path, num_of_levels, gamma):
    d_img_path_16bit = os.path.join(dataset_path, 'd_max90m_16bit')
    max_d_16bit = 90.0
    d_img_path_8bit = os.path.join(dataset_path, 'd_max80m_8bit')
    max_d_8bit = 80.0
    # 2D Conv is equivalent to SurfConv1, where all pixels are projected to dataset mean depth,
    # and projected image has width specified by surfconv1_lvl1_width
    # For multi-level SurfConv, we are gonna resample levels based on the same value,
    # so that the ConvNet always has the same real-world receptive field
    # This guarantees fair comparison
    surfconv1_lvl1_width = 960.0
    input_types = ['rgb', 'hha']
    splits = ['train', 'val']
    print('Computing dataset mean gamma-weighted depth...')
    depth_levels = get_depth_levels(
        d_img_path_16bit + '/train', 1, gamma, max_d_16bit)
    sc1_d_avg = depth_levels[0, 2]
    print('Computing SurfConv level boundaries...')
    depth_levels = get_depth_levels(
        d_img_path_16bit +
        '/train',
        num_of_levels,
        gamma,
        max_d_16bit)
    lvl1_width = surfconv1_lvl1_width / sc1_d_avg * depth_levels[0, 2]
    print('Resampling images...')
    for tp in input_types:
        for sp in splits:
            imgpath_in = os.path.join(dataset_path, tp + '/' + sp)
            imgpath_out = os.path.join(
                dataset_path,
                tp +
                '_SurfConv' +
                str(num_of_levels) +
                '_gamma' +
                str(gamma) +
                '/' +
                sp)
            if not os.path.isdir(imgpath_out):
                os.makedirs(imgpath_out)
            img_list = os.listdir(imgpath_in)
            for f in img_list:
                img = m.imread(os.path.join(imgpath_in, f))
                for k in range(num_of_levels):
                    width = round(lvl1_width /
                                  depth_levels[0, 2] *
                                  depth_levels[k, 2])
                    height = round(width / img.shape[1] * img.shape[0])
                    img_k = m.imresize(img, (int(height), int(width)), interp = 'bicubic')
                    m.imsave(os.path.join(imgpath_out,
                                          f[:-4] + 'p' + str(k + 1) + '.png'), img_k)
    print('Resampling & masking labels...')
    for sp in splits:
        imgpath_l = os.path.join(dataset_path, 'label/' + sp)
        imgpath_d = os.path.join(d_img_path_8bit, sp)
        imgpath_out = os.path.join(
            dataset_path,
            'label_SurfConv' +
            str(num_of_levels) +
            '_gamma' +
            str(gamma) +
            '/' +
            sp)
        if not os.path.isdir(imgpath_out):
            os.makedirs(imgpath_out)
        img_list = os.listdir(imgpath_l)
        for f in img_list:
            lbl = m.imread(os.path.join(imgpath_l, f))
            d = m.imread(os.path.join(imgpath_d, f))
            for k in range(num_of_levels):
                width = round(lvl1_width /
                              depth_levels[0, 2] *
                              depth_levels[k, 2])
                height = round(width / lbl.shape[1] * lbl.shape[0])
                l_k = m.imresize(
                    lbl, (int(height), int(width)), interp='nearest')
                d_k = m.imresize(
                    d, (int(height), int(width)), interp='nearest')
                d_k = d_k.astype(float) / 255.0 * max_d_8bit
                mask = np.logical_and(
                    d_k > depth_levels[k, 0], d_k <= depth_levels[k, 1])
                l_k_new = np.zeros((int(height), int(width)), dtype=np.uint8)
                l_k_new[mask] = l_k[mask]
                m.imsave(os.path.join(
                    imgpath_out, f[:-4] + 'p' + str(k + 1) + '.png'), l_k_new)

'''
Testing if main
'''
if __name__ == '__main__':
    dataset_path = '../dataset/KITTI/'
    num_of_levels = 4
    gamma = 1.0
    resample_images_KITTI('../dataset/KITTI/', 4, 1.0)
