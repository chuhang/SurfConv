'''
Data-Dependent Depth Discretization (D4)
Get depth level boundary values
- Hang Chu
'''
import os
import png
import numpy as np

'''
Function: read_img_16bit
    -Read a 16-bit png
in:
    -imgpath: path of image
out:
    -px_array: uint16 numpy array of image
'''
def read_img_16bit(imgpath):
    reader = png.Reader(imgpath)
    pngdata = reader.read()
    px_array = np.array(map(np.uint16, pngdata[2]))
    return px_array

'''
Function: get_depth_levels
    -Get depth level values with the proposed D4 scheme
-in:
    -d_img_path: 16-bit png depth image directory
    -num_of_levels: SurfConv levels
    -gamma: importance index of SurfConv
    -max_d: depth in meters corresponding to max 16-bit png value
    -num_of_bins: histogram resolution
-out:
    -depth_levels: colums are min/max/avg
'''
def get_depth_levels(
        d_img_path,
        num_of_levels,
        gamma,
        max_d=90.0,
        num_of_bins=1000):
    img_list = os.listdir(d_img_path)
    d_all = np.zeros((0))
    print('Collecting pixels...')
    for f in img_list:
        img = read_img_16bit(os.path.join(d_img_path, f))
        img = img.astype(float) / 65535.0 * max_d
        mask = img > 0
        d_all = np.append(d_all, img[mask])
    print('Computing histogram...')
    h, h_edges = np.histogram(d_all, bins=num_of_bins)
    h_c = (h_edges[:-1] + h_edges[1:]) / 2.0
    h_volume = np.multiply(h.astype(float), np.power(h_c, gamma))
    print('Finding boundary values...')
    idx = 0
    volume_thres = np.sum(h_volume) / num_of_levels
    depth_divide_id = np.zeros((num_of_levels, 2), dtype=int)
    for i in range(num_of_levels):
        volume_acc = 0.0
        depth_divide_id[i, 0] = int(idx)
        while True:
            volume_acc += h_volume[idx]
            idx += 1
            if idx > (num_of_bins - 1):
                idx -= 1
                break
            if volume_acc >= volume_thres:
                break
        depth_divide_id[i, 1] = int(idx)
    depth_levels = np.zeros((num_of_levels, 3))
    for i in range(num_of_levels):
        depth_levels[i, 0] = h_c[depth_divide_id[i, 0]]
        depth_levels[i, 1] = h_c[depth_divide_id[i, 1]]
        h_c_subset = h_c[depth_divide_id[i, 0]:(depth_divide_id[i, 1] + 1)]
        h_volume_subset = h_volume[depth_divide_id[i, 0]:(depth_divide_id[i, 1] + 1)]
        depth_levels[i, 2] = np.sum(np.multiply(h_c_subset, h_volume_subset)) / np.sum(h_volume_subset)
    depth_levels[0, 0] = 0.0
    depth_levels[num_of_levels - 1, 1] = np.inf
    return depth_levels

'''
Testing if main
'''
if __name__ == '__main__':
    d_img_path = '../dataset/KITTI/d_max90m_16bit/train/'
    num_of_levels = 4
    gamma = 1.0
    max_d = 90.0
    depth_levels = get_depth_levels(d_img_path, num_of_levels, gamma, max_d)
    print('--Depth images path: ' + d_img_path)
    print('--Number of SurfConv levels: ' + str(num_of_levels))
    print('--SurfConv importance index value: ' + str(gamma))
    print('Result depth levels (min/max/avg):')
    print(depth_levels)
