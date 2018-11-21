import numpy as np

import cv2
import matplotlib.pyplot as plt
from PIL import Image

import config


'''
    load image to ndarray
    :param image:   path of image
    :return:        ndarray with shape (H, W, 3)
'''
def load_image(image_path):
    img = cv2.imread(image_path)
    # grey image
    if img.shape[-1] == 1:
        expand_img = np.zeros((img.shape[0], img.shape[1], 3))
        expand_img[:, :, :] = img
        img = expand_img
    # BGR to RGB
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float)
    
    return img

    
'''
    convert img to blob (transpose)
    :param img:     ndarray with shape (num_image, H, W, 3)
                    or (H, W, 3)
    :return:        blob
'''
def image_to_data(img):
    if len(img.shape) == 3:
        img = img.transpose((2, 0, 1))
    elif len(img.shape) == 4:
        img = img.transpose((0, 3, 1, 2))
    return img

'''
    subtract mean
    :param:     img, ndarray with shape (num_image, H, W, 3)
                or shape (H, W, 3)
'''
def sub_mean(img):
    mean = config.RGB_MEAN
    blob = img.copy()
    blob -= mean
    return blob


'''
    draw attention
    :param image:       np.ndarray with shape (3, H, W)
    :param args:        list of mask with shape (H, W)
'''

def draw_attention(img, *masks):
    cmap = plt.get_cmap('jet')
    imgs = []
    for mask in masks:
        # convert to heat map
        rgba_img = cmap(mask)
        rgb_img = np.delete(rgba_img, 3, 2)
        rgb_img = (rgb_img * 255)
        # mean
        mean_img = ((rgb_img + img) / 2).astype(np.uint8)
        # convert to PIL.Image
        mean_img = Image.fromarray(mean_img, "RGB")
        imgs.append(mean_img)

    return imgs

'''
    blob resize
    :param blob:        a ndarray with shape (num, H, W)
    :param dest_shape:  tuple (h, w)
    :return:            a ndarray with shape (num, h, w)
    NOTE: If image size is too small, it will cause an error.
          It may be a bug of opencv2.10 for python
'''
def resize_blob(blob, dest_shape=None, im_scale=None, method=None):
    assert dest_shape != None or im_scale != None
    img = blob.transpose((1, 2, 0)) 
    if method is None:
        method = cv2.INTER_LINEAR
    if dest_shape is not None:
        dest_shape = dest_shape[1], dest_shape[0]
        img = cv2.resize(img, dest_shape, interpolation=method)
    else:
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=method)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    blob = img.transpose((2, 0, 1))
    return blob



'''
    visualize bbox
    :param image: (h, w, c) ndarray
    :param bbs:  (n, 4) ndarray(h,w order)
'''
def visualize_bbs(image, bbs):
    n = len(bbs)
    h, w = image.shape[:2]
    masks = np.zeros((n, h, w))
    for i in range(n):
        x0, x1, y0, y1 = max(0, bbs[i,0]), min(h, bbs[i,2]), max(0, bbs[i,1]), min(w, bbs[i,3])
        masks[i, x0: x1, y0: y1] = 1
    imgs = draw_attention(image, *list(masks))
    for img in imgs:
        plt.figure()
        plt.imshow(img)
    plt.show()

'''
    visualize masks
    :param image: (h, w, c) ndarray
    :param masks: (n, h, w) ndarray
'''
def visualize_masks(image, masks):
    imgs = draw_attention(image, *list(masks))
    n = 1
    while n * n < len(masks):
        n += 1
    plt.figure()
    _ = 1
    for img in imgs:
        plt.subplot(n, n, _)
        plt.imshow(img)
        _ += 1
    plt.show()
