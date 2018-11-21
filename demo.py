'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import sys
import os
import argparse
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
import config

import numpy as np
import cv2

from alchemy.utils.image import load_image, sub_mean
from alchemy.utils.load_config import load_config

from utils import gen_masks_new, interp
from skimage.segmentation import mark_boundaries

'''
    python image_demo.py gpu_id model input_image
'''

colors = [(170, 255, 0), (170, 85, 170), (170, 85, 0), (0, 255, 255),
          (85, 255, 170), (170, 255, 85), (170, 0, 85), (255, 170, 170),
          (0, 170, 85), (170, 0, 170), (85, 255, 85), (0, 85, 170), 
          (255, 170, 0), (255, 0, 170), (85, 170, 0), (255, 170, 85), 
          (0, 0, 85), (85, 170, 85), (85, 170, 255), (0, 170, 255),
          (170, 170, 0), (170, 255, 170), (255, 255, 170), (255, 85, 255), 
          (0, 0, 255), (85, 0, 255), (0, 255, 0), (255, 85, 0), (170, 85, 255),
          (85, 0, 85), (255, 255, 85), (255, 0, 85), (85, 85, 0),
          (255, 85, 85), (170, 255, 255), (85, 0, 170), (0, 170, 170), 
          (0, 85, 0), (0, 255, 85), (255, 85, 170), (0, 255, 170),
          (85, 255, 0), (85, 85, 170), (255, 170, 255), (0, 85, 255),
          (85, 255, 255), (85, 85, 255), (170, 170, 85), (170, 0, 0),
          (255, 0, 255), (170, 85, 85), (0, 170, 0), (255, 255, 0),
          (0, 85, 85), (170, 170, 255), (85, 170, 170), (0, 0, 170),
          (85, 0, 0), (170, 0, 255)]

def parse_args():
    parser = argparse.ArgumentParser('process image')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('input_image', type=str)
    parser.add_argument('--init_weights', type=str,
                        default='', dest='init_weights')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # caffe setup
    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))

    net = caffe.Net(
            'models/' + args.model + '.test.prototxt',
            'params/' + args.init_weights,
            caffe.TEST)
    
    # surgeries
    interp_layers = [layer for layer in net.params.keys() if 'up' in layer]
    interp(net, interp_layers)

    # load config
    if os.path.exists("configs/%s.json" % args.model):
        load_config("configs/%s.json" % args.model)
    else:
        print "Specified config does not exists, use the default config..."
        
    # image pre-processing
    img = load_image(args.input_image)
    img_org = img.copy()[:,:,::-1]
    origin_height = img.shape[0]
    origin_width = img.shape[1]

    if img.shape[0] > img.shape[1]:
        scale = 1.0 * config.TEST_SCALE / img.shape[0]
    else:
        scale = 1.0 * config.TEST_SCALE / img.shape[1]

    h, w = int(origin_height * scale), int(origin_width * scale)
    h, w = h - h % 16, w - w % 16

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = sub_mean(img)

    img_blob = img.transpose((2, 0, 1))
    img_blob = img_blob[np.newaxis, ...]

    # all masks with all scores (1000 in total)
    ret_masks, ret_scores = gen_masks_new(net, img_blob, config, dest_shape=(origin_height, origin_width))

    # display top 20 masks
    for i in np.argsort(ret_scores)[::-1][:20]:
        mask = ret_masks[i].copy()
        color = np.array(colors[i%len(colors)])
        mask = np.dstack([mask]*3)
        colorMask = color*mask
        img_org=img_org*(1-mask)+mask*img_org*.5+colorMask*.5
        img_org = mark_boundaries(img_org,mask[:,:,0],color=color.tolist(),mode='thick')

    img_org = img_org.astype(np.uint8)
    cv2.imshow('image', img_org)
    cv2.waitKey(100000)