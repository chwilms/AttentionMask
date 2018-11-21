'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import sys
import os
import argparse
import time
import cjson
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
import setproctitle 

from alchemy.utils.mask import encode
from alchemy.utils.load_config import load_config
from alchemy.utils.progress_bar import printProgress

import config
import utils
from config import *

from utils import gen_masks_new

'''
python test.py gpu_id model [--init_weights=*.caffemodel] [--dataset=val2014] \
                            [--end=5000]
'''

def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('--init_weights', dest='init_weights', type=str,
                        default=None)
    parser.add_argument('--dataset', dest='dataset', type=str, 
                        default='val2014')
    parser.add_argument('--end', dest='end', type=int, default=5000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    setproctitle.setproctitle(args.model)

    net = caffe.Net(
            'models/' + args.model + ".test.prototxt",
            'params/' + args.init_weights,
            caffe.TEST)

    # surgeries
    interp_layers = [layer for layer in net.params.keys() if 'up' in layer]
    utils.interp(net, interp_layers)

    if os.path.exists("configs/%s.json" % args.model):
        load_config("configs/%s.json" % args.model)
    else:
        print "Specified config does not exists, use the default config..."
        
    time.sleep(2)

    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = args.dataset
    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    spider.dataset.sort(key=lambda item: int(item.image_path[-10:-4]))
    ds = spider.dataset[:args.end]

    results = []
    for i in range(len(ds)):
        spider.fetch()
        img = spider.img_blob
        image_id = int(ds[i].image_path[-10:-4])

        ret = gen_masks_new(net, img, config, 
                dest_shape=(spider.origin_height, spider.origin_width)) 
        ret_masks, ret_scores = ret

        printProgress(i, len(ds), prefix='Progress: ', suffix='Complete', barLength=50)
        for _ in range(len(ret_masks)):
            score = float(ret_scores[_])
            objn = float(ret_scores[_])
            results.append({
                'image_id': image_id,
                'category_id': 1, #as we are doing class-agnostic proposal 
                                  #generation, cat_id is irrelevant
                'segmentation': encode(ret_masks[_]),
                'score': score,
                'objn': objn
                })

    with open('results/%s.json' % args.model, "wb") as f:
        f.write(cjson.encode(results))

