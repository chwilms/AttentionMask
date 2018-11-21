'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import argparse
import config
from config import *

from pycocotools.cocoeval import COCOeval

'''
python evalCOCO.py model [--useSegm=True] [--dataset=val2014] [--end=5000]
'''

def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('model', type=str)
    parser.add_argument('--useSegm', dest='useSegm', type=str, default='True')
    parser.add_argument('--end', dest='end', type=int, default=5000)
    parser.add_argument('--dataset', dest='dataset', type=str, default='val2014')

    args = parser.parse_args()
    args.useSegm = args.useSegm == 'True'
    return args

if __name__ == '__main__':
    args = parse_args()

    max_dets = [1, 10, 100, 1000]

    config.ANNOTATION_TYPE = args.dataset
    config.IMAGE_SET = args.dataset
    from spiders.coco_ssm_spider import COCOSSMDemoSpider
    spider = COCOSSMDemoSpider()
    ds = spider.dataset

    cocoGt = ds
    cocoDt = cocoGt.loadRes("results/%s.json" % args.model)
    cocoEval = COCOeval(cocoGt, cocoDt)

    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())[:args.end]
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = args.useSegm
    cocoEval.params.useCats = False
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
