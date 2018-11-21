'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

from config import *
import numpy as np
from alchemy.datasets.coco import COCO_DS

from base_coco_ssm_spider import BaseCOCOSSMSpiderAttentionMask, BaseCOCOSSMSpiderAttSizeTest, NoLabelException

class COCOSSMSpiderAttentionMask8_128(BaseCOCOSSMSpiderAttentionMask):

    attr = ['image', 'objAttMask_8', 'objAttMask_16', 'objAttMask_24', 'objAttMask_32', 'objAttMask_48', 'objAttMask_64', 'objAttMask_96', 'objAttMask_128', 'objAttMask_8_org', 'objAttMask_16_org', 'objAttMask_24_org', 'objAttMask_32_org', 'objAttMask_48_org', 'objAttMask_64_org', 'objAttMask_96_org', 'objAttMask_128_org']

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpiderAttentionMask8_128, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE

class COCOSSMSpiderAttentionMask8_192(BaseCOCOSSMSpiderAttentionMask):

    attr = ['image', 'objAttMask_8', 'objAttMask_16', 'objAttMask_24', 'objAttMask_32', 'objAttMask_48', 'objAttMask_64', 'objAttMask_96', 'objAttMask_128', 'objAttMask_192', 'objAttMask_8_org', 'objAttMask_16_org', 'objAttMask_24_org', 'objAttMask_32_org', 'objAttMask_48_org', 'objAttMask_64_org', 'objAttMask_96_org', 'objAttMask_128_org', 'objAttMask_192_org']

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpiderAttentionMask8_192, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE
        
class COCOSSMSpiderAttentionMask16_192(BaseCOCOSSMSpiderAttentionMask):

    attr = ['image', 'objAttMask_16', 'objAttMask_24', 'objAttMask_32', 'objAttMask_48', 'objAttMask_64', 'objAttMask_96', 'objAttMask_128', 'objAttMask_192', 'objAttMask_16_org', 'objAttMask_24_org', 'objAttMask_32_org', 'objAttMask_48_org', 'objAttMask_64_org', 'objAttMask_96_org', 'objAttMask_128_org', 'objAttMask_192_org']

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
            self.__class__.cats_to_labels = dict([(self.dataset.getCatIds()[i], i+1) for i in range(len(self.dataset.getCatIds()))])
        super(COCOSSMSpiderAttentionMask16_192, self).__init__(*args, **kwargs)
        self.RFs = RFs
        self.SCALE = SCALE

class COCOSSMDemoSpider(BaseCOCOSSMSpiderAttSizeTest):

    def __init__(self, *args, **kwargs):
        if getattr(self.__class__, 'dataset', None) is None:
            self.__class__.dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, False)
        super(COCOSSMDemoSpider, self).__init__(*args, **kwargs)
        try:
            self.RFs = RFs
        except Exception:
            pass
        try:
            self.SCALE = TEST_SCALE
        except Exception:
            pass
        
    def fetch(self):
        idx = self.get_idx()
        item = self.dataset[idx]
        self.image_path = item.image_path
        self.anns = item.imgToAnns
        self.max_edge = self.SCALE
        self.fetch_image()
        return {"image": self.img_blob}
