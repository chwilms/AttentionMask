'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import numpy as np
import os
import cv2
import config
from config import *

from alchemy.spiders.dataset_spider import DatasetSpider, DatasetSpiderTest

from pycocotools.mask import frPyObjects

from alchemy.utils.image import (load_image, sub_mean,resize_blob)
from alchemy.utils.mask import decode

class NoLabelException(Exception):
    pass

class BaseCOCOSSMSpiderAttentionMask(DatasetSpider):
    
    def __init__(self, *args, **kwargs):
        super(BaseCOCOSSMSpiderAttentionMask, self).__init__(*args, **kwargs)
        
    def fetch(self,i,flip,zoom):
        SCALE = self.SCALE
        self.flipped = flip
        tiny_zoom = zoom
        self.max_edge = tiny_zoom + SCALE
        item = self.dataset[i]
        self.image_path = item.image_path
        self.anns = item.imgToAnns
        batch = {}

        batch.update(self.fetch_image())
        self.fetch_masks()
        self.cal_centers_of_masks_and_bbs()

        try:
            batch.update(self.fetch_label())
            batch = self.filter_sample(batch)
            return batch
        except NoLabelException:
            return None
            
    def fetch_label(self):
        for rf in self.RFs:
            h, w, ratio = (self.height/rf) + (self.height%rf>0), (self.width/rf) + (self.width%rf>0), rf
            self.gen_single_scale_label(h, w, ratio)
        return {}

    def gen_single_scale_label(self, h, w, ratio):
        self.feat_h = h + 1
        self.feat_w = w + 1
        self.ratio = ratio
        self.find_matched_masks()

    def filter_sample(self, batch):
        for i, objAttMask in zip(self.RFs,self.objAttMaskScales): 
            batch['objAttMask_'+str(i)+'_org']=np.expand_dims(np.expand_dims(np.copy(objAttMask),0),0) #objectness attention gt before negative sample mining
            numPosSamples = int(np.sum(objAttMask)) 
            xs,ys = np.where(objAttMask==0) #all negative samples
            if len(xs) > numPosSamples*3: #if too many negative samples...
                coords = zip(xs,ys)
                np.random.shuffle(coords) #... randomly...
                coords = coords[numPosSamples*3:] 
                objAttMask[zip(*coords)]=2 #... remove samples
            batch['objAttMask_'+str(i)]=np.expand_dims(np.expand_dims(objAttMask,0),0) #objectness attention gt after negative sample mining
            
        return batch
        
    def fetch_image(self):
        # load
        if os.path.exists(self.image_path) is not True:
            raise IOError("File does not exist: %s" % self.image_path)
        img = load_image(self.image_path)
        self.origin_height = img.shape[0]
        self.origin_width = img.shape[1]

        # resize
        if img.shape[0] > img.shape[1]:
            scale = 1.0 * self.max_edge / img.shape[0]
        else:
            scale = 1.0 * self.max_edge / img.shape[1]

        h, w = int(self.origin_height * scale), int(self.origin_width * scale)
        h, w = h - h%16, w - w%16
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        img = sub_mean(img)
        if self.flipped:
            img = img[:, ::-1, :]
        self.height = img.shape[0]
        self.width = img.shape[1]

        self.image = img
        self.img_blob = self.image.transpose((2, 0, 1))
        self.img_blob = self.img_blob[np.newaxis, ...]

        self.scale = scale
        self.objAttMaskScales = []

        return {'image': self.img_blob}


    def fetch_masks(self):
        rles = []
        
        for item in self.anns:
            try:
                rles.append(frPyObjects(item['segmentation'], self.origin_height, self.origin_width)[0])
            except Exception:
                pass
        if rles == []:
            raise NoLabelException
        else:
            self.masks = decode(rles).astype(np.float)
        self.masks = resize_blob(self.masks, self.image.shape[:2])
        if self.flipped:
            self.masks = self.masks[:, :, ::-1]

    def cal_centers_of_masks_and_bbs(self):
        # (num, 4)
        scale = self.scale
        self.bbs = np.array([np.round(item['bbox']) for item in self.anns]) * scale
        if len(self.bbs) == 0:
            self.bbs=np.array([[0.0,0.0,1.0,1.0]])
        self.bbs_hw = self.bbs[:, (3, 2)]

    def find_matched_masks(self):
        #selection of segmentation masks, attention for attentional head and 
        #objectness score as well as positive and negative samples is now 
        #deferred to he BoxSelectionLayer
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio
        
        #merge masks fitting to the scale
        objAttMask = np.zeros((h,w))
        for i in range(n):
            # condition 1: neither too large nor too small
            if ((self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()):
                objAttMask += cv2.resize(self.masks[i,:,:], (w,h), interpolation=cv2.INTER_NEAREST)
            
        objAttMask = (objAttMask>0).astype(np.uint8) #due to interpolation masks may not only contain 0 and 1
        self.objAttMaskScales.append(objAttMask) #add attention ground truth for this scale
        
class BaseCOCOSSMSpiderAttSizeTest(DatasetSpiderTest):
    
    def __init__(self, *args, **kwargs):
        super(BaseCOCOSSMSpiderAttSizeTest, self).__init__(*args, **kwargs)

    def fetch(self):
        raise NotImplementedError
        
    def fetch_image(self):
        # load
        if os.path.exists(self.image_path) is not True:
            raise IOError("File does not exist: %s" % self.image_path)
        img = load_image(self.image_path)
        self.origin_height = img.shape[0]
        self.origin_width = img.shape[1]

        # resize
        if img.shape[0] > img.shape[1]:
            scale = 1.0 * self.max_edge / img.shape[0]
        else:
            scale = 1.0 * self.max_edge / img.shape[1]

        h, w = int(self.origin_height * scale), int(self.origin_width * scale)
        h, w = h - h % 16, w - w % 16

        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img = sub_mean(img)

        self.image = img
        self.img_blob = self.image.transpose((2, 0, 1))
        self.img_blob = self.img_blob[np.newaxis, ...]

        return {'image': self.img_blob}