'''
With code from Hu et al.

@author Christian Wilms
@author Hu et al.
@date 11/15/18
'''

import sys
import os
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.getcwd())
import caffe

import numpy as np

from alchemy.utils.image import resize_blob
from alchemy.utils.mask import pts_in_bbs

import time
import multiprocessing
from Queue import Empty

from config import *
from alchemy.datasets.coco import COCO_DS

from pycocotools.mask import frPyObjects

from alchemy.utils.image import load_image
from alchemy.utils.mask import decode
                        
def dataSupply(supplyConnsParent):
    dataset = COCO_DS(ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE, True)
    while True:
        for pid, conn in enumerate(supplyConnsParent):
            pollResult = conn.poll()
            if pollResult:
                index=conn.recv()
                conn.send(dataset[index])
        time.sleep(0.1)
        
class NoLabelException(Exception):
    pass

class BoxSelectionLayer(caffe.Layer):
    
    iterations = 0
    def setup(self, bottom, top):
        assert len(bottom) == 1
        assert len(top) == 5
        self.process_num = BOX_SELECTION_PROCESS        
        
        self.processes = []

        # load the order of images for training, so data is provided in the
        # same order in this layer and the data layer as well as flip and zoom
        # are handled identically
        with open('shuffledData.txt', 'r') as myFile: 
            lines = myFile.readlines()
        
        # queue of jobs ('images')
        self.q = multiprocessing.Queue(len(lines)) 
        for index,line in enumerate(lines):
            i,flip,zoom=map(int,line.split(';'))
            self.q.put([index, i, flip, zoom]) 
        
        # lock and value to synchronize maintain the order of data delivered by
        # the sub processes
        self.v = multiprocessing.Value('i', 0)
        self.lock = multiprocessing.Lock()
        
        # communication pipes between data supply sub process and sub processes
        # for data pre-processing
        supplyConnsChild = []
        supplyConnsParent = []
        for _ in range(self.process_num):
            parent_conn, child_conn = multiprocessing.Pipe(True)
            supplyConnsParent.append(parent_conn)
            supplyConnsChild.append(child_conn)
            
        # one process to load annotations for reduced memory usage
        self.process = multiprocessing.Process(target=dataSupply, args=(supplyConnsParent,))
        self.process.start()
        self.conns=[]
        
        # further processes to pre-process the data and extract the necessary 
        # ground truth
        for _ in range(self.process_num):
            parent_conn, child_conn = multiprocessing.Pipe(True)
            self.conns.append(parent_conn)
            self.processes.append(multiprocessing.Process(target=self.fetch, args=(child_conn, supplyConnsChild[_],_)))
            self.processes[_].start()
            
    def reshape(self, bottom, top):
        assert bottom[0].shape[0] == 1
        assert bottom[0].shape[1] == 1
        assert bottom[0].shape[2] == 1
        
        fetched = False
        while not fetched:
            for conn in self.conns:
                # ask sub processes for data no. self.v
                pollResult = conn.poll()
                # if data is already preprocessed, recv it
                if pollResult:
                    self.item, self.zoom_negtive_samples,self.shift_negtive_samples,i =conn.recv()
                    with self.lock:
                        self.v.value += 1
                    fetched = True
                    break
        self.numExamples = max(1,int(np.sum(bottom[0].data[0,0,0,:]))) 
        top[0].reshape(1) 
        top[1].reshape(1,MASK_SIZE,MASK_SIZE) 
        top[2].reshape(1,SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE)
        top[3].reshape(1) 
        top[4].reshape(1) 

    def fetch(self, conn, supplyConn, pid):
        while self.q.qsize() > 0:
            try:
                index, i, flip,zoom = self.q.get(True)
                supplyConn.send(i)
                item = supplyConn.recv()
                item_dict = self.fetchNext(i,flip,zoom,item)
                if item_dict == None:
                    print index
                    while self.v.value != index:
                        pass
                    with self.lock:
                        self.v.value += 1
                    continue
                item_dict['gt_masks']=item_dict['gt_masks'][item_dict['mask_filter']==1]
                item_dict['gt_atts']=item_dict['gt_atts'][item_dict['mask_filter']==1]
                while self.v.value != index:
                    time.sleep(0.1)
                    
                # if master process asks for data no. v, send it
                conn.send([item_dict,self.zoom_negtive_samples,self.shift_negtive_samples,i])
            except Empty:
                pass

    def forward(self, bottom, top):
        self.flags = bottom[0].data[0,0,0,:] #locations to attend to
        
        self.iterations+=1

        self.GT_OBJNS='gt_objns'
        self.GT_MASKS='gt_masks'
        self.GT_ATTS='gt_atts'
        self.MASK_FILTER='mask_filter'
        self.OBJN_FILTER='objn_filter'

        newAtts = np.zeros((len(self.item[self.GT_OBJNS]),SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE))
        newMasks = np.zeros((len(self.item[self.GT_OBJNS]),MASK_SIZE,MASK_SIZE))
        
        newAtts[self.item[self.MASK_FILTER]==1]=self.item[self.GT_ATTS]
        self.item[self.GT_ATTS]=newAtts
        newMasks[self.item[self.MASK_FILTER]==1]=self.item[self.GT_MASKS]
        self.item[self.GT_MASKS]=newMasks
        
        self.item[self.MASK_FILTER] = self.item[self.MASK_FILTER]*self.flags #filter positive examples
        self.item[self.GT_MASKS] = self.item[self.GT_MASKS][self.item[self.MASK_FILTER] == 1]
        self.item[self.GT_ATTS] = self.item[self.GT_ATTS][self.item[self.MASK_FILTER] == 1]
        
        if self.item[self.GT_ATTS].shape[0] == 0: #create a fake pos. example
            self.item[self.GT_ATTS]= np.zeros((1,SLIDING_WINDOW_SIZE,SLIDING_WINDOW_SIZE))
            self.item[self.GT_MASKS]= np.zeros((1,MASK_SIZE,MASK_SIZE))
            self.item[self.MASK_FILTER][0]=1
            self.flags[0]=1
            self.item[self.GT_OBJNS][0]=1

        self.filter_sample_new()
        
        self.item[self.OBJN_FILTER]=self.item[self.OBJN_FILTER][self.flags==1]
        self.item[self.MASK_FILTER]=self.item[self.MASK_FILTER][self.flags==1]

        top[0].reshape(*self.item[self.GT_OBJNS].shape)
        top[1].reshape(*self.item[self.GT_MASKS].shape)
        top[2].reshape(*self.item[self.GT_ATTS].shape)
        top[3].reshape(*self.item[self.OBJN_FILTER].shape)
        top[4].reshape(*self.item[self.MASK_FILTER].shape)
        top[0].data[...] = self.item[self.GT_OBJNS]
        top[1].data[...] = self.item[self.GT_MASKS]
        top[2].data[...] = self.item[self.GT_ATTS]
        top[3].data[...] = self.item[self.OBJN_FILTER]
        top[4].data[...] = self.item[self.MASK_FILTER]
        
        #properly terminate the sub processes, once an epoch is finished
        if self.iterations == 80000:
            for _ in range(self.process_num):
                self.processes[_].terminate()
            self.process.terminate()
            
    '''
    The folowing code is mainly from the original spiders/coco_ssm_spider.py 
    and spiders/base_coco_ssm_spider.py
    '''
                
    def fetchNext(self, i,flip,zoom,item):
        self.flipped = flip
        tiny_zoom = zoom
        self.max_edge = tiny_zoom + SCALE
        
        self.image_path = item.image_path
        self.anns = item.imgToAnns
        self.id = int(self.image_path.split('_')[-1].split('.')[0])
        batch = {'id':i}
        batch.update(self.fetch_image())
        self.fetch_masks()
        self.cal_centers_of_masks_and_bbs()

        self.zoom_negtive_samples = None
        self.shift_negtive_samples = None
        try:
            batch.update(self.fetch_label())
            return batch
        except NoLabelException:
            return None
            
    def fetch_image(self):
        if os.path.exists(self.image_path) is not True:
            raise IOError("File does not exist: %s" % self.image_path)
        img = load_image(self.image_path)
        self.origin_height = img.shape[0]
        self.origin_width = img.shape[1]

        if img.shape[0] > img.shape[1]:
            scale = 1.0 * self.max_edge / img.shape[0]
        else:
            scale = 1.0 * self.max_edge / img.shape[1]

        h, w = int(self.origin_height * scale), int(self.origin_width * scale)
        h, w = h - h%16, w - w%16

        if self.flipped:
            img = img[:, ::-1, :]
        self.height = h
        self.width = w
        
        self.scale = scale

        return {}
            
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
        self.masks = resize_blob(self.masks, (self.height,self.width))
        if self.flipped:
            self.masks = self.masks[:, :, ::-1]

    def cal_centers_of_masks_and_bbs(self):
        scale = self.scale
        self.bbs = np.array([np.round(item['bbox']) for item in self.anns]) * scale
        if len(self.bbs) == 0:
            self.bbs=np.array([[0.0,0.0,1.0,1.0]])
        self.bbs_hw = self.bbs[:, (3, 2)]
        self.bbs[:, 2:] += self.bbs[:, :2]
        self.bbs = self.bbs[:, (1, 0, 3, 2)]
        if self.flipped:
            self.bbs = self.bbs[:, (0, 3, 2, 1)]
            self.bbs[:, (1, 3)] = self.width - self.bbs[:, (1, 3)]
        self.centers = np.array(
            ((self.bbs[:, 0] + self.bbs[:, 2])/2.0,
            (self.bbs[:, 1]+ self.bbs[:, 3])/2.0)).transpose((1, 0))
        
    def fetch_label(self):

        gt_objns = []
        mask_filter = []
        gt_masks = []
        gt_atts = []
        
        for rf in RFs:
            h, w, ratio = (self.height/rf) + (self.height%rf>0), (self.width/rf) + (self.width%rf>0), rf
            self.gen_single_scale_label(h, w, ratio)
            gt_objns.append(self.gt_objns.copy())
            mask_filter.append(self.mask_filter.copy())
            gt_masks.append(self.gt_masks.copy())
            gt_atts.append(self.gt_atts.copy())

        gt_objns = np.concatenate(gt_objns)
        mask_filter = np.concatenate(mask_filter)
        gt_masks = np.concatenate(gt_masks)
        gt_atts = np.concatenate(gt_atts)
        
        
        ret = {
                'gt_objns': gt_objns,
                'gt_masks': gt_masks,
                'gt_atts':  gt_atts,
                'mask_filter': mask_filter
                }

        return ret
        
    def gen_single_scale_label(self, h, w, ratio):
        self.feat_h = h + 1
        self.feat_w = w + 1
        self.ratio = ratio
        self.find_matched_masks()
        self.assign_gt()
        
    def find_matched_masks(self):
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        win_pts = np.array((np.arange(h*w, dtype=np.int)/w, np.arange(h*w, dtype=np.int)%w))
        win_pts = win_pts.transpose((1, 0)).astype(np.float)
        win_pts *= ratio
        self.win_pts = win_pts
        objn_win_cens = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio * OBJN_CENTER_RATIO/ 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio * OBJN_CENTER_RATIO / 2.0)))
        mask_win_cens = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio * MASK_CENTER_RATIO/ 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio * MASK_CENTER_RATIO / 2.0)))
        self.objn_win_cens = objn_win_cens
        self.mask_win_cens = mask_win_cens

        win_bbs = np.hstack((win_pts - (SLIDING_WINDOW_SIZE * ratio / 2.0), win_pts + (SLIDING_WINDOW_SIZE * ratio / 2.0)))
        self.win_bbs = win_bbs

        self.objn_match = np.ones((h * w, n), np.int8) 
        self.mask_match = np.ones((h * w, n), np.int8) 

        # condition 1: neither too large nor too small
        for i in range(n):
            self.objn_match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()
        for i in range(n):
            self.mask_match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * MASK_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * MASK_UPPER_BOUND_RATIO).all()

        # condition 2: roughly contained
        for i in range(n):
            self.objn_match[:, i] = self.objn_match[:, i] & pts_in_bbs(self.centers[i], win_bbs)
        for i in range(n):
            self.mask_match[:, i] = self.mask_match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        # condition 3: roughly centered
        for i in range(n):
            self.objn_match[:, i] = self.objn_match[:, i] & pts_in_bbs(self.centers[i], objn_win_cens)
        for i in range(n):
            self.mask_match[:, i] = self.mask_match[:, i] & pts_in_bbs(self.centers[i], mask_win_cens)

        # choose the closest one
        dist = self.objn_match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        self.objn_match[np.arange(h * w), obj_ids] += 1
        self.objn_match[self.objn_match < 2] = 0
        self.objn_match[self.objn_match == 2] = 1

        dist = self.mask_match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        self.mask_match[np.arange(h * w), obj_ids] += 1
        self.mask_match[self.mask_match < 2] = 0
        self.mask_match[self.mask_match == 2] = 1
        
        self.get_zoom_negtive_samples()
        self.get_shift_negtive_samples()


    def get_shift_negtive_samples(self):
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        match = np.ones((h * w, n), np.int8)
        win_bbs = self.win_bbs
        win_pts = self.win_pts
        # condition 1: not too large or too small
        for i in range(n):
            match[:, i] = (self.bbs_hw[i].max() >= SLIDING_WINDOW_SIZE * ratio * OBJN_LOWER_BOUND_RATIO).all() & (self.bbs_hw[i].max() <= SLIDING_WINDOW_SIZE * ratio * OBJN_UPPER_BOUND_RATIO).all()

        # condition 2: roughly contained
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        # choose the closest one
        dist = match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        match[np.arange(h * w), obj_ids] += 1
        match[match < 2] = 0
        match[match == 2] = 1
        self.shift_negtive_match = match

        secondary_objns = np.zeros((h * w))
        secondary_objns[match.any(axis=1)] = 1 
        objns = np.zeros((h * w))
        objns[self.objn_match.any(axis=1)] = 1 
        negtive_samples = np.zeros((h * w))
        negtive_samples[(secondary_objns == 1) & (objns == 0)] = 1 
        if self.shift_negtive_samples is None:
            self.shift_negtive_samples = negtive_samples
        else:
            self.shift_negtive_samples = np.concatenate((self.shift_negtive_samples, negtive_samples), axis=0)


    def get_zoom_negtive_samples(self):
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        match = np.ones((h * w, n), np.int8)
        win_bbs = self.win_bbs
        objn_win_cens = self.objn_win_cens
        win_pts = self.win_pts
        # condition 2: roughly contained
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], win_bbs)

        # condition 3: roughly centered
        for i in range(n):
            match[:, i] = match[:, i] & pts_in_bbs(self.centers[i], objn_win_cens)

        # choose the closest one
        dist = match * -1e9
        for i in range(n):
            dist[:,i] += np.linalg.norm(win_pts - self.centers[i], axis=1)
        obj_ids = np.argmin(dist, axis=1)
        match[np.arange(h * w), obj_ids] += 1
        match[match < 2] = 0
        match[match == 2] = 1
        self.zoom_negtive_match = match

        secondary_objns = np.zeros((h * w))
        secondary_objns[match.any(axis=1)] = 1
        objns = np.zeros((h * w))
        objns[self.objn_match.any(axis=1)] = 1
        negtive_samples = np.zeros((h * w))
        negtive_samples[(secondary_objns == 1) & (objns == 0)] = 1
        if self.zoom_negtive_samples is None:
            self.zoom_negtive_samples = negtive_samples
        else:
            self.zoom_negtive_samples = np.concatenate((self.zoom_negtive_samples, negtive_samples), axis=0)



    def assign_gt(self):
        n, h, w, ratio = len(self.masks), self.feat_h, self.feat_w, self.ratio

        gt_objns = np.ones((h * w))
        gt_objns[np.where(self.objn_match.any(axis=1) == 0)] = 0
        self.gt_objns = gt_objns

        try:
            assert (np.nonzero(self.gt_objns)[0] == np.nonzero(self.objn_match.any(axis=1))[0]).all()
        except Exception as e:
            print np.nonzero(self.gt_objns)[0], np.nonzero(self.objn_match.any(axis=1))[0]
            raise e
        
        # mask_filter
        mask_filter = np.ones((h * w))
        mask_filter[np.where(self.mask_match.any(axis=1) == 0)] = 0
        mask_ids = np.where(mask_filter == 1)[0]
        self.mask_filter = mask_filter

        # masks
        mask_scale = 1.0 / ratio * MASK_SIZE / SLIDING_WINDOW_SIZE
        masks = resize_blob(self.masks, None, mask_scale)
        mh, mw = masks.shape[1:]

        # pad
        pad_masks = np.zeros((n, int(mh+MASK_SIZE*1.5), int(mw+MASK_SIZE*1.5)), np.int8)
        pad_masks[:, MASK_SIZE/2: mh+MASK_SIZE/2, MASK_SIZE/2: mw+MASK_SIZE/2] = masks
        masks = pad_masks

        # gt masks
        self.gt_masks = np.zeros((h * w, MASK_SIZE, MASK_SIZE), np.int8)
        obj_ids = np.argmax(self.mask_match[mask_ids, :], axis=1)
        i = 0
        scale = MASK_SIZE / SLIDING_WINDOW_SIZE
        for idx in mask_ids:
            self.gt_masks[idx, :, :] = masks[obj_ids[i], idx/w*scale: idx/w*scale+MASK_SIZE, idx%w*scale: idx%w*scale+MASK_SIZE]
            i += 1

        # gt attention:
        masks = np.zeros((n, h * scale + MASK_SIZE, w * scale + MASK_SIZE))
        bbs = self.bbs.copy()
        bbs *= mask_scale
        bbs[:, :2] = np.floor(bbs[:, :2])
        bbs[:, 2:] = np.ceil(bbs[:, 2:]).astype(np.int)
        for i in range(n):
            masks[i, int(bbs[i, 0]+MASK_SIZE/2): int(bbs[i, 2] + MASK_SIZE/2), int(bbs[i, 1] + MASK_SIZE/2):int(bbs[i, 3] + MASK_SIZE/2)] = 1
        masks = resize_blob(masks, None, 1.0/scale)
        self.gt_atts = np.zeros((h * w, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_SIZE), np.int8)
        _ = 0
        for idx in mask_ids:
            i = obj_ids[_]
            x = idx/w
            y = idx%w
            try:
                self.gt_atts[idx, :, :] = masks[i, x: x+SLIDING_WINDOW_SIZE, y: y+SLIDING_WINDOW_SIZE]
            except Exception as e:
                raise e
            _ += 1

    def filter_sample_new(self):
        self.negtive_samples = (self.zoom_negtive_samples == 1) | (self.shift_negtive_samples == 1)
        OBJN_BATCH_SIZE = 64
        MASK_BATCH_SIZE = 64
        
        self.item[self.GT_OBJNS] = self.item[self.GT_OBJNS] *self.flags
        
        if np.sum(self.item[self.GT_OBJNS]) == 0:
            self.item[self.GT_OBJNS][0]=1
            self.flags[0]=1
        self.negtive_samples = self.negtive_samples*self.flags
        positive_num = np.count_nonzero(self.item[self.GT_OBJNS])
        positive_samples = self.item[self.GT_OBJNS]#da sind erstmal alle windows drin
        positive_sample_ids = np.where(self.item[self.GT_OBJNS])[0]#das sind die pos bsp, wo gt_objns == 1 ist
        if positive_num * 2 > OBJN_BATCH_SIZE:
            positive_sample_ids = np.random.choice(positive_sample_ids, OBJN_BATCH_SIZE/2, replace=False)
            positive_samples[...] = 0
            positive_samples[positive_sample_ids] = 1
            positive_num = OBJN_BATCH_SIZE/2

        negtive_samples = self.negtive_samples #da sind erstmal alle windows drin
        negtive_sample_ids = np.where(negtive_samples)[0]
        negtive_num = len(negtive_sample_ids)
        if negtive_num * 2 > OBJN_BATCH_SIZE:
            negtive_sample_ids = np.random.choice(negtive_sample_ids, OBJN_BATCH_SIZE/2, replace=False) #replace false heisst, dass die mit 0 ersetzt werden -> newgative_samples danach genauso gross wie vorher
            negtive_samples[...] = 0
            negtive_samples[negtive_sample_ids] = 1
            negtive_num = OBJN_BATCH_SIZE/2

        objn_filter = np.zeros(len(positive_samples))
        objn_filter[(positive_samples > 0)| (negtive_samples > 0)] = 1#die windows, die entweder pos oder neg bsp sind
        self.positive_samples = positive_samples
        self.negtive_samples = negtive_samples
        self.item['objn_filter'] = objn_filter
        self.item[self.GT_OBJNS] = self.item[self.GT_OBJNS][objn_filter > 0]

        mask_ids = np.random.choice(len(self.item[self.GT_MASKS]), min(len(self.item[self.GT_MASKS]), MASK_BATCH_SIZE), replace=False)
        mask_bool = np.zeros(len(self.item[self.GT_MASKS]), np.int8)
        mask_bool[mask_ids] = 1
        mask_filter_ids = np.where(self.item[self.MASK_FILTER])[0]
        mask_filter_ids = mask_filter_ids[mask_ids]
        self.item[self.MASK_FILTER][...] = 0
        self.item[self.MASK_FILTER][mask_filter_ids] = 1
        self.item[self.GT_MASKS] = self.item[self.GT_MASKS][mask_bool == 1]
        self.item[self.GT_ATTS] = self.item[self.GT_ATTS][mask_bool == 1]
        
    def backward(self, top, propagate_down, bottom):
        pass