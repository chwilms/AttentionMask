'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import multiprocessing
import time
import caffe
import config

from Queue import Empty

class AlchemyDataLayer(caffe.Layer):

    iterations = 0

    def setup(self, bottom, top):
        assert len(bottom) == 0
        assert len(self.__class__.spider.attr) == len(top)

        q = multiprocessing.Queue(len(self.spider().dataset)) # queue of job (all images in dataset)
        self.q = q
        self.q2 =  multiprocessing.Queue(100) # queue prep-rocessed data
        
        # load the order of images for training, so data is provided in the
        # same order in this layer and the box selection layer as well as flip 
        # and zoom are handled identically
        with open('shuffledData.txt', 'r') as myFile:
            lines = myFile.readlines()
        for index,line in enumerate(lines):
            i,flip,zoom=map(int,line.split(';'))
            q.put([index,i, flip, zoom],True) 
            
        # process to pre-process the data and extract the objectness attention 
        # ground truth
        self.process = multiprocessing.Process(target=self.fetch, args=(self.spider(),))
        self.process.start()

    def fetch(self, spider):
        while self.q.qsize() > 0:
            try:
                index,i,flip, zoom = self.q.get(True) # get image info
                item_dict = spider.fetch(i, flip, zoom)
                item = []
                if item_dict == None:
                    continue
                for attr_name in spider.attr:
                    item.append(item_dict[attr_name])
                self.q2.put([index,i,item], True) # send pre-processed datat back to master
            except Empty:
                pass

    def reshape(self, bottom, top):
        fetched = False
        while not fetched:
            if self.q2.qsize() == 0 and self.q.qsize() == 0:
                break
            while self.q2.qsize() == 0 and self.q.qsize() > 0:
                print 'waiting for spider...', self.q.qsize(),  
                time.sleep(0.2)
            try :
                index,ii,item = self.q2.get(True)
                for i in range(len(item)):
                    top[i].reshape(*item[i].shape)
                self.item = item
                fetched = True
            except Empty:
                pass
            if self.q2.qsize() == 0 and self.q.qsize() == 0:
                break
                
    def forward(self, bottom, top):
        self.iterations+=1
        item = self.item
        for i in range(len(item)):
            top[i].data[...] = item[i]
        #properly terminate the sub process, once an epoch is finished
        if self.iterations == config.steps:
            self.process.terminate()

    def backward(self, bottom, propagate_down, top):
        pass

