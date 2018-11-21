import unittest
import sys
import os

import numpy as np
import cv2

import pycocotools.mask
from pycocotools.coco import COCO


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
from utils.image import *
from utils.load_config import load_config
load_config()
import config


class TestImage(unittest.TestCase):

    def setUp(self):
        self.annotations_file = "data/coco/annotations/instances_minival2014.json"
        if getattr(self, 'coco', None) is None:
            self.coco = COCO(self.annotations_file)
        image_name = config.IMAGE_NAME_FORMAT % ('val2014', self.coco.getImgIds()[0])
        image_path = config.IMAGE_PATH_FORMAT % ('val2014', image_name)
        self.image = load_image(image_path)

    def test_load_image(self):
        blob = sub_mean(self.image)
        image_mean = np.mean(np.mean(self.image, axis=0), axis=0)
        blob_mean = np.mean(np.mean(blob, axis=0), axis=0)
        assert (np.fabs(config.RGB_MEAN - (image_mean - blob_mean)) < 1e-9).all()

    def test_image_to_data(self):
        blob = image_to_data(sub_mean(self.image))
        assert blob.shape[0] == self.image.shape[2]
        assert blob.shape[1] == self.image.shape[0]
        assert blob.shape[2] == self.image.shape[1]

    def test_visualization(self):
        # TODO
        pass

    def test_blob_resize(self):
        blob = np.zeros((2, 100, 100))
        blob[:, 0:50, 50:100] = 1
        resized_blob = resize_blob(blob, (2, 2))
        expected_blob = np.array([
                [[0, 1],
                [0, 0]],
                [[0, 1],
                [0, 0]]
        ])
        assert (resized_blob == expected_blob).all()

if __name__ == '__main__':
    unittest.main()
