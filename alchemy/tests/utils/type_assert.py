import unittest
import sys
import os

import numpy as np

from pycocotools.mask import *

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
from utils.type_assert import * 


class TestRLEAssert(unittest.TestCase):

    def setUp(self):
        # build some cases
        self.box = np.array([
                123, 123, 345, 456
            ], np.float)
        self.boxes = np.array([
                [0, 1, 20, 30],
                [10, 20, 30, 40]
            ], np.float)
        self.fake_boxes = [{'counts': "1111"}]
        self.shape = (800, 800)

    def test_tobbox(self):
        # frPyObjects only accept boxes list
        # it will cause exception
        try:
            frPyObjects(self.box, *self.shape)
        except Exception as e:
            pass
        else:
            raise Exception("API changes.")

        try:
            self.RLEs = frPyObjects(self.boxes, *self.shape)
        except Exception as e:
            print "API changes"
            raise e

    def test_RLEs(self):
        boxes = frPyObjects(self.boxes, *self.shape)
        fake_boxes = self.fake_boxes
        assert is_RLE(boxes) is not True
        assert is_RLE(boxes[0])
        assert is_RLEs(boxes)
        assert is_RLE(fake_boxes) is not True
        assert is_RLEs(fake_boxes) is not True

class TestMasksAssert(unittest.TestCase):

    def test_all(self):
        mask = np.asarray([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])

        masks = np.array([np.ones((4, 5))])

        assert is_mask(mask) is True
        assert is_masks(masks) is True
        assert is_mask(masks) is not True
        assert is_masks(mask) is not True


class TestBoxesAssert(unittest.TestCase):

    def test_all(self):

        box = np.array([1, 2, 3, 4])
        boxes = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
            ])
        assert is_box(box)
        assert is_box(boxes) is not True
        assert is_boxes(boxes)
        assert is_boxes(box) is not True

if __name__ == '__main__':
    unittest.main()
