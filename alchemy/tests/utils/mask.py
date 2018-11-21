import unittest
import sys
import os

import numpy as np

import pycocotools.mask


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
from utils.mask import *


class TestMask(unittest.TestCase):
    
    def test_main(self):

        # set up
        mask = np.asarray([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]
        ])
        mask2 = np.ones((4, 5))
        masks = np.array([mask])

        # test area
        assert area(encode(mask)) == 7
        assert area(encode(masks)[0]) == 7
        assert np.array_equal(decode(encode(mask)), mask)

        # test iou
        assert isinstance(iou(encode(masks), encode(masks), [0]), np.ndarray)
        assert iou(encode(mask), encode(mask), [0]) == 1
        assert equal(iou(encode(np.array([mask, mask])), encode(mask2), [0]), 7.0/20).all()

        # test toBbox
        assert isinstance(toBbox(masks), np.ndarray)
        assert np.equal(toBbox(encode(mask)), np.array([1, 0, 4, 3])).all()
        assert np.equal(toBbox(encode(mask2)), np.array([0, 0, 5, 4])).all()

    def test_bbs_in_bbs(self):
        bbs_a = np.array([1, 1, 2.0, 3])
        bbs_b = np.array([1, 0, 4, 5])
        bbs_c = np.array([0, 0, 2, 2])
        assert bbs_in_bbs(bbs_a, bbs_b).all()
        assert bbs_in_bbs(bbs_b, bbs_c).any() is not True
        assert bbs_in_bbs(bbs_a, bbs_c).any() is not True
        bbs_d = np.array([
            [0, 0, 5, 5],
            [1, 2, 4, 4],
            [2, 3, 4, 5]
            ])
        assert (bbs_in_bbs(bbs_a, bbs_d) == np.array([1, 0, 0], dtype=np.bool)).all()
        assert (bbs_in_bbs(bbs_d, bbs_d) == np.ones((3), dtype=np.bool)).all()
        bbs_a *= 100
        bbs_d *= 100
        assert (bbs_in_bbs(bbs_a, bbs_d) == np.array([1, 0, 0], dtype=np.bool)).all()

    def test_pts_in_bbs(self):
        pt = np.array([1, 2])
        bbs_a = np.array([1, 2, 3, 4])
        assert isinstance(pts_in_bbs(pt, bbs_a), np.bool_)
        assert pts_in_bbs(pt, bbs_a)
        pts = np.array([
            [1, 2],
            [2, 3],
            [3, 4]
        ])
        bbs_b = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [2, 3, 4, 5]
        ])
        assert (pts_in_bbs(pts, bbs_b) == np.array([1, 0, 1], dtype=np.bool)).all()
        

if __name__ == '__main__':
    unittest.main()
