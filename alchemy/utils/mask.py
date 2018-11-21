import numpy as np
import math

from type_assert import *

from pycocotools import mask as _mask


EPS = 1e-9

#  bbs     - [nx4] Bounding box(es) stored as [x y w h]

# fix ndarray order problem
# you don't need to transpose((1, 2, 0))

def _masks_as_fortran_order(masks):
    masks = masks.transpose((1, 2, 0))
    masks = np.asfortranarray(masks)
    masks = masks.astype(np.uint8)
    return masks


def _masks_as_c_order(masks):
    masks = masks.transpose((2, 0, 1))
    masks = np.ascontiguousarray(masks)
    return masks


def encode(obj):
    # return single RLE
    if len(obj.shape) == 2:
        mask = obj
        masks = np.array(np.asarray([mask]))
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        rle = rles[0]
        return rle
    # return RLEs
    elif len(obj.shape) == 3:
        masks = obj
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        return rles
    else:
        raise Exception("Not Implement")


def decode(obj):
    # return single mask
    if is_RLE(obj):
        rles = [obj]
        masks = _mask.decode(rles)
        masks = _masks_as_c_order(masks)
        mask = masks[0]
        return mask
    # return masks
    elif is_RLEs(obj):
        rles = obj
        masks = _mask.decode(rles)
        masks = _masks_as_c_order(masks)
        return masks
    else:
        raise Exception("Not Implement")

def area(obj):
    # single RLE
    if is_RLE(obj):
        rles = [obj]
        areas = _mask.area(rles)
        area = areas[0]
        return area
    # RLEs
    elif is_RLEs(obj):
        rles = obj
        areas = _mask.area(rles)
        return areas
    else:
        raise Exception("Not Implement")

def iou(dt, gt, crowds):
    flatten_count = 0
    if is_RLE(dt):
        flatten_count += 1
        dt = [dt]
    if is_RLE(gt):
        flatten_count += 1
        gt = [gt]
    ret = _mask.iou(dt, gt, crowds)
    if flatten_count == 0:
        return ret
    elif flatten_count == 1:
        return ret.flatten()
    elif flatten_count == 2:
        return ret[0][0]
    else:
        raise Exception("Unknown Error")

def toBbox(obj):
    if is_RLE(obj):
        rles = [obj]
        boxes = _mask.toBbox(rles)
        box = boxes[0]
        return box
    elif is_RLEs(obj):
        rles = obj
        boxes = _mask.toBbox(rles)
        return boxes
    elif is_mask(obj):
        rle = encode(obj)
        rles = [rle]
        boxes = _mask.toBbox(rles)
        box = boxes[0]
        return box
    elif is_masks(obj):
        rles = encode(obj)
        boxes = _mask.toBbox(rles)
        return boxes
    else:
        raise Exception("Not Implement")


def polygon_resize(polygon, scale):
    polygon = np.asarray(polygon[0])
    polygon = polygon * scale
    return [list(polygon)]


# 2 float number
def equal(a, b):
    return np.fabs(a - b) < EPS

# (4), (4)
# (4), (num, 4)
# or (num, 4), (num, 4)
# (num, 4), (4)
def bbs_in_bbs(sm, big):
    if len(sm.shape) == 1 and len(big.shape) == 1:
        return (sm[0] >= big[: 0]) & (sm[1] >= big[1]) & \
                (sm[2] <= big[2]) & \
                (sm[3] <= big[3])
    if len(sm.shape) == 1:
        return (sm[0] >= big[:, 0]) & (sm[1] >= big[:, 1]) & \
                (sm[2] <= big[:, 2]) & \
                (sm[3] <= big[:, 3])
    if len(big.shape) == 1:
        return (sm[:, 0] >= big[0]) & (sm[:, 1] >= big[1]) & \
                (sm[:, 2] <= big[2]) & \
                (sm[:, 3] <= big[3])
    else:
        return (sm[:, 0] >= big[:, 0]) & (sm[:, 1] >= big[:, 1]) & \
                (sm[:, 2] <= big[:, 2]) & \
                (sm[:, 3] <= big[:, 3])

# (2), (2)
# or (2), (num, 4)
# or (num, 2), (2)
# or (num, 2), (num, 4)
def pts_in_bbs(pts, bbs):
    if len(bbs.shape) == 1:
        return ((pts >= bbs[:2]) & (pts <= bbs[2:])).all()
    else:
        return ((pts >= bbs[:,:2]) & (pts <= bbs[:,2:])).all(axis=1)


# (1, h, w) (4) (2)
def crop(masks, src_shape, dest_shape):
    xb, xe, yb, ye = src_shape
    oh, ow = dest_shape
    size = masks.shape[1:]
    masks = masks[:, max(0, -xb): size[0] + min(0, oh-xe), max(0, -yb): size[1] + min(0, ow-ye)]
    return masks
