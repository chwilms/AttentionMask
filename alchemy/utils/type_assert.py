import numpy as np


def is_RLE(obj):
    if isinstance(obj, dict) is not True:
        return False
    if ('counts' in obj.keys()) is not True:
        return False
    if ('size' in obj.keys()) is not True:
        return False
    if (isinstance(obj['counts'], str) or isinstance(obj['counts'], unicode)) is not True:
        return False
    if isinstance(obj['size'], list) is not True:
        return False
    if len(obj['size']) != 2:
        return False
    
    return True


def is_RLEs(objs):
    if isinstance(objs, list) is not True:
        return False

    for obj in objs:
        if is_RLE(obj) is not True:
            return False

    return True


def is_masks(objs):
    if isinstance(objs, np.ndarray) is not True:
        return False
    if len(objs.shape) != 3:
        return False
    return True

def is_mask(obj):
    if isinstance(obj, np.ndarray) is not True:
        return False
    if len(obj.shape) != 2:
        return False
    return True
    

def is_box(obj):
    if isinstance(obj, np.ndarray) is not True:
        return False
    if len(obj.shape) != 1:
        return False
    if len(obj) != 4:
        return False
    return True

def is_boxes(obj):
    if isinstance(obj, np.ndarray) is not True:
        return False
    if len(obj.shape) != 2:
        return False
    if obj.shape[1] != 4:
        return False
    return True
