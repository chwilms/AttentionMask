import numpy as np


class Dummy(object):
    pass


class BaseDataset(list):

    def __init__(self):
        super(BaseDataset, self).__init__()
