import unittest
import sys
import os
import json

import numpy as np


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
from utils.load_config import load_config
import config


class TestImage(unittest.TestCase):

    def setUp(self):
        self.config_path = "examples/config/coco.json"

    def test_load_config(self):
        load_config(self.config_path)
        with open(self.config_path, "r") as f:
            obj = json.load(f)
        for k,v in obj.iteritems():
            assert getattr(config, k, None) == v
        assert config.RGB_MEAN == [123.68, 116.779, 103.939]


if __name__ == '__main__':
    unittest.main()
