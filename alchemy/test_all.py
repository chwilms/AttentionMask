import unittest
import importlib
import os
import sys
import os.path as osp


def test_all(path):
    if os.path.exists(osp.join(path, '__init__.py')):
        for f_name in os.listdir(path):
            f_name = osp.join(path, f_name)
            if f_name.endswith(".py"):
                assert os.system('python ' + f_name) == 0
            else:
                test_all(f_name)



if __name__ == "__main__":
    test_all('tests')

