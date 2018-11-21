'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''

import sys
import os

os.environ['GLOG_minloglevel'] = '0' 

import argparse
sys.path.append(os.path.abspath("caffe/python"))
sys.path.append(os.path.abspath("python_layers"))
sys.path.append(os.getcwd())
import caffe
from IPython import embed
import config

import setproctitle 
from alchemy.utils.load_config import load_config
import utils

'''
python train.py gpu_id model [--restore=*.solverstate] [--debug=False] [--init_weights=*.caffemodel] [--step=num]
'''

def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('gpu_id', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('--restore', dest='restore', type=str)
    parser.add_argument('--debug', dest='debug', type=bool, default=False)
    parser.add_argument('--init_weights', dest='init_weights', type=str,
                        default='ResNet-50-model.caffemodel')
    parser.add_argument('--step', dest='step', type=int, default=int(1e5))

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    config.model = args.model
    config.solver_prototxt = args.model + '.solver.prototxt'
    if os.path.exists("configs/%s.json" % config.model):
        load_config("configs/%s.json" % config.model)
    else:
        print "The config file does not exist, use the default config..."

    caffe.set_mode_gpu()
    caffe.set_device(int(args.gpu_id))
    setproctitle.setproctitle('spider ' + args.model)
    print 'models/%s' % config.solver_prototxt
    solver = caffe.SGDSolver('models/%s' % config.solver_prototxt)
    setproctitle.setproctitle(args.model)

    # restore
    if getattr(args, 'restore', None) is not None:
        solver.restore("params/%s" % args.restore)
    # finetune
    else:
        solver.net.copy_from("params/%s" % args.init_weights)
        # div3 branch
        for name in solver.net.params.keys():
            if 'div3' in name:
                print 'copy params from %s to %s.' % (name[:name.rfind('_div3')], name)
                for i in range(len(solver.net.params[name])):
                    solver.net.params[name][i].data[...] = solver.net.params[name[:name.rfind('_div3')]][i].data
            if '_att' in name:
                print name
                print 'copy params from %s to %s.' % (name[:name.rfind('_att')], name)
                for i in range(len(solver.net.params[name])):
                    solver.net.params[name][i].data[...] = solver.net.params[name[:name.rfind('_att')]][i].data
    
    # surgeries (for upsample layer)
    interp_layers = [layer for layer in solver.net.params.keys() if 'up' in layer]
    utils.interp(solver.net, interp_layers)

    # debug
    if args.debug:
        embed()
    
    # start training
    solver.step(args.step)



