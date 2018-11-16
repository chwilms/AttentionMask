# AttentionMask
[AttentionMask: Attentive, Efficient Object Proposal Generation Focusing on Small Objects (ACCV 2018, accepted as oral)](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms/attentionmask.html)

We propose a novel approach for class-agnostic object proposal generation, which is efficient and especially well-suited to detect small objects. Efficiency is achieved by scale-specific objectness attention maps which focus the processing on promising parts of the image and reduce the amount of sampled windows strongly. This leads to a system, which is 33% faster than the state-of-the-art and clearly outperforming state-of-the-art in terms of average recall. Secondly, we add a module for detecting small objects, which are often missed by recent models. We show that this module improves the average recall for small objects by about 53%.

The system is based on [FastMask](https://arxiv.org/abs/1612.08843).

If you find this software useful in your research, please cite our paper.

```
@inproceedings{WilmsFrintropACCV2018,
title = {{AttentionMask}: Attentive, Efficient Object Proposal Generation Focusing on Small Objects},
author = {Christian Wilms and Simone Frintrop},
booktitle = {Asien Conference on Computer Vision (ACCV)},
year = {2018}
}
```
# Requirements
- Ubuntu 16.04 
- Cuda 9.0
- Python 2.7
- OpenCV-Python
- Python packages: scipy, numpy, python-cjson, setproctitle, scikit-image
- [COCOApi](https://github.com/pdollar/coco)
- Caffe (already part of this git)
- [Alchemy](https://github.com/voidrank/alchemy) (already part of this git)

# Installation
We assume Ubuntu 16.04 with Cuda 9.0, Python 2.7 and pip already installed

First, install OpenCV-Python:

```
$ sudo apt-get install python-opencv
```

Then, clone and install COCOApi as described [here](https://github.com/pdollar/coco). 

Now, clone this repository, install the Python packages from `requirements.txt` if necessary and install the requirements of Caffe (PyCaffe) following [the official instructions](http://caffe.berkeleyvision.org/installation.html). Edit the `Makefile.config` according to your system settings.
```
$ git clone https://github.com/chwilms/AttentionMask
$ cd AttentionMask
$ pip install -r requirements.txt
$ cd caffe
$ make pycaffe -j6
$ cd ..
```

Create new subdirectories for weights `params` and results `results`:
```
$ mkdir params results
```

# Usage
After sucessfull installation, AttentionMask can immediatly be used without any training. Just download the weights (and the COCO dataset). However, you can also train AttentionMask with your own data. Note however, that your own data should be in the COCO format.

## Download dataset
Download the `train2014` and `val2014` splits from [COCO dataset](http://cocodataset.org/#download). The `train2014` split is exclusively used for training, while the first 5000 image from the `val2014` split are used for testing. After downloading, extract the data in the following structure:

```
AttentionMask
-> data
   -> coco
      -> annotations
         | some annotation files
      -> train2014
         | image files
      -> val2014
         | image files
```

## Download weights
For inference, you have to download the model weights for the final AttentionMask model from our [webpage](https://fiona.uni-hamburg.de/f746e4ae/attentionmask-8-128final.caffemodel). If you want to do training yourself, download the [initial ImageNet weights for the ResNet](https://fiona.uni-hamburg.de/f746e4ae/resnet-50-model.caffemodel). Both files should be moved into the `params` subdirectory.

## Inference
There are two options for inference. You can either generate proposals for the COCO dataset (or any other dataset following that format) or you can generate proposals for one image.

### COCO dataset
For inference on the COCO dataset, use the `testAttentionMask.py` script with the gpu id, the model name, the weights and the dataset you want to test on (e.g., val2014):

```
$ python testAttentionMask.py 0 attentionMask-8-128 --init_weights attentionMask-8-128_final.caffemodel --dataset val2014 --end 5000
```

By default, only the first 5000 images of a dataset are used.

### Individual image
If you want to test AttentionMask on one of your images, call the `demo.py` script with the path to your image, the gpu id, the model name and the weights:

```
$ python demo.py 0 attentionMask-8-128 <your image path here> --init_weights=attentionMask-8-128_final.caffemodel
```

As a result you get an image with the best 20 proposals as overlays. If you want to dive deeper into the set of proposals, you can store them all using the `ret_masks` variable in the script with the `ret_scores` variable for the objectness scores.

## Evaluation
For evaluation on the COCO dataset, you can use the `evalCOCO.py` script with the model name and the dataset used. `--useSegm` is  a flag for using segmentation masks instead of bounding boxes.

```
$ python evalCOCO.py attentionMask-8-128 --dataset val2014 --useSegm True --end 5000
```

By default, only the first 5000 images of a dataset are used.
## Training
To train AttentionMask on a dataset, you can use the `train.sh` script. It iterates over several epochs and saves as well as evaluates the result of each epoch (outputs in `trainEval.txt`). For validation, currently the frist 5000 images of the training set are used, however, we encourage you to use a split of `val2014` that is disjunct to the first 5000 images as your own validation set. Lowering the learning rate has to be done manually. We lowered the learning rate after 3 consecutive epochs of not improving results on our validation set. The environmental variable `EPOCH` determines the next epoch to be run and is automatically incremented.

```
$ export EPOCH=1
$ ./train.sh
```
