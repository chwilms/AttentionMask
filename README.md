# AttentionMask
[AttentionMask: Attentive, Efficient Object Proposal Generation Focusing on Small Objects (ACCV 2018, accepted as oral)](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people/wilms/attentionmask.html)

We propose a novel approach for class-agnostic object proposal generation, which is efficient and especially well-suited to detect small objects. Efficiency is achieved by scale-specific objectness attention maps which focus the processing on promising parts of the image and reduce the amount of sampled windows strongly. This leads to a system, which is 33% faster than the state-of-the-art and clearly outperforming state-of-the-art in terms of average recall. Secondly, we add a module for detecting small objects, which are often missed by recent models. We show that this module improves the average recall for small objects by about 53%.

![Example](/example.png)

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

# Hardware specifications
For the results in the paper we used the following hardware:
- Intel i7-5930K 6 core CPU
- 32 GB RAM
- GTX Titan X GPU with 12 GB RAM

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
|
---- data
     |
     ---- coco
          |
          ---- annotations
          |    |
          |    ---- instances_train2014.json
          |    |
          |    ---- instances_val2014.json
          |
          ---- train2014
          |    |
          |    ---- COCO_train2014_000000000009.jpg
          |    |
          |    ---- ...
          |
          ---- val2014
               |
               ---- COCO_val2014_000000000042.jpg
               |
               ---- ...
```

## Download weights
For inference, you have to download the model weights for one of the final AttentionMask models: [AttentionMask-8-128](https://fiona.uni-hamburg.de/f746e4ae/attentionmask-8-128final.caffemodel), [AttentionMask-8-192](https://fiona.uni-hamburg.de/f746e4ae/attentionmask-8-192final.caffemodel), [AttentionMask-16-192](https://fiona.uni-hamburg.de/f746e4ae/attentionmask-16-192final.caffemodel)

If you want to do training yourself, download the [initial ImageNet weights for the ResNet](https://fiona.uni-hamburg.de/f746e4ae/resnet-50-model.caffemodel). Weight files should be moved into the `params` subdirectory.



## Inference
There are two options for inference. You can either generate proposals for the COCO dataset (or any other dataset following that format) or you can generate proposals for one image.

### COCO dataset
For inference on the COCO dataset, use the `testAttentionMask.py` script with the gpu id, the model name, the weights and the dataset you want to test on (e.g., val2014):

```
$ python testAttentionMask.py 0 attentionMask-8-128 --init_weights attentionmask-8-128final.caffemodel --dataset val2014 --end 5000
```

By default, only the first 5000 images of a dataset are used.

### Individual image
If you want to test AttentionMask on one of your images, call the `demo.py` script with the path to your image, the gpu id, the model name and the weights:

```
$ python demo.py 0 attentionMask-8-128 <your image path here> --init_weights=attentionmask-8-128final.caffemodel
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

### Training on your own dataset
If you want to change the dataset form COCO to something else you have to follow the subsequent steps.

1. You have to provide the annotations in COCO-style. COCO-style means the annotation file has to be a json file similar to the COCO annotations. There are many tools on the web to change or create annotations accordingly.

2. Change the `shuffledData.txt`. The only purpose of this file is to keep the data preprocessing in the data layer and the box selection layer in sync (loading the identical image, determining if the image should be flipped or slightly zoomed). Therefore this file keeps a randomly shuffled list of all indices of the dataset. In case of COCO dataset it is a list of numbers from 0 to 82080 (82081 images in training set). Additionally for each number there is a random flag (0 or 1) for horizontally flipping the image for training as well as a number between 0 and 69 as a tiny zoom. Tiny zoom is a small number that is added on top of the max edge length to get some more variety in image sizes (check `fetch()` and `fetch_image()` in `base_coco_ssm_spider.py` or `boxSelectionLayerMP.py` for details). All three values are separated by a semicolon and each line has one entry.

3. In `config.py` adjust `ANNOTATION_TYPE` and `IMAGE_SET` according to your new dataset. Furthermore, you may have to adjust `ANNOTATION_FILE_FORMAT` for the path to the annotations or `IMAGE_PATH_FORMAT` for the path to the images. The image format strings are used in `alchemy/data/coco.py` for locating the images. Changes may have to be applied there as well, e.g., if the image file name does not start with the dataset name.

4. The solver (`models/*.solver.prototxt`) has to be adapted if the dataset is of different length than COCO. Change snapshot, display and average_loss according to the number of images in your dataset.

5. Change the value of the `--step` parameter when calling the training script to the number of images in your dataset.

6. For inference, you may have to change the extraction of `image_id` in the `testAttentionMask.py` script according to the image IDs in your dataset.

## Speed up training or testing / Decrease network size
To speed up the training or testing phase as well as decreasing the network's memory footprint, removing one or multiple scales is the straightforward solution. However, this comes at the cost of (slightly) decreased performance in terms of AR. Usually removing scales `24`, `48` and `96` results only in little changes in terms of AR. Removing scale `8` on the other hand results in large gains in terms of speed up and memory footprint, however, decreases the performance on small objects significantly. For removing one or multiple scales follow the subsequent steps.

### Testing
Usually it is no problem to test with fewer scales than you used for training. We will show here how to remove scale `128` from `myModel`. Removing other scales works accordingly.

1. From the `configs/myModel.json` file remove `128` from the list `RFs`.
2. In the `models/myModel.test.prototxt` remove the following elements: 
   1. Remove `bottom: "sample_128"` from `sample_concat layer`.
   2. Remove the `layers extractor_128` and `conv_feat_1_128s`.
   3. Remove `top: "obj128_flags"` and `bottom: "obj128_checked"` from the `SplitIndices` layer.
   4. Remove `bottom: "obj128_checked"` and `bottom: "nonObj128_checked"` from `concatFlattendObj` and `concatFlattendNonObj` layer, respectively.
   5. Remove the entire `objectness attention at scale 128` block. This step is not necessary for scale `8`, scale `16` and scale `24`.
   6. Remove the entire `shared neck at scale 128` block. This step is only necessary, if you remove a final scale from one of the branches (e.g. scale `96` or scale `128` from `attentionMask-8-128`). This step is not necessary for scale `8`, scale `16` and scale `24`.

### Training
1. Follow the instructions for testing.
2. Remove all elements in the `models/myModel.train.prototxt` file that have been removed from the `models/myModel.test.prototxt` file.
3. Furthermore, remove `top: "objAttMask_128"` and `top: "objAttMask_128_org"` from the `data` layer.
4. Remove `128` from the `attr` in `spider/coco_ssm_spider.py`.

### Removing an entire branch

If you remove scales `24`, `48`, `96`(, `192`), you can also remove the `div3 branch` from the base net (layers marked with `_div3`).

# FAQs

### I'm getting `Check failed: error == cudaSuccess (4 vs. 0)  driver shutting down` at the end of executing AttentionMask
Most likely, everything is just fine. You can ignore the error. Check, if a json-file with your chosen model name was created in the `results` subdirectory in case you ran `testAttentionMask.py`. The model name is the second parameter in `testAttentionMask.py`.
