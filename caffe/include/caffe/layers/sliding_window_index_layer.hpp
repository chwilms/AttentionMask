#ifndef CAFFE_SLIDING_WINDOW_INDEX_LAYER_HPP_
#define CAFFE_SLIDING_WINDOW_INDEX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SlidingWindowIndexLayer : public Layer<Dtype> {
  public:
    explicit SlidingWindowIndexLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "SlidingWindowIndex"; }

    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }


  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int channels_;
    int bottom_height_;
    int bottom_width_;
    int top_height_;
    int top_width_;
    int window_h_;
    int window_w_;
    int stride_h_;
    int stride_w_;
    int numResults_;
};


}

#endif
