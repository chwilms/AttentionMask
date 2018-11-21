#ifndef CAFFE_TOP_K_OLD_LAYER_HPP_
#define CAFFE_TOP_K_OLD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TopKOldLayer : public Layer<Dtype> {
  public:
    explicit TopKOldLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    
    virtual inline const char* type() const { return "TopKOld"; }

    virtual inline int MinBottomBlobs() const { return 3; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }


  protected:
    struct SortByScore {
      // closure
      vector<Dtype> score_;
      bool operator()(int x, int y) {
        return score_[x] > score_[y];
      }
    } sort_by_score;
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    Blob<int> ids_;
    vector<Dtype> max_buf_;
    vector<int> shape_;
    int k_;
    int k2_;
};


}

#endif
