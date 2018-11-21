#include "caffe/layers/top_k_old_layer.hpp"


namespace caffe {

  template <typename Dtype>
  void TopKOldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) { }

  template <typename Dtype>
  void TopKOldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    shape_ = bottom[0]->shape();
    shape_[0] = bottom[2]->shape(0);
    k2_ = std::min(bottom[2]->shape(0),bottom[0]->shape(0));
    k_ = shape_[0];
    top[0]->Reshape(shape_);
    if (top.size() == 2) 
      top[1]->Reshape(bottom[2]->shape(0), 1, 1, 1);
  }

  template <typename Dtype>
  void TopKOldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void TopKOldLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

#ifdef CPU_ONLY
STUB_GPU(TopKOldLayer);
#endif

INSTANTIATE_CLASS(TopKOldLayer);
REGISTER_LAYER_CLASS(TopKOld);


}
