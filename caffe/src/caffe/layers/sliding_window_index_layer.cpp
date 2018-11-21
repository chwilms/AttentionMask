#include "caffe/layers/sliding_window_index_layer.hpp"
#include <iostream>


namespace caffe {

  template <typename Dtype>
  void SlidingWindowIndexLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    SlidingWindowParameter sliding_window_param = this->layer_param_.sliding_window_param();
    CHECK_GT(sliding_window_param.window_h(), 0) << "window_h must be > 0";
    CHECK_EQ(sliding_window_param.window_h()&1, 0) << "window_h must be an even int";
    CHECK_GT(sliding_window_param.window_w(), 0) << "window_w must be > 0";
    CHECK_EQ(sliding_window_param.window_w()&1, 0) << "window_w must be an even int";
    CHECK_GT(sliding_window_param.stride_h(), 0);
    CHECK_GT(sliding_window_param.stride_w(), 0);
    window_h_ = sliding_window_param.window_h();
    window_w_ = sliding_window_param.window_w();
    stride_h_ = sliding_window_param.stride_h();
    stride_w_ = sliding_window_param.stride_w();
  }

  template <typename Dtype>
  void SlidingWindowIndexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->shape(0), 1);
    CHECK_EQ(bottom[1]->shape(0), 1);
    CHECK_EQ(bottom[1]->shape(1), 1);
    CHECK_EQ(bottom[1]->shape(2), 1);
    channels_ = bottom[0]->shape(1);
    bottom_height_ = bottom[0]->shape(2);
    bottom_width_ = bottom[0]->shape(3);
    top_height_ = bottom_height_ + 1;
    top_width_ = bottom_width_ + 1;
    int counter = 0;
const Dtype* bottom_data = bottom[1]->cpu_data();
    for (int i = 0; i < bottom[1]->count(); ++i)
      counter+=bottom_data[i];
    numResults_ = bottom[1]->asum_data();
    if (numResults_ == 0) 
      numResults_=1; 
    CHECK_EQ(bottom[1]->shape(3), top_height_ * top_width_);
    top[0]->Reshape(numResults_, channels_, window_h_, window_w_);
  }

  template <typename Dtype>
  void SlidingWindowIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void SlidingWindowIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

#ifdef CPU_ONLY
STUB_GPU(SlidingWindowLayer);
#endif

INSTANTIATE_CLASS(SlidingWindowIndexLayer);
REGISTER_LAYER_CLASS(SlidingWindowIndex);


}
