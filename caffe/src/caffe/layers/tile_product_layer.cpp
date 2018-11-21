#include "caffe/layers/tile_product_layer.hpp"


namespace caffe {

  template <typename Dtype>
  void TileProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  }

  template <typename Dtype>
  void TileProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[1]->shape(1), 1);
    batch_ = bottom[0]->shape(0);
    channels_ = bottom[0]->shape(1);
    height_ = bottom[0]->shape(2);
    width_ = bottom[0]->shape(3);
    top[0]->Reshape(batch_, channels_, height_, width_);
  }

  template <typename Dtype>
  void TileProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void TileProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }

#ifdef CPU_ONLY
STUB_GPU(TileProductLayer);
#endif

INSTANTIATE_CLASS(TileProductLayer);
REGISTER_LAYER_CLASS(TileProduct);


}
