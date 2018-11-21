#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/flagsToIndices_layer.hpp"

namespace caffe {

template <typename Dtype>
void FlagsToIndicesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void FlagsToIndicesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top_k_ = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int i = 0; i < bottom[0]->shape(3); i++){
    top_k_ += bottom_data[i];
  }
  top[0]->Reshape(top_k_,1,1,1);
}

template <typename Dtype>
void FlagsToIndicesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  int counter = 0;
  int index = 0;
  for (int i = 0; i < bottom[0]->shape(3); i++){
    if (bottom_data[i] == 1){
      top_data[index]=counter;
      index+=1;
    }
    counter += 1;
  }
}

template <typename Dtype>
void FlagsToIndicesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //bottom[0]->ShareDiff(*top[0]);
}

INSTANTIATE_CLASS(FlagsToIndicesLayer);
REGISTER_LAYER_CLASS(FlagsToIndices);

}  // namespace caffe
