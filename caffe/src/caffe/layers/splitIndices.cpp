#include "caffe/layers/splitIndices.hpp"
#include <iostream>
#include <algorithm> 

namespace caffe {

  template <typename Dtype>
  void SplitIndicesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  }

  template <typename Dtype>
  void SplitIndicesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    int feld[bottom.size()-1];
    int summeFelder = 0;
    CHECK_EQ(bottom[1]->shape(0), 1);
    CHECK_EQ(bottom[1]->shape(1), 1);
    CHECK_EQ(bottom[1]->shape(2), 1);    
    for (int i = 1; i < bottom.size(); ++i) {
      feld[i-1]=bottom[i]->shape(3);
      summeFelder+=bottom[i]->shape(3);
    }
    CHECK_EQ(bottom[0]->shape(0), 1);
    CHECK_EQ(bottom[0]->shape(1), 1);
    CHECK_EQ(bottom[0]->shape(2), 1);
    CHECK_EQ(bottom[0]->shape(3), summeFelder); 
    bottomNum_ = bottom[0]->shape(3);
    for (int i = 1; i < bottom.size(); ++i) {
      top[i-1]->Reshape(1, 1, 1, feld[i-1]);
    }
  }

  template <typename Dtype>
  void SplitIndicesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    int feld[bottom.size()];
    feld[0] = 0;
    for (int i = 1; i < bottom.size(); ++i) {
      feld[i]=bottom[i]->shape(3);
    }
    int startIndex = 0;
    for (int i = 1; i < bottom.size(); ++i) {
      Dtype* top_data = top[i-1]->mutable_cpu_data();
      memcpy(top_data, &bottom_data[startIndex] , sizeof(Dtype) * feld[i]);
      startIndex+=feld[i];
    }
    for (int ii = 0; ii< 4; ++ii){
      const Dtype* top_data = top[ii]->cpu_data();
      int summe = 0;
      for (int i = 0; i < top[ii]->shape(3); ++i){
        summe+=top_data[i];
      }
    }
  }

  template <typename Dtype>
  void SplitIndicesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return;
  }

#ifdef CPU_ONLY
STUB_GPU(SplitIndicesLayer);
#endif

INSTANTIATE_CLASS(SplitIndicesLayer);
REGISTER_LAYER_CLASS(SplitIndices);


}
