#include "caffe/layers/check_att_layer.hpp"
#include <iostream>


namespace caffe {

  template <typename Dtype>
  void CheckAttLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  }

  template <typename Dtype>
  void CheckAttLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
    CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
    CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));
    top[0]->Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2),bottom[0]->shape(3));
    top[1]->Reshape(bottom[0]->shape(0),bottom[0]->shape(1),bottom[0]->shape(2),bottom[0]->shape(3));
  }

  template <typename Dtype>
  void CheckAttLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_obj_data = bottom[0]->cpu_data();
    const Dtype* bottom_noObj_data = bottom[1]->cpu_data();
    Dtype* top_obj_data = top[0]->mutable_cpu_data();
    Dtype* top_noObj_data = top[1]->mutable_cpu_data();
    int summe = 0;
    for (int i = 0; i<bottom[0]->count();++i){
      if(bottom_obj_data[i] > bottom_noObj_data[i]){
        summe+=1;
      }
      top_obj_data[i]=bottom_obj_data[i];
      top_noObj_data[i]=bottom_noObj_data[i];
    }
    if (summe == 0){
      top_obj_data[0]=0.51;//bottom_noObj_data[0];
      top_noObj_data[0]=0.49;//bottom_obj_data[0];
    }
  }

  template <typename Dtype>
  void CheckAttLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
//bottom[0]->ShareDiff(*top[0]);
  }

#ifdef CPU_ONLY
STUB_GPU(CheckAttLayer);
#endif

INSTANTIATE_CLASS(CheckAttLayer);
REGISTER_LAYER_CLASS(CheckAtt);


}
