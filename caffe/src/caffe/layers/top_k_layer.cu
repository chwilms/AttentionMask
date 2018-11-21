#include "caffe/layers/top_k_layer.hpp"
#include <cstdio>
#include <algorithm>
#include <iostream>


namespace caffe {

using std::sort;
using std::max;

template <typename Dtype>
__global__ void TopKForward(const int nthreads, const Dtype* bottom_data,
    const int* ids, const int chw, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    int ind = index % chw;
    int b_ind = index / chw;
    int bottom_index = ids[b_ind] * chw + ind;
    top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
void TopKLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // top batch size
  k_ = std::min(bottom[2]->shape(0),bottom[0]->shape(0));//bottom[2]->shape(0);
  // score data
  Dtype* v = bottom[1]->mutable_cpu_data();
  int ssize = bottom[1]->count() / bottom[1]->shape(0);
  max_buf_.resize(bottom[1]->shape(0));
  // objn
  if (ssize == 1) {
    for (int i = 0; i < bottom[1]->shape(0); ++i)
        max_buf_[i] = v[i];
  }
  // cls (ignore background)
  else {
      for (int i = 0; i < bottom[1]->shape(0); ++i) {
        max_buf_[i] = 0;
        for (int j = 1; j < ssize; ++j)
            max_buf_[i] = max(max_buf_[i], v[i * ssize + j]);
      }
  }
  sort_by_score.score_ = max_buf_;
  // buffer
  ids_.Reshape(bottom[1]->shape(0), 1, 1, 1);
  int *ids_cpu_data = ids_.mutable_cpu_data();
  for (int i = 0; i < bottom[0]->shape(0); ++i)
    ids_cpu_data[i] = i;

//  std::cout<<ids_cpu_data<<" "<<bottom[0]->shape(0)<<" hello from topk\n";
  // sort by score
  sort(ids_cpu_data, ids_cpu_data + bottom[0]->shape(0), sort_by_score);
  // choose first k_
  sort(ids_cpu_data, ids_cpu_data + k_);

  // copy
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int* ids_gpu_data = ids_.gpu_data();
  int count = k_;
  int item_size = 1;
  for (int i = 1; i < shape_.size(); ++i)
    item_size *= shape_[i];
  count *= item_size;

  TopKForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
      count, bottom_data, ids_gpu_data, item_size, top_data);
  CUDA_POST_KERNEL_CHECK;
  if (top.size() == 2) {
    for (int i = 0; i < top[1]->shape(0); ++i)
      top[1]->mutable_cpu_data()[i] = ids_cpu_data[i];
  }
}

template <typename Dtype>
__global__ void TopKBackward(const int nthreads, const Dtype* top_diff,
    const int *ids, const int chw, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int ind = index % chw;
    int b_ind = index / chw;
    int bottom_index = ids[b_ind] * chw + ind;
    bottom_diff[bottom_index] = top_diff[index];
  }
}

template <typename Dtype>
void TopKLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;
  const Dtype* top_diff = top[0]->gpu_diff();
  const int* ids_gpu_data = ids_.gpu_data();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = k_;
  int item_size = 1;
  for (int i = 1; i < shape_.size(); ++i)
    item_size *= shape_[i];
  count *= item_size;

  caffe_gpu_set(bottom[0]->shape(0) * item_size, Dtype(0.), bottom_diff);
  TopKBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, ids_gpu_data, item_size, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(TopKLayer);


}
