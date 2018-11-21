#include "caffe/layers/tile_product_layer.hpp"
#include <cstdio>


namespace caffe {

template <typename Dtype>
__global__ void TileProductForward(const int nthreads, const Dtype* bottom_data,
    const Dtype* tile_bottom_data, const int channels, const int height, 
    const int width, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / channels / height / width;
    int h = index / width % height;
    int w = index % width;
    top_data[index] = bottom_data[index] * tile_bottom_data[n * width * height + h * width + w];
  }
}


template <typename Dtype>
void TileProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype *bottom_data = bottom[0]->gpu_data(),
              *tile_bottom_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = batch_ * channels_ * height_ * width_;
  TileProductForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
      count, bottom_data, tile_bottom_data, channels_, height_, width_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TileProductBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* bottom_data, const Dtype* tile_bottom_data, const int channels, 
    const int height, const int width, Dtype *bottom_diff, Dtype *tile_bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int n = index / channels / height / width;
      int h = index / width % height;
      int w = index % width;
      int tile_index = n * height * width + h * width + w;
      bottom_diff[index] = top_diff[index] * tile_bottom_data[tile_index];
      tile_bottom_diff[tile_index] += top_diff[index] * bottom_data[index];
  }
}

template <typename Dtype>
void TileProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;
  const Dtype *top_diff = top[0]->gpu_diff(),
              *bottom_data = bottom[0]->gpu_data(),
              *tile_bottom_data = bottom[1]->gpu_data();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff(),
        *tile_bottom_diff = bottom[1]->mutable_gpu_diff();
  const int count = batch_ * channels_ * height_ * width_;
  caffe_gpu_set(batch_ * height_ * width_, Dtype(0.), tile_bottom_diff);
  TileProductBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, bottom_data, tile_bottom_data, channels_, height_, width_,
      bottom_diff, tile_bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(TileProductLayer);


}
