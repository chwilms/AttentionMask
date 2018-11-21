#include "caffe/layers/sliding_window_index_layer.hpp"
#include <cstdio>
#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void SlidingWindowIndexForward(const int nthreads, const Dtype* bottom_data, const Dtype* bottom_flags,
    const int channels, const int bottom_height, const int bottom_width, 
    const int top_height, const int top_width, const int stride_h, const int stride_w, 
    const int window_h, const int window_w, const int numResults, Dtype* top_data) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    /* (((height/stride_h) + (height%stride_h>0)) * ((width/stride_w) + (width%stride_w>0)),
     *   channels,
     *   window_h,
     *   window_w)
     */

    int c = index % channels;
    int n = index / channels; 
    if (bottom_flags[n] > 0){ 
      int origin_h = n / top_width;
      int origin_w = n % top_width;

      int windowIndex = 0;
      for (int i = 0; i < n; ++i){
          windowIndex += bottom_flags[i]; 
      }
      for (int h = -window_h/2; h < window_h/2; ++h)
        for (int w = -window_w/2; w < window_w/2; ++w) {
          int top_idx = windowIndex * channels * window_h * window_w +
                    c * window_h * window_w +
                    (h + window_h/2) * window_w +
                    (w + window_w/2);
          
          if (origin_h + h >= 0 && origin_h + h < bottom_height &&
              origin_w + w >= 0 && origin_w + w < bottom_width) {
            int bottom_idx =  c * bottom_height * bottom_width +
                          (origin_h + h) * bottom_width +
                          (origin_w + w);
            top_data[top_idx] = bottom_data[bottom_idx];
          }
          else
            top_data[top_idx] = 0;
        }
    }
}
  
}


template <typename Dtype>
void SlidingWindowIndexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_flags = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top_height_ * top_width_ * channels_;
  SlidingWindowIndexForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
      count, bottom_data, bottom_flags, channels_, bottom_height_, bottom_width_, top_height_, 
      top_width_, stride_h_, stride_w_, window_h_, window_w_, numResults_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SlidingWindowIndexBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int bottom_height, const int bottom_width, 
    const int top_height, const int top_width, const int stride_h, const int stride_w, 
    const int window_h, const int window_w, Dtype *bottom_diff, const Dtype* bottom_flags, const int numResults) {
  CUDA_KERNEL_LOOP(index, nthreads){ 
    /* (((height/stride_h) + (height%stride_h>0)) * ((width/stride_w) + (width%stride_w>0)),
     *   channels,
     *   window_h,
     *   window_w)
     */

    int w = index % bottom_width;
    int h = (index / bottom_width) % bottom_height;
    int c = index / bottom_width / bottom_height;
    
    for (int off_h = -window_h + 1; off_h <= 0; ++off_h)
      for (int off_w = -window_w + 1; off_w <= 0; ++off_w) {
        int top_h = h + off_h;
        int top_w = w + off_w;
        if (top_h < -window_h/2 || top_w < -window_w/2 ||
           top_h >= top_height - window_h/2 || top_w >= top_width - window_w/2)
           continue;
        int top_n = (top_h + window_h/2) * top_width + (top_w + window_w/2);
        if (bottom_flags[top_n] > 0){
          
          int windowIndex = 0;
          for (int i = 0; i < top_n; ++i){
            windowIndex += bottom_flags[i]; 
          }
          int top_idx = windowIndex * channels * window_h * window_w +
                         c * window_h * window_w +
                        (h - top_h) * window_w +
                        (w - top_w);
          bottom_diff[index] += top_diff[top_idx];
        }
                
      }
  }
}

template <typename Dtype>
void SlidingWindowIndexLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_flags = bottom[1]->gpu_data();
  const int count = channels_ * bottom_height_ * bottom_width_;
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  SlidingWindowIndexBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, channels_, bottom_height_, bottom_width_, top_height_,
      top_width_, stride_h_, stride_w_, window_h_, window_w_, bottom_diff, bottom_flags, numResults_);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SlidingWindowIndexLayer);


}
