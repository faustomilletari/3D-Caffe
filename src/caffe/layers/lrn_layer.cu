#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LRNFillScale(const int nthreads, const Dtype* in,
    const int num_spatial_axes, const int* shape, const int size, 
		const Dtype alpha_over_size, const Dtype k, Dtype* scale) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // sub index of current index for each spatial dimension 
  	int *sub_index = new int[num_spatial_axes];
  	sub_index[num_spatial_axes-1] = index % shape[num_spatial_axes+1];
  	// spatial size 
  	int step = shape[num_spatial_axes+1];	
  	for (int i = num_spatial_axes-2; i >= 0; i--)
  	{
  		sub_index[i]  = (index / step) % shape[i+2];
  		step *= shape[i+2];
  	} 	
  	// retrieve the local offset
    int n = index / step;
    int offset = n * shape[1];
    for (int i = 0; i < num_spatial_axes; i++){
    	offset = (offset * shape[i+2]) + sub_index[i];
    }    
    delete[] sub_index;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < shape[1]) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // both add and subtract
    while (head < shape[1]) {
      accum_scale += in[head * step] * in[head * step];
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < shape[1] + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      }
      scale[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}


template <typename Dtype>
void LRNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelForward_gpu(bottom, top);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelForward(bottom, top);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

// TODO: check if it would be faster to just put it into the previous kernel.
template <typename Dtype>
__global__ void LRNComputeOutput(const int nthreads, const Dtype* in,
    const Dtype* scale, const Dtype negative_beta, Dtype* out) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, compute scale
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  const int* shape_data = shape_.cpu_data();
  // We will launch one kernel for each pixel location, and have the kernel
  // go through all the channels.
  int n_threads = shape_data[0];
  for (int i = 2; i < num_axes_; i++){
  	n_threads *= shape_data[i];
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNFillScale<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, num_spatial_axes_, shape_.gpu_data(), 
			size_, alpha_ / size_, k_, scale_data);
  CUDA_POST_KERNEL_CHECK;
  n_threads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeOutput<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom_data, scale_data, -beta_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template void LRNLayer<float>::CrossChannelForward_gpu(
    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
template void LRNLayer<double>::CrossChannelForward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);


template <typename Dtype>
void LRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_gpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
__global__ void LRNComputeDiff(const int nthreads, const Dtype* bottom_data,
    const Dtype* top_data, const Dtype* scale, const Dtype* top_diff,
    const int num_spatial_axes, const int* shape, const int size, 
		const Dtype negative_beta, const Dtype cache_ratio, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	// sub index of current index for each spatial dimension 
		int *sub_index = new int[num_spatial_axes];
		sub_index[num_spatial_axes-1] = index % shape[num_spatial_axes+1];
		// spatial size LRNFillScale
		int step = shape[num_spatial_axes+1];	
		for (int i = num_spatial_axes-2; i >= 0; i--)
		{
			sub_index[i]  = (index / step) % shape[i+2];
			step *= shape[i+2];
		} 	
		// retrieve the local offset
		int n = index / step;
		int offset = n * shape[1];
		for (int i = 0; i < num_spatial_axes; i++){
			offset = (offset * shape[i+2]) + sub_index[i];
		}  
    delete[] sub_index;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < shape[1]) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // both add and subtract
    while (head < shape[1]) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
            top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < shape[1] + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff[(head - size) * step] *
            top_data[(head - size) * step] / scale[(head - size) * step];
      }
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const int* shape_data = shape_.cpu_data();
	int n_threads = shape_data[0];
	for (int i = 2; i < num_axes_; i++){
		n_threads *= shape_data[i];
	}
  // NOLINT_NEXT_LINE(whitespace/operators)
  LRNComputeDiff<<<CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS>>>(
      n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
      scale_.gpu_data(), top[0]->gpu_diff(), num_spatial_axes_, 
			shape_.gpu_data(), size_, -beta_, Dtype(2. * alpha_ * beta_ / size_),
      bottom[0]->mutable_gpu_diff());
}
template void LRNLayer<float>::CrossChannelBackward_gpu(
    const vector<Blob<float>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<float>*>& bottom);
template void LRNLayer<double>::CrossChannelBackward_gpu(
    const vector<Blob<double>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom);



INSTANTIATE_LAYER_GPU_FUNCS(LRNLayer);

}  // namespace caffe
