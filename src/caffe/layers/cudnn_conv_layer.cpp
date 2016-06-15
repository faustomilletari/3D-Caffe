#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
  workspaceSizeInBytes = 0;
  workspace = NULL;

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
  }

  // determine absolute kernel size
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  int kernel_size = kernel_shape_data[0];
  for(int axes_idx = 1; axes_idx < this->num_spatial_axes_; axes_idx ++){
  	kernel_size *= kernel_shape_data[axes_idx];
  }

  // Set the indexing parameters.
  weight_offset_ = (this->num_output_ / this->group_)
      * (this->channels_ / this->group_) * kernel_size;
  bias_offset_ = (this->num_output_ / this->group_);

	// Create filter descriptor.
	int kernel_shape[this->num_spatial_axes_+2];
	kernel_shape[0] = this->num_output_ / this->group_;
	kernel_shape[1] = this->channels_ / this->group_;
	for(int i = 0; i < this->num_spatial_axes_; i++){
		kernel_shape[i+2] = kernel_shape_data[i];
	}
	cudnn::createFilterNdDesc<Dtype>(&filter_desc_,
			this->num_spatial_axes_+2,kernel_shape);

	// Create tensor descriptor(s) for data and corresponding convolution(s).
	for (int i = 0; i < bottom.size(); i++) {
		cudnnTensorDescriptor_t bottom_desc;
		cudnn::createTensorDesc<Dtype>(&bottom_desc);
		bottom_descs_.push_back(bottom_desc);
		cudnnTensorDescriptor_t top_desc;
		cudnn::createTensorDesc<Dtype>(&top_desc);
		top_descs_.push_back(top_desc);
		cudnnConvolutionDescriptor_t conv_desc;
		cudnn::createConvolutionDesc<Dtype>(&conv_desc);
		conv_descs_.push_back(conv_desc);
	}

	// Tensor descriptor for bias.
	if (this->bias_term_) {
		cudnn::createTensorDesc<Dtype>(&bias_desc_);
	}
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);

  // input channel, height, width, (depth)ibias_shape
  const int* input_shape_data = this->input_shape_.cpu_data();
  // stride
  const int* stride_data = this->stride_.cpu_data();
  // padding
	const int* pad_data = this->pad_.cpu_data();

	int stride[this->num_spatial_axes_];
	int pad[this->num_spatial_axes_];
	for (int i = 0; i < this->num_spatial_axes_; i++){
		stride[i] = stride_data[i];
		pad[i] = pad_data[i];
	}

	bottom_offset_ = (this->channels_ / this->group_);
	top_offset_    = (this->num_output_ / this->group_);
	for (int i = 0; i < this->num_spatial_axes_; i++){
		bottom_offset_ *= input_shape_data[i+1];
		top_offset_    *= this->output_shape_[i];
	}

	int input_shape[this->num_spatial_axes_ + 2];
	int output_shape[this->num_spatial_axes_ + 2];
	input_shape[0]  = this->num_;
	output_shape[0] = this->num_;
	input_shape[1]  = this->channels_/ this->group_;
	output_shape[1] = this->num_output_/ this->group_;
	for (int i = 0; i < this->num_spatial_axes_; i++){
		input_shape[i+2]  = input_shape_data[i+1];
		output_shape[i+2] = this->output_shape_[i];
	}

	int input_stride[this->num_spatial_axes_ + 2];
	int output_stride[this->num_spatial_axes_ + 2];
	input_stride[this->num_spatial_axes_ + 1]  = 1;
	output_stride[this->num_spatial_axes_ + 1] = 1;
	for(int i = this->num_spatial_axes_; i > 0; i--)
	{
		input_stride[i]  = input_stride[i+1] * input_shape_data[i];
		output_stride[i] = output_stride[i+1] * this->output_shape_[i-1];
	}
	input_stride[0]  = input_stride[1] * this->channels_;
	output_stride[0] = output_stride[1] * this->num_output_;

	for (int i = 0; i < bottom.size(); i++) {
		cudnn::setTensorNdDesc<Dtype>(&bottom_descs_[i],
				this->num_spatial_axes_+2, input_shape, input_stride);
		cudnn::setTensorNdDesc<Dtype>(&top_descs_[i],
				this->num_spatial_axes_+2, output_shape, output_stride);
		cudnn::setConvolutionNdDesc<Dtype>(&conv_descs_[i],
				this->num_spatial_axes_, pad, stride);
	}
	// Tensor descriptor for bias.
	int bias_shape[this->num_spatial_axes_ +2];
	bias_shape[0] = 1;
	bias_shape[1] = this->num_output_/ this->group_;
	for (int i = 0; i< this->num_spatial_axes_; i++){
		bias_shape[i+2] = 1;
	}
	if (this->bias_term_) {
		cudnn::setTensorNdDesc<Dtype>(&bias_desc_,
				this->num_spatial_axes_+2, bias_shape);
	}
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  delete [] stream_;
  delete [] handle_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
