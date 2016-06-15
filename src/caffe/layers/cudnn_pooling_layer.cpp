#ifdef USE_CUDNN
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);

  // stride
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	// stride
	const int* stride_data = this->stride_.cpu_data();
	// padding
	const int* pad_data = this->pad_.cpu_data();

	int kernel_shape[this->num_spatial_axes_];
	int stride[this->num_spatial_axes_];
	int pad[this->num_spatial_axes_];
	for (int i = 0; i < this->num_spatial_axes_; i++){
		kernel_shape[i] = kernel_shape_data[i];
		stride[i] = stride_data[i];
		pad[i] = pad_data[i];
	}

	CUDNN_CHECK(cudnnCreate(&handle_));

	cudnn::createTensorDesc<Dtype>(&bottom_desc_);
	cudnn::createTensorDesc<Dtype>(&top_desc_);
	cudnn::createPoolingNdDesc<Dtype>(&pooling_desc_,
			this->layer_param_.pooling_param().pool(), &mode_,
			this->num_spatial_axes_, kernel_shape,
			pad, stride);
	handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);

  // input channel, height, width, (depth)
	const int* input_shape_data = this->input_shape_.cpu_data();
	int input_shape[this->num_spatial_axes_+2];
	input_shape[0] = bottom[0]->shape(0);
	for(int i=1; i<this->num_spatial_axes_+2; i++){
		input_shape[i] = input_shape_data[i-1];
	}
	// output channel, height, width, (depth)
	const int* pooled_shape_data = this->output_shape_.cpu_data();
	int output_shape[this->num_spatial_axes_+2];
	output_shape[0] = bottom[0]->shape(0);
	output_shape[1] = input_shape_data[0];
	for(int i=2; i<this->num_spatial_axes_+2; i++){
		output_shape[i] = pooled_shape_data[i-2];
	}

	cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, this->num_spatial_axes_+2, input_shape);
	cudnn::setTensorNdDesc<Dtype>(&top_desc_, this->num_spatial_axes_+2, output_shape);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
