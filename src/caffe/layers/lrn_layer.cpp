#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  size_ = this->layer_param_.lrn_param().local_size();
  CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();
  if (this->layer_param_.lrn_param().norm_region() ==
      LRNParameter_NormRegion_WITHIN_CHANNEL) {
    // Set up split_layer_ to use inputs in the numerator and denominator.
    split_top_vec_.clear();
    split_top_vec_.push_back(&product_input_);
    split_top_vec_.push_back(&square_input_);
    LayerParameter split_param;
    split_layer_.reset(new SplitLayer<Dtype>(split_param));
    split_layer_->SetUp(bottom, split_top_vec_);
    // Set up square_layer_ to square the inputs.
    square_bottom_vec_.clear();
    square_top_vec_.clear();
    square_bottom_vec_.push_back(&square_input_);
    square_top_vec_.push_back(&square_output_);
    LayerParameter square_param;
    square_param.mutable_power_param()->set_power(Dtype(2));
    square_layer_.reset(new PowerLayer<Dtype>(square_param));
    square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
    // Set up pool_layer_ to sum over square neighborhoods of the input.
    pool_top_vec_.clear();
    pool_top_vec_.push_back(&pool_output_);
    LayerParameter pool_param;
    pool_param.mutable_pooling_param()->set_pool(
        PoolingParameter_PoolMethod_AVE);
    pool_param.mutable_pooling_param()->add_pad(pre_pad_);
    pool_param.mutable_pooling_param()->add_kernel_size(size_);
    pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
    pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
    // Set up power_layer_ to compute (1 + alpha_ s)^-beta_, where s is
    // the sum of a squared neighborhood divided by the kernel size
    // (the output of pool_layer_).
    power_top_vec_.clear();
    power_top_vec_.push_back(&power_output_);
    LayerParameter power_param;
    power_param.mutable_power_param()->set_power(-beta_);
    power_param.mutable_power_param()->set_scale(alpha_);
    power_param.mutable_power_param()->set_shift(Dtype(1));
    power_layer_.reset(new PowerLayer<Dtype>(power_param));
    power_layer_->SetUp(pool_top_vec_, power_top_vec_);
    // Set up a product_layer_ to compute outputs by multiplying inputs by the
    // inverse demoninator computed by the power layer.
    product_bottom_vec_.clear();
    product_bottom_vec_.push_back(&product_input_);
    product_bottom_vec_.push_back(&power_output_);
    LayerParameter product_param;
    EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
    eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
    product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
    product_layer_->SetUp(product_bottom_vec_, top);
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	num_axes_ = bottom[0]->num_axes();
	num_spatial_axes_ = num_axes_ - 2;
	vector<int> dim_blob_shape(1, num_axes_);
	shape_.Reshape(dim_blob_shape);
	int* shape_data = shape_.mutable_cpu_data();
  for (int i = 0; i< num_axes_; i++){
  	shape_data[i] = bottom[0]->shape(i);
  }
	switch (this->layer_param_.lrn_param().norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			top[0]->ReshapeLike(*bottom[0]);
			scale_.ReshapeLike(*bottom[0]);
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
			split_layer_->Reshape(bottom, split_top_vec_);
			square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
			pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
			power_layer_->Reshape(pool_top_vec_, power_top_vec_);
			product_layer_->Reshape(product_bottom_vec_, top);
			break;
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			CrossChannelForward_cpu(bottom, top);
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
			WithinChannelForward(bottom, top);
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* scale_data = scale_.mutable_cpu_data();
	// start with the constant value
	for (int i = 0; i < scale_.count(); ++i) {
		scale_data[i] = k_;
	}

	const int* shape_data = shape_.cpu_data();
	vector<int> padded_square_shape(num_axes_);
	padded_square_shape[0] = 1;
	padded_square_shape[1] = shape_data[1] + size_ - 1;
	for (int i = 2; i < num_axes_; i++){
		padded_square_shape[i] = shape_data[i];
	}
	Blob<Dtype> padded_square(padded_square_shape);
	Dtype* padded_square_data = padded_square.mutable_cpu_data();
	caffe_set(padded_square.count(), Dtype(0), padded_square_data);

	Dtype alpha_over_size = alpha_ / size_;
	int spatial_size = 1;
	for (int i = 2; i < num_axes_; i++){
		spatial_size *= shape_data[i];
	}

	// auxiliary vectors to determine offset
	vector<int> offset_vector(num_axes_,0);
	vector<int> offset_vector2(num_axes_,0);
	vector<int> padded_offset_vector(num_axes_,0);

	// go through the images
	for (int n = 0; n < shape_data[0]; ++n) {
		// compute the padded square
		offset_vector[0] = n;
		offset_vector[1] = 0;
		offset_vector2[0] = n;
		padded_offset_vector[1] = pre_pad_;
		caffe_sqr(shape_data[1] * spatial_size,
				bottom_data + bottom[0]->offset(offset_vector), 									// offset(n,0)
				padded_square_data + padded_square.offset(padded_offset_vector)); // offset(0,pre_pad_)
		// Create the first channel scale
		for (int c = 0; c < size_; ++c) {
			padded_offset_vector[1] = c;
			caffe_axpy<Dtype>(spatial_size, alpha_over_size,
					padded_square_data + padded_square.offset(padded_offset_vector),// offset(0,c)
					scale_data + scale_.offset(offset_vector)); 										// offset(n,0)
		}
		for (int c = 1; c < shape_data[1]; ++c) {
			offset_vector[1]  = c;

			// copy previous scale
			offset_vector2[1] = c -1;
			caffe_copy<Dtype>(spatial_size,
					scale_data + scale_.offset(offset_vector2),  // offset(n, c - 1)
					scale_data + scale_.offset(offset_vector));	 // offset(n, c)

			// add head
			padded_offset_vector[1]  = c + size_ - 1;
			caffe_axpy<Dtype>(spatial_size, alpha_over_size,
					padded_square_data + padded_square.offset(padded_offset_vector), // offset(0, c + size_ - 1)
					scale_data + scale_.offset(offset_vector));										   // offset(n, c)

			// subtract tail
			padded_offset_vector[1]  = c - 1;
			caffe_axpy<Dtype>(spatial_size, -alpha_over_size,
					padded_square_data + padded_square.offset(padded_offset_vector), // offset(0, c - 1)
					scale_data + scale_.offset(offset_vector));										   // offset(n, c)
		}
	}

	// In the end, compute output
	caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
	caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  split_layer_->Forward(bottom, split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, square_top_vec_);
  pool_layer_->Forward(square_top_vec_, pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* scale_data = scale_.cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int* shape_data = shape_.cpu_data();

	vector<int> blob_shape(num_axes_);
	blob_shape[0] = 1;
	blob_shape[1] = shape_data[1] + size_ - 1;
	for (int i = 2; i < num_axes_; i++){
		blob_shape[i] = shape_data[i];
	}
	Blob<Dtype> padded_ratio(blob_shape);

	blob_shape[1] = 1;
	Blob<Dtype> accum_ratio(blob_shape);

	Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
	Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();

	// We hack a little bit by using the diff() to store an additional result
	Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
	caffe_set(padded_ratio.count(), Dtype(0), padded_ratio_data);
	Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

	caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
	caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

	// compute spatial size
	int spatial_size = 1;
	for (int i = 2; i < num_axes_; i++){
		spatial_size *= shape_data[i];
	}

	// auxiliarry vector
	vector<int> offset_vector(num_axes_,0);

	// go through individual data
	int inverse_pre_pad = size_ - (size_ + 1) / 2;
	for (int n = 0; n < shape_data[0]; ++n) {
		offset_vector[0] = n;
		offset_vector[1] = 0;
		int block_offset = scale_.offset(offset_vector); // offset(n, 0)

		// first, compute diff_i * y_i / s_i
		offset_vector[0] = 0;
		offset_vector[1] = inverse_pre_pad;
		caffe_mul<Dtype>(shape_data[1] * spatial_size,
				top_diff + block_offset, top_data + block_offset,
				padded_ratio_data + padded_ratio.offset(offset_vector));		// offset(0, inverse_pre_pad)
		caffe_div<Dtype>(shape_data[1] * spatial_size,
				padded_ratio_data + padded_ratio.offset(offset_vector), 		// offset(0, inverse_pre_pad)
				scale_data + block_offset,
				padded_ratio_data + padded_ratio.offset(offset_vector));		// offset(0, inverse_pre_pad)

		// Now, compute the accumulated ratios and the bottom diff
		caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
		for (int c = 0; c < size_ - 1; ++c) {
			offset_vector[1] = c;
			caffe_axpy<Dtype>(spatial_size, 1.,
					padded_ratio_data + padded_ratio.offset(offset_vector), accum_ratio_data); // offset(0,c)
		}
		for (int c = 0; c < shape_data[1]; ++c) {
			offset_vector[0] = 0;
			offset_vector[1] = c + size_ - 1;
			caffe_axpy<Dtype>(spatial_size, 1.,
					padded_ratio_data + padded_ratio.offset(offset_vector), // offset(0, c + size_ - 1)
					accum_ratio_data);
			// compute bottom diff
			offset_vector[0] = n;
			offset_vector[1] = c;
			caffe_mul<Dtype>(spatial_size,
					bottom_data + top[0]->offset(offset_vector),
					accum_ratio_data, accum_ratio_times_bottom);															// offset(n,c)
			caffe_axpy<Dtype>(spatial_size, -cache_ratio_value,
					accum_ratio_times_bottom, bottom_diff + top[0]->offset(offset_vector));		// offset(n,c)
			offset_vector[0] = 0;
			caffe_axpy<Dtype>(spatial_size, -1.,
					padded_ratio_data + padded_ratio.offset(offset_vector), accum_ratio_data); // offset(0,c)
		}
	}
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    vector<bool> product_propagate_down(2, true);
    product_layer_->Backward(top, product_propagate_down, product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, propagate_down, pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, propagate_down, square_top_vec_);
    square_layer_->Backward(square_top_vec_, propagate_down,
                            square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LRNLayer);
STUB_GPU_FORWARD(LRNLayer, CrossChannelForward);
STUB_GPU_BACKWARD(LRNLayer, CrossChannelBackward);
#endif

INSTANTIATE_CLASS(LRNLayer);
REGISTER_LAYER_CLASS(LRN);

}  // namespace caffe
