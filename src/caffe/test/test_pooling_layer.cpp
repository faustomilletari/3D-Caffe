#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
    pooling_param->add_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
      for (int i = 0; i < 8 * num * channels; i += 8) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(3);
    pooling_param->set_kernel_w(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 4);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4    17    17]
        // [ 8    21    21    17    17]
        // [13    27    27    17    17]
        // [32    32    27    35    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 34);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(2);
    pooling_param->set_kernel_w(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    26]
    // [32    32    27    27]
    // [33    33    33    27]
    // [34    34    34    17]
    // [36    36    34    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4     4]
        // [ 8     8    17    17]
        // [21    21    21    17]
        // [27    27    27    22]
        // [32    32    27    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 21);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for 2x2 pooling with padding and stride layer
	void TestForwardStridePad() {
		LayerParameter layer_param;
		PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
		pooling_param->clear_kernel_size();
		pooling_param->add_kernel_size(2);
		pooling_param->clear_pad();
		pooling_param->add_pad(1);
		pooling_param->clear_stride();
		pooling_param->add_stride(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 4, 5);
		// Input: 2x 2 channels of:
		//     [1 2 5 2 3]
		//     [9 4 1 4 8]
		//     [1 2 5 2 3]
		//     [4 3 1 2 1]
		for (int i = 0; i < 20 * num * channels; i += 20) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 1;
			blob_bottom_->mutable_cpu_data()[i +  1] = 2;
			blob_bottom_->mutable_cpu_data()[i +  2] = 5;
			blob_bottom_->mutable_cpu_data()[i +  3] = 2;
			blob_bottom_->mutable_cpu_data()[i +  4] = 3;
			blob_bottom_->mutable_cpu_data()[i +  5] = 9;
			blob_bottom_->mutable_cpu_data()[i +  6] = 4;
			blob_bottom_->mutable_cpu_data()[i +  7] = 1;
			blob_bottom_->mutable_cpu_data()[i +  8] = 4;
			blob_bottom_->mutable_cpu_data()[i +  9] = 8;
			blob_bottom_->mutable_cpu_data()[i + 10] = 1;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2;
			blob_bottom_->mutable_cpu_data()[i + 12] = 5;
			blob_bottom_->mutable_cpu_data()[i + 13] = 2;
			blob_bottom_->mutable_cpu_data()[i + 14] = 3;
			blob_bottom_->mutable_cpu_data()[i + 15] = 4;
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 1;
			blob_bottom_->mutable_cpu_data()[i + 18] = 2;
			blob_bottom_->mutable_cpu_data()[i + 19] = 1;
		}
		PoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 3);
		EXPECT_EQ(blob_top_->width(), 3);
		if (blob_top_vec_.size() > 1) {
			EXPECT_EQ(blob_top_mask_->num(), num);
			EXPECT_EQ(blob_top_mask_->channels(), channels);
			EXPECT_EQ(blob_top_mask_->height(), 3);
			EXPECT_EQ(blob_top_mask_->width(), 3);
		}
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		// Expected output: 2x2 channels of:
		//     [1 5 3]
		//     [9 5 8]
		//     [4 3 2]
		for (int i = 0; i < 9 * num * channels; i += 9) {
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 2);
		}
		if (blob_top_vec_.size() > 1) {
			// Expected mask output: 2x 2 channels of:
			//     [0  2  4]
			//     [5  12 9]
			//     [15 16 18]
			for (int i = 0; i < 9 * num * channels; i += 9) {
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  0);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  4);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  5);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4], 12);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  9);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 15);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7], 16);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8], 18);
			}
		}
	}
};

TYPED_TEST_CASE(PoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(PoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(PoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(PoolingLayerTest, TestSetupGlobalPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

/*
TYPED_TEST(PoolingLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(PoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

TYPED_TEST(PoolingLayerTest, TestForwardMaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

TYPED_TEST(PoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->clear_pad();
      pooling_param->add_pad(1);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0] = 1;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 4;
  this->blob_bottom_->mutable_cpu_data()[3] = 2;
  this->blob_bottom_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_->mutable_cpu_data()[5] = 2;
  this->blob_bottom_->mutable_cpu_data()[6] = 4;
  this->blob_bottom_->mutable_cpu_data()[7] = 2;
  this->blob_bottom_->mutable_cpu_data()[8] = 1;
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 1 4 4 ]
  //     [ 4 4 4 ]
  //     [ 4 4 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 1, epsilon);
}

TYPED_TEST(PoolingLayerTest, TestGradientMaxTopMask) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      this->blob_top_vec_.pop_back();
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestForwardAve) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(1);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
}

TYPED_TEST(PoolingLayerTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(PoolingLayerTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->clear_pad();
      pooling_param->add_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      PoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

/******************************************************************
 *  3D Pooling Layer Tests
 ******************************************************************/

template <typename TypeParam>
class PoolingLayer3DTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PoolingLayer3DTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    int blobShape[5]={2,3,6,5,4};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PoolingLayer3DTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x2x2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
    pooling_param->clear_pad();
    pooling_param->clear_stride();
    pooling_param->add_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    int blobShape[5]={num,channels,3,5,4};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // Input: 2x2 channels of:
    //     [1 2 5 2 3] [11 5 5 2 7] [1 0 3 2 3] [3 1 2 2 0]
    //     [9 4 1 4 8] [1 8 5 1 1]  [0 2 4 2 1] [2 0 1 1 4]
    //     [1 2 5 2 3] [2 2 2 1 1]  [1 2 4 1 3] [1 2 1 2 6]
    for (int i = 0; i < 60 * num * channels; i += 60) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 11;
      blob_bottom_->mutable_cpu_data()[i +  2] = 1;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3; //
      blob_bottom_->mutable_cpu_data()[i +  4] = 2;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 0;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1; //
      blob_bottom_->mutable_cpu_data()[i +  8] = 5;
      blob_bottom_->mutable_cpu_data()[i +  9] = 5;
      blob_bottom_->mutable_cpu_data()[i + 10] = 3;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2; //
      blob_bottom_->mutable_cpu_data()[i + 12] = 2;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 7;
			blob_bottom_->mutable_cpu_data()[i + 18] = 3;
			blob_bottom_->mutable_cpu_data()[i + 19] = 0; //row 1
			blob_bottom_->mutable_cpu_data()[i + 20] = 9;
			blob_bottom_->mutable_cpu_data()[i + 21] = 1;
			blob_bottom_->mutable_cpu_data()[i + 22] = 0;
			blob_bottom_->mutable_cpu_data()[i + 23] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 24] = 4;
			blob_bottom_->mutable_cpu_data()[i + 25] = 8;
			blob_bottom_->mutable_cpu_data()[i + 26] = 2;
			blob_bottom_->mutable_cpu_data()[i + 27] = 0; //
			blob_bottom_->mutable_cpu_data()[i + 28] = 1;
			blob_bottom_->mutable_cpu_data()[i + 29] = 5;
			blob_bottom_->mutable_cpu_data()[i + 30] = 4;
			blob_bottom_->mutable_cpu_data()[i + 31] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 32] = 4;
			blob_bottom_->mutable_cpu_data()[i + 33] = 1;
			blob_bottom_->mutable_cpu_data()[i + 34] = 2;
			blob_bottom_->mutable_cpu_data()[i + 35] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 36] = 8;
			blob_bottom_->mutable_cpu_data()[i + 37] = 1;
			blob_bottom_->mutable_cpu_data()[i + 38] = 1;
			blob_bottom_->mutable_cpu_data()[i + 39] = 4; //row 2
			blob_bottom_->mutable_cpu_data()[i + 40] = 1;
			blob_bottom_->mutable_cpu_data()[i + 41] = 2;
			blob_bottom_->mutable_cpu_data()[i + 42] = 1;
			blob_bottom_->mutable_cpu_data()[i + 43] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 44] = 2;
			blob_bottom_->mutable_cpu_data()[i + 45] = 2;
			blob_bottom_->mutable_cpu_data()[i + 46] = 2;
			blob_bottom_->mutable_cpu_data()[i + 47] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 48] = 5;
			blob_bottom_->mutable_cpu_data()[i + 49] = 2;
			blob_bottom_->mutable_cpu_data()[i + 50] = 4;
			blob_bottom_->mutable_cpu_data()[i + 51] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 52] = 2;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 1;
			blob_bottom_->mutable_cpu_data()[i + 55] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 56] = 3;
			blob_bottom_->mutable_cpu_data()[i + 57] = 1;
			blob_bottom_->mutable_cpu_data()[i + 58] = 3;
			blob_bottom_->mutable_cpu_data()[i + 59] = 6;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 2);
    EXPECT_EQ(blob_top_->shape(3), 4);
    EXPECT_EQ(blob_top_->shape(4), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 2);
      EXPECT_EQ(blob_top_mask_->shape(3), 4);
      EXPECT_EQ(blob_top_mask_->shape(4), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x2 channels of:
    //     [11 8 5 8] [11 8 5 7] [3 4 4 4]
    //     [9 8 5 8]  [8 8 5 3]  [2 4 4 6]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0],  11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1],  11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2],  3); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 3],  8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4],  8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5],  4); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 6],  5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7],  5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8],  4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 9],  8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 7);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 4); // row 1
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 16], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 17], 4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 18], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 19], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 20], 4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 6);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
    	//     [1  25  8 36] [1  25 9  17] [3  30 30 39]
    	//     [20 25 29 36] [25 25 29 58] [23 30 30 59]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  3); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6],  8);
			  EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8],  30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 9],  36);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 17);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 39); // row 1
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 20);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 23); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 29);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 29);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 36);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 58);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 59);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->clear_stride();
    pooling_param->add_kernel_size(4);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    int blobShape[5]={num,channels,5,5,5};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // Input: 2x2 channels of:
    // [35     1     6    26    19] [4      25     2     17    19] [29     3      15    14    12]
    // [ 3    32     7    21    23] [31     4      9     15    19] [9      17     11    24    24]
    // [31     9     2    22    27] [7      17     11    22    19] [23     7      3     20    19]
    // [ 8    28    33    17    10] [27     8      22    18    19] [2      22     29    13    6]
    // [30     5    34    12    14] [3      31     33    21    19] [37     4      40    11    8]
    //
    // [29     3     16     6    9] [14     22     32    37    9]
		// [ 6    21     17     2    3] [3      13     19    5     8]
		// [ 1    19     12    12    7] [17     5      16    2     6]
		// [28     8     31    17    1] [7      28      2    8    22]
		// [ 3    25     4     32    4] [33     26     3     26   31]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 125 * num * channels; i += 125) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 4;
      blob_bottom_->mutable_cpu_data()[i +  2] = 29;
      blob_bottom_->mutable_cpu_data()[i +  3] = 29;
      blob_bottom_->mutable_cpu_data()[i +  4] = 14;
      blob_bottom_->mutable_cpu_data()[i +  5] = 1;
      blob_bottom_->mutable_cpu_data()[i +  6] = 25;
      blob_bottom_->mutable_cpu_data()[i +  7] = 3;
      blob_bottom_->mutable_cpu_data()[i +  8] = 3;
      blob_bottom_->mutable_cpu_data()[i +  9] = 22;
      blob_bottom_->mutable_cpu_data()[i + 10] = 6;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 15;
      blob_bottom_->mutable_cpu_data()[i + 13] = 16;
      blob_bottom_->mutable_cpu_data()[i + 14] = 32;
      blob_bottom_->mutable_cpu_data()[i + 15] = 26;
      blob_bottom_->mutable_cpu_data()[i + 16] = 17;
      blob_bottom_->mutable_cpu_data()[i + 17] = 14;
      blob_bottom_->mutable_cpu_data()[i + 18] = 6;
      blob_bottom_->mutable_cpu_data()[i + 19] = 37;
      blob_bottom_->mutable_cpu_data()[i + 20] = 19;
      blob_bottom_->mutable_cpu_data()[i + 21] = 19;
      blob_bottom_->mutable_cpu_data()[i + 22] = 12;
      blob_bottom_->mutable_cpu_data()[i + 23] = 9;
      blob_bottom_->mutable_cpu_data()[i + 24] = 9;
      blob_bottom_->mutable_cpu_data()[i + 25] = 3; //
      blob_bottom_->mutable_cpu_data()[i + 26] = 31;
      blob_bottom_->mutable_cpu_data()[i + 27] = 9;
      blob_bottom_->mutable_cpu_data()[i + 28] = 6;
      blob_bottom_->mutable_cpu_data()[i + 29] = 3;
      blob_bottom_->mutable_cpu_data()[i + 30] = 32;
      blob_bottom_->mutable_cpu_data()[i + 31] = 4;
      blob_bottom_->mutable_cpu_data()[i + 32] = 17;
      blob_bottom_->mutable_cpu_data()[i + 33] = 21;
      blob_bottom_->mutable_cpu_data()[i + 34] = 13;
      blob_bottom_->mutable_cpu_data()[i + 35] = 7;
			blob_bottom_->mutable_cpu_data()[i + 36] = 9;
			blob_bottom_->mutable_cpu_data()[i + 37] = 11;
			blob_bottom_->mutable_cpu_data()[i + 38] = 17;
			blob_bottom_->mutable_cpu_data()[i + 39] = 19;
			blob_bottom_->mutable_cpu_data()[i + 40] = 21;
			blob_bottom_->mutable_cpu_data()[i + 41] = 15;
			blob_bottom_->mutable_cpu_data()[i + 42] = 24;
			blob_bottom_->mutable_cpu_data()[i + 43] = 2;
			blob_bottom_->mutable_cpu_data()[i + 44] = 5;
			blob_bottom_->mutable_cpu_data()[i + 45] = 23;
			blob_bottom_->mutable_cpu_data()[i + 46] = 19;
			blob_bottom_->mutable_cpu_data()[i + 47] = 24;
			blob_bottom_->mutable_cpu_data()[i + 48] = 3;
			blob_bottom_->mutable_cpu_data()[i + 49] = 8;
			blob_bottom_->mutable_cpu_data()[i + 50] = 31; //
			blob_bottom_->mutable_cpu_data()[i + 51] = 7;
			blob_bottom_->mutable_cpu_data()[i + 52] = 23;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 17;
			blob_bottom_->mutable_cpu_data()[i + 55] = 9;
			blob_bottom_->mutable_cpu_data()[i + 56] = 17;
			blob_bottom_->mutable_cpu_data()[i + 57] = 7;
			blob_bottom_->mutable_cpu_data()[i + 58] = 19;
			blob_bottom_->mutable_cpu_data()[i + 59] = 5;
			blob_bottom_->mutable_cpu_data()[i + 60] = 2;
			blob_bottom_->mutable_cpu_data()[i + 61] = 11;
			blob_bottom_->mutable_cpu_data()[i + 62] = 3;
			blob_bottom_->mutable_cpu_data()[i + 63] = 12;
			blob_bottom_->mutable_cpu_data()[i + 64] = 16;
			blob_bottom_->mutable_cpu_data()[i + 65] = 22;
			blob_bottom_->mutable_cpu_data()[i + 66] = 22;
			blob_bottom_->mutable_cpu_data()[i + 67] = 20;
			blob_bottom_->mutable_cpu_data()[i + 68] = 12;
			blob_bottom_->mutable_cpu_data()[i + 69] = 2;
			blob_bottom_->mutable_cpu_data()[i + 70] = 27;
			blob_bottom_->mutable_cpu_data()[i + 71] = 19;
			blob_bottom_->mutable_cpu_data()[i + 72] = 19;
			blob_bottom_->mutable_cpu_data()[i + 73] = 7;
			blob_bottom_->mutable_cpu_data()[i + 74] = 6;
			blob_bottom_->mutable_cpu_data()[i + 75] = 8; //
			blob_bottom_->mutable_cpu_data()[i + 76] = 27;
			blob_bottom_->mutable_cpu_data()[i + 77] = 2;
			blob_bottom_->mutable_cpu_data()[i + 78] = 28;
			blob_bottom_->mutable_cpu_data()[i + 79] = 28;
			blob_bottom_->mutable_cpu_data()[i + 80] = 28;
			blob_bottom_->mutable_cpu_data()[i + 81] = 8;
			blob_bottom_->mutable_cpu_data()[i + 82] = 22;
			blob_bottom_->mutable_cpu_data()[i + 83] = 8;
			blob_bottom_->mutable_cpu_data()[i + 84] = 28;
			blob_bottom_->mutable_cpu_data()[i + 85] = 33;
			blob_bottom_->mutable_cpu_data()[i + 86] = 22;
			blob_bottom_->mutable_cpu_data()[i + 87] = 29;
			blob_bottom_->mutable_cpu_data()[i + 88] = 31;
			blob_bottom_->mutable_cpu_data()[i + 89] = 2;
			blob_bottom_->mutable_cpu_data()[i + 90] = 17;
			blob_bottom_->mutable_cpu_data()[i + 91] = 18;
			blob_bottom_->mutable_cpu_data()[i + 92] = 13;
			blob_bottom_->mutable_cpu_data()[i + 93] = 17;
			blob_bottom_->mutable_cpu_data()[i + 94] = 8;
			blob_bottom_->mutable_cpu_data()[i + 95] = 10;
			blob_bottom_->mutable_cpu_data()[i + 96] = 19;
			blob_bottom_->mutable_cpu_data()[i + 97] = 6;
			blob_bottom_->mutable_cpu_data()[i + 98] = 1;
			blob_bottom_->mutable_cpu_data()[i + 99] = 22;
			blob_bottom_->mutable_cpu_data()[i + 100] = 30; //
			blob_bottom_->mutable_cpu_data()[i + 101] = 3;
			blob_bottom_->mutable_cpu_data()[i + 102] = 37;
			blob_bottom_->mutable_cpu_data()[i + 103] = 3;
			blob_bottom_->mutable_cpu_data()[i + 104] = 33;
			blob_bottom_->mutable_cpu_data()[i + 105] = 5;
			blob_bottom_->mutable_cpu_data()[i + 106] = 31;
			blob_bottom_->mutable_cpu_data()[i + 107] = 4;
			blob_bottom_->mutable_cpu_data()[i + 108] = 25;
			blob_bottom_->mutable_cpu_data()[i + 109] = 26;
			blob_bottom_->mutable_cpu_data()[i + 110] = 34;
			blob_bottom_->mutable_cpu_data()[i + 111] = 33;
			blob_bottom_->mutable_cpu_data()[i + 112] = 40;
			blob_bottom_->mutable_cpu_data()[i + 113] = 4;
			blob_bottom_->mutable_cpu_data()[i + 114] = 3;
			blob_bottom_->mutable_cpu_data()[i + 115] = 12;
			blob_bottom_->mutable_cpu_data()[i + 116] = 21;
			blob_bottom_->mutable_cpu_data()[i + 117] = 11;
			blob_bottom_->mutable_cpu_data()[i + 118] = 32;
			blob_bottom_->mutable_cpu_data()[i + 119] = 26;
			blob_bottom_->mutable_cpu_data()[i + 120] = 14;
			blob_bottom_->mutable_cpu_data()[i + 121] = 19;
			blob_bottom_->mutable_cpu_data()[i + 122] = 8;
			blob_bottom_->mutable_cpu_data()[i + 123] = 4;
			blob_bottom_->mutable_cpu_data()[i + 124] = 31;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 2);
    EXPECT_EQ(blob_top_->shape(3), 4);
    EXPECT_EQ(blob_top_->shape(4), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 2);
      EXPECT_EQ(blob_top_mask_->shape(3), 4);
      EXPECT_EQ(blob_top_mask_->shape(4), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x2 channels of:
    // [35    33    33    27] [31    31    31    24] [29    32    37    37]
    // [37    40    40    27] [37    40    40    32] [37    40    40    32]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 37); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 27);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 32);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 32);
    }
    if (blob_top_vec_.size() > 1) {
      // [0     85    85    70] [26    88    88    42] [ 2    14    19    19]
  		// [102  112    112   70] [102  112   112   118] [102   112   112  118]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5], 14);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 70);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 42);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 70);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 118);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 118);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->clear_stride();
		pooling_param->add_kernel_size(3);
		pooling_param->add_kernel_size(4);
		pooling_param->add_kernel_size(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		int blobShape[5]={num,channels,5,5,5};
		blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
		// Input: 2x2 channels of:
		// [35     1     6    26    19] [4      25     2     17    19] [29     3      15    14    12]
		// [ 3    32     7    21    23] [31     4      9     15    19] [9      17     11    24    24]
		// [31     9     2    22    27] [7      17     11    22    19] [23     7      3     20    19]
		// [ 8    28    33    17    10] [27     8      22    18    19] [2      22     29    13    6]
		// [30     5    34    12    14] [3      31     33    21    19] [37     4      40    11    8]
		//
		// [29     3     16     6    9] [14     22     32    37    9]
		// [ 6    21     17     2    3] [3      13     19    5     8]
		// [ 1    19     12    12    7] [17     5      16    2     6]
		// [28     8     31    17    1] [7      28      2    8    22]
		// [ 3    25     4     32    4] [33     26     3     26   31]
		// (this is generated by magic(6) in MATLAB)
		for (int i = 0; i < 125 * num * channels; i += 125) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 35;
			blob_bottom_->mutable_cpu_data()[i +  1] = 4;
			blob_bottom_->mutable_cpu_data()[i +  2] = 29;
			blob_bottom_->mutable_cpu_data()[i +  3] = 29;
			blob_bottom_->mutable_cpu_data()[i +  4] = 14;
			blob_bottom_->mutable_cpu_data()[i +  5] = 1;
			blob_bottom_->mutable_cpu_data()[i +  6] = 25;
			blob_bottom_->mutable_cpu_data()[i +  7] = 3;
			blob_bottom_->mutable_cpu_data()[i +  8] = 3;
			blob_bottom_->mutable_cpu_data()[i +  9] = 22;
			blob_bottom_->mutable_cpu_data()[i + 10] = 6;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2;
			blob_bottom_->mutable_cpu_data()[i + 12] = 15;
			blob_bottom_->mutable_cpu_data()[i + 13] = 16;
			blob_bottom_->mutable_cpu_data()[i + 14] = 32;
			blob_bottom_->mutable_cpu_data()[i + 15] = 26;
			blob_bottom_->mutable_cpu_data()[i + 16] = 17;
			blob_bottom_->mutable_cpu_data()[i + 17] = 14;
			blob_bottom_->mutable_cpu_data()[i + 18] = 6;
			blob_bottom_->mutable_cpu_data()[i + 19] = 37;
			blob_bottom_->mutable_cpu_data()[i + 20] = 19;
			blob_bottom_->mutable_cpu_data()[i + 21] = 19;
			blob_bottom_->mutable_cpu_data()[i + 22] = 12;
			blob_bottom_->mutable_cpu_data()[i + 23] = 9;
			blob_bottom_->mutable_cpu_data()[i + 24] = 9;
			blob_bottom_->mutable_cpu_data()[i + 25] = 3; //
			blob_bottom_->mutable_cpu_data()[i + 26] = 31;
			blob_bottom_->mutable_cpu_data()[i + 27] = 9;
			blob_bottom_->mutable_cpu_data()[i + 28] = 6;
			blob_bottom_->mutable_cpu_data()[i + 29] = 3;
			blob_bottom_->mutable_cpu_data()[i + 30] = 32;
			blob_bottom_->mutable_cpu_data()[i + 31] = 4;
			blob_bottom_->mutable_cpu_data()[i + 32] = 17;
			blob_bottom_->mutable_cpu_data()[i + 33] = 21;
			blob_bottom_->mutable_cpu_data()[i + 34] = 13;
			blob_bottom_->mutable_cpu_data()[i + 35] = 7;
			blob_bottom_->mutable_cpu_data()[i + 36] = 9;
			blob_bottom_->mutable_cpu_data()[i + 37] = 11;
			blob_bottom_->mutable_cpu_data()[i + 38] = 17;
			blob_bottom_->mutable_cpu_data()[i + 39] = 19;
			blob_bottom_->mutable_cpu_data()[i + 40] = 21;
			blob_bottom_->mutable_cpu_data()[i + 41] = 15;
			blob_bottom_->mutable_cpu_data()[i + 42] = 24;
			blob_bottom_->mutable_cpu_data()[i + 43] = 2;
			blob_bottom_->mutable_cpu_data()[i + 44] = 5;
			blob_bottom_->mutable_cpu_data()[i + 45] = 23;
			blob_bottom_->mutable_cpu_data()[i + 46] = 19;
			blob_bottom_->mutable_cpu_data()[i + 47] = 24;
			blob_bottom_->mutable_cpu_data()[i + 48] = 3;
			blob_bottom_->mutable_cpu_data()[i + 49] = 8;
			blob_bottom_->mutable_cpu_data()[i + 50] = 31; //
			blob_bottom_->mutable_cpu_data()[i + 51] = 7;
			blob_bottom_->mutable_cpu_data()[i + 52] = 23;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 17;
			blob_bottom_->mutable_cpu_data()[i + 55] = 9;
			blob_bottom_->mutable_cpu_data()[i + 56] = 17;
			blob_bottom_->mutable_cpu_data()[i + 57] = 7;
			blob_bottom_->mutable_cpu_data()[i + 58] = 19;
			blob_bottom_->mutable_cpu_data()[i + 59] = 5;
			blob_bottom_->mutable_cpu_data()[i + 60] = 2;
			blob_bottom_->mutable_cpu_data()[i + 61] = 11;
			blob_bottom_->mutable_cpu_data()[i + 62] = 3;
			blob_bottom_->mutable_cpu_data()[i + 63] = 12;
			blob_bottom_->mutable_cpu_data()[i + 64] = 16;
			blob_bottom_->mutable_cpu_data()[i + 65] = 22;
			blob_bottom_->mutable_cpu_data()[i + 66] = 22;
			blob_bottom_->mutable_cpu_data()[i + 67] = 20;
			blob_bottom_->mutable_cpu_data()[i + 68] = 12;
			blob_bottom_->mutable_cpu_data()[i + 69] = 2;
			blob_bottom_->mutable_cpu_data()[i + 70] = 27;
			blob_bottom_->mutable_cpu_data()[i + 71] = 19;
			blob_bottom_->mutable_cpu_data()[i + 72] = 19;
			blob_bottom_->mutable_cpu_data()[i + 73] = 7;
			blob_bottom_->mutable_cpu_data()[i + 74] = 6;
			blob_bottom_->mutable_cpu_data()[i + 75] = 8; //
			blob_bottom_->mutable_cpu_data()[i + 76] = 27;
			blob_bottom_->mutable_cpu_data()[i + 77] = 2;
			blob_bottom_->mutable_cpu_data()[i + 78] = 28;
			blob_bottom_->mutable_cpu_data()[i + 79] = 28;
			blob_bottom_->mutable_cpu_data()[i + 80] = 28;
			blob_bottom_->mutable_cpu_data()[i + 81] = 8;
			blob_bottom_->mutable_cpu_data()[i + 82] = 22;
			blob_bottom_->mutable_cpu_data()[i + 83] = 8;
			blob_bottom_->mutable_cpu_data()[i + 84] = 28;
			blob_bottom_->mutable_cpu_data()[i + 85] = 33;
			blob_bottom_->mutable_cpu_data()[i + 86] = 22;
			blob_bottom_->mutable_cpu_data()[i + 87] = 29;
			blob_bottom_->mutable_cpu_data()[i + 88] = 31;
			blob_bottom_->mutable_cpu_data()[i + 89] = 2;
			blob_bottom_->mutable_cpu_data()[i + 90] = 17;
			blob_bottom_->mutable_cpu_data()[i + 91] = 18;
			blob_bottom_->mutable_cpu_data()[i + 92] = 13;
			blob_bottom_->mutable_cpu_data()[i + 93] = 17;
			blob_bottom_->mutable_cpu_data()[i + 94] = 8;
			blob_bottom_->mutable_cpu_data()[i + 95] = 10;
			blob_bottom_->mutable_cpu_data()[i + 96] = 19;
			blob_bottom_->mutable_cpu_data()[i + 97] = 6;
			blob_bottom_->mutable_cpu_data()[i + 98] = 1;
			blob_bottom_->mutable_cpu_data()[i + 99] = 22;
			blob_bottom_->mutable_cpu_data()[i + 100] = 30; //
			blob_bottom_->mutable_cpu_data()[i + 101] = 3;
			blob_bottom_->mutable_cpu_data()[i + 102] = 37;
			blob_bottom_->mutable_cpu_data()[i + 103] = 3;
			blob_bottom_->mutable_cpu_data()[i + 104] = 33;
			blob_bottom_->mutable_cpu_data()[i + 105] = 5;
			blob_bottom_->mutable_cpu_data()[i + 106] = 31;
			blob_bottom_->mutable_cpu_data()[i + 107] = 4;
			blob_bottom_->mutable_cpu_data()[i + 108] = 25;
			blob_bottom_->mutable_cpu_data()[i + 109] = 26;
			blob_bottom_->mutable_cpu_data()[i + 110] = 34;
			blob_bottom_->mutable_cpu_data()[i + 111] = 33;
			blob_bottom_->mutable_cpu_data()[i + 112] = 40;
			blob_bottom_->mutable_cpu_data()[i + 113] = 4;
			blob_bottom_->mutable_cpu_data()[i + 114] = 3;
			blob_bottom_->mutable_cpu_data()[i + 115] = 12;
			blob_bottom_->mutable_cpu_data()[i + 116] = 21;
			blob_bottom_->mutable_cpu_data()[i + 117] = 11;
			blob_bottom_->mutable_cpu_data()[i + 118] = 32;
			blob_bottom_->mutable_cpu_data()[i + 119] = 26;
			blob_bottom_->mutable_cpu_data()[i + 120] = 14;
			blob_bottom_->mutable_cpu_data()[i + 121] = 19;
			blob_bottom_->mutable_cpu_data()[i + 122] = 8;
			blob_bottom_->mutable_cpu_data()[i + 123] = 4;
			blob_bottom_->mutable_cpu_data()[i + 124] = 31;
    }
    PoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 3);
    EXPECT_EQ(blob_top_->shape(3), 2);
    EXPECT_EQ(blob_top_->shape(4), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 3);
      EXPECT_EQ(blob_top_mask_->shape(3), 2);
      EXPECT_EQ(blob_top_mask_->shape(4), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32] [31    25] [29    24] [37    37]
    // [33    33] [31    29] [31    31] [31    31]
    // [34    34] [40    40] [40    40] [33    32]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 25);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 34);
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 32);
    }
    if (blob_top_vec_.size() > 1) {
      // [0     30] [26     6] [2     42] [19    19]
      // [85    85] [26    87] [88    88] [88    88]
      // [110  110] [112  112] [112  112] [104  118]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  30);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  6);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 42);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 87);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 110);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 104);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 110);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 118);
      }
    }
  }
  // Test for 2x2x2 square pooling layer
	void TestForwardStridePad() {
		LayerParameter layer_param;
		PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
		pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->add_pad(1);
		pooling_param->clear_stride();
		pooling_param->add_stride(2);
		pooling_param->add_kernel_size(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		int blobShape[5]={num,channels,3,5,4};
		blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
		// Input: 2x2 channels of:
		//     [1 2 5 2 3] [11 5 5 2 7] [1 0 3 2 3] [3 1 2 2 0]
		//     [9 4 1 4 8] [1 8 5 1 1]  [0 2 4 2 1] [2 0 1 1 4]
		//     [1 2 5 2 3] [2 2 2 1 1]  [1 2 4 1 3] [1 2 1 2 6]
		for (int i = 0; i < 60 * num * channels; i += 60) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 1;
			blob_bottom_->mutable_cpu_data()[i +  1] = 11;
			blob_bottom_->mutable_cpu_data()[i +  2] = 1;
			blob_bottom_->mutable_cpu_data()[i +  3] = 3; //
			blob_bottom_->mutable_cpu_data()[i +  4] = 2;
			blob_bottom_->mutable_cpu_data()[i +  5] = 5;
			blob_bottom_->mutable_cpu_data()[i +  6] = 0;
			blob_bottom_->mutable_cpu_data()[i +  7] = 1; //
			blob_bottom_->mutable_cpu_data()[i +  8] = 5;
			blob_bottom_->mutable_cpu_data()[i +  9] = 5;
			blob_bottom_->mutable_cpu_data()[i + 10] = 3;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 12] = 2;
			blob_bottom_->mutable_cpu_data()[i + 13] = 2;
			blob_bottom_->mutable_cpu_data()[i + 14] = 2;
			blob_bottom_->mutable_cpu_data()[i + 15] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 7;
			blob_bottom_->mutable_cpu_data()[i + 18] = 3;
			blob_bottom_->mutable_cpu_data()[i + 19] = 0; //row 1
			blob_bottom_->mutable_cpu_data()[i + 20] = 9;
			blob_bottom_->mutable_cpu_data()[i + 21] = 1;
			blob_bottom_->mutable_cpu_data()[i + 22] = 0;
			blob_bottom_->mutable_cpu_data()[i + 23] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 24] = 4;
			blob_bottom_->mutable_cpu_data()[i + 25] = 8;
			blob_bottom_->mutable_cpu_data()[i + 26] = 2;
			blob_bottom_->mutable_cpu_data()[i + 27] = 0; //
			blob_bottom_->mutable_cpu_data()[i + 28] = 1;
			blob_bottom_->mutable_cpu_data()[i + 29] = 5;
			blob_bottom_->mutable_cpu_data()[i + 30] = 4;
			blob_bottom_->mutable_cpu_data()[i + 31] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 32] = 4;
			blob_bottom_->mutable_cpu_data()[i + 33] = 1;
			blob_bottom_->mutable_cpu_data()[i + 34] = 2;
			blob_bottom_->mutable_cpu_data()[i + 35] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 36] = 8;
			blob_bottom_->mutable_cpu_data()[i + 37] = 1;
			blob_bottom_->mutable_cpu_data()[i + 38] = 1;
			blob_bottom_->mutable_cpu_data()[i + 39] = 4; //row 2
			blob_bottom_->mutable_cpu_data()[i + 40] = 1;
			blob_bottom_->mutable_cpu_data()[i + 41] = 2;
			blob_bottom_->mutable_cpu_data()[i + 42] = 1;
			blob_bottom_->mutable_cpu_data()[i + 43] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 44] = 2;
			blob_bottom_->mutable_cpu_data()[i + 45] = 2;
			blob_bottom_->mutable_cpu_data()[i + 46] = 2;
			blob_bottom_->mutable_cpu_data()[i + 47] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 48] = 5;
			blob_bottom_->mutable_cpu_data()[i + 49] = 2;
			blob_bottom_->mutable_cpu_data()[i + 50] = 4;
			blob_bottom_->mutable_cpu_data()[i + 51] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 52] = 2;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 1;
			blob_bottom_->mutable_cpu_data()[i + 55] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 56] = 3;
			blob_bottom_->mutable_cpu_data()[i + 57] = 1;
			blob_bottom_->mutable_cpu_data()[i + 58] = 3;
			blob_bottom_->mutable_cpu_data()[i + 59] = 6;
		}
		PoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->shape(0), num);
		EXPECT_EQ(blob_top_->shape(1), channels);
		EXPECT_EQ(blob_top_->shape(2), 2);
		EXPECT_EQ(blob_top_->shape(3), 3);
		EXPECT_EQ(blob_top_->shape(4), 3);
		if (blob_top_vec_.size() > 1) {
			EXPECT_EQ(blob_top_mask_->shape(0), num);
			EXPECT_EQ(blob_top_mask_->shape(1), channels);
			EXPECT_EQ(blob_top_mask_->shape(2), 2);
			EXPECT_EQ(blob_top_mask_->shape(3), 3);
			EXPECT_EQ(blob_top_mask_->shape(4), 3);
		}
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		// Expected output: 2x2 channels of:
		//     [1  5   3] [11  5  7] [3  2  2]
		//     [9  5   8] [2   8  3] [2  2  6]
		for (int i = 0; i < 18 * num * channels; i += 18) {
			EXPECT_EQ(blob_top_->cpu_data()[i + 0],  1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1],  11);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2],  3); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 3],  5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4],  5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5],  2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 6],  3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7],  7);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8],  2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 9],  9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 16], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 17], 6); //
		}
		if (blob_top_vec_.size() > 1) {
			// Expected mask output: 2x 2 channels of:
			//     [0   8 16] [1   5 17] [3  11 15]
			//     [20 48 36] [41 25 58] [23 47 59]
			for (int i = 0; i < 18 * num * channels; i += 18) {
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  0);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  3); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  8);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  11); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6],  16);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  17);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8],  15); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 9],  20);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 41);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 23); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 48); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 47);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 36); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 58);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 59);
			}
		}
	}
};

TYPED_TEST_CASE(PoolingLayer3DTest, TestDtypesAndDevices);

TYPED_TEST(PoolingLayer3DTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
}

TYPED_TEST(PoolingLayer3DTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
	EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
	EXPECT_EQ(this->blob_top_->shape(2), 4);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
}

TYPED_TEST(PoolingLayer3DTest, TestSetupGlobalPooling) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
	EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
	EXPECT_EQ(this->blob_top_->shape(2), 1);
	EXPECT_EQ(this->blob_top_->shape(3), 1);
	EXPECT_EQ(this->blob_top_->shape(3), 1);
}

/*
TYPED_TEST(PoolingLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(PoolingLayer3DTest, TestForwardMax) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

TYPED_TEST(PoolingLayer3DTest, TestForwardMaxTopMask) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

TYPED_TEST(PoolingLayer3DTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->clear_pad();
				pooling_param->add_pad(1);
				pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
				PoolingLayer<Dtype> layer(layer_param);
				GradientChecker<Dtype> checker(1e-4, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
    	}
    }
  }
}

TYPED_TEST(PoolingLayer3DTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  int blobShape[5]={1,1,3,3,3};
  this->blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
  // Input:
  //     [ 1 2 4 ] [ 1 2 1 ] [ 3 1 2 ]
  //     [ 2 3 2 ] [ 1 1 2 ] [ 0 2 0 ]
  //     [ 4 2 1 ] [ 1 1 2 ] [ 1 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0]  = 1;
  this->blob_bottom_->mutable_cpu_data()[1]  = 1;
  this->blob_bottom_->mutable_cpu_data()[2]  = 3;
  this->blob_bottom_->mutable_cpu_data()[3]  = 2;
  this->blob_bottom_->mutable_cpu_data()[4]  = 2;
  this->blob_bottom_->mutable_cpu_data()[5]  = 1;
  this->blob_bottom_->mutable_cpu_data()[6]  = 4;
  this->blob_bottom_->mutable_cpu_data()[7]  = 1;
  this->blob_bottom_->mutable_cpu_data()[8]  = 2;
  this->blob_bottom_->mutable_cpu_data()[9]  = 2;
	this->blob_bottom_->mutable_cpu_data()[10] = 1;
	this->blob_bottom_->mutable_cpu_data()[11] = 0;
	this->blob_bottom_->mutable_cpu_data()[12] = 3;
	this->blob_bottom_->mutable_cpu_data()[13] = 1;
	this->blob_bottom_->mutable_cpu_data()[14] = 2;
	this->blob_bottom_->mutable_cpu_data()[15] = 2;
	this->blob_bottom_->mutable_cpu_data()[16] = 2;
	this->blob_bottom_->mutable_cpu_data()[17] = 0;
	this->blob_bottom_->mutable_cpu_data()[18] = 4;
	this->blob_bottom_->mutable_cpu_data()[19] = 1;
	this->blob_bottom_->mutable_cpu_data()[20] = 1;
	this->blob_bottom_->mutable_cpu_data()[21] = 2;
	this->blob_bottom_->mutable_cpu_data()[22] = 1;
	this->blob_bottom_->mutable_cpu_data()[23] = 2;
	this->blob_bottom_->mutable_cpu_data()[24] = 1;
	this->blob_bottom_->mutable_cpu_data()[25] = 2;
	this->blob_bottom_->mutable_cpu_data()[26] = 1;
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  //     [ 1 2 4 ] [ 1 2 1 ] [ 3 1 2 ]
	//     [ 2 3 2 ] [ 1 1 2 ] [ 0 2 0 ]
	//     [ 4 2 1 ] [ 1 1 2 ] [ 1 2 1 ]
  // Output:
  //     [ 1 4 4 ] [ 3 4 4 ] [ 3 3 2 ]
  //     [ 4 4 4 ] [ 4 4 4 ] [ 3 3 2 ]
  //     [ 4 4 1 ] [ 4 4 2 ] [ 1 2 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0],  1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8],  2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[9],  4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[10], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[11], 3, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[12], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[13], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[14], 3, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[15], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[16], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[17], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[18], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[19], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[20], 1, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[21], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[22], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[23], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[24], 1, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[25], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[26], 1, epsilon);
}

TYPED_TEST(PoolingLayer3DTest, TestGradientMaxTopMask) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
				this->blob_top_vec_.push_back(this->blob_top_mask_);
				PoolingLayer<Dtype> layer(layer_param);
				GradientChecker<Dtype> checker(1e-4, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
				this->blob_top_vec_.pop_back();
    	}
    }
  }
}

TYPED_TEST(PoolingLayer3DTest, TestForwardAve) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(1);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  int blobShape[5]={1,1,3,3,3};
	this->blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  PoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
	EXPECT_EQ(this->blob_top_->shape(1), 1);
	EXPECT_EQ(this->blob_top_->shape(2), 3);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
	EXPECT_EQ(this->blob_top_->shape(4), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4],  36.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[9],  24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[10], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[11], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[12], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[13], 54.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[14], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[15], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[16], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[17], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[18], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[19], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[20], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[21], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[22], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[23], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[24], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[25], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[26], 16.0 / 27, epsilon);
}

TYPED_TEST(PoolingLayer3DTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
				PoolingLayer<Dtype> layer(layer_param);
				GradientChecker<Dtype> checker(1e-2, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
			}
    }
  }
}

TYPED_TEST(PoolingLayer3DTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->clear_pad();
				pooling_param->add_pad(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
				PoolingLayer<Dtype> layer(layer_param);
				GradientChecker<Dtype> checker(1e-2, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
			}
    }
  }
}

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNPoolingLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
    pooling_param->add_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * num * channels; i += 15) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 1;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 5;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 3;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 2);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < 8 * num * channels; i += 8) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
      //     [5  2  2 9]
      //     [5 12 12 9]
      for (int i = 0; i < 8 * num * channels; i += 8) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(3);
    pooling_param->set_kernel_w(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 4);
    EXPECT_EQ(blob_top_->width(), 5);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 4);
      EXPECT_EQ(blob_top_mask_->width(), 5);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    27    27]
    // [32    33    33    27    27]
    // [31    34    34    27    27]
    // [36    36    34    18    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 18);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4    17    17]
        // [ 8    21    21    17    17]
        // [13    27    27    17    17]
        // [32    32    27    35    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 12);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 34);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_h(2);
    pooling_param->set_kernel_w(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    blob_bottom_->Reshape(num, channels, 6, 6);
    // Input: 2x 2 channels of:
    // [35     1     6    26    19    24]
    // [ 3    32     7    21    23    25]
    // [31     9     2    22    27    20]
    // [ 8    28    33    17    10    15]
    // [30     5    34    12    14    16]
    // [ 4    36    29    13    18    11]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 36 * num * channels; i += 36) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 1;
      blob_bottom_->mutable_cpu_data()[i +  2] = 6;
      blob_bottom_->mutable_cpu_data()[i +  3] = 26;
      blob_bottom_->mutable_cpu_data()[i +  4] = 19;
      blob_bottom_->mutable_cpu_data()[i +  5] = 24;
      blob_bottom_->mutable_cpu_data()[i +  6] = 3;
      blob_bottom_->mutable_cpu_data()[i +  7] = 32;
      blob_bottom_->mutable_cpu_data()[i +  8] = 7;
      blob_bottom_->mutable_cpu_data()[i +  9] = 21;
      blob_bottom_->mutable_cpu_data()[i + 10] = 23;
      blob_bottom_->mutable_cpu_data()[i + 11] = 25;
      blob_bottom_->mutable_cpu_data()[i + 12] = 31;
      blob_bottom_->mutable_cpu_data()[i + 13] = 9;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 22;
      blob_bottom_->mutable_cpu_data()[i + 16] = 27;
      blob_bottom_->mutable_cpu_data()[i + 17] = 20;
      blob_bottom_->mutable_cpu_data()[i + 18] = 8;
      blob_bottom_->mutable_cpu_data()[i + 19] = 28;
      blob_bottom_->mutable_cpu_data()[i + 20] = 33;
      blob_bottom_->mutable_cpu_data()[i + 21] = 17;
      blob_bottom_->mutable_cpu_data()[i + 22] = 10;
      blob_bottom_->mutable_cpu_data()[i + 23] = 15;
      blob_bottom_->mutable_cpu_data()[i + 24] = 30;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 34;
      blob_bottom_->mutable_cpu_data()[i + 27] = 12;
      blob_bottom_->mutable_cpu_data()[i + 28] = 14;
      blob_bottom_->mutable_cpu_data()[i + 29] = 16;
      blob_bottom_->mutable_cpu_data()[i + 30] = 4;
      blob_bottom_->mutable_cpu_data()[i + 31] = 36;
      blob_bottom_->mutable_cpu_data()[i + 32] = 29;
      blob_bottom_->mutable_cpu_data()[i + 33] = 13;
      blob_bottom_->mutable_cpu_data()[i + 34] = 18;
      blob_bottom_->mutable_cpu_data()[i + 35] = 11;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), 5);
    EXPECT_EQ(blob_top_->width(), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), 5);
      EXPECT_EQ(blob_top_mask_->width(), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32    26    26]
    // [32    32    27    27]
    // [33    33    33    27]
    // [34    34    34    17]
    // [36    36    34    18]
    for (int i = 0; i < 20 * num * channels; i += 20) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 26);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 17);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 36);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 18);
    }
    if (blob_top_vec_.size() > 1) {
        // [ 1     8     4     4]
        // [ 8     8    17    17]
        // [21    21    21    17]
        // [27    27    27    22]
        // [32    32    27    35]
      for (int i = 0; i < 20 * num * channels; i += 20) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  3);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  7);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 20);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 16);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 21);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 31);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 34);
      }
    }
  }
  // Test for 2x2 pooling with padding and stride layer
	void TestForwardStridePad() {
		LayerParameter layer_param;
		PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
		pooling_param->clear_kernel_size();
		pooling_param->add_kernel_size(2);
		pooling_param->clear_pad();
		pooling_param->add_pad(1);
		pooling_param->clear_stride();
		pooling_param->add_stride(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		blob_bottom_->Reshape(num, channels, 4, 5);
		// Input: 2x 2 channels of:
		//     [1 2 5 2 3]
		//     [9 4 1 4 8]
		//     [1 2 5 2 3]
		//     [4 3 1 2 1]
		for (int i = 0; i < 20 * num * channels; i += 20) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 1;
			blob_bottom_->mutable_cpu_data()[i +  1] = 2;
			blob_bottom_->mutable_cpu_data()[i +  2] = 5;
			blob_bottom_->mutable_cpu_data()[i +  3] = 2;
			blob_bottom_->mutable_cpu_data()[i +  4] = 3;
			blob_bottom_->mutable_cpu_data()[i +  5] = 9;
			blob_bottom_->mutable_cpu_data()[i +  6] = 4;
			blob_bottom_->mutable_cpu_data()[i +  7] = 1;
			blob_bottom_->mutable_cpu_data()[i +  8] = 4;
			blob_bottom_->mutable_cpu_data()[i +  9] = 8;
			blob_bottom_->mutable_cpu_data()[i + 10] = 1;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2;
			blob_bottom_->mutable_cpu_data()[i + 12] = 5;
			blob_bottom_->mutable_cpu_data()[i + 13] = 2;
			blob_bottom_->mutable_cpu_data()[i + 14] = 3;
			blob_bottom_->mutable_cpu_data()[i + 15] = 4;
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 1;
			blob_bottom_->mutable_cpu_data()[i + 18] = 2;
			blob_bottom_->mutable_cpu_data()[i + 19] = 1;
		}
		CuDNNPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->num(), num);
		EXPECT_EQ(blob_top_->channels(), channels);
		EXPECT_EQ(blob_top_->height(), 3);
		EXPECT_EQ(blob_top_->width(), 3);
		if (blob_top_vec_.size() > 1) {
			EXPECT_EQ(blob_top_mask_->num(), num);
			EXPECT_EQ(blob_top_mask_->channels(), channels);
			EXPECT_EQ(blob_top_mask_->height(), 3);
			EXPECT_EQ(blob_top_mask_->width(), 3);
		}
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		// Expected output: 2x2 channels of:
		//     [1 5 3]
		//     [9 5 8]
		//     [4 3 2]
		for (int i = 0; i < 9 * num * channels; i += 9) {
			EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 3], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 6], 4);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8], 2);
		}
		if (blob_top_vec_.size() > 1) {
			// Expected mask output: 2x 2 channels of:
			//     [0  2  4]
			//     [5  12 9]
			//     [15 16 18]
			for (int i = 0; i < 9 * num * channels; i += 9) {
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  0);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  4);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  5);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4], 12);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  9);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 15);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7], 16);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8], 18);
			}
		}
	}
};

TYPED_TEST_CASE(CuDNNPoolingLayerTest, TestDtypes);

TYPED_TEST(CuDNNPoolingLayerTest, TestSetupCuDNN) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->add_kernel_size(3);
  pooling_param->clear_stride();
  pooling_param->add_stride(2);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
}

TYPED_TEST(CuDNNPoolingLayerTest, TestSetupPaddedCuDNN) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->add_kernel_size(3);
  pooling_param->clear_stride();
  pooling_param->add_stride(2);
  pooling_param->clear_pad();
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

/*
TYPED_TEST(CuDNNPoolingLayerTest, PrintBackwardCuDNN) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxCuDNN) {
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

// Currently, cuDNN does not support a top mask, so we comment this and
// the corresponding backward test.
/*
TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxTopMaskCuDNN) {
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}*/


TYPED_TEST(CuDNNPoolingLayerTest, TestGradientMaxCuDNN) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      // currenty, cuDNN pooling does not support padding
      pooling_param->clear_pad();
      pooling_param->add_pad(0);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMaxPaddedCuDNN) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->add_kernel_size(3);
  pooling_param->clear_stride();
  pooling_param->add_stride(2);
  pooling_param->clear_pad();
  pooling_param->add_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0] = 1;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 4;
  this->blob_bottom_->mutable_cpu_data()[3] = 2;
  this->blob_bottom_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_->mutable_cpu_data()[5] = 2;
  this->blob_bottom_->mutable_cpu_data()[6] = 4;
  this->blob_bottom_->mutable_cpu_data()[7] = 2;
  this->blob_bottom_->mutable_cpu_data()[8] = 1;
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam epsilon = 1e-8;
  // Output:
  //     [ 1 4 4 ]
  //     [ 4 4 4 ]
  //     [ 4 4 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 1, epsilon);
}

/*
TYPED_TEST(CuDNNPoolingLayerTest, TestGradientMaxTopMaskCuDNN) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->set_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
      this->blob_top_vec_.push_back(this->blob_top_mask_);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      this->blob_top_vec_.pop_back();
    }
  }
}
*/

TYPED_TEST(CuDNNPoolingLayerTest, TestForwardAveCuDNN) {
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->add_kernel_size(3);
  pooling_param->clear_stride();
  pooling_param->add_stride(1);
  // Currently, cuDNN pooling does not support padding, so we use
  // a simplified version of this test.
  pooling_param->clear_pad();
  pooling_param->add_pad(0);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  this->blob_bottom_->Reshape(1, 1, 3, 3);
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 2.0, epsilon);
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientAveCuDNN) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

TYPED_TEST(CuDNNPoolingLayerTest, TestGradientAvePaddedCuDNN) {
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      LayerParameter layer_param;
      PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
      pooling_param->set_kernel_h(kernel_h);
      pooling_param->set_kernel_w(kernel_w);
      pooling_param->clear_stride();
      pooling_param->add_stride(2);
      pooling_param->clear_pad();
      pooling_param->add_pad(2);
      pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
      CuDNNPoolingLayer<TypeParam> layer(layer_param);
      GradientChecker<TypeParam> checker(1e-2, 1e-2);
      checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
    }
  }
}

/******************************************************************
 *  3D Pooling Layer Tests
 ******************************************************************/

template <typename Dtype>
class CuDNNPoolingLayer3DTest : public ::testing::Test {
 protected:
  CuDNNPoolingLayer3DTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    int blobShape[5]={2,3,6,5,4};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNPoolingLayer3DTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x2x2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
    pooling_param->clear_pad();
    pooling_param->clear_stride();
    pooling_param->add_kernel_size(2);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    int blobShape[5]={num,channels,3,5,4};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // Input: 2x2 channels of:
    //     [1 2 5 2 3] [11 5 5 2 7] [1 0 3 2 3] [3 1 2 2 0]
    //     [9 4 1 4 8] [1 8 5 1 1]  [0 2 4 2 1] [2 0 1 1 4]
    //     [1 2 5 2 3] [2 2 2 1 1]  [1 2 4 1 3] [1 2 1 2 6]
    for (int i = 0; i < 60 * num * channels; i += 60) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 11;
      blob_bottom_->mutable_cpu_data()[i +  2] = 1;
      blob_bottom_->mutable_cpu_data()[i +  3] = 3; //
      blob_bottom_->mutable_cpu_data()[i +  4] = 2;
      blob_bottom_->mutable_cpu_data()[i +  5] = 5;
      blob_bottom_->mutable_cpu_data()[i +  6] = 0;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1; //
      blob_bottom_->mutable_cpu_data()[i +  8] = 5;
      blob_bottom_->mutable_cpu_data()[i +  9] = 5;
      blob_bottom_->mutable_cpu_data()[i + 10] = 3;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2; //
      blob_bottom_->mutable_cpu_data()[i + 12] = 2;
      blob_bottom_->mutable_cpu_data()[i + 13] = 2;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 7;
			blob_bottom_->mutable_cpu_data()[i + 18] = 3;
			blob_bottom_->mutable_cpu_data()[i + 19] = 0; //row 1
			blob_bottom_->mutable_cpu_data()[i + 20] = 9;
			blob_bottom_->mutable_cpu_data()[i + 21] = 1;
			blob_bottom_->mutable_cpu_data()[i + 22] = 0;
			blob_bottom_->mutable_cpu_data()[i + 23] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 24] = 4;
			blob_bottom_->mutable_cpu_data()[i + 25] = 8;
			blob_bottom_->mutable_cpu_data()[i + 26] = 2;
			blob_bottom_->mutable_cpu_data()[i + 27] = 0; //
			blob_bottom_->mutable_cpu_data()[i + 28] = 1;
			blob_bottom_->mutable_cpu_data()[i + 29] = 5;
			blob_bottom_->mutable_cpu_data()[i + 30] = 4;
			blob_bottom_->mutable_cpu_data()[i + 31] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 32] = 4;
			blob_bottom_->mutable_cpu_data()[i + 33] = 1;
			blob_bottom_->mutable_cpu_data()[i + 34] = 2;
			blob_bottom_->mutable_cpu_data()[i + 35] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 36] = 8;
			blob_bottom_->mutable_cpu_data()[i + 37] = 1;
			blob_bottom_->mutable_cpu_data()[i + 38] = 1;
			blob_bottom_->mutable_cpu_data()[i + 39] = 4; //row 2
			blob_bottom_->mutable_cpu_data()[i + 40] = 1;
			blob_bottom_->mutable_cpu_data()[i + 41] = 2;
			blob_bottom_->mutable_cpu_data()[i + 42] = 1;
			blob_bottom_->mutable_cpu_data()[i + 43] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 44] = 2;
			blob_bottom_->mutable_cpu_data()[i + 45] = 2;
			blob_bottom_->mutable_cpu_data()[i + 46] = 2;
			blob_bottom_->mutable_cpu_data()[i + 47] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 48] = 5;
			blob_bottom_->mutable_cpu_data()[i + 49] = 2;
			blob_bottom_->mutable_cpu_data()[i + 50] = 4;
			blob_bottom_->mutable_cpu_data()[i + 51] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 52] = 2;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 1;
			blob_bottom_->mutable_cpu_data()[i + 55] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 56] = 3;
			blob_bottom_->mutable_cpu_data()[i + 57] = 1;
			blob_bottom_->mutable_cpu_data()[i + 58] = 3;
			blob_bottom_->mutable_cpu_data()[i + 59] = 6;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 2);
    EXPECT_EQ(blob_top_->shape(3), 4);
    EXPECT_EQ(blob_top_->shape(4), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 2);
      EXPECT_EQ(blob_top_mask_->shape(3), 4);
      EXPECT_EQ(blob_top_mask_->shape(4), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x2 channels of:
    //     [11 8 5 8] [11 8 5 7] [3 4 4 4]
    //     [9 8 5 8]  [8 8 5 3]  [2 4 4 6]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0],  11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1],  11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2],  3); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 3],  8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4],  8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5],  4); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 6],  5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7],  5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8],  4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 9],  8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 7);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 4); // row 1
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 16], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 17], 4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 18], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 19], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 20], 4); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 6);
    }
    if (blob_top_vec_.size() > 1) {
      // Expected mask output: 2x 2 channels of:
    	//     [1  25  8 36] [1  25 9  17] [3  30 30 39]
    	//     [20 25 29 36] [25 25 29 58] [23 30 30 59]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  3); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6],  8);
			  EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8],  30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 9],  36);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 17);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 39); // row 1
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 20);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 23); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 29);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 29);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 30); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 36);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 58);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 59);
      }
    }
  }
  // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
  void TestForwardRectHigh() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->clear_stride();
    pooling_param->add_kernel_size(4);
    pooling_param->add_kernel_size(2);
    pooling_param->add_kernel_size(3);
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 2;
    int blobShape[5]={num,channels,5,5,5};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // Input: 2x2 channels of:
    // [35     1     6    26    19] [4      25     2     17    19] [29     3      15    14    12]
    // [ 3    32     7    21    23] [31     4      9     15    19] [9      17     11    24    24]
    // [31     9     2    22    27] [7      17     11    22    19] [23     7      3     20    19]
    // [ 8    28    33    17    10] [27     8      22    18    19] [2      22     29    13    6]
    // [30     5    34    12    14] [3      31     33    21    19] [37     4      40    11    8]
    //
    // [29     3     16     6    9] [14     22     32    37    9]
		// [ 6    21     17     2    3] [3      13     19    5     8]
		// [ 1    19     12    12    7] [17     5      16    2     6]
		// [28     8     31    17    1] [7      28      2    8    22]
		// [ 3    25     4     32    4] [33     26     3     26   31]
    // (this is generated by magic(6) in MATLAB)
    for (int i = 0; i < 125 * num * channels; i += 125) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 35;
      blob_bottom_->mutable_cpu_data()[i +  1] = 4;
      blob_bottom_->mutable_cpu_data()[i +  2] = 29;
      blob_bottom_->mutable_cpu_data()[i +  3] = 29;
      blob_bottom_->mutable_cpu_data()[i +  4] = 14;
      blob_bottom_->mutable_cpu_data()[i +  5] = 1;
      blob_bottom_->mutable_cpu_data()[i +  6] = 25;
      blob_bottom_->mutable_cpu_data()[i +  7] = 3;
      blob_bottom_->mutable_cpu_data()[i +  8] = 3;
      blob_bottom_->mutable_cpu_data()[i +  9] = 22;
      blob_bottom_->mutable_cpu_data()[i + 10] = 6;
      blob_bottom_->mutable_cpu_data()[i + 11] = 2;
      blob_bottom_->mutable_cpu_data()[i + 12] = 15;
      blob_bottom_->mutable_cpu_data()[i + 13] = 16;
      blob_bottom_->mutable_cpu_data()[i + 14] = 32;
      blob_bottom_->mutable_cpu_data()[i + 15] = 26;
      blob_bottom_->mutable_cpu_data()[i + 16] = 17;
      blob_bottom_->mutable_cpu_data()[i + 17] = 14;
      blob_bottom_->mutable_cpu_data()[i + 18] = 6;
      blob_bottom_->mutable_cpu_data()[i + 19] = 37;
      blob_bottom_->mutable_cpu_data()[i + 20] = 19;
      blob_bottom_->mutable_cpu_data()[i + 21] = 19;
      blob_bottom_->mutable_cpu_data()[i + 22] = 12;
      blob_bottom_->mutable_cpu_data()[i + 23] = 9;
      blob_bottom_->mutable_cpu_data()[i + 24] = 9;
      blob_bottom_->mutable_cpu_data()[i + 25] = 3; //
      blob_bottom_->mutable_cpu_data()[i + 26] = 31;
      blob_bottom_->mutable_cpu_data()[i + 27] = 9;
      blob_bottom_->mutable_cpu_data()[i + 28] = 6;
      blob_bottom_->mutable_cpu_data()[i + 29] = 3;
      blob_bottom_->mutable_cpu_data()[i + 30] = 32;
      blob_bottom_->mutable_cpu_data()[i + 31] = 4;
      blob_bottom_->mutable_cpu_data()[i + 32] = 17;
      blob_bottom_->mutable_cpu_data()[i + 33] = 21;
      blob_bottom_->mutable_cpu_data()[i + 34] = 13;
      blob_bottom_->mutable_cpu_data()[i + 35] = 7;
			blob_bottom_->mutable_cpu_data()[i + 36] = 9;
			blob_bottom_->mutable_cpu_data()[i + 37] = 11;
			blob_bottom_->mutable_cpu_data()[i + 38] = 17;
			blob_bottom_->mutable_cpu_data()[i + 39] = 19;
			blob_bottom_->mutable_cpu_data()[i + 40] = 21;
			blob_bottom_->mutable_cpu_data()[i + 41] = 15;
			blob_bottom_->mutable_cpu_data()[i + 42] = 24;
			blob_bottom_->mutable_cpu_data()[i + 43] = 2;
			blob_bottom_->mutable_cpu_data()[i + 44] = 5;
			blob_bottom_->mutable_cpu_data()[i + 45] = 23;
			blob_bottom_->mutable_cpu_data()[i + 46] = 19;
			blob_bottom_->mutable_cpu_data()[i + 47] = 24;
			blob_bottom_->mutable_cpu_data()[i + 48] = 3;
			blob_bottom_->mutable_cpu_data()[i + 49] = 8;
			blob_bottom_->mutable_cpu_data()[i + 50] = 31; //
			blob_bottom_->mutable_cpu_data()[i + 51] = 7;
			blob_bottom_->mutable_cpu_data()[i + 52] = 23;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 17;
			blob_bottom_->mutable_cpu_data()[i + 55] = 9;
			blob_bottom_->mutable_cpu_data()[i + 56] = 17;
			blob_bottom_->mutable_cpu_data()[i + 57] = 7;
			blob_bottom_->mutable_cpu_data()[i + 58] = 19;
			blob_bottom_->mutable_cpu_data()[i + 59] = 5;
			blob_bottom_->mutable_cpu_data()[i + 60] = 2;
			blob_bottom_->mutable_cpu_data()[i + 61] = 11;
			blob_bottom_->mutable_cpu_data()[i + 62] = 3;
			blob_bottom_->mutable_cpu_data()[i + 63] = 12;
			blob_bottom_->mutable_cpu_data()[i + 64] = 16;
			blob_bottom_->mutable_cpu_data()[i + 65] = 22;
			blob_bottom_->mutable_cpu_data()[i + 66] = 22;
			blob_bottom_->mutable_cpu_data()[i + 67] = 20;
			blob_bottom_->mutable_cpu_data()[i + 68] = 12;
			blob_bottom_->mutable_cpu_data()[i + 69] = 2;
			blob_bottom_->mutable_cpu_data()[i + 70] = 27;
			blob_bottom_->mutable_cpu_data()[i + 71] = 19;
			blob_bottom_->mutable_cpu_data()[i + 72] = 19;
			blob_bottom_->mutable_cpu_data()[i + 73] = 7;
			blob_bottom_->mutable_cpu_data()[i + 74] = 6;
			blob_bottom_->mutable_cpu_data()[i + 75] = 8; //
			blob_bottom_->mutable_cpu_data()[i + 76] = 27;
			blob_bottom_->mutable_cpu_data()[i + 77] = 2;
			blob_bottom_->mutable_cpu_data()[i + 78] = 28;
			blob_bottom_->mutable_cpu_data()[i + 79] = 28;
			blob_bottom_->mutable_cpu_data()[i + 80] = 28;
			blob_bottom_->mutable_cpu_data()[i + 81] = 8;
			blob_bottom_->mutable_cpu_data()[i + 82] = 22;
			blob_bottom_->mutable_cpu_data()[i + 83] = 8;
			blob_bottom_->mutable_cpu_data()[i + 84] = 28;
			blob_bottom_->mutable_cpu_data()[i + 85] = 33;
			blob_bottom_->mutable_cpu_data()[i + 86] = 22;
			blob_bottom_->mutable_cpu_data()[i + 87] = 29;
			blob_bottom_->mutable_cpu_data()[i + 88] = 31;
			blob_bottom_->mutable_cpu_data()[i + 89] = 2;
			blob_bottom_->mutable_cpu_data()[i + 90] = 17;
			blob_bottom_->mutable_cpu_data()[i + 91] = 18;
			blob_bottom_->mutable_cpu_data()[i + 92] = 13;
			blob_bottom_->mutable_cpu_data()[i + 93] = 17;
			blob_bottom_->mutable_cpu_data()[i + 94] = 8;
			blob_bottom_->mutable_cpu_data()[i + 95] = 10;
			blob_bottom_->mutable_cpu_data()[i + 96] = 19;
			blob_bottom_->mutable_cpu_data()[i + 97] = 6;
			blob_bottom_->mutable_cpu_data()[i + 98] = 1;
			blob_bottom_->mutable_cpu_data()[i + 99] = 22;
			blob_bottom_->mutable_cpu_data()[i + 100] = 30; //
			blob_bottom_->mutable_cpu_data()[i + 101] = 3;
			blob_bottom_->mutable_cpu_data()[i + 102] = 37;
			blob_bottom_->mutable_cpu_data()[i + 103] = 3;
			blob_bottom_->mutable_cpu_data()[i + 104] = 33;
			blob_bottom_->mutable_cpu_data()[i + 105] = 5;
			blob_bottom_->mutable_cpu_data()[i + 106] = 31;
			blob_bottom_->mutable_cpu_data()[i + 107] = 4;
			blob_bottom_->mutable_cpu_data()[i + 108] = 25;
			blob_bottom_->mutable_cpu_data()[i + 109] = 26;
			blob_bottom_->mutable_cpu_data()[i + 110] = 34;
			blob_bottom_->mutable_cpu_data()[i + 111] = 33;
			blob_bottom_->mutable_cpu_data()[i + 112] = 40;
			blob_bottom_->mutable_cpu_data()[i + 113] = 4;
			blob_bottom_->mutable_cpu_data()[i + 114] = 3;
			blob_bottom_->mutable_cpu_data()[i + 115] = 12;
			blob_bottom_->mutable_cpu_data()[i + 116] = 21;
			blob_bottom_->mutable_cpu_data()[i + 117] = 11;
			blob_bottom_->mutable_cpu_data()[i + 118] = 32;
			blob_bottom_->mutable_cpu_data()[i + 119] = 26;
			blob_bottom_->mutable_cpu_data()[i + 120] = 14;
			blob_bottom_->mutable_cpu_data()[i + 121] = 19;
			blob_bottom_->mutable_cpu_data()[i + 122] = 8;
			blob_bottom_->mutable_cpu_data()[i + 123] = 4;
			blob_bottom_->mutable_cpu_data()[i + 124] = 31;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 2);
    EXPECT_EQ(blob_top_->shape(3), 4);
    EXPECT_EQ(blob_top_->shape(4), 3);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 2);
      EXPECT_EQ(blob_top_mask_->shape(3), 4);
      EXPECT_EQ(blob_top_mask_->shape(4), 3);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x2 channels of:
    // [35    33    33    27] [31    31    31    24] [29    32    37    37]
    // [37    40    40    27] [37    40    40    32] [37    40    40    32]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 27);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 37); //
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 27);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 32);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 32);
    }
    if (blob_top_vec_.size() > 1) {
      // [0     85    85    70] [26    88    88    42] [ 2    14    19    19]
  		// [102  112    112   70] [102  112   112   118] [102   112   112  118]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5], 14);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 70);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 42);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 102);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 70);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 118);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 118);
      }
    }
  }
  // Test for rectangular pooling layer with kernel_w > kernel_h
  void TestForwardRectWide() {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->clear_stride();
		pooling_param->add_kernel_size(3);
		pooling_param->add_kernel_size(4);
		pooling_param->add_kernel_size(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		int blobShape[5]={num,channels,5,5,5};
		blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
		// Input: 2x2 channels of:
		// [35     1     6    26    19] [4      25     2     17    19] [29     3      15    14    12]
		// [ 3    32     7    21    23] [31     4      9     15    19] [9      17     11    24    24]
		// [31     9     2    22    27] [7      17     11    22    19] [23     7      3     20    19]
		// [ 8    28    33    17    10] [27     8      22    18    19] [2      22     29    13    6]
		// [30     5    34    12    14] [3      31     33    21    19] [37     4      40    11    8]
		//
		// [29     3     16     6    9] [14     22     32    37    9]
		// [ 6    21     17     2    3] [3      13     19    5     8]
		// [ 1    19     12    12    7] [17     5      16    2     6]
		// [28     8     31    17    1] [7      28      2    8    22]
		// [ 3    25     4     32    4] [33     26     3     26   31]
		// (this is generated by magic(6) in MATLAB)
		for (int i = 0; i < 125 * num * channels; i += 125) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 35;
			blob_bottom_->mutable_cpu_data()[i +  1] = 4;
			blob_bottom_->mutable_cpu_data()[i +  2] = 29;
			blob_bottom_->mutable_cpu_data()[i +  3] = 29;
			blob_bottom_->mutable_cpu_data()[i +  4] = 14;
			blob_bottom_->mutable_cpu_data()[i +  5] = 1;
			blob_bottom_->mutable_cpu_data()[i +  6] = 25;
			blob_bottom_->mutable_cpu_data()[i +  7] = 3;
			blob_bottom_->mutable_cpu_data()[i +  8] = 3;
			blob_bottom_->mutable_cpu_data()[i +  9] = 22;
			blob_bottom_->mutable_cpu_data()[i + 10] = 6;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2;
			blob_bottom_->mutable_cpu_data()[i + 12] = 15;
			blob_bottom_->mutable_cpu_data()[i + 13] = 16;
			blob_bottom_->mutable_cpu_data()[i + 14] = 32;
			blob_bottom_->mutable_cpu_data()[i + 15] = 26;
			blob_bottom_->mutable_cpu_data()[i + 16] = 17;
			blob_bottom_->mutable_cpu_data()[i + 17] = 14;
			blob_bottom_->mutable_cpu_data()[i + 18] = 6;
			blob_bottom_->mutable_cpu_data()[i + 19] = 37;
			blob_bottom_->mutable_cpu_data()[i + 20] = 19;
			blob_bottom_->mutable_cpu_data()[i + 21] = 19;
			blob_bottom_->mutable_cpu_data()[i + 22] = 12;
			blob_bottom_->mutable_cpu_data()[i + 23] = 9;
			blob_bottom_->mutable_cpu_data()[i + 24] = 9;
			blob_bottom_->mutable_cpu_data()[i + 25] = 3; //
			blob_bottom_->mutable_cpu_data()[i + 26] = 31;
			blob_bottom_->mutable_cpu_data()[i + 27] = 9;
			blob_bottom_->mutable_cpu_data()[i + 28] = 6;
			blob_bottom_->mutable_cpu_data()[i + 29] = 3;
			blob_bottom_->mutable_cpu_data()[i + 30] = 32;
			blob_bottom_->mutable_cpu_data()[i + 31] = 4;
			blob_bottom_->mutable_cpu_data()[i + 32] = 17;
			blob_bottom_->mutable_cpu_data()[i + 33] = 21;
			blob_bottom_->mutable_cpu_data()[i + 34] = 13;
			blob_bottom_->mutable_cpu_data()[i + 35] = 7;
			blob_bottom_->mutable_cpu_data()[i + 36] = 9;
			blob_bottom_->mutable_cpu_data()[i + 37] = 11;
			blob_bottom_->mutable_cpu_data()[i + 38] = 17;
			blob_bottom_->mutable_cpu_data()[i + 39] = 19;
			blob_bottom_->mutable_cpu_data()[i + 40] = 21;
			blob_bottom_->mutable_cpu_data()[i + 41] = 15;
			blob_bottom_->mutable_cpu_data()[i + 42] = 24;
			blob_bottom_->mutable_cpu_data()[i + 43] = 2;
			blob_bottom_->mutable_cpu_data()[i + 44] = 5;
			blob_bottom_->mutable_cpu_data()[i + 45] = 23;
			blob_bottom_->mutable_cpu_data()[i + 46] = 19;
			blob_bottom_->mutable_cpu_data()[i + 47] = 24;
			blob_bottom_->mutable_cpu_data()[i + 48] = 3;
			blob_bottom_->mutable_cpu_data()[i + 49] = 8;
			blob_bottom_->mutable_cpu_data()[i + 50] = 31; //
			blob_bottom_->mutable_cpu_data()[i + 51] = 7;
			blob_bottom_->mutable_cpu_data()[i + 52] = 23;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 17;
			blob_bottom_->mutable_cpu_data()[i + 55] = 9;
			blob_bottom_->mutable_cpu_data()[i + 56] = 17;
			blob_bottom_->mutable_cpu_data()[i + 57] = 7;
			blob_bottom_->mutable_cpu_data()[i + 58] = 19;
			blob_bottom_->mutable_cpu_data()[i + 59] = 5;
			blob_bottom_->mutable_cpu_data()[i + 60] = 2;
			blob_bottom_->mutable_cpu_data()[i + 61] = 11;
			blob_bottom_->mutable_cpu_data()[i + 62] = 3;
			blob_bottom_->mutable_cpu_data()[i + 63] = 12;
			blob_bottom_->mutable_cpu_data()[i + 64] = 16;
			blob_bottom_->mutable_cpu_data()[i + 65] = 22;
			blob_bottom_->mutable_cpu_data()[i + 66] = 22;
			blob_bottom_->mutable_cpu_data()[i + 67] = 20;
			blob_bottom_->mutable_cpu_data()[i + 68] = 12;
			blob_bottom_->mutable_cpu_data()[i + 69] = 2;
			blob_bottom_->mutable_cpu_data()[i + 70] = 27;
			blob_bottom_->mutable_cpu_data()[i + 71] = 19;
			blob_bottom_->mutable_cpu_data()[i + 72] = 19;
			blob_bottom_->mutable_cpu_data()[i + 73] = 7;
			blob_bottom_->mutable_cpu_data()[i + 74] = 6;
			blob_bottom_->mutable_cpu_data()[i + 75] = 8; //
			blob_bottom_->mutable_cpu_data()[i + 76] = 27;
			blob_bottom_->mutable_cpu_data()[i + 77] = 2;
			blob_bottom_->mutable_cpu_data()[i + 78] = 28;
			blob_bottom_->mutable_cpu_data()[i + 79] = 28;
			blob_bottom_->mutable_cpu_data()[i + 80] = 28;
			blob_bottom_->mutable_cpu_data()[i + 81] = 8;
			blob_bottom_->mutable_cpu_data()[i + 82] = 22;
			blob_bottom_->mutable_cpu_data()[i + 83] = 8;
			blob_bottom_->mutable_cpu_data()[i + 84] = 28;
			blob_bottom_->mutable_cpu_data()[i + 85] = 33;
			blob_bottom_->mutable_cpu_data()[i + 86] = 22;
			blob_bottom_->mutable_cpu_data()[i + 87] = 29;
			blob_bottom_->mutable_cpu_data()[i + 88] = 31;
			blob_bottom_->mutable_cpu_data()[i + 89] = 2;
			blob_bottom_->mutable_cpu_data()[i + 90] = 17;
			blob_bottom_->mutable_cpu_data()[i + 91] = 18;
			blob_bottom_->mutable_cpu_data()[i + 92] = 13;
			blob_bottom_->mutable_cpu_data()[i + 93] = 17;
			blob_bottom_->mutable_cpu_data()[i + 94] = 8;
			blob_bottom_->mutable_cpu_data()[i + 95] = 10;
			blob_bottom_->mutable_cpu_data()[i + 96] = 19;
			blob_bottom_->mutable_cpu_data()[i + 97] = 6;
			blob_bottom_->mutable_cpu_data()[i + 98] = 1;
			blob_bottom_->mutable_cpu_data()[i + 99] = 22;
			blob_bottom_->mutable_cpu_data()[i + 100] = 30; //
			blob_bottom_->mutable_cpu_data()[i + 101] = 3;
			blob_bottom_->mutable_cpu_data()[i + 102] = 37;
			blob_bottom_->mutable_cpu_data()[i + 103] = 3;
			blob_bottom_->mutable_cpu_data()[i + 104] = 33;
			blob_bottom_->mutable_cpu_data()[i + 105] = 5;
			blob_bottom_->mutable_cpu_data()[i + 106] = 31;
			blob_bottom_->mutable_cpu_data()[i + 107] = 4;
			blob_bottom_->mutable_cpu_data()[i + 108] = 25;
			blob_bottom_->mutable_cpu_data()[i + 109] = 26;
			blob_bottom_->mutable_cpu_data()[i + 110] = 34;
			blob_bottom_->mutable_cpu_data()[i + 111] = 33;
			blob_bottom_->mutable_cpu_data()[i + 112] = 40;
			blob_bottom_->mutable_cpu_data()[i + 113] = 4;
			blob_bottom_->mutable_cpu_data()[i + 114] = 3;
			blob_bottom_->mutable_cpu_data()[i + 115] = 12;
			blob_bottom_->mutable_cpu_data()[i + 116] = 21;
			blob_bottom_->mutable_cpu_data()[i + 117] = 11;
			blob_bottom_->mutable_cpu_data()[i + 118] = 32;
			blob_bottom_->mutable_cpu_data()[i + 119] = 26;
			blob_bottom_->mutable_cpu_data()[i + 120] = 14;
			blob_bottom_->mutable_cpu_data()[i + 121] = 19;
			blob_bottom_->mutable_cpu_data()[i + 122] = 8;
			blob_bottom_->mutable_cpu_data()[i + 123] = 4;
			blob_bottom_->mutable_cpu_data()[i + 124] = 31;
    }
    CuDNNPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->shape(0), num);
    EXPECT_EQ(blob_top_->shape(1), channels);
    EXPECT_EQ(blob_top_->shape(2), 3);
    EXPECT_EQ(blob_top_->shape(3), 2);
    EXPECT_EQ(blob_top_->shape(4), 4);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->shape(0), num);
      EXPECT_EQ(blob_top_mask_->shape(1), channels);
      EXPECT_EQ(blob_top_mask_->shape(2), 3);
      EXPECT_EQ(blob_top_mask_->shape(3), 2);
      EXPECT_EQ(blob_top_mask_->shape(4), 4);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output: 2x 2 channels of:
    // [35    32] [31    25] [29    24] [37    37]
    // [33    33] [31    29] [31    31] [31    31]
    // [34    34] [40    40] [40    40] [33    32]
    for (int i = 0; i < 24 * num * channels; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i +  0], 35);
      EXPECT_EQ(blob_top_->cpu_data()[i +  1], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i +  2], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i +  3], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  4], 32);
      EXPECT_EQ(blob_top_->cpu_data()[i +  5], 25);
      EXPECT_EQ(blob_top_->cpu_data()[i +  6], 24);
      EXPECT_EQ(blob_top_->cpu_data()[i +  7], 37);
      EXPECT_EQ(blob_top_->cpu_data()[i +  8], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i +  9], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 29);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 31);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 34);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 40);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 33);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 34);
			EXPECT_EQ(blob_top_->cpu_data()[i + 21], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 22], 40);
			EXPECT_EQ(blob_top_->cpu_data()[i + 23], 32);
    }
    if (blob_top_vec_.size() > 1) {
      // [0     30] [26     6] [2     42] [19    19]
      // [85    85] [26    87] [88    88] [88    88]
      // [110  110] [112  112] [112  112] [104  118]
      for (int i = 0; i < 24 * num * channels; i += 24) {
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  0],  0);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  1],  26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  2],  2);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  3],  19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  4],  30);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  5],  6);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  6], 42);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  7], 19);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  8], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i +  9], 26);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 85);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 87);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 88);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 110);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 18], 112);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 19], 104);
        EXPECT_EQ(blob_top_mask_->cpu_data()[i + 20], 110);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 21], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 22], 112);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 23], 118);
      }
    }
  }
  // Test for 2x2x2 square pooling layer
	void TestForwardStridePad() {
		LayerParameter layer_param;
		PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
		pooling_param->clear_kernel_size();
		pooling_param->clear_pad();
		pooling_param->add_pad(1);
		pooling_param->clear_stride();
		pooling_param->add_stride(2);
		pooling_param->add_kernel_size(2);
		pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
		const int num = 2;
		const int channels = 2;
		int blobShape[5]={num,channels,3,5,4};
		blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
		// Input: 2x2 channels of:
		//     [1 2 5 2 3] [11 5 5 2 7] [1 0 3 2 3] [3 1 2 2 0]
		//     [9 4 1 4 8] [1 8 5 1 1]  [0 2 4 2 1] [2 0 1 1 4]
		//     [1 2 5 2 3] [2 2 2 1 1]  [1 2 4 1 3] [1 2 1 2 6]
		for (int i = 0; i < 60 * num * channels; i += 60) {
			blob_bottom_->mutable_cpu_data()[i +  0] = 1;
			blob_bottom_->mutable_cpu_data()[i +  1] = 11;
			blob_bottom_->mutable_cpu_data()[i +  2] = 1;
			blob_bottom_->mutable_cpu_data()[i +  3] = 3; //
			blob_bottom_->mutable_cpu_data()[i +  4] = 2;
			blob_bottom_->mutable_cpu_data()[i +  5] = 5;
			blob_bottom_->mutable_cpu_data()[i +  6] = 0;
			blob_bottom_->mutable_cpu_data()[i +  7] = 1; //
			blob_bottom_->mutable_cpu_data()[i +  8] = 5;
			blob_bottom_->mutable_cpu_data()[i +  9] = 5;
			blob_bottom_->mutable_cpu_data()[i + 10] = 3;
			blob_bottom_->mutable_cpu_data()[i + 11] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 12] = 2;
			blob_bottom_->mutable_cpu_data()[i + 13] = 2;
			blob_bottom_->mutable_cpu_data()[i + 14] = 2;
			blob_bottom_->mutable_cpu_data()[i + 15] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 16] = 3;
			blob_bottom_->mutable_cpu_data()[i + 17] = 7;
			blob_bottom_->mutable_cpu_data()[i + 18] = 3;
			blob_bottom_->mutable_cpu_data()[i + 19] = 0; //row 1
			blob_bottom_->mutable_cpu_data()[i + 20] = 9;
			blob_bottom_->mutable_cpu_data()[i + 21] = 1;
			blob_bottom_->mutable_cpu_data()[i + 22] = 0;
			blob_bottom_->mutable_cpu_data()[i + 23] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 24] = 4;
			blob_bottom_->mutable_cpu_data()[i + 25] = 8;
			blob_bottom_->mutable_cpu_data()[i + 26] = 2;
			blob_bottom_->mutable_cpu_data()[i + 27] = 0; //
			blob_bottom_->mutable_cpu_data()[i + 28] = 1;
			blob_bottom_->mutable_cpu_data()[i + 29] = 5;
			blob_bottom_->mutable_cpu_data()[i + 30] = 4;
			blob_bottom_->mutable_cpu_data()[i + 31] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 32] = 4;
			blob_bottom_->mutable_cpu_data()[i + 33] = 1;
			blob_bottom_->mutable_cpu_data()[i + 34] = 2;
			blob_bottom_->mutable_cpu_data()[i + 35] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 36] = 8;
			blob_bottom_->mutable_cpu_data()[i + 37] = 1;
			blob_bottom_->mutable_cpu_data()[i + 38] = 1;
			blob_bottom_->mutable_cpu_data()[i + 39] = 4; //row 2
			blob_bottom_->mutable_cpu_data()[i + 40] = 1;
			blob_bottom_->mutable_cpu_data()[i + 41] = 2;
			blob_bottom_->mutable_cpu_data()[i + 42] = 1;
			blob_bottom_->mutable_cpu_data()[i + 43] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 44] = 2;
			blob_bottom_->mutable_cpu_data()[i + 45] = 2;
			blob_bottom_->mutable_cpu_data()[i + 46] = 2;
			blob_bottom_->mutable_cpu_data()[i + 47] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 48] = 5;
			blob_bottom_->mutable_cpu_data()[i + 49] = 2;
			blob_bottom_->mutable_cpu_data()[i + 50] = 4;
			blob_bottom_->mutable_cpu_data()[i + 51] = 1; //
			blob_bottom_->mutable_cpu_data()[i + 52] = 2;
			blob_bottom_->mutable_cpu_data()[i + 53] = 1;
			blob_bottom_->mutable_cpu_data()[i + 54] = 1;
			blob_bottom_->mutable_cpu_data()[i + 55] = 2; //
			blob_bottom_->mutable_cpu_data()[i + 56] = 3;
			blob_bottom_->mutable_cpu_data()[i + 57] = 1;
			blob_bottom_->mutable_cpu_data()[i + 58] = 3;
			blob_bottom_->mutable_cpu_data()[i + 59] = 6;
		}
		CuDNNPoolingLayer<Dtype> layer(layer_param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);
		EXPECT_EQ(blob_top_->shape(0), num);
		EXPECT_EQ(blob_top_->shape(1), channels);
		EXPECT_EQ(blob_top_->shape(2), 2);
		EXPECT_EQ(blob_top_->shape(3), 3);
		EXPECT_EQ(blob_top_->shape(4), 3);
		if (blob_top_vec_.size() > 1) {
			EXPECT_EQ(blob_top_mask_->shape(0), num);
			EXPECT_EQ(blob_top_mask_->shape(1), channels);
			EXPECT_EQ(blob_top_mask_->shape(2), 2);
			EXPECT_EQ(blob_top_mask_->shape(3), 3);
			EXPECT_EQ(blob_top_mask_->shape(4), 3);
		}
		layer.Forward(blob_bottom_vec_, blob_top_vec_);
		// Expected output: 2x2 channels of:
		//     [1  5   3] [11  5  7] [3  2  2]
		//     [9  5   8] [2   8  3] [2  2  6]
		for (int i = 0; i < 18 * num * channels; i += 18) {
			EXPECT_EQ(blob_top_->cpu_data()[i + 0],  1);
			EXPECT_EQ(blob_top_->cpu_data()[i + 1],  11);
			EXPECT_EQ(blob_top_->cpu_data()[i + 2],  3); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 3],  5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 4],  5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 5],  2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 6],  3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 7],  7);
			EXPECT_EQ(blob_top_->cpu_data()[i + 8],  2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 9],  9);
			EXPECT_EQ(blob_top_->cpu_data()[i + 10], 2);
			EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);
			EXPECT_EQ(blob_top_->cpu_data()[i + 13], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 14], 2); //
			EXPECT_EQ(blob_top_->cpu_data()[i + 15], 8);
			EXPECT_EQ(blob_top_->cpu_data()[i + 16], 3);
			EXPECT_EQ(blob_top_->cpu_data()[i + 17], 6); //
		}
		if (blob_top_vec_.size() > 1) {
			// Expected mask output: 2x 2 channels of:
			//     [0   8 16] [1   5 17] [3  11 15]
			//     [20 48 36] [41 25 58] [23 47 59]
			for (int i = 0; i < 18 * num * channels; i += 18) {
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  0);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  1);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  3); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  8);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5],  11); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6],  16);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  17);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 8],  15); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 9],  20);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 10], 41);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 11], 23); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 12], 48); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 13], 25);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 14], 47);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 15], 36); //
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 16], 58);
				EXPECT_EQ(blob_top_mask_->cpu_data()[i + 17], 59);
			}
		}
	}
};

TYPED_TEST_CASE(CuDNNPoolingLayer3DTest, TestDtypes);

TYPED_TEST(CuDNNPoolingLayer3DTest, TestSetup) {
	Caffe::set_mode(Caffe::GPU);
	LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
}

TYPED_TEST(CuDNNPoolingLayer3DTest, TestSetupPadded) {
	Caffe::set_mode(Caffe::GPU);
	LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
	EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
	EXPECT_EQ(this->blob_top_->shape(2), 4);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
}

TYPED_TEST(CuDNNPoolingLayer3DTest, TestSetupGlobalPooling) {
	Caffe::set_mode(Caffe::GPU);
	LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_global_pooling(true);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
	EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
	EXPECT_EQ(this->blob_top_->shape(2), 1);
	EXPECT_EQ(this->blob_top_->shape(3), 1);
	EXPECT_EQ(this->blob_top_->shape(3), 1);
}

/*
TYPED_TEST(CuDNNPoolingLayer3DTest, PrintBackward) {
	Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, this->blob_bottom_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(CuDNNPoolingLayer3DTest, TestForwardMax) {
	Caffe::set_mode(Caffe::GPU);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}

// Currently, cuDNN does not support the extra top blob.
/*TYPED_TEST(CuDNNPoolingLayer3DTest, TestForwardMaxTopMask) {
	Caffe::set_mode(Caffe::GPU);
	this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardSquare();
  this->TestForwardRectHigh();
  this->TestForwardRectWide();
  this->TestForwardStridePad();
}*/

TYPED_TEST(CuDNNPoolingLayer3DTest, TestGradientMax) {
	Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->clear_pad();
				pooling_param->add_pad(1);
				pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
				CuDNNPoolingLayer<TypeParam> layer(layer_param);
				GradientChecker<TypeParam> checker(1e-4, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
    	}
    }
  }
}

TYPED_TEST(CuDNNPoolingLayer3DTest, TestForwardMaxPadded) {
	Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(2);
  pooling_param->add_pad(2);
  pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
  int blobShape[5]={1,1,3,3,3};
  this->blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
  // Input:
  //     [ 1 2 4 ] [ 1 2 1 ] [ 3 1 2 ]
  //     [ 2 3 2 ] [ 1 1 2 ] [ 0 2 0 ]
  //     [ 4 2 1 ] [ 1 1 2 ] [ 1 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0]  = 1;
  this->blob_bottom_->mutable_cpu_data()[1]  = 1;
  this->blob_bottom_->mutable_cpu_data()[2]  = 3;
  this->blob_bottom_->mutable_cpu_data()[3]  = 2;
  this->blob_bottom_->mutable_cpu_data()[4]  = 2;
  this->blob_bottom_->mutable_cpu_data()[5]  = 1;
  this->blob_bottom_->mutable_cpu_data()[6]  = 4;
  this->blob_bottom_->mutable_cpu_data()[7]  = 1;
  this->blob_bottom_->mutable_cpu_data()[8]  = 2;
  this->blob_bottom_->mutable_cpu_data()[9]  = 2;
	this->blob_bottom_->mutable_cpu_data()[10] = 1;
	this->blob_bottom_->mutable_cpu_data()[11] = 0;
	this->blob_bottom_->mutable_cpu_data()[12] = 3;
	this->blob_bottom_->mutable_cpu_data()[13] = 1;
	this->blob_bottom_->mutable_cpu_data()[14] = 2;
	this->blob_bottom_->mutable_cpu_data()[15] = 2;
	this->blob_bottom_->mutable_cpu_data()[16] = 2;
	this->blob_bottom_->mutable_cpu_data()[17] = 0;
	this->blob_bottom_->mutable_cpu_data()[18] = 4;
	this->blob_bottom_->mutable_cpu_data()[19] = 1;
	this->blob_bottom_->mutable_cpu_data()[20] = 1;
	this->blob_bottom_->mutable_cpu_data()[21] = 2;
	this->blob_bottom_->mutable_cpu_data()[22] = 1;
	this->blob_bottom_->mutable_cpu_data()[23] = 2;
	this->blob_bottom_->mutable_cpu_data()[24] = 1;
	this->blob_bottom_->mutable_cpu_data()[25] = 2;
	this->blob_bottom_->mutable_cpu_data()[26] = 1;
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam epsilon = 1e-8;
  //     [ 1 2 4 ] [ 1 2 1 ] [ 3 1 2 ]
	//     [ 2 3 2 ] [ 1 1 2 ] [ 0 2 0 ]
	//     [ 4 2 1 ] [ 1 1 2 ] [ 1 2 1 ]
  // Output:
  //     [ 1 4 4 ] [ 3 4 4 ] [ 3 3 2 ]
  //     [ 4 4 4 ] [ 4 4 4 ] [ 3 3 2 ]
  //     [ 4 4 1 ] [ 4 4 2 ] [ 1 2 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0],  1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5],  3, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7],  4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8],  2, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[9],  4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[10], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[11], 3, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[12], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[13], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[14], 3, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[15], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[16], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[17], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[18], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[19], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[20], 1, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[21], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[22], 4, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[23], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[24], 1, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[25], 2, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[26], 1, epsilon);
}

// Currently, cuDNN does not support the extra top blob.
/*TYPED_TEST(CuDNNPoolingLayer3DTest, TestGradientMaxTopMask) {
	Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
				this->blob_top_vec_.push_back(this->blob_top_mask_);
				CuDNNPoolingLayer<TypeParam> layer(layer_param);
				GradientChecker<TypeParam> checker(1e-4, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
				this->blob_top_vec_.pop_back();
    	}
    }
  }
}*/

TYPED_TEST(CuDNNPoolingLayer3DTest, TestForwardAve) {
	Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->clear_kernel_size();
  pooling_param->clear_stride();
  pooling_param->clear_pad();
  pooling_param->add_kernel_size(3);
  pooling_param->add_stride(1);
  pooling_param->add_pad(1);
  pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
  int blobShape[5]={1,1,3,3,3};
	this->blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
  FillerParameter filler_param;
  filler_param.set_value(TypeParam(2));
  ConstantFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  CuDNNPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
	EXPECT_EQ(this->blob_top_->shape(1), 1);
	EXPECT_EQ(this->blob_top_->shape(2), 3);
	EXPECT_EQ(this->blob_top_->shape(3), 3);
	EXPECT_EQ(this->blob_top_->shape(4), 3);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  TypeParam epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4],  36.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7],  24.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8],  16.0 / 27, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[9],  24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[10], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[11], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[12], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[13], 54.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[14], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[15], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[16], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[17], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[18], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[19], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[20], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[21], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[22], 36.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[23], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[24], 16.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[25], 24.0 / 27, epsilon);
	EXPECT_NEAR(this->blob_top_->cpu_data()[26], 16.0 / 27, epsilon);
}

TYPED_TEST(CuDNNPoolingLayer3DTest, TestGradientAve) {
	Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
				CuDNNPoolingLayer<TypeParam> layer(layer_param);
				GradientChecker<TypeParam> checker(1e-2, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
			}
    }
  }
}

TYPED_TEST(CuDNNPoolingLayer3DTest, TestGradientAvePadded) {
	Caffe::set_mode(Caffe::GPU);
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
    	for (int kernel_d = 3; kernel_d <= 4; kernel_d++) {
				LayerParameter layer_param;
				PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
				pooling_param->clear_kernel_size();
				pooling_param->add_kernel_size(kernel_h);
				pooling_param->add_kernel_size(kernel_w);
				pooling_param->add_kernel_size(kernel_d);
				pooling_param->clear_stride();
				pooling_param->add_stride(2);
				pooling_param->clear_pad();
				pooling_param->add_pad(2);
				pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
				CuDNNPoolingLayer<TypeParam> layer(layer_param);
				GradientChecker<TypeParam> checker(1e-2, 1e-2);
				checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
						this->blob_top_vec_);
			}
    }
  }
}

#endif

}  // namespace caffe
