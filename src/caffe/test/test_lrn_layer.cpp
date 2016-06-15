#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class LRNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LRNLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LRNLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LRNLayerTest<TypeParam>::ReferenceLRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LRNParameter lrn_param = layer_param.lrn_param();
  Dtype alpha = lrn_param.alpha();
  Dtype beta = lrn_param.beta();
  int size = lrn_param.local_size();
  switch (lrn_param.norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    for (int n = 0; n < blob_bottom.num(); ++n) {
      for (int c = 0; c < blob_bottom.channels(); ++c) {
        for (int h = 0; h < blob_bottom.height(); ++h) {
          for (int w = 0; w < blob_bottom.width(); ++w) {
            int c_start = c - (size - 1) / 2;
            int c_end = min(c_start + size, blob_bottom.channels());
            c_start = max(c_start, 0);
            Dtype scale = 1.;
            for (int i = c_start; i < c_end; ++i) {
              Dtype value = blob_bottom.data_at(n, i, h, w);
              scale += value * value * alpha / size;
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    for (int n = 0; n < blob_bottom.num(); ++n) {
      for (int c = 0; c < blob_bottom.channels(); ++c) {
        for (int h = 0; h < blob_bottom.height(); ++h) {
          int h_start = h - (size - 1) / 2;
          int h_end = min(h_start + size, blob_bottom.height());
          h_start = max(h_start, 0);
          for (int w = 0; w < blob_bottom.width(); ++w) {
            Dtype scale = 1.;
            int w_start = w - (size - 1) / 2;
            int w_end = min(w_start + size, blob_bottom.width());
            w_start = max(w_start, 0);
            for (int nh = h_start; nh < h_end; ++nh) {
              for (int nw = w_start; nw < w_end; ++nw) {
                Dtype value = blob_bottom.data_at(n, c, nh, nw);
                scale += value * value * alpha / (size * size);
              }
            }
            *(top_data + blob_top->offset(n, c, h, w)) =
              blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
          }
        }
      }
    }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

TYPED_TEST_CASE(LRNLayerTest, TestDtypesAndDevices);

TYPED_TEST(LRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestForwardAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestGradientAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestSetupWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayerTest, TestGradientWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


template <typename TypeParam>
class LRNLayer3DTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LRNLayer3DTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    int blobShape[5]={2,7,3,3,3};
    blob_bottom_->Reshape(std::vector<int>(blobShape,blobShape+5));
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LRNLayer3DTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LRNLayer3DTest<TypeParam>::ReferenceLRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->ReshapeLike(blob_bottom);
  Dtype* top_data = blob_top->mutable_cpu_data();
  LRNParameter lrn_param = layer_param.lrn_param();
  Dtype alpha = lrn_param.alpha();
  Dtype beta = lrn_param.beta();
  int size = lrn_param.local_size();
  vector<int> data_offset(5);
  switch (lrn_param.norm_region()) {
		case LRNParameter_NormRegion_ACROSS_CHANNELS:
			for (int n = 0; n < blob_bottom.shape(0); ++n) {
				data_offset[0] = n;
				for (int c = 0; c < blob_bottom.shape(1); ++c) {
					data_offset[1] = c;
					for (int h = 0; h < blob_bottom.shape(2); ++h) {
						data_offset[2] = h;
						for (int w = 0; w < blob_bottom.shape(3); ++w) {
							data_offset[3] = w;
							for (int d = 0; d < blob_bottom.shape(4); ++d) {
								data_offset[4] = d;
								int c_start = c - (size - 1) / 2;
								int c_end = min(c_start + size, blob_bottom.shape(1));
								c_start = max(c_start, 0);
								Dtype scale = 1.;
								for (int i = c_start; i < c_end; ++i) {
									data_offset[1] = i;
									Dtype value = blob_bottom.data_at(data_offset);
									scale += value * value * alpha / size;
								}
								data_offset[1] = c;
								*(top_data + blob_top->offset(data_offset)) =
									blob_bottom.data_at(data_offset) / pow(scale, beta);
							}
						}
					}
				}
			}
			break;
		case LRNParameter_NormRegion_WITHIN_CHANNEL:
			for (int n = 0; n < blob_bottom.shape(0); ++n) {
				data_offset[0] = n;
				for (int c = 0; c < blob_bottom.shape(1); ++c) {
					data_offset[1] = c;
					for (int h = 0; h < blob_bottom.shape(2); ++h) {
						int h_start = h - (size - 1) / 2;
						int h_end = min(h_start + size, blob_bottom.shape(2));
						h_start = max(h_start, 0);
						for (int w = 0; w < blob_bottom.shape(3); ++w) {
							int w_start = w - (size - 1) / 2;
							int w_end = min(w_start + size, blob_bottom.shape(3));
							w_start = max(w_start, 0);
							for (int d = 0; d < blob_bottom.shape(4); ++d) {
								int d_start = d - (size - 1) / 2;
								int d_end = min(d_start + size, blob_bottom.shape(4));
								d_start = max(d_start, 0);
								Dtype scale = 1.;
								for (int nh = h_start; nh < h_end; ++nh) {
									for (int nw = w_start; nw < w_end; ++nw) {
										for (int nd = d_start; nd < d_end; ++nd) {
											data_offset[2] = nh;
											data_offset[3] = nw;
											data_offset[4] = nd;
											Dtype value = blob_bottom.data_at(data_offset);
											scale += (value * value * alpha / (size * size * size));
										}
									}
								}
								data_offset[2] = h;
								data_offset[3] = w;
								data_offset[4] = d;
								*(top_data + blob_top->offset(data_offset)) =
									blob_bottom.data_at(data_offset) / pow(scale, beta);
							}
						}
					}
				}
			}
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
  }
}

TYPED_TEST_CASE(LRNLayer3DTest, TestDtypesAndDevices);

TYPED_TEST(LRNLayer3DTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 7);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
}

TYPED_TEST(LRNLayer3DTest, TestForwardAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayer3DTest, TestForwardAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayer3DTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayer3DTest, TestGradientAcrossChannelsLargeRegion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_local_size(15);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down,
                 this->blob_bottom_vec_);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(LRNLayer3DTest, TestSetupWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 7);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
}

TYPED_TEST(LRNLayer3DTest, TestForwardWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNLayer3DTest, TestGradientWithinChannel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_lrn_param()->set_norm_region(
      LRNParameter_NormRegion_WITHIN_CHANNEL);
  layer_param.mutable_lrn_param()->set_local_size(3);
  LRNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
