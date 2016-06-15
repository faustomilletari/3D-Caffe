#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

namespace caffe {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    int num_dim, int dim_tensor[], int stride[]) {
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, dataType<Dtype>::type,
        num_dim, dim_tensor, stride));
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    int num_dim, int dim_tensor[]) {
	int stride[num_dim];
	stride[num_dim-1] = 1;
	for(int i = num_dim-2; i>=0; i--)
	{
		stride[i] = stride[i+1] * dim_tensor[i+1];
	}
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, dataType<Dtype>::type,
        num_dim, dim_tensor, stride));
}

template <typename Dtype>
inline void createFilter4dDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
      n, c, h, w));
}

template <typename Dtype>
inline void createFilterNdDesc(cudnnFilterDescriptor_t* desc,
    int num_dim, int dim_filter[]) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(*desc, dataType<Dtype>::type,CUDNN_TENSOR_NCHW, num_dim, dim_filter));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolution2dDesc(cudnnConvolutionDescriptor_t* conv,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void setConvolutionNdDesc(cudnnConvolutionDescriptor_t* conv,
    int num_dim, int pad[], int stride[]) {
	int upscale[num_dim];
	for(int i = 0; i<num_dim; i++){
		upscale[i] = 1;
	}
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(*conv, num_dim,
      pad, stride, upscale, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
}

template <typename Dtype>
inline void createPooling2dDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, CUDNN_NOT_PROPAGATE_NAN,h, w,
        pad_h, pad_w, stride_h, stride_w));
}

template <typename Dtype>
inline void createPoolingNdDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
		int num_dim, int filter_dim[], int pad[], int stride[]) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc, *mode,CUDNN_NOT_PROPAGATE_NAN,
  		num_dim, filter_dim, pad, stride));
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
