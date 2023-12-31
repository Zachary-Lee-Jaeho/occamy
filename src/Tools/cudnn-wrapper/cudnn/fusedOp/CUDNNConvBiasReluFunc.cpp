#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>

#define DEBUG 0

extern "C"
float* CUDNNConvBiasActivFunc(cudnnHandle_t cudnnHandle,
    float *inData_d, int64_t dimX[4],
    float *filterData_d, int64_t dimw[4],
    float *biasData_d, int64_t dimB[4],
    float *workspace, int64_t workspaceSize, int64_t algoValue,
    int64_t pads[2], int64_t strides[2],
    int64_t activMode, float *outData_d) {


#if DEBUG
  printf("\ndimX -> %ld, %ld, %ld, %ld\n", dimX[0] , dimX[1] , dimX[2] ,dimX[3]);
  printf("dimw -> %ld, %ld, %ld, %ld\n\n", dimw[0] , dimw[1] , dimw[2] ,dimw[3]);


  float *X;
  X = (float*) malloc(sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3]);
  cudaMemcpy(X, inData_d, sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3], (cudaMemcpyKind) 2);

  printf("[ConvFused] inData_t Addr -> %p, Size -> %ld\n", inData_d, sizeof(float) * dimX[0] * dimX[1] * dimX[2] * dimX[3]);
  printf("[ConvFused] inData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7]);
  free(X);

  float *f;
  f = (float*) malloc(sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3]);
  cudaMemcpy(f, filterData_d, sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3], (cudaMemcpyKind) 2);

  printf("[ConvFused] filterData_t Addr -> %p, Size -> %ld\n", filterData_d, sizeof(float) * dimw[0] * dimw[1] * dimw[2] * dimw[3]);
  printf("[ConvFused] filterData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n",
      f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  free(f);

  float *B;
  B = (float*) malloc(sizeof(float) * dimB[0] * dimB[1] * dimB[2] * dimB[3]);
  cudaMemcpy(B, biasData_d, sizeof(float) * dimB[0] * dimB[1] * dimB[2] * dimB[3], (cudaMemcpyKind) 2);

  printf("[ConvFused] biasData_t Addr -> %p, Size -> %ld\n", inData_d, sizeof(float) * dimB[0] * dimB[1] * dimB[2] * dimB[3]);
  printf("[ConvFused] biasData_t -> %.9f, %.9f, %.9f\n",
      B[0], B[1], B[2]);
  free(B);
#endif
  int error = 5555;

  cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc;
  cudnnFilterDescriptor_t filterDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnActivationDescriptor_t activDesc;

  cudnnCreateTensorDescriptor(&inTensorDesc);
  cudnnCreateTensorDescriptor(&outTensorDesc);
  cudnnCreateTensorDescriptor(&biasTensorDesc);
  cudnnCreateFilterDescriptor(&filterDesc);
  cudnnCreateConvolutionDescriptor(&convDesc);

  // Make activation discriptor
  // Only ReLU operation is supported
  cudnnCreateActivationDescriptor(&activDesc);
  cudnnSetActivationDescriptor(activDesc,
      (cudnnActivationMode_t) activMode,
      /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
      /*relu_coef=*/0);

  cudnnSetTensor4dDescriptor(
      inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimX[0], dimX[1], dimX[2], dimX[3]);

  cudnnSetTensor4dDescriptor(
      biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dimB[0], dimB[1], dimB[2], dimB[3]);

  cudnnSetFilter4dDescriptor(
      filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
      dimw[0], dimw[1], dimw[2], dimw[3]);


   cudnnSetConvolution2dDescriptor(convDesc,
      (int)pads[0], (int)pads[1],
      (int)strides[0], (int)strides[1],
      1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

  int out_n, out_c, out_h, out_w;
  cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w);
  cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

  float alpha = 1;
  float beta = 0;
  error = (int)cudnnConvolutionBiasActivationForward(cudnnHandle,
      &alpha,
      inTensorDesc,
      inData_d,
      filterDesc,
      filterData_d,
      convDesc,
      (cudnnConvolutionFwdAlgo_t) algoValue,
      (void*) workspace,
      workspaceSize,
      &beta,
      outTensorDesc,
      outData_d,
      biasTensorDesc,
      biasData_d,
      activDesc,
      outTensorDesc,
      outData_d);

#if DEBUG
  printf("[ConvFused] conv result -> %d\n", error);

  printf("\npads[0, 1], strides[0, 1] : %ld, %ld, %ld, %ld\n", pads[0], pads[1], strides[0], strides[1]);

  float *y;
  y = (float*) malloc(sizeof(float) * out_n * out_c * out_h * out_w);
  printf("%p\n", y);
  error = cudaMemcpy(y, outData_d, sizeof(float) * out_n * out_c * out_h * out_w, (cudaMemcpyKind) 2);
  printf("[ConvFused] out memcpy result -> %d\n", error);


  printf("[ConvFused] outData_t Addr -> %p, Size -> %ld\n", (void*)outData_d, sizeof(float) * out_n * out_c * out_h * out_w);
  printf("[ConvFused] outData_t -> %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f\n\n",
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  free(y);
#endif

  cudnnDestroyTensorDescriptor(inTensorDesc);
  cudnnDestroyTensorDescriptor(biasTensorDesc);
  cudnnDestroyTensorDescriptor(outTensorDesc);
  cudnnDestroyFilterDescriptor(filterDesc);
  cudnnDestroyConvolutionDescriptor(convDesc);
  cudnnDestroyActivationDescriptor(activDesc);

  return outData_d;
}
