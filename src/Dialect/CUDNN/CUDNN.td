#ifndef CUDNN_OPS
#define CUDNN_OPS

//===---- CUDNN.td - CUDNN Dialect Operation Definitions ---===//
//
//===----------------- XXX corelab jaeho XXX ------------------===//
//
//===------------ defines CUDNN Dialect operations. -----------===//

include "mlir/IR/OpBase.td"

def CUDNN_Dialect : Dialect {
    let name = "cudnn";
    let cppNamespace = "::mlir";
}

class CUDNN_Op<string mnemonic, list<Trait> traits = []> :
    Op<CUDNN_Dialect, mnemonic, traits> ;

def CUDNNMallocOp : Op<CUDNN_Dialect, "malloc"> {
  let summary = "cudnn malloc operation";
  let description = [{
      The "cudnn.malloc" operation allocates memory on the device.
  }];

  let arguments = (ins I64:$size);
  let results = (outs AnyMemRef:$devPtr);
}

def CUDNNMemPoolInitOp : Op<CUDNN_Dialect, "mempool-init"> {
  let summary = "cudnn memory pool initializing operation";
  let description = [{
      The "cudnn.mempool_init" operation initializes the global memory pool.
  }];

  let arguments = (ins I64:$size);
  let results = (outs AnyMemRef:$devPtr);
}

def CUDNNMemOffsetOp : Op<CUDNN_Dialect, "mem-offset"> {
  let summary = "cudnn memeory offset operation";
  let description = [{
      The "cudnn.mem_offset" operation returns the baseAddr + memOffset
      from the global memory pool.
  }];

  let arguments = (ins AnyMemRef:$basePtr, I64:$offset, I64:$mallocSize);
  let results = (outs AnyMemRef:$devPtr);
}

def CUDNNDeallocOp : Op<CUDNN_Dialect, "dealloc"> {
  let summary = "cudnn dealloc operation";
  let description = [{
      The "cudnn.dealloc" operation deallocates memory on the device.
  }];

  let arguments = (ins AnyMemRef:$devPtr);
  let results = (outs I32:$cudaError);
}

def CUDNNMemcpyOp : Op<CUDNN_Dialect, "memcpy"> {
  let summary = "cudnn memcpy operation";
  let description = [{
      The "cudnn.memcpy" operation copies data from src to dst.

      cudaMemcpyHostToHost = 0
        Host -> Host
      cudaMemcpyHostToDevice = 1
        Host -> Device
      cudaMemcpyDeviceToHost = 2
        Device -> Host
      cudaMemcpyDeviceToDevice = 3
        Device -> Device
      cudaMemcpyDefault = 4
        Direction of the transfer is inferred from the pointer values.
        Requires unified virtual addressing
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$dst,
                    AnyTypeOf<[AnyMemRef, AnyTensor]>:$src,
                    I64:$count, I32Attr:$mode);
  let results = (outs I32:$cudaError);
}

def CUDNNHandleOp : CUDNN_Op<"handle"> {
  let summary = "cudnn handle making operation";
  let description = [{
      The "cudnn.handle" operation is used to make handle for cudnn library.
  }];

  let arguments = (ins );
  let results = (outs AnyMemRef:$handle);
}

def CUDNNConvForwardOp : CUDNN_Op<"convfwd"> {
  let summary = "cudnn conv forward operation";
  let description = [{
      The "cudnn.convfwd" operation is used to perform forward convolution.
      The order of the padding argument is [padding_h, padding_w]
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                      I64ArrayAttr:$dimX,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$W,
                      I64ArrayAttr:$dimW,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$workspace,
                      I64Attr:$workspaceSize,
                      I64ArrayAttr:$pads,
                      I64ArrayAttr:$strides,
                      I64Attr:$convAlgorithm,
                      I64Attr:$group,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$out);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, NoneType]>:$output);
}

def CUDNNConvBiasActivForwardOp : CUDNN_Op<"conv-bias-relufwd"> {
  let summary = "cudnn fused conv, add, relu operation";
  let description = [{
      The "cudnn.conv-bias-relufwd" operation is used to perform forward convolution with bias addition and activation ReLU.
      The only supported activation function is ReLU.
      The order of the padding argument is [padding_h, padding_w]
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                      I64ArrayAttr:$dimX,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$W,
                      I64ArrayAttr:$dimW,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$B,
                      I64ArrayAttr:$dimB,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$workspace,
                      I64Attr:$workspaceSize,
                      I64ArrayAttr:$pads,
                      I64ArrayAttr:$strides,
                      I64Attr:$activMode,
                      I64Attr:$convAlgorithm,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$out);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, NoneType]>:$output);
}

def CUDNNActivationForwardOp : CUDNN_Op<"activationfwd"> {
  let summary = "cudnn activation forward operation";
  let description = [{
      The "cudnn.activationfwd" operation is used to perform forward activation.

      Values
      CUDNN_ACTIVATION_SIGMOID
      Selects the sigmoid function.

      CUDNN_ACTIVATION_RELU
      Selects the rectified linear function.

      CUDNN_ACTIVATION_TANH
      Selects the hyperbolic tangent function.

      CUDNN_ACTIVATION_CLIPPED_RELU
      Selects the clipped rectified linear function.

      CUDNN_ACTIVATION_ELU
      Selects the exponential linear function.

      CUDNN_ACTIVATION_IDENTITY
      Selects the identity function, intended for bypassing the activation step
      in cudnnConvolutionBiasActivationForward(). (The
      cudnnConvolutionBiasActivationForward() function must use
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM.)
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$Y,
                      I64ArrayAttr:$dimX,
                      I32Attr:$mode);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, NoneType]>:$output);
}

def CUDNNPReluOp : Op<CUDNN_Dialect, "prelu"> {
  let summary = "cudnn prelu tensor operation";
  let description = [{
      The "cudnn.prelu" operation prelu of tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$slope,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNLeakyReluOp : CUDNN_Op<"leakyrelu"> {
  let summary = "cudnn leaky relu operation";
  let description = [{
      The "cudnn.leakyrelu" operation is used to perform leaky relu funtion.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X,
                      AnyTypeOf<[AnyMemRef, AnyTensor]>:$Y,
                      I64ArrayAttr:$dimX,
                      F32Attr:$alpha);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, AnyMemRef, NoneType]>:$output);
}

def CUDNNAddOp : Op<CUDNN_Dialect, "add"> {
  let summary = "cudnn add tensor operation";
  let description = [{
      The "cudnn.add" operation adds two tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       F32Attr:$biasB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNNegativeOp : Op<CUDNN_Dialect, "negative"> {
  let summary = "cudnn negative tensor operation";
  let description = [{
    The "cudnn.negative" operation computes elementwise negative.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

// Use negative and add to compute sub
def CUDNNSubOp : Op<CUDNN_Dialect, "sub"> {
  let summary = "cudnn subtraction tensor operation";
  let description = [{
      The "cudnn.sub" operation subtracts two tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       F32Attr:$biasB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNMulOp : Op<CUDNN_Dialect, "mul"> {
  let summary = "cudnn mul tensor operation";
  let description = [{
      The "cudnn.mul" operation multiplies two tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNReciprocalOp : Op<CUDNN_Dialect, "reciprocal"> {
  let summary = "cudnn reciprocal tensor operation";
  let description = [{
    The "cudnn.reciprocal" operation computes elementwise reciprocal.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

// Use reciprocal and mul to compute div
def CUDNNDivOp : Op<CUDNN_Dialect, "div"> { // DEPRICATED OP
  let summary = "cudnn div tensor operation";
  let description = [{
      The "cudnn.div" operation divides two tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNSqrtOp : Op<CUDNN_Dialect, "sqrt"> {
  let summary = "cudnn sqrt tensor operation";
  let description = [{
      The "cudnn.sqrt" operation takes elementwise square root to input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNReduceOp: Op<CUDNN_Dialect, "reduce"> {
  let summary = "cudnn reduce tensor operation";
  let description = [{
      The "cudnn.reduce" operation reduces the input tensor.

      Mode Enum Values
      CUDNN_REDUCE_TENSOR_ADD : The operation to be performed is addition.
      CUDNN_REDUCE_TENSOR_MUL : The operation to be performed is multiplication.
      CUDNN_REDUCE_TENSOR_MIN : The operation to be performed is a minimum comparison.
      CUDNN_REDUCE_TENSOR_MAX : The operation to be performed is a maximum comparison.
      CUDNN_REDUCE_TENSOR_AMAX : The operation to be performed is a maximum comparison of absolute values.
      CUDNN_REDUCE_TENSOR_AVG : The operation to be performed is averaging.
      CUDNN_REDUCE_TENSOR_NORM1 : The operation to be performed is addition of absolute values.
      CUDNN_REDUCE_TENSOR_NORM2 : The operation to be performed is a square root of the sum of squares.
      CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS : The operation to be performed is multiplication, not including elements of value zero
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I64ArrayAttr:$preFlattenOutDim,
                       I32Attr:$mode);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNMaxPoolOp: Op<CUDNN_Dialect, "maxpool"> {
  let summary = "cudnn reduce tensor operation";
  let description = [{
      The "cudnn.maxpool" operation reduces the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I64ArrayAttr:$dilations,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$pads,
                       I64ArrayAttr:$strides);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNAveragePoolOp: Op<CUDNN_Dialect, "averagepool"> {
  let summary = "cudnn reduce tensor operation";
  let description = [{
      The "cudnn.averagepool" operation reduces the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$pads,
                       I64ArrayAttr:$strides);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNSoftmaxOp: Op<CUDNN_Dialect, "softmax"> {
  let summary = "cudnn reduce tensor operation";
  let description = [{
      The "cudnn.softmax" operation calculates softmax to the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNFlattenOp: Op<CUDNN_Dialect, "flatten"> {
  let summary = "cudnn flatten tensor operation";
  let description = [{
      The "cudnn.flatten" operation flattens the input tensor to 2 dimension.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I32Attr:$axis);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNUnsqueezeOp: Op<CUDNN_Dialect, "unsqueeze"> {
  let summary = "cudnn unsqueeze tensor operation";
  let description = [{
      The "cudnn.unsqueeze" operation unsqueezes the axis given of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I64ArrayAttr:$axes);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNSqueezeOp: Op<CUDNN_Dialect, "squeeze"> {
  let summary = "cudnn squeeze tensor operation";
  let description = [{
      The "cudnn.squeeze" operation squeezes the axis given of the input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimInput,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimOutput,
                       I64ArrayAttr:$axes);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNMatmul2dOp: Op<CUDNN_Dialect, "matmul2d"> {
  let summary = "cudnn general mat-mat multiply operation";
  let description = [{
      The "cudnn.matmul2d" operation multiply two 2 dimensions tensor.
      (with transpose feature.)
      output = alpha * (A matmul2d B).
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       F32Attr:$alpha,
                       F32Attr:$beta,
                       I64Attr:$transA,
                       I64Attr:$transB);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNMatmulNdOp: Op<CUDNN_Dialect, "matmulnd"> {
  let summary = "cudnn general mat-mat multiply operation";
  let description = [{
      The "cudnn.matmulnd" operation multiply two N dimensions tensor.
      Matrix product that behaves like numpy.matmul
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$A,
                       I64ArrayAttr:$dimA,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$B,
                       I64ArrayAttr:$dimB,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNGatherOp: Op<CUDNN_Dialect, "gather"> {
  let summary = "cuda coded tensor gather operation";
  let description = [{
      The "cudnn.gather" operation gathers the input tensor
      within the given indicies.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$indices,
                       I64ArrayAttr:$dimIndices,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNConcatOp: Op<CUDNN_Dialect, "concat"> {
  let summary = "cuda coded tensor concat operation";
  let description = [{
      The "cudnn.concat" operation concatenate the input tensor
      along the given axis.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       I64Attr:$axis);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNReshapeOp: Op<CUDNN_Dialect, "reshape"> {
  let summary = "cuda coded tensor reshape operation";
  let description = [{
      The "cudnn.reshape" operation reshape the input tensor
      into the given shape. (-1 shape supported. Similar with numpy.reshape.)
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNNonZeroOp: Op<CUDNN_Dialect, "nonzero"> {
  let summary = "cuda coded nonzero operation";
  let description = [{
      The "cudnn.nonzero" operation Returns the indices of the elements
      that are non-zero (in row-major order - by dimension). NonZero
      behaves similar to numpy.nonzero.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNTransposeOp: Op<CUDNN_Dialect, "transpose"> {
  let summary = "cuda coded transpose operation";
  let description = [{
      The "cudnn.transpose" operation transposes the input tensor similar to
      numpy.transpose. For example, when perm=(1, 0, 2), given an input
      tensor of shape (1, 2, 3), the output shape will be (2, 1, 3).
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       I64ArrayAttr:$perm);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNExpandOp: Op<CUDNN_Dialect, "expand"> {
  let summary = "cuda coded expanded operation";
  let description = [{
      The "cudnn.expand" operation broadcasts the input tensor following the
      given shape and the broadcast rule.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       I64ArrayAttr:$dimX,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$shape,
                       I64ArrayAttr:$dimShape);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNPowOp: Op<CUDNN_Dialect, "pow"> {
  let summary = "cuda coded elementwise power operation";
  let description = [{
      The "cudnn.pow" operation computes elementwise power of the input tensor.
      Now only supports a single number exponent, not a tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X, // Input
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y, // Exponent
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Z, // Output (malloc)
                       I64ArrayAttr:$dimZ);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNErfOp: Op<CUDNN_Dialect, "erf"> {
  let summary = "cuda coded Gauss error function calculation operation";
  let description = [{
      The "cudnn.erf" operation computes Gauss error function of the given
      input tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$result,
                       I64ArrayAttr:$dimResult);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNCastOp: Op<CUDNN_Dialect, "cast"> {
  let summary = "cuda coded data casting operation";
  let description = [{
      The "cudnn.cast" operation casts the data of the given
      input tensor to given type.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY,
                       I64Attr:$to);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

def CUDNNClipOp: Op<CUDNN_Dialect, "clip"> {
  let summary = "cuda coded tensor clipping operation";
  let description = [{
      The "cudnn.clip" operation limits the given input within an interval.
  }];

  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$X,
                       F32Attr:$min,
                       F32Attr:$max,
                       AnyTypeOf<[AnyTensor, AnyMemRef]>:$Y,
                       I64ArrayAttr:$dimY);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$output);
}

#endif // CUDNN_OPS
