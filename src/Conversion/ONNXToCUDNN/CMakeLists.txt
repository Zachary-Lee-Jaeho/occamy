# SPDX-License-Identifier: Apache-2.0
add_core_dnn_library(CDONNXToCUDNN_COMMON_CUDA
    ONNXToCUDNNCommonCUDA.cu

    LINK_LIBS PUBLIC
    cudart
    cudnn
    )

target_include_directories(CDONNXToCUDNN_COMMON_CUDA
    PRIVATE
    /usr/local/cuda/include
    )


add_core_dnn_library(CDONNXToCUDNN
    ConvertONNXToCUDNN.cpp
    ONNXToCUDNNCommon.cpp
    Ops/Constant.cpp
    Ops/ConstantOfShape.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDCUDNNOps
    CDONNXToCUDNN_COMMON_CUDA
    CDONNXToCUDNN_CUDNN
    CDONNXToCUDNN_CUBLAS
    CDONNXToCUDNN_CUDA
    )

add_subdirectory(Ops)

