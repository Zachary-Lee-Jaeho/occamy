add_core_dnn_library(CDCUDNNRuntime
    CUDNNRuntimeAPI.cpp
    CUDNNRuntimeAPI.hpp

    LINK_LIBS PUBLIC
    CDCUDNNOps
)

add_core_dnn_library(CDCUDNNToLLVM
    CUDNNToLLVMCommon.cpp
    ConvertCUDNNToLLVM.cpp

    LINK_LIBS PUBLIC
    CDCUDNNRuntime
    CDCUDNNToLLVM_CUDNN
    CDCUDNNToLLVM_CUBLAS
    CDCUDNNToLLVM_CUDA
    CDCUDNNOps
)

add_subdirectory(Ops)
