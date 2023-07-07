add_core_dnn_library(CDCUDNNToLLVM_CUDNN
    ConvForward.cpp
    ActivationForward.cpp
    Add.cpp
    Mul.cpp
    Sqrt.cpp
    Reduce.cpp
    MaxPool.cpp
    AveragePool.cpp

    LINK_LIBS PUBLIC
    CDCUDNNRuntime
    CDCUDNNToLLVM_CUDNN_FUSED
    CDCUDNNOps
)

add_subdirectory(fusedOps)
