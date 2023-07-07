add_core_dnn_library(CDCUDNNToLLVM_CUBLAS
    Matmul2d.cpp

    LINK_LIBS PUBLIC
    CDCUDNNRuntime
    CDCUDNNOps
)

