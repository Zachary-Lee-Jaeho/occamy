add_core_dnn_library(CDCUDNNToLLVM_CUDNN_FUSED
    ConvBiasActiv.cpp

    LINK_LIBS PUBLIC
    CDCUDNNRuntime
    CDCUDNNOps
)

