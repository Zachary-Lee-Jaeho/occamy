add_core_dnn_library(CDONNXToCUDNN_CUBLAS
    Gemm.cpp

    LINK_LIBS PUBLIC
    OMONNXOps
    CDCUDNNOps
)
