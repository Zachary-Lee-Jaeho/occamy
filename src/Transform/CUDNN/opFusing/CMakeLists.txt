add_core_dnn_library(CDCUDNNTransformOpFusing
  fuseConvBiasRelu.cpp

  LINK_LIBS PUBLIC
  CDCUDNNOps
  MLIRTransformUtils
)
