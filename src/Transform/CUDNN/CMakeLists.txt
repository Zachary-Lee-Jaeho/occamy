add_core_dnn_library(CDCUDNNTransform
  CUDNNDeallocOpt.cpp
  eraseDummyConstants.cpp
  FunctionArgument.cpp
  mallocPoolOpt.cpp

  LINK_LIBS PUBLIC
  CDCUDNNOps
  CDCUDNNTransformOpFusing
  MLIRTransformUtils
)

add_subdirectory(opFusing)
