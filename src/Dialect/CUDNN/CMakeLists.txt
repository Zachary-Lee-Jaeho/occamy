# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the CUDNNOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_core_dnn_dialect(CUDNN cudnn)
add_core_dnn_dialect_doc(cudnn CUDNN.td)

add_core_dnn_library(CDCUDNNOps
    CUDNNOps.cpp

    DEPENDS
    CDCUDNNIncGen
    )
