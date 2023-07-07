
# Must unset LLVM_DIR in cache. Otherwise, when MLIR_DIR changes LLVM_DIR
# won't change accordingly.
unset(LLVM_DIR CACHE)
if (NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR is not configured but it is required. "
    "Set the cmake option MLIR_DIR, e.g.,\n"
    "    cmake -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir ..\n "
    )
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# add_definitions(${LLVM_DEFINITIONS})
#
# set(BUILD_SHARED_LIBS ${LLVM_ENABLE_SHARED_LIBS} CACHE BOOL "" FORCE)
# message(STATUS "BUILD_SHARED_LIBS       : " ${BUILD_SHARED_LIBS})
#
# onnx uses exceptions, so we need to make sure that LLVM_REQUIRES_EH is set to ON, so that
# the functions from HandleLLVMOptions and AddLLVM don't disable exceptions.
set(LLVM_REQUIRES_EH ON)
message(STATUS "LLVM_REQUIRES_EH        : " ${LLVM_REQUIRES_EH})
#
# # LLVM_HOST_TRIPLE is exported as part of the llvm config, so we should be able to leverage it.
# # If, for some reason, it is not set, default to an empty string which is the old default behavior of on
# set(ONNX_MLIR_DEFAULT_TRIPLE "${LLVM_HOST_TRIPLE}" CACHE STRING "Default triple for onnx-mlir.")
# message(STATUS "ONNX_MLIR_DEFAULT_TRIPLE: " ${ONNX_MLIR_DEFAULT_TRIPLE})
#
# # If CMAKE_INSTALL_PREFIX was not provided explicitly and we are not using an install of
# # LLVM and a CMakeCache.txt exists,
# # force CMAKE_INSTALL_PREFIX to be the same as the LLVM build.
# if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND NOT LLVM_INSTALL_PREFIX)
#   if (EXISTS ${LLVM_BINARY_DIR}/CMakeCache.txt)
#     file(STRINGS ${LLVM_BINARY_DIR}/CMakeCache.txt prefix REGEX CMAKE_INSTALL_PREFIX)
#     string(REGEX REPLACE "CMAKE_INSTALL_PREFIX:PATH=" "" prefix ${prefix})
#     string(REGEX REPLACE "//.*" "" prefix ${prefix})
#     set(CMAKE_INSTALL_PREFIX ${prefix} CACHE PATH "" FORCE)
#   endif()
# endif()
# message(STATUS "CMAKE_INSTALL_PREFIX    : " ${CMAKE_INSTALL_PREFIX})

# Support compilation of .cu files (nvcc)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_library(CUDA_RUNTIME_LIBRARY cuda)
  add_definitions(-DENABLE_CUDA_RUNTIME=1)
endif()

# The tablegen functions below are modeled based on the corresponding functions
# in mlir: https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIR.cmake
function(add_core_dnn_dialect_doc dialect dialect_tablegen_file)
  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_tablegen_file})
  tablegen(MLIR ${dialect}.md -gen-op-doc -I${ONNX_MLIR_SRC_ROOT} -I${CORE_DNN_SRC_ROOT})
  set(GEN_DOC_FILE ${CORE_DNN_SRC_ROOT}/docs/Dialects/${dialect}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect}.md)
  add_custom_target(${dialect}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(core-dnn-docs ${dialect}DocGen)
endfunction()
add_custom_target(core-dnn-docs)

# If an extra parameter, the dialect name, is provided,
# this function will generate dialect and type from the td file
function(add_core_dnn_dialect dialect dialect_name)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}Ops.hpp.inc -gen-op-decls -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Ops.cpp.inc -gen-op-defs -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Dialect.hpp.inc -gen-dialect-decls -dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Types.hpp.inc -gen-typedef-decls -typedefs-dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Attributes.hpp.inc -gen-attrdef-decls -attrdefs-dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  mlir_tablegen(${dialect}Attributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=${dialect_name} -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  add_public_tablegen_target(CD${dialect}IncGen)
endfunction()

function(add_core_dnn_rewriter rewriter)
  set(LLVM_TARGET_DEFINITIONS ${rewriter}.td)
  mlir_tablegen(CORE${rewriter}.inc -gen-rewriters -I${CORE_DNN_SRC_ROOT} -I${ONNX_MLIR_SRC_ROOT})
  add_public_tablegen_target(CDCORE${rewriter}IncGen)
endfunction()


# add_core_dnn_library(name sources...
#   This function (generally) has the same semantic as add_library. In
#   addition it supports the arguments below and it does the following
#   by default (unless an argument overrides this):
#   1. Add the library
#   2. Add the default target_include_directories
#   3. Add the library to a global property CORE_DNN_LIBS
#   4. Add an install target for the library
#   EXCLUDE_FROM_CD_LIBS
#     Do not add the library to the CORE_DNN_LIBS property.
#   NO_INSTALL
#     Do not add an install target for the library.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   INCLUDE_DIRS include_dirs...
#     Same semantics as target_include_directories().
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   LINK_COMPONENTS llvm_components...
#     Link the specified LLVM components.
#     Note: only one linkage mode can be specified.
#   )
function(add_core_dnn_library name)
  cmake_parse_arguments(ARG
    "EXCLUDE_FROM_CD_LIBS;NO_INSTALL"
    ""
    "DEPENDS;INCLUDE_DIRS;ACCEL_INCLUDE_DIRS;LINK_LIBS;LINK_COMPONENTS"
    ${ARGN}
    )

  if (NOT ARG_EXCLUDE_FROM_CD_LIBS)
    set_property(GLOBAL APPEND PROPERTY CORE_DNN_LIBS ${name})
  endif()

  add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
  llvm_update_compile_flags(${name})

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_INCLUDE_DIRS})
  endif()

  if (ARG_ACCEL_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_ACCEL_INCLUDE_DIRS})
  endif()

  target_include_directories(${name}
    PUBLIC
    ${ONNX_MLIR_SRC_ROOT}
    ${ONNX_MLIR_BIN_ROOT}
    ${CORE_DNN_SRC_ROOT}
    ${CORE_DNN_BIN_ROOT}
    )

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} ${ARG_LINK_LIBS})
  endif()

  if (ARG_LINK_COMPONENTS)
    set(LinkageMode)
    if (ARG_LINK_COMPONENTS MATCHES "^(PUBLIC|PRIVATE|INTERFACE)")
      list(POP_FRONT ARG_LINK_COMPONENTS LinkageMode)
    endif()

    llvm_map_components_to_libnames(COMPONENT_LIBS ${ARG_LINK_COMPONENTS})

    if (LinkageMode)
      target_link_libraries(${name} ${LinkageMode} ${COMPONENT_LIBS})
    else()
      target_link_libraries(${name} PRIVATE ${COMPONENT_LIBS})
    endif()
  endif()

  if (NOT ARG_NO_INSTALL)
    install(TARGETS ${name}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      RUNTIME DESTINATION bin
      )
  endif()
endfunction(add_core_dnn_library)


# add_core_dnn_executable(name sources...
#   This function (generally) has the same semantic as add_executable.
#   In addition is supports the arguments below and it does the following
#   by default (unless an argument overrides this):
#   1. Add the executable
#   2. Add an install target for the executable
#   NO_INSTALL
#     Do not add an install target for the executable.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   INCLUDE_DIRS include_dirs...
#     Same semantics as target_include_directories().
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   DEFINE define_targets...
#     Same semantics as target_compile_definitions()
#   )
function(add_core_dnn_executable name)
  cmake_parse_arguments(ARG
    "NO_INSTALL"
    ""
    "DEPENDS;INCLUDE_DIRS;LINK_LIBS;DEFINE"
    ${ARGN}
    )

  if (EXCLUDE_FROM_ALL)
    add_executable(${name} EXCLUDE_FROM_ALL ${ARG_UNPARSED_ARGUMENTS})
  else()
    add_executable(${name} ${ARG_UNPARSED_ARGUMENTS})
  endif()

  llvm_update_compile_flags(${name})

  if (ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()

  if (ARG_INCLUDE_DIRS)
    target_include_directories(${name} ${ARG_INCLUDE_DIRS})
  endif()

  if (ARG_LINK_LIBS)
    target_link_libraries(${name} ${ARG_LINK_LIBS})
  endif()

  if (NOT ARG_NO_INSTALL)
    install(TARGETS ${name} DESTINATION bin)
  endif()

  if (ARG_DEFINE)
    target_compile_definitions(${name} ${ARG_DEFINE})
  endif()
endfunction(add_core_dnn_executable)

