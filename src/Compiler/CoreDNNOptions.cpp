#include "llvm/Support/Debug.h"

#include "ExternalUtil.hpp"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Compiler/CoreDNNOptions.hpp"

#define DEBUG_TYPE "corednn_options"

namespace core_dnn {
  // CoreDNN optimization options
  llvm::cl::OptionCategory CoreDnnOptions("CoreDNN optimization Options", "");

  // the option is used in this file, so defined here
  llvm::cl::opt<bool> onnxConstHoisting("onnx-const-hoisting",
      llvm::cl::desc("move all onnx constantOp to top of the funcOp"),
      llvm::cl::init(false), llvm::cl::cat(CoreDnnOptions));

  llvm::cl::opt<bool> onnxConstAtUse("onnx-const-at-use",
      llvm::cl::desc("move all onnx constantOp right before its first use"),
      llvm::cl::init(true), llvm::cl::cat(CoreDnnOptions));

  llvm::cl::opt<bool> cudnnKernelFusion("cudnn-kernel-fusion",
      llvm::cl::desc("cuDNN kernel fusing option (conv-bias-relu, conv-bias)"),
      llvm::cl::init(false), llvm::cl::cat(CoreDnnOptions));

  llvm::cl::opt<bool> cudnnDeallocOpt("cudnn-dealloc-opt",
      llvm::cl::desc("cuDNN constant liveness check optimization (fixing GPU OOM)."),
      llvm::cl::init(false), llvm::cl::cat(CoreDnnOptions));

  llvm::cl::opt<bool> mallocPoolOpt("malloc-pool-opt",
      llvm::cl::desc("forming the pool of memories for reducing mallocs."),
      llvm::cl::init(false), llvm::cl::cat(CoreDnnOptions));
}
