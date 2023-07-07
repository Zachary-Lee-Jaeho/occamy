#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace core_dnn {

/// Pass for lowering ONNX dialect to CUDNN dialect
std::unique_ptr<Pass> createConvertONNXToCUDNNPass();

/// Pass for lowering CUDNN dialect to LLVM dialect
std::unique_ptr<Pass> createConvertCUDNNToLLVMPass();

/// Pass for transforming CUDNN dialect
std::unique_ptr<Pass> createONNXConstantHoistingPass();
std::unique_ptr<Pass> createONNXConstantAtUsePass();
std::unique_ptr<Pass> createFuncOpArgumentToCUDNNPass();
std::unique_ptr<Pass> createCUDNNDeallocOptPass();
std::unique_ptr<Pass> createeraseDummyConstantsPass();
std::unique_ptr<Pass> createmallocPoolOptPass();
std::unique_ptr<Pass> createfuseConvBiasActivPass();
}

