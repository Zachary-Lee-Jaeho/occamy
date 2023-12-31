#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CoreDNNOptions.hpp"
#include "src/Compiler/CoreDNNPasses.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Pass/CDPasses.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace core_dnn {

void addONNXToCUDNNPasses(mlir::PassManager &pm) {
  pm.addPass(core_dnn::createConvertONNXToCUDNNPass());
  pm.addNestedPass<func::FuncOp>(core_dnn::createFuncOpArgumentToCUDNNPass());
  pm.addPass(core_dnn::createeraseDummyConstantsPass());
}

void addCUDNNToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(core_dnn::createConvertCUDNNToLLVMPass());
}

} // namespace core_dnn
