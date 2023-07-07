#pragma once
#include "core-dnn/Compiler/CDCompilerTypes.h"
#include "mlir/Pass/PassManager.h"

namespace core_dnn {

void addONNXToCUDNNPasses(mlir::PassManager &pm);
void addCUDNNToLLVMPasses(mlir::PassManager &pm);
void addONNXToNPUPasses(mlir::PassManager &pm);
void addNPUToLLVMPasses(mlir::PassManager &pm);

} // namespace core_dnn
