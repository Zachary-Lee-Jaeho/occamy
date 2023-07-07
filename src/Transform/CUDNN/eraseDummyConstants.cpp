//===--------- Start of eraseDummyConstantsPass ----------===//

#include <iostream>
#include "mlir/Analysis/Liveness.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

/*!
 *  Pass that inserts memcpy to device for arguments.
 */
class eraseDummyConstantsPass
    : public PassWrapper<eraseDummyConstantsPass, OperationPass<ModuleOp>> {

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<Operation*, 1> opToErase;

    module.walk([&](CUDNNMallocOp op) {
      bool isDummy = true;
      for (Operation* user: op->getUsers()) {
        if (!isa<mlir::CUDNNMemcpyOp>(user) &&
           !isa<mlir::CUDNNDeallocOp>(user)) {
          isDummy = false;
        }
      }

      if (isDummy) {
        Operation* krnlConstOp;
        Operation* cpySizeOp;
        for (Operation* user: op->getUsers()) {
          if (auto memcpyOp = dyn_cast_or_null<mlir::CUDNNMemcpyOp>(user)) {
            krnlConstOp = memcpyOp.getSrc().getDefiningOp();
            cpySizeOp = memcpyOp.getCount().getDefiningOp();
            opToErase.emplace_back(user);
          } else {
            opToErase.emplace_back(user);
          }
        }
        opToErase.emplace_back(krnlConstOp);
        opToErase.emplace_back(op);
        opToErase.emplace_back(cpySizeOp);
      }
    });

    for (long unsigned int i=0; i<opToErase.size(); i++) {
      opToErase[i]->erase();
    }
  }
};

std::unique_ptr<Pass> core_dnn::createeraseDummyConstantsPass() {
  return std::make_unique<eraseDummyConstantsPass>();
}

