//===--------- Start of FuncOpArgumentToCUDNN ----------===//

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
class FuncOpArgumentToCUDNNPass
    : public PassWrapper<FuncOpArgumentToCUDNNPass, OperationPass<func::FuncOp>> {

public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getOperation());

    auto *context  = funcOp.getContext();
    auto loc = funcOp.getLoc();
    auto numArg = funcOp.getNumArguments();

    auto int32Ty = builder.getIntegerType(32);
    auto int64Ty = builder.getIntegerType(64);

    for (unsigned i = 0; i < numArg; i++) {
      auto arg = funcOp.getArgument(i);
      auto argTy = convertToMemRefType(arg.getType());
      auto shape = argTy.getShape();
      int64_t numElements = 1;
      for (int j = 0; j < shape.size(); ++j)
        numElements *= shape[j];

      auto size = builder.create<arith::ConstantOp>(loc, int64Ty,
          builder.getI64IntegerAttr(numElements *
            argTy.getElementType().getIntOrFloatBitWidth() / 8));
      auto malloc = builder.create<CUDNNMallocOp>(loc, argTy, size);

      arg.replaceAllUsesWith(malloc);

      auto memcpy = builder.create<CUDNNMemcpyOp>(loc, int32Ty,
          malloc.getResult(), arg, size, builder.getI32IntegerAttr(1));
      auto &parentBlock = funcOp.front();
      size.getOperation()->moveBefore(&parentBlock.front());
      malloc.getOperation()->moveAfter(size);
      memcpy.getOperation()->moveAfter(malloc);
      auto dealloc = builder.create<CUDNNDeallocOp>(loc, int32Ty, malloc);
      dealloc.getOperation()->moveBefore(&parentBlock.back());
    }

  }
};

std::unique_ptr<Pass> core_dnn::createFuncOpArgumentToCUDNNPass() {
  return std::make_unique<FuncOpArgumentToCUDNNPass>();
}
//===---------- End of ONNXConvOpToCUDNN -----------===//

