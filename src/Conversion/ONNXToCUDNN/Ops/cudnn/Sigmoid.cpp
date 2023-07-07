//===--------- Start of ONNXSigmoidOpToCUDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

struct ONNXSigmoidOpToCUDNN : public ConversionPattern {
  ONNXSigmoidOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSigmoidOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSigmoidOpAdaptor operandAdaptor(operands);

    auto inputOperand = operandAdaptor.getX();
    MemRefType XMemRef;
    if (inputOperand.getType().isa<MemRefType>())
      XMemRef = inputOperand.getType().cast<MemRefType>();
    else if (auto wTensor = inputOperand.getType().dyn_cast<RankedTensorType>())
      XMemRef = convertToMemRefType(inputOperand.getType());
    else
      assert(0);

    auto resultMemRefType = convertToMemRefType(*op->result_type_begin());

    unsigned int i;
    int64_t resultNumElement = 1, resultSize = 1;

    //---------- Malloc input, kernel, and result ----------//

    for (i=0;i<resultMemRefType.getShape().size();i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    auto resultSizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);

    //---------- Making CUDNNActivation Operation ----------//

    auto malloc = rewriter.create<CUDNNMallocOp>(loc, resultMemRefType, resultSizeConst);
    auto cudnnActivateFwd =
      rewriter.create<CUDNNActivationForwardOp>(
          loc, resultMemRefType,
          // cudaMallocX.getResult(), rewriter.getI64ArrayAttr(XMemRef.getShape()),
          inputOperand, malloc, rewriter.getI64ArrayAttr(XMemRef.getShape()),
          /* SIGMOID = 0 */ rewriter.getI32IntegerAttr(0));

    insertDealloc(malloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, malloc, loc, rewriter);
    if (!ret)
      ret = cudnnActivateFwd.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSigmoidOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSigmoidOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXSigmoidOpToCUDNN -----------===//
