//===--------- Start of ONNXSqrtOpToCUDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

struct ONNXSqrtOpToCUDNN : public ConversionPattern {
  ONNXSqrtOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSqrtOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSqrtOpAdaptor operandAdaptor(operands);
    auto input = operandAdaptor.getX();
    auto resultMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t resultSize = 1, resultNumElement = 1;

    for (unsigned int i=0; i<resultMemRefType.getShape().size(); i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    //---------- Making CUDNNSqrt Operation ----------//

    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto resultMalloc = rewriter.create<CUDNNMallocOp>(
        loc, resultMemRefType, sizeConst);
    auto cudnnSqrt = rewriter.create<CUDNNSqrtOp>( loc, resultMemRefType,
        input, resultMalloc,
        rewriter.getI64ArrayAttr(resultMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnSqrt.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSqrtOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSqrtOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXSqrtOpToCUDNN -----------===//
