#include <iostream>

//===--------- Start of ONNXConcatOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXConcatOpToCUDNN : public ConversionPattern {
  ONNXConcatOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXConcatOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXConcatOpAdaptor operandAdaptor(operands);
    ONNXConcatOp concatOp = dyn_cast<ONNXConcatOp>(op);

    auto input = operandAdaptor.getInputs();
    auto axis = concatOp.getAxis();

    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto shape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNConcat Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnConcatOp = rewriter.create<CUDNNConcatOp>(loc, outMemRefType,
        input, outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()),
        rewriter.getI64IntegerAttr(axis));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnConcatOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXConcatOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXConcatOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXConcatOpToCUDNN -----------===//

