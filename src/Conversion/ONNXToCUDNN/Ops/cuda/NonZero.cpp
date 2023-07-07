#include <iostream>

//===--------- Start of ONNXNonZeroOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXNonZeroOpToCUDNN : public ConversionPattern {
  ONNXNonZeroOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXNonZeroOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto nonzeroOp = dyn_cast<ONNXNonZeroOp>(op);
    ONNXNonZeroOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getX();

    auto inputMemRefType = convertToMemRefType(input.getType());
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto shape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNNonZero Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnNonZeroOp = rewriter.create<CUDNNNonZeroOp>(loc, outMemRefType,
        input, rewriter.getI64ArrayAttr(inputMemRefType.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnNonZeroOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXNonZeroOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXNonZeroOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXNonZeroOpToCUDNN -----------===//

