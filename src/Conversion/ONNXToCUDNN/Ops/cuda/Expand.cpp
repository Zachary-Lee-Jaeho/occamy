#include <iostream>

//===--------- Start of ONNXExpandOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXExpandOpToCUDNN : public ConversionPattern {
  ONNXExpandOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXExpandOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto expandOp = dyn_cast<ONNXExpandOp>(op);
    ONNXExpandOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getInput();
    auto shape = operandAdaptor.getShape();

    auto inputMemRefType = convertToMemRefType(input.getType());
    auto shapeMemRefType = convertToMemRefType(shape.getType());
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNExpand Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnExpandOp = rewriter.create<CUDNNExpandOp>(loc, outMemRefType,
        input, rewriter.getI64ArrayAttr(inputMemRefType.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()),
        shape, rewriter.getI64ArrayAttr(shapeMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnExpandOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXExpandOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXExpandOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXExpandOpToCUDNN -----------===//

