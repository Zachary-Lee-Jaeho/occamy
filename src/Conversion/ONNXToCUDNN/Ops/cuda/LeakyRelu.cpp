#include <iostream>

//===--------- Start of ONNXLeakyReluOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXLeakyReluOpToCUDNN : public ConversionPattern {
  ONNXLeakyReluOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXLeakyReluOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto leakyReluOp = dyn_cast<ONNXLeakyReluOp>(op);
    ONNXLeakyReluOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getX();
    auto alpha = leakyReluOp.getAlpha();

    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNLeakyRelu Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnLeakyReluOp = rewriter.create<CUDNNLeakyReluOp>(loc, outMemRefType,
        input, outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()), alpha);

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnLeakyReluOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXLeakyReluOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXLeakyReluOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXLeakyReluOpToCUDNN -----------===//

