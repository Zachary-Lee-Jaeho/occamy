#include <iostream>

//===--------- Start of ONNXPowOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXPowOpToCUDNN : public ConversionPattern {
  ONNXPowOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXPowOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto powOp = dyn_cast<ONNXPowOp>(op);
    ONNXPowOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getX();
    auto exponent = operandAdaptor.getY();

    auto exponentMemRefType = convertToMemRefType(exponent.getType());
    auto exponentRank = exponentMemRefType.getShape().size();
    assert(exponentRank == 0 && "Only supports a single number exponent not a tensor");
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto outShape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outShape.size(); ++i)
      numElements *= outShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNPow Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnPowOp = rewriter.create<CUDNNPowOp>(loc, outMemRefType,
        input, exponent, outMalloc,
        rewriter.getI64ArrayAttr(outMemRefType.getShape()));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnPowOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXPowOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXPowOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXPowOpToCUDNN -----------===//

