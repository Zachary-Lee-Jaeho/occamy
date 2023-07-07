#include <iostream>

//===--------- Start of ONNXTransposeOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXTransposeOpToCUDNN : public ConversionPattern {
  ONNXTransposeOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXTransposeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto transposeOp = dyn_cast<ONNXTransposeOp>(op);
    ONNXTransposeOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getData();
    auto perm = transposeOp.getPerm();

    auto inputMemRefType = convertToMemRefType(input.getType());
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    auto shape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNTranspose Operation ----------//

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnTransposeOp = rewriter.create<CUDNNTransposeOp>(loc, outMemRefType,
        input, rewriter.getI64ArrayAttr(inputMemRefType.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outMemRefType.getShape()),
        perm.value());

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnTransposeOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXTransposeOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXTransposeOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXTransposeOpToCUDNN -----------===//

