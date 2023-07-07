#include <iostream>

//===--------- Start of ONNXMatMulOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXMatMulOpToCUDNN : public ConversionPattern {
  ONNXMatMulOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXMatMulOpAdaptor operandAdaptor(operands);
    auto onnxMatMulOp = dyn_cast<ONNXMatMulOp>(op);

    auto inputA = operandAdaptor.getA();
    auto inputB = operandAdaptor.getB();
    auto output = onnxMatMulOp.getResult();

    auto inputAMemRef = convertToMemRefType(inputA.getType());
    auto inputBMemRef = convertToMemRefType(inputB.getType());
    auto outputYMemRef = convertToMemRefType(output.getType());
    auto outputYShape = outputYMemRef.getShape();

    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputYShape.size(); i++)
      numElements *= outputYShape[i];
    int64_t sizeBytes = numElements *
      outputYMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making CUDNNMatMulOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto resultMalloc = rewriter.create<CUDNNMallocOp>(loc, outputYMemRef, sizeConst);

    auto cudnnMatMul = rewriter.create<CUDNNMatmulNdOp>(loc, outputYMemRef,
        inputA, rewriter.getI64ArrayAttr(inputAMemRef.getShape()),
        inputB, rewriter.getI64ArrayAttr(inputBMemRef.getShape()),
        resultMalloc, rewriter.getI64ArrayAttr(outputYMemRef.getShape()));

    // Insert dealloc.
    insertDealloc(resultMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnMatMul.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXMatMulOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXMatMulOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXMatMulOpToCUDNN -----------===//

