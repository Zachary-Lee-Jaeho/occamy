#include <iostream>

//===--------- Start of ONNXGemmOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXGemmOpToCUDNN : public ConversionPattern {
  ONNXGemmOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXGemmOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXGemmOpAdaptor operandAdaptor(operands);
    auto onnxGemmOp = dyn_cast<ONNXGemmOp>(op);

    auto inputA = operandAdaptor.getA();
    auto inputB = operandAdaptor.getB();
    auto inputC = operandAdaptor.getC();
    auto output = onnxGemmOp.getResult();

    auto alpha = onnxGemmOp.getAlpha().convertToFloat();
    auto beta = onnxGemmOp.getBeta().convertToFloat();
    auto transA = onnxGemmOp.getTransA();
    auto transB = onnxGemmOp.getTransB();

    auto inputAMemRef = convertToMemRefType(inputA.getType());
    auto inputBMemRef = convertToMemRefType(inputB.getType());
    auto inputCMemRef = convertToMemRefType(inputC.getType());
    auto outputYMemRef = convertToMemRefType(output.getType());

    auto inputAShape = inputAMemRef.getShape();
    auto inputBShape = inputBMemRef.getShape();
    auto outputYShape = outputYMemRef.getShape();

    // Compute broadcasting Dim for InputC
    SmallVector<int64_t> broadcastedDim;
    broadcastOnlyDimension(&broadcastedDim, outputYMemRef, inputCMemRef);

    int64_t numElements = 1;
    for (size_t i = 0; i < outputYShape.size(); i++)
      numElements *= outputYShape[i];
    int64_t sizeBytes = numElements *
      outputYMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    //-------------- Making CUDNNGemmOperation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto matmulMalloc = rewriter.create<CUDNNMallocOp>(loc, outputYMemRef, sizeConst);

    // create cudnnMatmul2d operator
    rewriter.create<CUDNNMatmul2dOp>(loc, outputYMemRef,
        inputA, rewriter.getI64ArrayAttr(inputAShape),
        inputB, rewriter.getI64ArrayAttr(inputBShape),
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape),
        rewriter.getF32FloatAttr(alpha),
        rewriter.getF32FloatAttr(beta),
        rewriter.getI64IntegerAttr(transA),
        rewriter.getI64IntegerAttr(transB));

    // create cudnnadd operator
    auto addOp = rewriter.create<CUDNNAddOp>(loc, outputYMemRef,
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape),
        // create cudnnadd operator
        inputC, rewriter.getI64ArrayAttr(broadcastedDim),
        FloatAttr::get(rewriter.getF32Type(), 1.f),
        matmulMalloc, rewriter.getI64ArrayAttr(outputYShape));

    // Insert dealloc.
    insertDealloc(matmulMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, matmulMalloc, loc, rewriter);
    if (!ret)
      ret = addOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXGemmOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXGemmOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXGemmOpToCUDNN -----------===//

