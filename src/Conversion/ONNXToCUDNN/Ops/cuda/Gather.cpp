#include <iostream>

//===--------- Start of ONNXGatherOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXGatherOpToCUDNN : public ConversionPattern {
  ONNXGatherOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXGatherOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto gatherOp = dyn_cast<ONNXGatherOp>(op);
    ONNXGatherOpAdaptor operandAdaptor(operands);

    auto input = operandAdaptor.getData();
    auto indices = operandAdaptor.getIndices();
    auto axis = gatherOp.getAxis();

    auto inputMemRefType = convertToMemRefType(input.getType());
    auto indicesMemRefType = convertToMemRefType(indices.getType());
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());

    int inputRank = inputMemRefType.getShape().size();
    int indicesRank = indicesMemRefType.getShape().size();
    int outputRank = outMemRefType.getShape().size();

    auto shape = outMemRefType.getShape();
    int64_t numElements = 1;
    for (unsigned int i = 0; i < shape.size(); ++i)
      numElements *= shape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    //---------- Making CUDNNGather Operation ----------//
    ArrayAttr inputArray;
    ArrayAttr outputArray;
    ArrayAttr indicesArray;
    int64_t singleArr[] = {1};
    if(inputRank == 0) inputArray = rewriter.getI64ArrayAttr(singleArr);
    else inputArray = rewriter.getI64ArrayAttr(inputMemRefType.getShape());

    if(indicesRank == 0) indicesArray = rewriter.getI64ArrayAttr(singleArr);
    else indicesArray = rewriter.getI64ArrayAttr(indicesMemRefType.getShape());

    if(outputRank == 0) outputArray = rewriter.getI64ArrayAttr(singleArr);
    else outputArray = rewriter.getI64ArrayAttr(outMemRefType.getShape());

    //------------ Lowering Pattern ------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnGatherOp = rewriter.create<CUDNNGatherOp>(loc, outMemRefType,
        input, inputArray, indices, indicesArray, outMalloc, outputArray,
        rewriter.getI64IntegerAttr(axis));

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //---------- Lowering Pattern End ----------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnGatherOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXGatherOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXGatherOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXGatherOpToCUDNN -----------===//

