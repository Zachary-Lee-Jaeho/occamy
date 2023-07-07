//===--------- Start of ONNXSubOpToCUDNN ----------===//

#include <iostream>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

struct ONNXSubOpToCUDNN : public ConversionPattern {
  ONNXSubOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSubOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSubOpAdaptor operandAdaptor(operands);

    auto inputOperandA = operandAdaptor.getA();
    auto inputOperandB = operandAdaptor.getB();
    auto inputAMemRef = convertToMemRefType(inputOperandA.getType());
    auto inputBMemRef = convertToMemRefType(inputOperandB.getType());

    int inputARank = inputAMemRef.getShape().size();
    int inputBRank = inputBMemRef.getShape().size();

    auto resultMemRefType = convertToMemRefType(*op->result_type_begin());

    // Compute broadcasting Dim for Input
    SmallVector<int64_t> broadcastedDim;
    broadcastOnlyDimension(&broadcastedDim, inputAMemRef, inputBMemRef);

    unsigned int i;
    int64_t resultSize = 1, resultNumElement = 1;
    int64_t BSize = 1, BNumElement = 1;

    for (i=0;i<resultMemRefType.getShape().size();i++) {
      resultNumElement *= resultMemRefType.getShape()[i];
    }
    resultSize = resultNumElement *
      resultMemRefType.getElementType().getIntOrFloatBitWidth() / 8;

    for (i=0;i<inputBMemRef.getShape().size();i++) {
      BNumElement *= inputBMemRef.getShape()[i];
    }
    BSize = BNumElement *
      inputBMemRef.getElementType().getIntOrFloatBitWidth() / 8;

    auto I64Ty = rewriter.getIntegerType(64);

    //---------- Making CUDNNSub Operation ----------//
    auto sizeConst = emitConstantOp(rewriter,
        loc, I64Ty, resultSize);
    auto BsizeConst = emitConstantOp(rewriter,
        loc, I64Ty, BSize);

    auto negativeResultMalloc = rewriter.create<CUDNNMallocOp>(
        loc, inputBMemRef, BsizeConst);
    auto resultMalloc = rewriter.create<CUDNNMallocOp>(
        loc, resultMemRefType, sizeConst);

    ArrayAttr AarrayAttr;
    ArrayAttr BarrayAttr;
    if(inputARank > inputBRank) {
      AarrayAttr = rewriter.getI64ArrayAttr(inputAMemRef.getShape());
      BarrayAttr = rewriter.getI64ArrayAttr(broadcastedDim);
    } else if (inputARank < inputBRank) {
      AarrayAttr = rewriter.getI64ArrayAttr(broadcastedDim);
      BarrayAttr = rewriter.getI64ArrayAttr(inputBMemRef.getShape());
    } else {
      AarrayAttr = rewriter.getI64ArrayAttr(inputAMemRef.getShape());
      BarrayAttr = rewriter.getI64ArrayAttr(inputBMemRef.getShape());
    }

    // create cudnnNegative operator
    rewriter.create<CUDNNNegativeOp>(
        loc, resultMemRefType, inputOperandB, negativeResultMalloc, BarrayAttr);

    CUDNNAddOp cudnnAddOp;
    if(inputARank >= inputBRank) {
      cudnnAddOp = rewriter.create<CUDNNAddOp>(loc, resultMemRefType,
          inputOperandA, AarrayAttr,
          negativeResultMalloc, BarrayAttr,
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    } else if (inputARank < inputBRank) {
      cudnnAddOp = rewriter.create<CUDNNAddOp>(loc, resultMemRefType,
          negativeResultMalloc, BarrayAttr,
          inputOperandA, AarrayAttr,
          FloatAttr::get(rewriter.getF32Type(), 1.f),
          resultMalloc, rewriter.getI64ArrayAttr(resultMemRefType.getShape()));
    }

    // Insert dealloc.
    insertDealloc(negativeResultMalloc, loc, rewriter);
    insertDealloc(resultMalloc, loc, rewriter);

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, resultMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnAddOp.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSubOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSubOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXSubOpToCUDNN -----------===//
