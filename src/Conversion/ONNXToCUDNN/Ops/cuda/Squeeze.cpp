#include <iostream>

//===--------- Start of ONNXSqueezeOpToCUDNN ----------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Conversion/ONNXToCUDNN/ONNXToCUDNNCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace std;

struct ONNXSqueezeOpToCUDNN : public ConversionPattern {
  ONNXSqueezeOpToCUDNN(TypeConverter &typeConverter, MLIRContext *context)
    : ConversionPattern(mlir::ONNXSqueezeOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXSqueezeOpAdaptor operandAdaptor(operands);
    auto squeezeOp = dyn_cast<ONNXSqueezeOp>(op);

    auto input = operandAdaptor.getData();
    auto output = squeezeOp.getResult();

    auto inputMemRef = convertToMemRefType(input.getType());
    auto outputMemRef = convertToMemRefType(output.getType());

    auto axesValue = operandAdaptor.getAxes();

    ArrayAttr sqAxes;
    if(isa<mlir::ONNXConstantOp>(axesValue.getDefiningOp())) {
      auto axesConst = dyn_cast<ONNXConstantOp>(axesValue.getDefiningOp());
      if(onnx_mlir::isDenseONNXConstant(axesConst)) {
        sqAxes = onnx_mlir::createArrayAttrFromConstantOp(rewriter, axesConst);
      }
    } else {
      assert(0 && "ONNXUnsqeezeOp: Unsupported axes value");
    }

    auto outputShape = outputMemRef.getShape();
    auto outMemRefType = convertToMemRefType(*op->result_type_begin());
    int64_t numElements = 1;
    for (unsigned int i = 0; i < outputShape.size(); ++i)
      numElements *= outputShape[i];
    int64_t sizeBytes = numElements *
      outMemRefType.getElementType().getIntOrFloatBitWidth() / 8;
    //-------------- Making CUDNNSqueeze Operation --------------//

    //-------------------- Lowering Pattern --------------------//
    auto int64Ty = rewriter.getIntegerType(64);
    auto sizeConst = emitConstantOp(rewriter, loc, int64Ty,
        sizeBytes);
    auto outMalloc = rewriter.create<CUDNNMallocOp>(loc, outMemRefType, sizeConst);
    auto cudnnSqueeze = rewriter.create<CUDNNSqueezeOp>(loc, outputMemRef,
        input, rewriter.getI64ArrayAttr(inputMemRef.getShape()),
        outMalloc, rewriter.getI64ArrayAttr(outputMemRef.getShape()),
        sqAxes);

    // Insert dealloc.
    insertDealloc(outMalloc, loc, rewriter);
    //----------------- Lowering Pattern Ends ------------------//

    // Insert memcpy if this op is returned.
    Value ret = nullptr;
    if (checkInsertMemcpy(op))
      ret = insertMemcpyToHost(op, outMalloc, loc, rewriter);
    if (!ret)
      ret = cudnnSqueeze.getResult();

    rewriter.replaceOp(op, ret);

    return success();
  }
};

void populateLoweringONNXSqueezeOpToCUDNNPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter, MLIRContext *context) {
  patterns.insert<ONNXSqueezeOpToCUDNN>(typeConverter, context);
}
//===---------- End of ONNXSqueezeOpToCUDNN -----------===//

