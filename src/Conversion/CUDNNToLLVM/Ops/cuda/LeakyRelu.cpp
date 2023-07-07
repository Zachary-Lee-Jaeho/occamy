//===----------------------------------------------------------------------===//
// CUDNN to LLVM: CUDNNPLeakyReluOpLowering
//===----------------------------------------------------------------------===//

#include <iostream>
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "src/Conversion/CUDNNToLLVM/CUDNNRuntimeAPI.hpp"
#include "src/Conversion/CUDNNToLLVM/CUDNNToLLVMCommon.hpp"
#include "src/Dialect/CUDNN/CUDNNOps.hpp"

using namespace mlir;
using namespace core_dnn;

class CUDNNPLeakyReluOpLowering : public ConvertToLLVMPattern {
public:
  CUDNNPLeakyReluOpLowering(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(
          mlir::CUDNNLeakyReluOp::getOperationName(), ctx, typeConverter) {}
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    CUDNNLeakyReluOp leakyReluOp = dyn_cast<CUDNNLeakyReluOp>(op);

    auto *context = op->getContext();
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    mlir::Type inType = op->getOperand(0).getType();
    const auto &apiRegistry = core_dnn::CUDNNRuntimeAPIRegistry(module, rewriter, inType);

    auto int32Ty = IntegerType::get(context, 32);
    auto int64Ty = IntegerType::get(context, 64);
    auto int64PtrTy = LLVM::LLVMPointerType::get(int64Ty);
    auto int64ArrayTy = LLVM::LLVMArrayType::get(int64Ty, 4);
    mlir::Type floatTy = FloatType::getF32(context);
    if (inType.isF64())
      floatTy = FloatType::getF64(context);
    auto floatPtrTy = LLVM::LLVMPointerType::get(floatTy);

    auto input = leakyReluOp.getX();
    auto alpha = leakyReluOp.getAlpha();
    auto output = leakyReluOp.getY();
    auto outputDimAttr = leakyReluOp.getDimX();

    auto inputMemRefType = input.getType().cast<mlir::MemRefType>();
    auto elemType = typeConverter->convertType(inputMemRefType.getElementType());
    auto llvmElemType = typeConverter->convertType(elemType).cast<mlir::Type>();
    auto llvmPtrType = LLVM::LLVMPointerType::get(llvmElemType);
    auto inputMemRefShape = inputMemRefType.getShape();

    int inputRank = inputMemRefShape.size();

    // Insert unrealized conversion cast op to convert memref to llvm struct type.
    auto convertedInput = castToLLVMStruct(context, typeConverter, rewriter, loc, input);
    auto convertedOutput = castToLLVMStruct(context, typeConverter, rewriter, loc, output);

    // Load input and output
    auto extractInput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedInput,
        llvm::ArrayRef<int64_t>{0});
    auto extractOutput = rewriter.create<LLVM::ExtractValueOp>(loc, llvmPtrType, convertedOutput,
        llvm::ArrayRef<int64_t>{0});

    // Create integer array from shape attribute
    auto alphaConst = rewriter.create<LLVM::ConstantOp>(loc, floatTy,
        rewriter.getF32FloatAttr(alpha.convertToFloat()));
    auto rankConst = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
        rewriter.getI64IntegerAttr(inputRank));
    auto outputDim = rewriter.create<LLVM::AllocaOp>(
        loc, int64PtrTy, rankConst, 0);

    for (int i = 0; i < inputRank; i++) {
      auto offset = rewriter.create<LLVM::ConstantOp>(loc, int32Ty,
          rewriter.getI32IntegerAttr(i));

      auto outputDimI = rewriter.create<LLVM::ConstantOp>(loc, int64Ty,
          outputDimAttr[i].cast<IntegerAttr>());
      auto outputGep = rewriter.create<LLVM::GEPOp>(loc, int64PtrTy, outputDim,
          ArrayRef<Value>({offset}));
      rewriter.create<LLVM::StoreOp>(loc, outputDimI, outputGep);
    }

    // Call C coded library
    // TODO: Complete lowering without the library
    auto callLeakyRelu = core_dnn::CUDNNRuntimeAPI::callApi(rewriter, loc,
        apiRegistry, CUDNNRuntimeAPI::API::CUDA_LEAKYRELU,
        {extractInput, extractOutput, outputDim, alphaConst, rankConst});

    Value callLeakyReluOutput = insertAndReturnOutputShapeInfo(
        context, loc, typeConverter, rewriter, op->getResult(0), callLeakyRelu);

    rewriter.replaceOp(op, callLeakyReluOutput);

    return success();
  }
};

void mlir::populateCUDNNLeakyReluToLLVMConversionPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, LLVMTypeConverter &typeConverter) {
  patterns.insert<CUDNNPLeakyReluOpLowering>(ctx, typeConverter);
}
